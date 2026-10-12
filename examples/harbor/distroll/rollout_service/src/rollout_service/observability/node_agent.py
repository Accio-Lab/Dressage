from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from rollout_service.config import load_inventory

from .docker_snapshot import (
    aggregate_image_cache,
    aggregate_managed_containers,
    local_image_refs,
    parse_json_lines,
)
from .procfs import (
    cpu_used_percent,
    read_cpu_times,
    read_load,
    read_memory,
    read_pressure,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run(args: list[str], *, timeout: float = 2.0) -> str:
    completed = subprocess.run(
        args,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-500:]
        raise RuntimeError(f"{args[0]} returned {completed.returncode}: {detail}")
    return completed.stdout


def _service_states(names: list[str]) -> dict[str, str]:
    states: dict[str, str] = {}
    for name in names:
        try:
            value = _run(["systemctl", "is-active", name], timeout=1.0).strip()
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            value = "unknown"
        states[name] = value or "unknown"
    return states


def _filesystem_snapshot(paths: dict[str, Path]) -> list[dict[str, Any]]:
    by_device: dict[int, dict[str, Any]] = {}
    for purpose, path in paths.items():
        try:
            selected = path.expanduser().resolve()
            device = selected.stat().st_dev
            usage = shutil.disk_usage(selected)
            statvfs = os.statvfs(selected)
        except OSError:
            continue
        entry = by_device.setdefault(
            device,
            {
                "device_id": device,
                "paths": [],
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_percent": round(
                    usage.used / usage.total * 100.0 if usage.total else 100.0,
                    3,
                ),
                "inodes_total": statvfs.f_files,
                "inodes_free": statvfs.f_favail,
                "inodes_free_percent": round(
                    statvfs.f_favail / statvfs.f_files * 100.0
                    if statvfs.f_files
                    else 0.0,
                    3,
                ),
            },
        )
        entry["paths"].append({"purpose": purpose, "path": str(selected)})
    return list(by_device.values())


class NodeSampler:
    def __init__(self) -> None:
        inventory = load_inventory(os.environ["ROLLOUT_INVENTORY"])
        self.worker_id = os.environ["ROLLOUT_WORKER_ID"]
        self.node = inventory.worker(self.worker_id)
        self.interval = float(os.environ.get("ROLLOUT_OBSERVER_INTERVAL_SEC", "2"))
        self.worker_url = f"http://{self.node.address}"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(1.0), trust_env=False)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._snapshot: dict[str, Any] | None = None
        self._previous_cpu: tuple[int, ...] | None = None
        self._docker: dict[str, Any] = {}
        self._images: dict[str, Any] = {}
        self._docker_sampled_at: str | None = None
        self._images_sampled_at: str | None = None
        self._container_error: str | None = None
        self._image_error: str | None = None
        self._filesystems: list[dict[str, Any]] = []
        self._services: dict[str, str] = {}
        self._cycle = 0

    @property
    def snapshot(self) -> dict[str, Any] | None:
        return self._snapshot

    async def start(self) -> None:
        await self._refresh()
        self._task = asyncio.create_task(self._loop(), name="node-observer-sampler")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._client.aclose()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except TimeoutError:
                pass
            if self._stop.is_set():
                return
            await self._refresh()

    async def _worker_snapshot(self) -> tuple[dict[str, Any], str | None]:
        payload: dict[str, Any] = {"healthy": False, "ready": False}
        errors: list[str] = []
        for endpoint, key in (("healthz", "healthy"), ("readyz", "ready")):
            try:
                response = await self._client.get(f"{self.worker_url}/{endpoint}")
                data = response.json()
                if isinstance(data, dict):
                    payload.update(data)
                payload[key] = response.status_code == 200
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                errors.append(f"{endpoint}:{type(exc).__name__}")
        return payload, ",".join(errors) or None

    async def _refresh_containers(self) -> None:
        try:
            containers_text = await asyncio.to_thread(
                _run,
                [
                    "docker",
                    "container",
                    "ls",
                    "--no-trunc",
                    "--format",
                    "{{json .}}",
                ],
                timeout=5.0,
            )
            container_rows = parse_json_lines(containers_text)
            self._docker = aggregate_managed_containers(
                container_rows, worker_id=self.worker_id
            )
            self._docker["all_running_containers"] = len(container_rows)
            self._docker["enumeration_scope"] = "running"
            self._docker_sampled_at = _utc_now()
            self._container_error = None
        except (
            json.JSONDecodeError,
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            self._container_error = f"{type(exc).__name__}: {str(exc)[:300]}"

    async def _refresh_images(self) -> None:
        try:
            images_text = await asyncio.to_thread(
                _run,
                [
                    "docker",
                    "image",
                    "ls",
                    "--digests",
                    "--no-trunc",
                    "--format",
                    "{{json .}}",
                ],
                timeout=10.0,
            )
            self._images = aggregate_image_cache(
                self.node.image_cache.cache_root.expanduser() / "records",
                local_image_refs(parse_json_lines(images_text)),
            )
            self._images_sampled_at = _utc_now()
            self._image_error = None
        except (
            json.JSONDecodeError,
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            self._image_error = f"{type(exc).__name__}: {str(exc)[:300]}"

    @staticmethod
    def _outbox_snapshot(state_dir: Path) -> dict[str, int]:
        now = time.time()
        pending = list((state_dir / "completion-outbox").glob("*.json"))
        rejected = list((state_dir / "completion-rejected").glob("*.json"))
        rejected_recent = 0
        for path in rejected:
            try:
                if now - path.stat().st_mtime <= 900:
                    rejected_recent += 1
            except OSError:
                continue
        return {
            "pending": len(pending),
            "rejected": len(rejected),
            "rejected_recent_15m": rejected_recent,
        }

    async def _refresh(self) -> None:
        self._cycle += 1
        sampled_at = _utc_now()
        source_errors: dict[str, str] = {}
        try:
            current_cpu = read_cpu_times()
            used = (
                None
                if self._previous_cpu is None
                else cpu_used_percent(self._previous_cpu, current_cpu)
            )
            self._previous_cpu = current_cpu
            cpu = {**read_load(), "used_percent": used}
        except (OSError, ValueError) as exc:
            cpu = {}
            source_errors["cpu"] = f"{type(exc).__name__}: {exc}"
        try:
            memory = read_memory()
        except (OSError, ValueError) as exc:
            memory = {}
            source_errors["memory"] = f"{type(exc).__name__}: {exc}"
        try:
            pressure = read_pressure()
        except (OSError, ValueError) as exc:
            pressure = {}
            source_errors["pressure"] = f"{type(exc).__name__}: {exc}"
        worker, worker_error = await self._worker_snapshot()
        if worker_error:
            source_errors["worker"] = worker_error
        if self._cycle == 1 or self._cycle % 3 == 0:
            await self._refresh_containers()
        if self._cycle == 1 or self._cycle % 15 == 0:
            await self._refresh_images()
        if self._container_error:
            source_errors["docker_containers"] = self._container_error
        if self._image_error:
            source_errors["docker_images"] = self._image_error

        filesystem_paths = {
            "docker_root": self.node.image_cache.docker_root,
            "worker_state": self.node.state_dir,
        }
        if registry_root := os.environ.get("DRESSAGE_REGISTRY_DATA_ROOT"):
            filesystem_paths["registry"] = Path(registry_root)
        if self._cycle == 1 or self._cycle % 3 == 0:
            self._filesystems = await asyncio.to_thread(
                _filesystem_snapshot, filesystem_paths
            )
        outbox = self._outbox_snapshot(self.node.state_dir)
        if self._cycle == 1 or self._cycle % 5 == 0:
            self._services = await asyncio.to_thread(
                _service_states,
                [
                    "docker.service",
                    "dressage-distroll-worker.service",
                    "dressage-distroll-image-cleaner.service",
                    *(
                        [
                            "dressage-distroll-control.service",
                            "dressage-distroll-network-gateway.service",
                        ]
                        if self.worker_id == "cpu-01"
                        else []
                    ),
                ],
            )
        self._snapshot = {
            "schema_version": "rollout-node-snapshot/v1",
            "worker_id": self.worker_id,
            "sampled_at": sampled_at,
            "worker": worker,
            "cpu": cpu,
            "memory": memory,
            "pressure": pressure,
            "filesystems": self._filesystems,
            "docker": {
                **self._docker,
                "sampled_at": self._docker_sampled_at,
                "stale": self._container_error is not None,
            },
            "images": {
                **self._images,
                "sampled_at": self._images_sampled_at,
                "stale": self._image_error is not None,
            },
            "outbox": outbox,
            "services": self._services,
            "sources": {
                "errors": source_errors,
                "fresh": not source_errors,
            },
        }


def create_node_observer_app(sampler: NodeSampler | None = None) -> FastAPI:
    selected = sampler or NodeSampler()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await selected.start()
        app.state.sampler = selected
        try:
            yield
        finally:
            await selected.close()

    app = FastAPI(title="Rollout Node Observer", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = selected.snapshot is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not-ready"},
        )

    @app.get("/api/v1/snapshot")
    async def snapshot() -> JSONResponse:
        payload = selected.snapshot
        return JSONResponse(
            status_code=200 if payload is not None else 503,
            content=payload or {"error": "snapshot-not-ready"},
        )

    return app


def main() -> None:
    host = os.environ.get("ROLLOUT_OBSERVER_HOST")
    if not host:
        inventory = load_inventory(os.environ["ROLLOUT_INVENTORY"])
        node = inventory.worker(os.environ["ROLLOUT_WORKER_ID"])
        host = urlsplit(f"http://{node.address}").hostname
    if not host:
        raise SystemExit("cannot resolve observer listen host")
    port = int(os.environ.get("ROLLOUT_OBSERVER_PORT", "19191"))
    uvicorn.run(create_node_observer_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
