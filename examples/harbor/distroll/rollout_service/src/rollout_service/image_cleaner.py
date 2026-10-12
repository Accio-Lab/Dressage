from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import shutil
import signal
import time
from typing import Any

from .config import load_inventory, worker_settings_from_env
from .image_cache import (
    AsyncFlock,
    ImageAdmissionInvariantError,
    ImageCache,
    ImagePullTransientError,
    ImageReferenceError,
    run_command,
)
from .task_resource_index import load_inventory_task_resource_index

logger = logging.getLogger(__name__)


class ImageCleaner:
    def __init__(self, cache: ImageCache) -> None:
        self.cache = cache
        self.settings = cache.settings
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            if self.settings.enabled and self.settings.cleaner_enabled:
                try:
                    result = await self.run_once()
                    logger.info("image cleaner pass result=%s", result)
                except Exception:
                    logger.exception("image cleaner pass failed")
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.settings.cleaner_scan_interval_sec
                )
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> dict[str, Any]:
        started = time.monotonic()
        usage = await asyncio.to_thread(shutil.disk_usage, self.settings.docker_root)
        used_percent = (usage.used / usage.total * 100.0) if usage.total else 100.0
        pressure = (
            used_percent >= self.settings.high_used_percent
            or usage.free < self.settings.min_free_bytes_before_pull
        )
        threshold = (
            self.settings.pressure_min_idle_sec
            if pressure
            else self.settings.idle_ttl_sec
        )
        now = time.time()
        candidates: list[tuple[float, str]] = []
        reconciled = 0
        for path in sorted(self.cache.records_dir.glob("*.json")):
            if time.monotonic() - started >= self.settings.max_scan_seconds:
                break
            digest_key = path.stem
            try:
                record = await self.cache.read_record(digest_key)
            except ImageAdmissionInvariantError as exc:
                logger.warning("skipping invalid image cache record %s: %s", path, exc)
                continue
            if record is None:
                continue
            if record.get("state") == "PULLING":
                if await self._reconcile_pulling(digest_key):
                    reconciled += 1
                record = await self.cache.read_record(digest_key)
                if record is None:
                    continue
            if record.get("state") != "READY" or not record.get("evictable"):
                continue
            if record.get("managed_origin") not in {
                "pulled_by_rollout_service",
                "imported_prewarm_proof",
            }:
                continue
            released = record.get("last_released_at")
            if (
                not isinstance(released, (int, float))
                or now - float(released) < threshold
            ):
                continue
            candidates.append((float(released), digest_key))

        deleted = 0
        ref_only = 0
        skipped = 0
        for _released, digest_key in sorted(candidates):
            if deleted >= self.settings.max_delete_per_scan:
                break
            if time.monotonic() - started >= self.settings.max_scan_seconds:
                break
            outcome = await self._evict(digest_key)
            if outcome == "deleted":
                deleted += 1
            elif outcome == "ref-only":
                ref_only += 1
            else:
                skipped += 1
            if pressure:
                usage = await asyncio.to_thread(
                    shutil.disk_usage, self.settings.docker_root
                )
                used_percent = (
                    usage.used / usage.total * 100.0 if usage.total else 100.0
                )
                if (
                    used_percent <= self.settings.low_used_percent
                    and usage.free >= self.settings.min_free_bytes_before_pull
                ):
                    break
        return {
            "pressure": pressure,
            "used_percent": round(used_percent, 3),
            "free_bytes": usage.free,
            "candidate_count": len(candidates),
            "deleted_count": deleted,
            "ref_only_count": ref_only,
            "skipped_count": skipped,
            "reconciled_count": reconciled,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    async def _reconcile_pulling(self, digest_key: str) -> bool:
        use_lock = AsyncFlock(self.cache._lock_path(digest_key, "use"), fcntl.LOCK_EX)
        if not await use_lock.acquire(wait=False):
            return False
        pull_lock = AsyncFlock(self.cache._lock_path(digest_key, "pull"), fcntl.LOCK_EX)
        try:
            if not await pull_lock.acquire(wait=False):
                return False
            record = await self.cache.read_record(digest_key)
            if record is None or record.get("state") != "PULLING":
                return False
            image_ref = str(record.get("image_ref") or "")
            try:
                _repository, digest, expected_key = self.cache.validate_execution_ref(
                    image_ref
                )
            except ImageReferenceError as exc:
                await self._quarantine_record(
                    digest_key,
                    record,
                    reason=f"invalid pulling execution image reference: {exc}",
                )
                return True
            if (
                expected_key != digest_key
                or str(record.get("image_digest") or "") != digest
            ):
                await self._quarantine_record(
                    digest_key,
                    record,
                    reason="pulling cache record key/digest disagrees with execution image",
                )
                return True
            inspected = await self.cache.inspect_image(image_ref)
            if inspected is None:
                record.update(
                    {
                        "state": "FAILED",
                        "active_attempt_id": None,
                        "last_error": "pull owner exited before image became ready",
                        "last_failed_at": time.time(),
                    }
                )
            else:
                record.update(
                    {
                        "state": "READY",
                        "image_id": inspected["image_id"],
                        "active_attempt_id": None,
                        "last_released_at": time.time(),
                        "last_outcome": "pull_reconciled_after_owner_exit",
                    }
                )
            await self.cache.write_record(digest_key, record)
            await self.cache.append_event(
                {
                    "event": "image_pull_reconciled",
                    "image_ref": image_ref,
                    "state": record["state"],
                }
            )
            return True
        finally:
            pull_lock.release()
            use_lock.release()

    async def _evict(self, digest_key: str) -> str:
        use_lock = AsyncFlock(self.cache._lock_path(digest_key, "use"), fcntl.LOCK_EX)
        if not await use_lock.acquire(wait=False):
            return "active"
        pull_lock = AsyncFlock(self.cache._lock_path(digest_key, "pull"), fcntl.LOCK_EX)
        try:
            if not await pull_lock.acquire(wait=False):
                return "pulling"
            record = await self.cache.read_record(digest_key)
            if record is None or not self._record_is_evictable(record):
                return "record-changed"
            image_ref = str(record.get("image_ref") or "")
            try:
                repository, digest, expected_key = self.cache.validate_execution_ref(
                    image_ref
                )
            except ImageReferenceError as exc:
                await self._quarantine_record(
                    digest_key,
                    record,
                    reason=f"invalid execution image reference: {exc}",
                )
                return "invalid-image-ref"
            if (
                expected_key != digest_key
                or str(record.get("image_digest") or "") != digest
            ):
                await self._quarantine_record(
                    digest_key,
                    record,
                    reason="cache record key/digest disagrees with execution image",
                )
                return "record-identity-mismatch"
            inspected = await self.cache.inspect_image(image_ref)
            image_id = str(record.get("image_id") or "")
            if inspected is not None and inspected["image_id"] != image_id:
                record.update(
                    {
                        "state": "QUARANTINED",
                        "evictable": False,
                        "last_error": "managed image ID changed before eviction",
                    }
                )
                await self.cache.write_record(digest_key, record)
                return "image-id-changed"
            managed_tags = [str(tag) for tag in record.get("managed_tags") or []]
            existing_managed_tags: list[str] = []
            for tag in managed_tags:
                if not tag.startswith(repository + ":") or "@" in tag:
                    record.update(
                        {
                            "state": "QUARANTINED",
                            "evictable": False,
                            "last_error": f"invalid managed tag before eviction: {tag}",
                        }
                    )
                    await self.cache.write_record(digest_key, record)
                    return "invalid-managed-tag"
                tag_image_id = await self._reference_image_id(tag)
                if tag_image_id is None:
                    continue
                if tag_image_id != image_id:
                    record.update(
                        {
                            "state": "QUARANTINED",
                            "evictable": False,
                            "last_error": f"managed tag image ID changed: {tag}",
                        }
                    )
                    await self.cache.write_record(digest_key, record)
                    return "managed-tag-id-changed"
                existing_managed_tags.append(tag)
            if inspected is None and not existing_managed_tags:
                if image_id and await self._image_id_exists(image_id):
                    await self._mark_ref_only(
                        digest_key, record, reason="managed-refs-already-absent"
                    )
                    return "ref-only"
                await self._mark_evicted(digest_key, record, reason="already-absent")
                return "deleted"
            containers = await run_command(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "--filter",
                    f"ancestor={image_id}",
                ],
                timeout_sec=30.0,
            )
            if containers.returncode != 0 or containers.stdout.strip():
                return "container-reference"
            for selected_ref in [
                *existing_managed_tags,
                *([image_ref] if inspected else []),
            ]:
                removed = await run_command(
                    ["docker", "image", "rm", selected_ref], timeout_sec=120.0
                )
                if removed.returncode != 0:
                    await self.cache.append_event(
                        {
                            "event": "image_evict_failed",
                            "image_ref": image_ref,
                            "selected_ref": selected_ref,
                            "stderr": removed.stderr[-1000:],
                        }
                    )
                    return "remove-failed"
            if await self._image_id_exists(image_id):
                await self._mark_ref_only(
                    digest_key, record, reason="other-image-reference-remains"
                )
                return "ref-only"
            await self._mark_evicted(digest_key, record, reason="cleaner")
            return "deleted"
        finally:
            pull_lock.release()
            use_lock.release()

    @staticmethod
    def _record_is_evictable(record: dict[str, Any]) -> bool:
        return bool(
            record.get("state") == "READY"
            and record.get("evictable") is True
            and record.get("managed_origin")
            in {"pulled_by_rollout_service", "imported_prewarm_proof"}
        )

    async def _quarantine_record(
        self, digest_key: str, record: dict[str, Any], *, reason: str
    ) -> None:
        record.update(
            {
                "state": "QUARANTINED",
                "evictable": False,
                "active_attempt_id": None,
                "last_error": reason,
                "quarantined_at": time.time(),
            }
        )
        await self.cache.write_record(digest_key, record)
        await self.cache.append_event(
            {
                "event": "image_cache_record_quarantined",
                "image_ref": record.get("image_ref"),
                "reason": reason,
            }
        )

    async def _mark_evicted(
        self, digest_key: str, record: dict[str, Any], *, reason: str
    ) -> None:
        record.update(
            {
                "state": "EVICTED",
                "image_id": None,
                "managed_tags": [],
                "active_attempt_id": None,
                "evicted_at": time.time(),
                "eviction_reason": reason,
            }
        )
        await self.cache.write_record(digest_key, record)
        await self.cache.append_event(
            {
                "event": "image_evicted",
                "image_ref": record.get("image_ref"),
                "reason": reason,
            }
        )

    async def _mark_ref_only(
        self, digest_key: str, record: dict[str, Any], *, reason: str
    ) -> None:
        record.update(
            {
                "state": "EVICTED_REF_ONLY",
                "active_attempt_id": None,
                "evictable": False,
                "evicted_at": time.time(),
                "eviction_reason": reason,
            }
        )
        await self.cache.write_record(digest_key, record)
        await self.cache.append_event(
            {
                "event": "image_ref_evicted_only",
                "image_ref": record.get("image_ref"),
                "image_id": record.get("image_id"),
                "reason": reason,
            }
        )

    @staticmethod
    async def _image_id_exists(image_id: str) -> bool:
        result = await run_command(
            ["docker", "image", "inspect", image_id], timeout_sec=30.0
        )
        if result.returncode == 0:
            return True
        detail = (result.stderr + result.stdout).lower()
        if "no such image" in detail or "no such object" in detail:
            return False
        raise ImagePullTransientError(
            f"cannot verify physical image deletion for {image_id}: "
            f"{result.stderr[-1000:]}"
        )

    @staticmethod
    async def _reference_image_id(image_ref: str) -> str | None:
        result = await run_command(
            ["docker", "image", "inspect", image_ref, "--format", "{{.Id}}"],
            timeout_sec=30.0,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        detail = (result.stderr + result.stdout).lower()
        if "no such image" in detail or "no such object" in detail:
            return None
        raise ImagePullTransientError(
            f"cannot inspect managed image tag {image_ref}: {result.stderr[-1000:]}"
        )


def _load_cache() -> ImageCache:
    worker_settings = worker_settings_from_env()
    inventory = load_inventory(worker_settings.inventory_path)
    node = inventory.worker(worker_settings.worker_id)
    index = load_inventory_task_resource_index(inventory)
    return ImageCache(node.image_cache, resource_index=index, worker_id=node.worker_id)


def image_cleaner_main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run managed task-image cache cleanup")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    async def run() -> None:
        cleaner = ImageCleaner(_load_cache())
        if args.once:
            print(json.dumps(await cleaner.run_once(), indent=2, sort_keys=True))
            return
        loop = asyncio.get_running_loop()
        for selected_signal in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(selected_signal, cleaner.stop)
        await cleaner.run_forever()

    asyncio.run(run())


__all__ = [
    "ImageCleaner",
    "image_cleaner_main",
]
