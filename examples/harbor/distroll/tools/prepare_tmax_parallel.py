#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from harbor.models.task.config import TaskConfig
from rollout_service.config import TaskResourceDefaults
from rollout_service.task_resource_index import (
    build_task_resource_index,
    write_task_resource_index,
)

try:
    from tools.resolve_harbor_task_source import resolve as resolve_source_image
    from tools.rewrite_tmax_task import rewrite
except ModuleNotFoundError:  # direct execution adds tools/, not project root
    from resolve_harbor_task_source import resolve as resolve_source_image
    from rewrite_tmax_task import rewrite


VALIDATION_PROFILE = "trusted-security"
TASK_STATES = (
    "PENDING",
    "WAITING_SOURCE",
    "BUILDING",
    "PUSHED",
    "PUBLISHED",
    "FAILED",
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)
_PRINT_LOCK = threading.Lock()


def _print(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _run(
    command: list[str],
    *,
    log_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed rc={result.returncode}; see {log_path}: {command[0]}"
        )
    return result


def read_task_ids(path: Path) -> list[str]:
    task_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    invalid = [
        task_id for task_id in task_ids if not _SAFE_COMPONENT.fullmatch(task_id)
    ]
    if invalid:
        raise ValueError(f"unsafe TMAX task IDs: {invalid[:5]}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("TMAX task list contains duplicate IDs")
    if not task_ids:
        raise ValueError("TMAX task list is empty")
    return task_ids


class TMAXParallelPreparer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = args.project_root.resolve()
        self.task_root = args.task_root.resolve()
        self.output_root = args.output_root.resolve()
        self.work_root = args.work_root.resolve()
        self.state_root = self.work_root / "state"
        self.log_root = self.work_root / "logs"
        self.catalog_root = self.output_root / "tasks" / args.namespace
        self.builder_config = self.work_root / "buildkitd.toml"
        self.runtime_tag = (
            f"{args.claude_runtime_repository}:claude-code-"
            f"{args.claude_code_version}-{args.release_id}"
        )
        self.runtime_image = ""
        self.builder_names: list[str] = []
        self.source_mirror_builder = f"tmax-source-mirror-{args.release_id}"
        self.source_mirror_builder_created = False
        self._source_mirror_lock = threading.Lock()
        self._next_source_mirror_at = 0.0
        self._result_lock = threading.Lock()

    def prepare(self) -> int:
        task_ids = read_task_ids(self.args.task_list)
        if len(task_ids) != self.args.expected_task_count:
            raise ValueError(
                "frozen TMAX task count mismatch: "
                f"expected={self.args.expected_task_count} actual={len(task_ids)}"
            )
        for path in (
            self.output_root,
            self.catalog_root.parent,
            self.state_root,
            self.log_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not self.args.resume:
            for name in (
                "fixed-tasks.json",
                "admission.json",
                "tmax1500.jsonl",
                "task-resource-index-v2.json",
            ):
                if (self.output_root / name).exists():
                    raise FileExistsError(
                        f"refusing to overwrite published output: {self.output_root / name}"
                    )
        self._validate_all_inputs(task_ids)
        self._write_or_verify_run_config()
        self._write_builder_config()
        for task_id in task_ids:
            state_path = self._state_path(task_id)
            if not state_path.exists():
                _atomic_json(state_path, {"state": "PENDING", "task_id": task_id})

        started = time.time()
        records: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        try:
            self.runtime_image = self._build_runtime()
            self._create_builders()
            work: queue.Queue[str] = queue.Queue()
            for task_id in task_ids:
                work.put(task_id)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.jobs
            ) as executor:
                futures = [
                    executor.submit(
                        self._run_slot, slot, builder, work, records, failures
                    )
                    for slot, builder in enumerate(self.builder_names)
                ]
                for future in futures:
                    future.result()
        finally:
            failures.extend(self._remove_builders())

        order = {task_id: position for position, task_id in enumerate(task_ids)}
        records.sort(key=lambda record: order[str(record["task_id"])])
        try:
            local_tags = self._local_release_tags()
        except Exception as exc:  # noqa: BLE001 - preserve the batch summary
            failures.append(
                {
                    "task_id": "<local-image-cleanliness>",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            if local_tags:
                failures.append(
                    {
                        "task_id": "<local-image-cleanliness>",
                        "error": (
                            "push-only publication unexpectedly left local release tags: "
                            + ", ".join(local_tags[:5])
                        ),
                    }
                )
        if not failures and len(records) == len(task_ids):
            try:
                self._publish_manifests(records)
            except Exception as exc:  # noqa: BLE001 - preserve a complete summary
                failures.append(
                    {
                        "task_id": "<manifest-publication>",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        self._write_attempt_report(task_ids, records, failures)
        summary = self._summary(task_ids, records, failures, started)
        _atomic_json(self.work_root / "summary.json", summary)
        _print(json.dumps(summary, indent=2, sort_keys=True))
        return 1 if failures or len(records) != len(task_ids) else 0

    def _run_slot(
        self,
        slot: int,
        builder: str,
        work: queue.Queue[str],
        records: list[dict[str, Any]],
        failures: list[dict[str, str]],
    ) -> None:
        while True:
            try:
                task_id = work.get_nowait()
            except queue.Empty:
                return
            try:
                for attempt in range(1, self.args.task_attempts + 1):
                    record: dict[str, Any] | None = None
                    error: BaseException | None = None
                    state_before = str(
                        self._read_state(self._state_path(task_id)).get("state") or ""
                    )
                    try:
                        record = self._prepare_task(
                            task_id,
                            builder=builder,
                            recover_existing=self.args.resume or attempt > 1,
                        )
                    except Exception as exc:  # noqa: BLE001 - task retry boundary
                        error = exc
                    if not (
                        self.args.resume
                        and state_before == "PUBLISHED"
                        and record is not None
                    ):
                        try:
                            self._recycle_builder(builder)
                        except Exception as exc:  # noqa: BLE001 - disk-safety boundary
                            with self._result_lock:
                                failures.append(
                                    {
                                        "task_id": f"<builder-recycle-slot-{slot}>",
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                            _print(f"[slot={slot}] BUILDER RECYCLE FAILED: {exc}")
                            return
                    if record is not None:
                        with self._result_lock:
                            records.append(record)
                        _print(
                            f"[slot={slot} task={task_id} attempt={attempt}] PUBLISHED"
                        )
                        break
                    assert error is not None
                    if attempt < self.args.task_attempts:
                        _print(
                            f"[slot={slot} task={task_id} attempt={attempt}] "
                            f"RETRY: {type(error).__name__}: {error}"
                        )
                        time.sleep(self.args.task_retry_backoff_sec)
                        continue
                    failure = {
                        "task_id": task_id,
                        "attempts": str(attempt),
                        "error": f"{type(error).__name__}: {error}",
                    }
                    current = self._read_state(self._state_path(task_id))
                    failed_state: dict[str, Any] = {
                        "state": "FAILED",
                        **failure,
                    }
                    if isinstance(current.get("record"), dict):
                        failed_state["record"] = current["record"]
                        failed_state["last_successful_state"] = current.get("state")
                    _atomic_json(self._state_path(task_id), failed_state)
                    with self._result_lock:
                        failures.append(failure)
                    _print(
                        f"[slot={slot} task={task_id} attempts={attempt}] "
                        f"SKIPPED_AFTER_FAILURE: {error}"
                    )
            finally:
                work.task_done()

    def _prepare_task(
        self,
        task_id: str,
        *,
        builder: str,
        recover_existing: bool = False,
    ) -> dict[str, Any]:
        self._wait_for_headroom(task_id)
        raw_task = self.task_root / task_id
        admitted_task = self.catalog_root / task_id
        state_path = self._state_path(task_id)
        task_log_root = self.log_root / task_id
        source_image = resolve_source_image(raw_task)
        if not source_image:
            raise ValueError(f"TMAX task has no declared source image: {task_id}")
        final_tag = (
            f"{self.args.output_repository}:{task_id}-claude-code-"
            f"{self.args.claude_code_version}-{self.args.release_id}"
        )
        task_log_root.mkdir(parents=True, exist_ok=True)
        metadata_path = task_log_root / "build-metadata.json"
        previous = self._read_state(state_path)
        if self.args.resume or recover_existing:
            resumed = self._resume_task(
                raw_task=raw_task,
                admitted_task=admitted_task,
                state=previous,
            )
            if resumed is not None:
                return resumed
        _atomic_json(
            state_path,
            {
                "state": "WAITING_SOURCE",
                "task_id": task_id,
                "source_image": source_image,
            },
        )
        mirrored_source, source_digest = self._ensure_source_mirror(
            task_id, source_image, task_log_root
        )
        existing_digest = self._registry_digest(final_tag)
        if (
            existing_digest is not None
            and (self.args.resume or recover_existing)
            and metadata_path.is_file()
        ):
            metadata_digest = self._digest_from_build_metadata(metadata_path)
            if metadata_digest != existing_digest:
                raise RuntimeError(
                    f"resume build metadata disagrees with Registry: "
                    f"metadata={metadata_digest} Registry={existing_digest}"
                )
            recovered = self._record(
                task_id=task_id,
                source_image=source_image,
                source_digest=source_digest,
                source_mirror_image=mirrored_source,
                final_tag=final_tag,
                digest=existing_digest,
            )
            _atomic_json(
                state_path,
                {"state": "PUSHED", "task_id": task_id, "record": recovered},
            )
            resumed = self._resume_task(
                raw_task=raw_task,
                admitted_task=admitted_task,
                state={"state": "PUSHED", "record": recovered},
            )
            if resumed is None:
                raise RuntimeError("cannot recover pushed TMAX task")
            return resumed
        if existing_digest is not None:
            raise RuntimeError(
                f"refusing to overwrite Registry tag without owned state: {final_tag}"
            )
        if admitted_task.exists():
            raise FileExistsError(
                f"refusing to overwrite admitted TMAX task: {admitted_task}"
            )

        _atomic_json(
            state_path,
            {
                "state": "BUILDING",
                "task_id": task_id,
                "source_image": source_image,
                "source_image_digest": source_digest,
                "admitted_image": final_tag,
            },
        )
        record: dict[str, Any] | None = None
        try:
            metadata_path.unlink(missing_ok=True)
            _run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--builder",
                    builder,
                    "--progress=plain",
                    "--provenance=false",
                    "--pull=false",
                    "--build-arg",
                    f"CLAUDE_RUNTIME_IMAGE={self.runtime_image}",
                    "--build-arg",
                    f"TMAX_SOURCE_IMAGE={mirrored_source}",
                    "--build-arg",
                    f"CLAUDE_CODE_VERSION={self.args.claude_code_version}",
                    "--metadata-file",
                    str(metadata_path),
                    "--tag",
                    final_tag,
                    "--push",
                    "--file",
                    str(
                        self.project_root
                        / "assets/tmax-claude-preinstalled-overlay.Dockerfile"
                    ),
                    str(self.project_root / "assets"),
                ],
                log_path=task_log_root / "build-push.log",
            )
            digest = self._digest_from_build_metadata(metadata_path)
            self._verify_build_material(metadata_path, mirrored_source)
            actual_digest = self._registry_digest(final_tag)
            if actual_digest != digest:
                raise RuntimeError(
                    f"Registry digest disagrees with build metadata: "
                    f"metadata={digest} Registry={actual_digest}"
                )
            record = self._record(
                task_id=task_id,
                source_image=source_image,
                source_digest=source_digest,
                source_mirror_image=mirrored_source,
                final_tag=final_tag,
                digest=digest,
            )
            _atomic_json(
                state_path,
                {"state": "PUSHED", "task_id": task_id, "record": record},
            )
            self._rewrite_task_atomic(
                raw_task,
                admitted_task,
                admitted_image=final_tag,
                source_image=source_image,
                source_image_digest=source_digest,
            )
            _atomic_json(
                state_path,
                {"state": "PUBLISHED", "task_id": task_id, "record": record},
            )
            return record
        except Exception:
            if record is not None:
                _atomic_json(
                    state_path,
                    {
                        "state": "FAILED",
                        "task_id": task_id,
                        "last_successful_state": "PUSHED",
                        "record": record,
                    },
                )
            raise

    def _record(
        self,
        *,
        task_id: str,
        source_image: str,
        source_digest: str,
        source_mirror_image: str,
        final_tag: str,
        digest: str,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "task_ref": f"{self.args.namespace}/{task_id}",
            "source_image": source_image,
            "source_image_digest": source_digest,
            "source_mirror_image": source_mirror_image,
            "admitted_image": final_tag,
            "execution_image": f"{self.args.output_repository}@{digest}",
            "claude_runtime_image": self.runtime_image,
            "claude_code_version": self.args.claude_code_version,
        }

    def _resume_task(
        self,
        *,
        raw_task: Path,
        admitted_task: Path,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        record = state.get("record")
        if not isinstance(record, dict):
            return None
        if state.get("state") not in {"PUSHED", "PUBLISHED", "FAILED"}:
            return None
        final_tag = str(record.get("admitted_image") or "")
        execution_image = str(record.get("execution_image") or "")
        expected_digest = execution_image.rpartition("@")[2]
        if not final_tag or not _DIGEST.fullmatch(expected_digest):
            return None
        actual_digest = self._registry_digest(final_tag)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"resume Registry tag drift: expected={expected_digest} "
                f"actual={actual_digest}"
            )
        if not admitted_task.exists():
            self._rewrite_task_atomic(
                raw_task,
                admitted_task,
                admitted_image=final_tag,
                source_image=str(record["source_image"]),
                source_image_digest=str(record["source_image_digest"]),
            )
        elif not self._catalog_is_current(admitted_task, final_tag):
            raise RuntimeError(
                f"resume catalog disagrees with Registry: {admitted_task}"
            )
        _atomic_json(
            self._state_path(str(record["task_id"])),
            {"state": "PUBLISHED", "task_id": record["task_id"], "record": record},
        )
        return dict(record)

    @staticmethod
    def _rewrite_task_atomic(
        source: Path,
        output: Path,
        *,
        admitted_image: str,
        source_image: str,
        source_image_digest: str,
    ) -> None:
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        if temporary.exists():
            raise FileExistsError(f"stale TMAX catalog staging path: {temporary}")
        try:
            rewrite(
                source,
                temporary,
                admitted_image=admitted_image,
                source_image=source_image,
                source_image_digest=source_image_digest,
            )
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _validate_all_inputs(self, task_ids: list[str]) -> None:
        selection = json.loads(self.args.selection_manifest.read_text(encoding="utf-8"))
        if (
            selection.get("schema_version") != "tmax-selection/v0"
            or selection.get("sample_size") != self.args.expected_task_count
        ):
            raise ValueError("TMAX selection manifest identity is invalid")
        selection_tasks = list(selection.get("tasks") or [])
        selected_ids = [str(task.get("task_id") or "") for task in selection_tasks]
        if selected_ids != task_ids:
            raise ValueError("task list order disagrees with TMAX selection manifest")
        for task_id in task_ids:
            task_dir = self.task_root / task_id
            for relative in ("task.toml", "instruction.md", "tmax-source.json"):
                if not (task_dir / relative).is_file():
                    raise FileNotFoundError(
                        f"TMAX input is missing: {task_dir / relative}"
                    )
            if resolve_source_image(task_dir) is None:
                raise ValueError(f"TMAX source image is unresolved: {task_id}")
        expected_digests = {
            str(task["task_id"]): str(task.get("tree_digest") or "")
            for task in selection_tasks
        }
        for task_id in task_ids:
            actual_digest = _tree_digest(self.task_root / task_id)
            if expected_digests[task_id] != actual_digest:
                raise ValueError(f"TMAX raw task tree drifted: {task_id}")

    @staticmethod
    def _catalog_is_current(task_dir: Path, admitted_image: str) -> bool:
        try:
            config = TaskConfig.model_validate_toml(
                (task_dir / "task.toml").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return False
        return config.environment.docker_image == admitted_image

    @staticmethod
    def _digest_from_build_metadata(path: Path) -> str:
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = str(payload.get("containerimage.digest") or "")
        if not _DIGEST.fullmatch(digest):
            raise RuntimeError(f"buildx metadata has no immutable digest: {path}")
        return digest

    @staticmethod
    def _verify_build_material(path: Path, source_ref: str) -> None:
        expected = source_ref.rpartition("@")[2].removeprefix("sha256:")
        payload = json.loads(path.read_text(encoding="utf-8"))
        materials = (payload.get("buildx.build.provenance") or {}).get(
            "materials"
        ) or []
        actual = {
            str((material.get("digest") or {}).get("sha256") or "")
            for material in materials
            if isinstance(material, dict)
        }
        if expected not in actual:
            raise RuntimeError(
                "build provenance does not contain the frozen source mirror digest"
            )

    def _ensure_source_mirror(
        self, task_id: str, source_image: str, log_root: Path
    ) -> tuple[str, str]:
        mirror_tag = (
            f"{self.args.source_mirror_repository}:{task_id}-{self.args.release_id}"
        )
        digest = self._registry_source_manifest_digest(mirror_tag)
        if digest is not None:
            return self._source_references(source_image, digest)

        log_path = log_root / "source-mirror.log"
        with self._source_mirror_lock:
            digest = self._registry_source_manifest_digest(mirror_tag)
            if digest is not None:
                return self._source_references(source_image, digest)
            for mirror_attempt in range(1, self.args.source_mirror_attempts + 1):
                delay = self._next_source_mirror_at - time.monotonic()
                if delay > 0:
                    _print(
                        f"[task={task_id}] WAITING_SOURCE_RATE_LIMIT "
                        f"seconds={delay:.0f}"
                    )
                    time.sleep(delay)
                previous_size = log_path.stat().st_size if log_path.exists() else 0
                result = _run(
                    [
                        "docker",
                        "buildx",
                        "imagetools",
                        "create",
                        "--builder",
                        self.source_mirror_builder,
                        "--prefer-index=false",
                        "--tag",
                        mirror_tag,
                        source_image,
                    ],
                    log_path=log_path,
                    check=False,
                )
                self._next_source_mirror_at = (
                    time.monotonic() + self.args.source_mirror_interval_sec
                )
                appended = ""
                try:
                    with log_path.open("r", encoding="utf-8") as stream:
                        stream.seek(previous_size)
                        appended = stream.read()
                except OSError:
                    pass
                self._recycle_source_mirror_builder()
                if result.returncode == 0:
                    digest = self._registry_source_manifest_digest(mirror_tag)
                    if digest is None:
                        raise RuntimeError(
                            f"source mirror tag is missing after copy: {mirror_tag}"
                        )
                    return self._source_references(source_image, digest)
                rate_limited = (
                    "429 Too Many Requests" in appended
                    or "toomanyrequests" in appended.lower()
                )
                if not rate_limited:
                    raise RuntimeError(f"source mirror copy failed; see {log_path}")
                if mirror_attempt >= self.args.source_mirror_attempts:
                    raise RuntimeError(
                        "Docker Hub rate limit did not recover before the configured "
                        f"mirror attempt limit; see {log_path}"
                    )
                self._next_source_mirror_at = max(
                    self._next_source_mirror_at,
                    time.monotonic() + self.args.source_rate_limit_cooldown_sec,
                )
                _print(
                    f"[task={task_id}] DOCKERHUB_RATE_LIMIT mirror_attempt="
                    f"{mirror_attempt}; global_cooldown_seconds="
                    f"{self.args.source_rate_limit_cooldown_sec}"
                )
        raise RuntimeError("source mirror loop exited unexpectedly")

    def _recycle_source_mirror_builder(self) -> None:
        error = self._remove_builder(self.source_mirror_builder)
        if error:
            raise RuntimeError(error)
        self._create_builder(self.source_mirror_builder)
        self.source_mirror_builder_created = True

    def _registry_source_manifest_digest(self, reference: str) -> str | None:
        top_digest = self._registry_digest(reference)
        if top_digest is None:
            return None
        repository, _, tag = reference.rpartition(":")
        registry, _, repository_path = repository.partition("/")
        request = urllib.request.Request(
            (
                f"{self.args.registry_scheme}://{registry}/v2/"
                f"{repository_path}/manifests/{tag}"
            ),
            headers={"Accept": _MANIFEST_ACCEPT},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.load(response)
        except (OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read source mirror manifest: {exc}") from exc
        manifests = payload.get("manifests")
        if not manifests:
            return top_digest
        matches = [
            item
            for item in manifests
            if isinstance(item, dict)
            and (item.get("platform") or {}).get("os") == "linux"
            and (item.get("platform") or {}).get("architecture") == "amd64"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "source mirror must contain exactly one linux/amd64 image"
            )
        digest = str(matches[0].get("digest") or "")
        if not _DIGEST.fullmatch(digest):
            raise RuntimeError("source mirror contains an invalid image digest")
        return digest

    def _source_references(self, source_image: str, digest: str) -> tuple[str, str]:
        source_repository = source_image.rpartition("@")[0] or source_image
        last_slash = source_repository.rfind("/")
        last_colon = source_repository.rfind(":")
        if last_colon > last_slash:
            source_repository = source_repository[:last_colon]
        return (
            f"{self.args.source_mirror_repository}@{digest}",
            f"{source_repository}@{digest}",
        )

    def _registry_digest(self, reference: str) -> str | None:
        repository, separator, tag = reference.rpartition(":")
        registry, slash, repository_path = repository.partition("/")
        if not separator or not slash or not tag or not repository_path:
            raise ValueError(f"invalid tagged image reference: {reference}")
        request = urllib.request.Request(
            (
                f"{self.args.registry_scheme}://{registry}/v2/"
                f"{repository_path}/manifests/{tag}"
            ),
            headers={"Accept": _MANIFEST_ACCEPT},
            method="HEAD",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                digest = str(response.headers.get("Docker-Content-Digest") or "")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RuntimeError(
                f"Registry manifest probe failed: HTTP {exc.code}"
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise RuntimeError(f"Registry manifest probe failed: {exc}") from exc
        if not _DIGEST.fullmatch(digest):
            raise RuntimeError(f"Registry returned invalid digest: {digest!r}")
        return digest

    def _create_builders(self) -> None:
        self._create_builder(self.source_mirror_builder)
        self.source_mirror_builder_created = True
        for slot in range(self.args.jobs):
            name = f"tmax-{self.args.release_id}-{slot}"
            self._create_builder(name)
            self.builder_names.append(name)

    def _build_runtime(self) -> str:
        existing_digest = self._registry_digest(self.runtime_tag)
        if existing_digest is not None:
            if not self.args.resume:
                raise RuntimeError(
                    f"refusing to overwrite Claude runtime tag: {self.runtime_tag}"
                )
            return f"{self.args.claude_runtime_repository}@{existing_digest}"

        builder = f"tmax-runtime-{self.args.release_id}"
        self._create_builder(builder)
        metadata_path = self.work_root / "runtime-build-metadata.json"
        metadata_path.unlink(missing_ok=True)
        try:
            _run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--builder",
                    builder,
                    "--progress=plain",
                    "--provenance=false",
                    "--pull=false",
                    "--build-arg",
                    f"NODE_RUNTIME_IMAGE={self.args.node_runtime_image}",
                    "--build-arg",
                    f"CLAUDE_CODE_VERSION={self.args.claude_code_version}",
                    "--metadata-file",
                    str(metadata_path),
                    "--tag",
                    self.runtime_tag,
                    "--push",
                    "--file",
                    str(self.project_root / "assets/tmax-claude-runtime.Dockerfile"),
                    str(self.project_root / "assets"),
                ],
                log_path=self.work_root / "runtime-build-push.log",
            )
        finally:
            error = self._remove_builder(builder)
            if error:
                raise RuntimeError(error)
        digest = self._digest_from_build_metadata(metadata_path)
        actual_digest = self._registry_digest(self.runtime_tag)
        if actual_digest != digest:
            raise RuntimeError(
                "Claude runtime Registry digest disagrees with build metadata"
            )
        return f"{self.args.claude_runtime_repository}@{digest}"

    def _recycle_builder(self, name: str) -> None:
        error = self._remove_builder(name)
        if error:
            raise RuntimeError(error)
        self._create_builder(name)

    def _create_builder(self, name: str) -> None:
        inspected = subprocess.run(
            ["docker", "buildx", "inspect", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if inspected.returncode == 0:
            if not self.args.resume:
                raise RuntimeError(f"dedicated builder already exists: {name}")
            error = self._remove_builder(name)
            if error:
                raise RuntimeError(error)
        _run(
            [
                "docker",
                "buildx",
                "create",
                "--name",
                name,
                "--driver",
                "docker-container",
                "--buildkitd-config",
                str(self.builder_config),
                "--bootstrap",
            ],
            log_path=self.work_root / "builder-create.log",
        )

    def _remove_builder(self, name: str) -> str | None:
        result = _run(
            ["docker", "buildx", "rm", name],
            log_path=self.work_root / "builder-cleanup.log",
            check=False,
        )
        return None if result.returncode == 0 else f"failed to remove builder {name}"

    def _remove_builders(self) -> list[dict[str, str]]:
        failures: list[dict[str, str]] = []
        for name in self.builder_names:
            error = self._remove_builder(name)
            if error:
                failures.append({"task_id": "<builder-cleanup>", "error": error})
        self.builder_names.clear()
        if self.source_mirror_builder_created:
            error = self._remove_builder(self.source_mirror_builder)
            if error:
                failures.append(
                    {"task_id": "<source-mirror-builder-cleanup>", "error": error}
                )
            self.source_mirror_builder_created = False
        return failures

    def _write_builder_config(self) -> None:
        registries = {
            self.args.output_repository.split("/", 1)[0],
            self.args.claude_runtime_repository.split("/", 1)[0],
            self.args.source_mirror_repository.split("/", 1)[0],
        }
        self.builder_config.write_text(
            "".join(
                f'[registry."{registry}"]\n  http = true\n  insecure = true\n'
                for registry in sorted(registries)
            ),
            encoding="utf-8",
        )

    def _local_release_tags(self) -> list[str]:
        suffix = f"-{self.args.release_id}"
        found: list[str] = []
        for repository in (
            self.args.output_repository,
            self.args.claude_runtime_repository,
        ):
            result = subprocess.run(
                [
                    "docker",
                    "image",
                    "ls",
                    repository,
                    "--format",
                    "{{.Repository}}:{{.Tag}}",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError("cannot inspect local images after publication")
            found.extend(
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip().endswith(suffix)
            )
        return sorted(found)

    def _write_or_verify_run_config(self) -> None:
        identity = {
            "schema_version": "tmax-parallel-prepare-config/v1",
            "validation_profile": VALIDATION_PROFILE,
            "release_id": self.args.release_id,
            "task_list": str(self.args.task_list),
            "task_list_digest": _file_digest(self.args.task_list),
            "selection_manifest": str(self.args.selection_manifest),
            "selection_manifest_digest": _file_digest(self.args.selection_manifest),
            "expected_task_count": self.args.expected_task_count,
            "task_root": str(self.task_root),
            "output_repository": self.args.output_repository,
            "claude_runtime_repository": self.args.claude_runtime_repository,
            "source_mirror_repository": self.args.source_mirror_repository,
            "node_runtime_image": self.args.node_runtime_image,
            "namespace": self.args.namespace,
            "claude_code_version": self.args.claude_code_version,
            "jobs": self.args.jobs,
            "task_attempts": self.args.task_attempts,
            "task_retry_backoff_sec": self.args.task_retry_backoff_sec,
            "min_docker_free_gib": self.args.min_docker_free_gib,
            "min_work_free_gib": self.args.min_work_free_gib,
            "min_memory_available_gib": self.args.min_memory_available_gib,
            "source_mirror_interval_sec": self.args.source_mirror_interval_sec,
            "source_rate_limit_cooldown_sec": (
                self.args.source_rate_limit_cooldown_sec
            ),
            "source_mirror_attempts": self.args.source_mirror_attempts,
        }
        path = self.work_root / "run-config.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != identity:
                source_mirror_keys = {
                    "source_mirror_repository",
                    "source_mirror_interval_sec",
                    "source_rate_limit_cooldown_sec",
                    "source_mirror_attempts",
                }
                legacy_identity = {
                    key: value
                    for key, value in identity.items()
                    if key not in source_mirror_keys
                }
                if self.args.resume and existing == legacy_identity:
                    backup = self.work_root / "run-config.pre-source-mirror.json"
                    if not backup.exists():
                        _atomic_json(backup, existing)
                    _atomic_json(path, identity)
                else:
                    raise RuntimeError(
                        "resume configuration differs from frozen release"
                    )
            if not self.args.resume:
                raise RuntimeError(
                    "work state exists; use --resume or a new release ID"
                )
            return
        _atomic_json(path, identity)

    def _publish_manifests(self, records: list[dict[str, Any]]) -> None:
        fixed = {
            "schema_version": "fixed-harbor-workload/v0",
            "validation_profile": VALIDATION_PROFILE,
            "namespace": self.args.namespace,
            "task_count": len(records),
            "tasks": [
                {"task_id": record["task_id"], "task_ref": record["task_ref"]}
                for record in records
            ],
        }
        _atomic_json(self.output_root / "fixed-tasks.json", fixed)
        publication = {
            "schema_version": "tmax-image-publication/v0",
            "release_id": self.args.release_id,
            "validation_profile": VALIDATION_PROFILE,
            "training_image_ready": True,
            "verifier_checked": False,
            "gold_checked": False,
            "lifecycle_checked": False,
            "user_accepted_unvalidated_tasks": True,
            "tasks": records,
        }
        publication_path = self.output_root / "admission.json"
        _atomic_json(publication_path, publication)

        prompt_path = self.output_root / "tmax1500.jsonl"
        temporary = prompt_path.with_name(f".{prompt_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                for record in records:
                    task_id = str(record["task_id"])
                    prompt = (self.task_root / task_id / "instruction.md").read_text(
                        encoding="utf-8"
                    )
                    stream.write(
                        json.dumps(
                            {
                                "prompt": prompt.strip(),
                                "label": None,
                                "metadata": {
                                    "harbor_task_ref": record["task_ref"],
                                    "benchmark": self.args.namespace,
                                    "task_id": task_id,
                                    "agent": "claude-code",
                                    "validation_profile": VALIDATION_PROFILE,
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            os.replace(temporary, prompt_path)
        finally:
            temporary.unlink(missing_ok=True)

        index = build_task_resource_index(
            self.output_root / "tasks",
            TaskResourceDefaults(
                cpus=self.args.default_cpus,
                memory_mb=self.args.default_memory_mb,
            ),
            publication_path,
        )
        write_task_resource_index(
            index, self.output_root / "task-resource-index-v2.json"
        )

    def _summary(
        self,
        task_ids: list[str],
        records: list[dict[str, Any]],
        failures: list[dict[str, str]],
        started: float,
    ) -> dict[str, Any]:
        states = {state: 0 for state in TASK_STATES}
        for task_id in task_ids:
            state = str(self._read_state(self._state_path(task_id)).get("state") or "")
            if state in states:
                states[state] += 1
        return {
            "schema_version": "tmax-parallel-prepare-summary/v1",
            "validation_profile": VALIDATION_PROFILE,
            "release_id": self.args.release_id,
            "requested_count": len(task_ids),
            "published_count": len(records),
            "failed_count": len(failures),
            "jobs": self.args.jobs,
            "task_attempts": self.args.task_attempts,
            "builder_cache_policy": "dedicated-builder-recreated-after-each-task",
            "claude_runtime_image": self.runtime_image,
            "final_image_load_policy": "push-only-never-load",
            "min_docker_free_gib": self.args.min_docker_free_gib,
            "min_work_free_gib": self.args.min_work_free_gib,
            "min_memory_available_gib": self.args.min_memory_available_gib,
            "source_mirror_policy": {
                "repository": self.args.source_mirror_repository,
                "minimum_interval_sec": self.args.source_mirror_interval_sec,
                "rate_limit_cooldown_sec": self.args.source_rate_limit_cooldown_sec,
                "attempts": self.args.source_mirror_attempts,
            },
            "elapsed_seconds": round(time.time() - started, 3),
            "state_counts": states,
            "failures": failures,
        }

    def _write_attempt_report(
        self,
        task_ids: list[str],
        records: list[dict[str, Any]],
        failures: list[dict[str, str]],
    ) -> None:
        _atomic_text(
            self.work_root / "successful-records.jsonl",
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        )
        _atomic_json(
            self.work_root / "failures.json",
            {
                "schema_version": "tmax-parallel-failures/v1",
                "release_id": self.args.release_id,
                "failures": failures,
            },
        )
        failed_ids: list[str] = []
        unresolved_ids: list[str] = []
        for task_id in task_ids:
            state = str(self._read_state(self._state_path(task_id)).get("state") or "")
            if state == "FAILED":
                failed_ids.append(task_id)
            if state != "PUBLISHED":
                unresolved_ids.append(task_id)
        _atomic_text(
            self.work_root / "failed-task-ids.txt",
            "".join(f"{task_id}\n" for task_id in failed_ids),
        )
        _atomic_text(
            self.work_root / "unresolved-task-ids.txt",
            "".join(f"{task_id}\n" for task_id in unresolved_ids),
        )

    def _state_path(self, task_id: str) -> Path:
        return self.state_root / f"{task_id}.json"

    @staticmethod
    def _read_state(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _check_headroom(self) -> None:
        docker_free = shutil.disk_usage(self.args.docker_root).free
        work_free = shutil.disk_usage(self.work_root).free
        mem_available_kib = 0
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_kib = int(line.split()[1])
                break
        if docker_free < self.args.min_docker_free_gib * 1024**3:
            raise RuntimeError("DockerRootDir free space is below configured floor")
        if work_free < self.args.min_work_free_gib * 1024**3:
            raise RuntimeError("work filesystem free space is below configured floor")
        if mem_available_kib < self.args.min_memory_available_gib * 1024**2:
            raise RuntimeError("MemAvailable is below configured floor")

    def _wait_for_headroom(self, task_id: str) -> None:
        deadline = time.monotonic() + self.args.headroom_wait_timeout_sec
        last_error = ""
        while True:
            try:
                self._check_headroom()
                return
            except RuntimeError as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"headroom did not recover before timeout: {last_error}"
                )
            _print(
                f"[task={task_id}] WAITING FOR HEADROOM: {last_error}; "
                f"retry_in={self.args.headroom_poll_sec}s"
            )
            time.sleep(self.args.headroom_poll_sec)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel-build and publish a frozen TMAX Claude-only workload"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--validation-profile", choices=(VALIDATION_PROFILE,), required=True
    )
    parser.add_argument("--ack-trusted-security", action="store_true")
    parser.add_argument("--task-list", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--expected-task-count", type=int, default=1500)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--task-attempts", type=int, default=2)
    parser.add_argument("--task-retry-backoff-sec", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-inputs-only", action="store_true")
    parser.add_argument("--registry-scheme", choices=("http", "https"), default="http")
    parser.add_argument(
        "--output-repository",
        required=True,
    )
    parser.add_argument(
        "--claude-runtime-repository",
        required=True,
    )
    parser.add_argument(
        "--source-mirror-repository",
        required=True,
    )
    parser.add_argument("--source-mirror-interval-sec", type=int, default=45)
    parser.add_argument("--source-rate-limit-cooldown-sec", type=int, default=3600)
    parser.add_argument("--source-mirror-attempts", type=int, default=24)
    parser.add_argument("--node-runtime-image", required=True)
    parser.add_argument("--namespace", default="tmax-15k")
    parser.add_argument("--claude-code-version", default="2.1.207")
    parser.add_argument("--docker-root", type=Path, default=Path("/var/lib/docker"))
    parser.add_argument("--min-docker-free-gib", type=int, default=50)
    parser.add_argument("--min-work-free-gib", type=int, default=500)
    parser.add_argument("--min-memory-available-gib", type=int, default=16)
    parser.add_argument("--headroom-wait-timeout-sec", type=int, default=1800)
    parser.add_argument("--headroom-poll-sec", type=int, default=10)
    parser.add_argument("--default-cpus", type=int, default=1)
    parser.add_argument("--default-memory-mb", type=int, default=4096)
    args = parser.parse_args(argv)
    if not args.ack_trusted_security:
        parser.error("--ack-trusted-security is required for trusted-security")
    if args.jobs < 1 or args.jobs > 4:
        parser.error("--jobs must be between 1 and 4")
    if args.task_attempts < 1 or args.task_attempts > 3:
        parser.error("--task-attempts must be between 1 and 3")
    if args.task_retry_backoff_sec < 0 or args.task_retry_backoff_sec > 300:
        parser.error("--task-retry-backoff-sec must be between 0 and 300")
    if args.source_mirror_interval_sec < 1:
        parser.error("--source-mirror-interval-sec must be positive")
    if args.source_rate_limit_cooldown_sec < 60:
        parser.error("--source-rate-limit-cooldown-sec must be at least 60")
    if args.source_mirror_attempts < 1 or args.source_mirror_attempts > 48:
        parser.error("--source-mirror-attempts must be between 1 and 48")
    if args.expected_task_count < 1:
        parser.error("--expected-task-count must be positive")
    if args.headroom_wait_timeout_sec < 1 or args.headroom_poll_sec < 1:
        parser.error("headroom wait and poll values must be positive")
    for name, value in (("release ID", args.release_id), ("namespace", args.namespace)):
        if not _SAFE_COMPONENT.fullmatch(value):
            parser.error(f"unsafe {name}: {value!r}")
    if not _VERSION.fullmatch(args.claude_code_version):
        parser.error("--claude-code-version must be numeric X.Y.Z")
    _, separator, node_digest = args.node_runtime_image.rpartition("@")
    if not separator or not _DIGEST.fullmatch(node_digest):
        parser.error("--node-runtime-image must be pinned by digest")
    for name in (
        "output_repository",
        "claude_runtime_repository",
        "source_mirror_repository",
    ):
        repository = getattr(args, name)
        if (
            "://" in repository
            or "@" in repository
            or any(character.isspace() for character in repository)
        ):
            parser.error(f"--{name.replace('_', '-')} must be an untagged repository")
    for name in (
        "project_root",
        "task_list",
        "selection_manifest",
        "task_root",
        "output_root",
        "work_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main() -> None:
    args = parse_args()
    preparer = TMAXParallelPreparer(args)
    if args.check_inputs_only:
        task_ids = read_task_ids(args.task_list)
        if len(task_ids) != args.expected_task_count:
            raise SystemExit(
                "frozen TMAX task count mismatch: "
                f"expected={args.expected_task_count} actual={len(task_ids)}"
            )
        preparer._validate_all_inputs(task_ids)
        print(
            f"TMAX inputs valid: tasks={len(task_ids)} "
            f"selection={args.selection_manifest}"
        )
        return
    raise SystemExit(preparer.prepare())


if __name__ == "__main__":
    main()
