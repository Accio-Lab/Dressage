from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import random
import re
import shutil
import signal
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .config import ImageCacheSettings
from .task_resource_index import TaskResourceIndex, TaskResourceIndexError

logger = logging.getLogger(__name__)

_IMAGE_REF = re.compile(
    r"^(?P<repository>[A-Za-z0-9][A-Za-z0-9._:/-]*)@"
    r"(?P<digest>sha256:[0-9a-f]{64})$"
)
_MANAGED_ORIGINS = frozenset({"pulled_by_rollout_service", "imported_prewarm_proof"})


class ImageCacheError(RuntimeError):
    failure_class = "RETRYABLE_INFRA"
    failure_stage = "PREPARING"
    retryable = True


class ImagePullTransientError(ImageCacheError):
    pass


class ImageDiskPressureError(ImageCacheError):
    pass


class ImageRegistryAuthError(ImageCacheError):
    failure_class = "INFRA_CONFIGURATION"
    retryable = False


class ImageAdmissionInvariantError(ImageCacheError):
    failure_class = "ADMISSION_INVARIANT"
    retryable = False


class ImageReferenceError(ImageCacheError):
    failure_class = "TASK_SETUP"
    retryable = False


class ImageNotFoundError(ImageAdmissionInvariantError):
    pass


class ImageDigestMismatchError(ImageAdmissionInvariantError):
    pass


@dataclass(frozen=True)
class DockerResult:
    returncode: int
    stdout: str
    stderr: str


class AsyncFlock:
    def __init__(self, path: Path, operation: int) -> None:
        self.path = path
        self.operation = operation
        self._descriptor: int | None = None

    async def acquire(self, *, wait: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            while True:
                try:
                    fcntl.flock(descriptor, self.operation | fcntl.LOCK_NB)
                    self._descriptor = descriptor
                    return True
                except BlockingIOError:
                    if not wait:
                        os.close(descriptor)
                        return False
                    await asyncio.sleep(0.05)
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.release()


@dataclass
class ImageUseHandle:
    cache: ImageCache
    use_lock: AsyncFlock
    attempt_id: str
    image_ref: str
    image_id: str
    cache_hit: bool
    lock_wait_seconds: float
    pull_lock_wait_seconds: float
    pull_attempts: int
    pull_seconds: float
    verify_seconds: float
    _released: bool = False

    async def release(self, *, outcome: str) -> None:
        if self._released:
            return
        self._released = True
        try:
            await self.cache.mark_released(
                self.image_ref,
                attempt_id=self.attempt_id,
                outcome=outcome,
            )
        finally:
            self.use_lock.release()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, _exc: object, _tb: object) -> None:
        await self.release(outcome="failed" if exc_type is not None else "completed")


class ImageCache:
    def __init__(
        self,
        settings: ImageCacheSettings,
        *,
        resource_index: TaskResourceIndex,
        worker_id: str,
    ) -> None:
        self.settings = settings
        self.resource_index = resource_index
        self.worker_id = worker_id
        self.root = settings.cache_root.expanduser()
        self.records_dir = self.root / "records"
        self.locks_dir = self.root / "locks"
        self.quarantine_dir = self.root / "quarantine"
        self.events_path = self.root / "events.jsonl"
        self._pull_gate = asyncio.Semaphore(settings.max_concurrent_pulls)
        for path in (self.records_dir, self.locks_dir, self.quarantine_dir):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def parse_image_ref(image_ref: str) -> tuple[str, str, str]:
        match = _IMAGE_REF.fullmatch(image_ref.strip())
        if match is None:
            raise ImageReferenceError(
                "execution image must use repository@sha256:<64 lowercase hex>"
            )
        repository = match.group("repository")
        digest = match.group("digest")
        return repository, digest, digest.removeprefix("sha256:")

    def validate_execution_ref(self, image_ref: str) -> tuple[str, str, str]:
        """Validate the internal immutable ref without applying a registry prefix policy."""

        return self.parse_image_ref(image_ref)

    async def acquire(
        self,
        *,
        attempt_id: str,
        task_ref: str,
        declared_image_ref: str,
        execution_image_ref: str,
        task_config_digest: str,
        worker_incarnation: str,
    ) -> ImageUseHandle:
        try:
            entry = self.resource_index.tasks[task_ref]
            admitted_image = self.resource_index.admitted_image(task_ref)
            execution_image = self.resource_index.execution_image(task_ref)
        except (KeyError, TaskResourceIndexError) as exc:
            raise ImageAdmissionInvariantError(str(exc)) from exc
        if entry.task_config_digest != task_config_digest:
            raise ImageAdmissionInvariantError(
                f"task config digest changed after admission: {task_ref}"
            )
        if admitted_image != declared_image_ref:
            raise ImageAdmissionInvariantError(
                f"task image alias disagrees with frozen admission entry: {task_ref}"
            )
        if execution_image != execution_image_ref:
            raise ImageAdmissionInvariantError(
                f"task execution image disagrees with frozen admission entry: {task_ref}"
            )

        image_ref = execution_image_ref
        _repository, digest, digest_key = self.validate_execution_ref(image_ref)
        use_lock = AsyncFlock(self._lock_path(digest_key, "use"), fcntl.LOCK_SH)
        started = time.monotonic()
        await use_lock.acquire()
        lock_wait_seconds = time.monotonic() - started
        pull_attempts = 0
        pull_seconds = 0.0
        verify_seconds = 0.0
        pull_lock_started = time.monotonic()
        try:
            async with self._lock(digest_key, "pull", fcntl.LOCK_EX):
                pull_lock_wait_seconds = time.monotonic() - pull_lock_started
                verify_started = time.monotonic()
                inspected = await self.inspect_image(image_ref)
                verify_seconds += time.monotonic() - verify_started
                cache_hit = inspected is not None
                record = await self.read_record(digest_key)
                now = time.time()
                if inspected is None:
                    await self._check_pull_headroom()
                    await self.write_record(
                        digest_key,
                        {
                            "schema_version": "rollout-image-cache/v0",
                            "state": "PULLING",
                            "image_ref": image_ref,
                            "declared_image_ref": declared_image_ref,
                            "image_digest": digest,
                            "image_id": None,
                            "managed_tags": [],
                            "catalog_digest": self.resource_index.digest,
                            "validation_profile": self.resource_index.validation_profile,
                            "admission_manifest_digest": self.resource_index.admission_manifest_digest,
                            "admission_entry_digest": entry.admission_entry_digest,
                            "managed_origin": "pulled_by_rollout_service",
                            "evictable": True,
                            "worker_id": self.worker_id,
                            "worker_incarnation": worker_incarnation,
                            "active_attempt_id": attempt_id,
                            "first_managed_at": now,
                            "last_acquired_at": now,
                            "last_released_at": None,
                        },
                    )
                    try:
                        pull_attempts, pull_seconds = await self._pull_with_retries(
                            image_ref
                        )
                        verify_started = time.monotonic()
                        inspected = await self.inspect_image(image_ref)
                        verify_seconds += time.monotonic() - verify_started
                        if inspected is None:
                            raise ImageDigestMismatchError(
                                f"pulled image is not inspectable by exact digest: {image_ref}"
                            )
                    except BaseException as exc:
                        await self._mark_pull_failed(
                            digest_key,
                            image_ref=image_ref,
                            digest=digest,
                            attempt_id=attempt_id,
                            error=exc,
                        )
                        raise
                    record = {
                        "schema_version": "rollout-image-cache/v0",
                        "managed_origin": "pulled_by_rollout_service",
                        "evictable": True,
                        "first_managed_at": now,
                    }

                assert inspected is not None
                origin = str(
                    (record or {}).get("managed_origin") or "observed_preexisting"
                )
                evictable = (
                    bool((record or {}).get("evictable")) and origin in _MANAGED_ORIGINS
                )
                if record is None:
                    origin = "observed_preexisting"
                    evictable = False
                ready_record = {
                    **(record or {}),
                    "schema_version": "rollout-image-cache/v0",
                    "state": "READY",
                    "image_ref": image_ref,
                    "declared_image_ref": declared_image_ref,
                    "image_digest": digest,
                    "image_id": inspected["image_id"],
                    "catalog_digest": self.resource_index.digest,
                    "validation_profile": self.resource_index.validation_profile,
                    "admission_manifest_digest": self.resource_index.admission_manifest_digest,
                    "admission_entry_digest": entry.admission_entry_digest,
                    "managed_origin": origin,
                    "evictable": evictable,
                    "worker_id": self.worker_id,
                    "worker_incarnation": worker_incarnation,
                    "active_attempt_id": attempt_id,
                    "first_managed_at": (record or {}).get("first_managed_at", now),
                    "last_acquired_at": now,
                    "last_released_at": (record or {}).get("last_released_at"),
                    "last_outcome": (record or {}).get("last_outcome"),
                }
                await self.write_record(digest_key, ready_record)
            await self.append_event(
                {
                    "event": "image_acquired",
                    "attempt_id": attempt_id,
                    "task_ref": task_ref,
                    "image_ref": image_ref,
                    "declared_image_ref": declared_image_ref,
                    "image_id": inspected["image_id"],
                    "cache_hit": cache_hit,
                    "lock_wait_seconds": lock_wait_seconds,
                    "pull_lock_wait_seconds": pull_lock_wait_seconds,
                    "pull_attempts": pull_attempts,
                    "pull_seconds": pull_seconds,
                    "verify_seconds": verify_seconds,
                }
            )
            return ImageUseHandle(
                cache=self,
                use_lock=use_lock,
                attempt_id=attempt_id,
                image_ref=image_ref,
                image_id=inspected["image_id"],
                cache_hit=cache_hit,
                lock_wait_seconds=lock_wait_seconds,
                pull_lock_wait_seconds=pull_lock_wait_seconds,
                pull_attempts=pull_attempts,
                pull_seconds=pull_seconds,
                verify_seconds=verify_seconds,
            )
        except BaseException:
            use_lock.release()
            raise

    async def mark_released(
        self, image_ref: str, *, attempt_id: str, outcome: str
    ) -> None:
        _repository, _digest, digest_key = self.parse_image_ref(image_ref)
        try:
            record = await self.read_record(digest_key)
            if record is None:
                return
            record.update(
                {
                    "active_attempt_id": None,
                    "last_released_at": time.time(),
                    "last_attempt_id": attempt_id,
                    "last_outcome": outcome,
                }
            )
            await self.write_record(digest_key, record)
            await self.append_event(
                {
                    "event": "image_released",
                    "attempt_id": attempt_id,
                    "image_ref": image_ref,
                    "outcome": outcome,
                }
            )
        except Exception:
            logger.warning("failed to update image release metadata", exc_info=True)

    async def inspect_image(self, image_ref: str) -> dict[str, Any] | None:
        result = await run_command(
            ["docker", "image", "inspect", image_ref], timeout_sec=30.0
        )
        if result.returncode != 0:
            combined = (result.stderr + result.stdout).lower()
            if "no such image" in combined or "no such object" in combined:
                return None
            raise ImagePullTransientError(
                f"docker image inspect failed: {result.stderr[-1000:]}"
            )
        try:
            payload = json.loads(result.stdout)
            item = payload[0]
            image_id = str(item["Id"])
            repo_digests = [str(value) for value in item.get("RepoDigests") or []]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ImageDigestMismatchError(
                "docker inspect returned invalid image data"
            ) from exc
        if image_ref not in repo_digests:
            raise ImageDigestMismatchError(
                f"local image RepoDigests do not contain expected ref: {image_ref}"
            )
        return {"image_id": image_id, "repo_digests": repo_digests}

    async def read_record(self, digest_key: str) -> dict[str, Any] | None:
        path = self._record_path(digest_key)
        if not path.exists():
            return None
        try:
            return await asyncio.to_thread(_read_json, path)
        except Exception as exc:
            target = self.quarantine_dir / f"{path.name}.{int(time.time())}.invalid"
            with contextlib.suppress(OSError):
                os.replace(path, target)
            raise ImageAdmissionInvariantError(
                f"invalid image cache record was quarantined: {path}"
            ) from exc

    async def write_record(self, digest_key: str, record: dict[str, Any]) -> None:
        async with self._lock(digest_key, "meta", fcntl.LOCK_EX):
            await asyncio.to_thread(
                _atomic_json_write, self._record_path(digest_key), record
            )

    async def append_event(self, event: dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), "worker_id": self.worker_id, **event}
        try:
            await asyncio.to_thread(_append_json_line, self.events_path, payload)
        except Exception:
            logger.warning("failed to append image cache event", exc_info=True)

    async def _pull_with_retries(self, image_ref: str) -> tuple[int, float]:
        total_seconds = 0.0
        last_error: ImageCacheError | None = None
        for attempt in range(1, self.settings.pull_max_attempts + 1):
            started = time.monotonic()
            try:
                async with self._pull_gate:
                    result = await run_command(
                        ["docker", "pull", image_ref],
                        timeout_sec=self.settings.pull_timeout_sec,
                        terminate_process_group=True,
                    )
            except TimeoutError:
                result = DockerResult(124, "", "docker pull timed out")
            total_seconds += time.monotonic() - started
            if result.returncode == 0:
                return attempt, total_seconds
            last_error = classify_pull_failure(result)
            if not isinstance(last_error, ImagePullTransientError):
                raise last_error
            if attempt < self.settings.pull_max_attempts:
                delay = self.settings.pull_retry_base_sec * (2 ** (attempt - 1))
                await asyncio.sleep(delay + random.uniform(0, max(0.1, delay * 0.1)))
        assert last_error is not None
        raise last_error

    async def _check_pull_headroom(self) -> None:
        try:
            usage = await asyncio.to_thread(
                shutil.disk_usage, self.settings.docker_root
            )
        except OSError as exc:
            raise ImageDiskPressureError(
                f"cannot inspect Docker root {self.settings.docker_root}: {exc}"
            ) from exc
        if usage.free < self.settings.min_free_bytes_before_pull:
            raise ImageDiskPressureError(
                f"Docker root free bytes {usage.free} is below pull floor "
                f"{self.settings.min_free_bytes_before_pull}"
            )

    async def _mark_pull_failed(
        self,
        digest_key: str,
        *,
        image_ref: str,
        digest: str,
        attempt_id: str,
        error: BaseException,
    ) -> None:
        with contextlib.suppress(Exception):
            record = await self.read_record(digest_key) or {}
            record.update(
                {
                    "schema_version": "rollout-image-cache/v0",
                    "state": "FAILED",
                    "image_ref": image_ref,
                    "image_digest": digest,
                    "managed_origin": "pulled_by_rollout_service",
                    "evictable": True,
                    "active_attempt_id": None,
                    "last_attempt_id": attempt_id,
                    "last_error": f"{type(error).__name__}: {error}",
                    "last_failed_at": time.time(),
                }
            )
            await self.write_record(digest_key, record)
            await self.append_event(
                {
                    "event": "image_pull_failed",
                    "attempt_id": attempt_id,
                    "image_ref": image_ref,
                    "error": record["last_error"],
                }
            )

    @contextlib.asynccontextmanager
    async def _lock(
        self, digest_key: str, kind: str, operation: int, *, wait: bool = True
    ) -> AsyncIterator[AsyncFlock]:
        lock = AsyncFlock(self._lock_path(digest_key, kind), operation)
        acquired = await lock.acquire(wait=wait)
        if not acquired:
            raise BlockingIOError(f"image cache {kind} lock is busy")
        try:
            yield lock
        finally:
            lock.release()

    def _record_path(self, digest_key: str) -> Path:
        return self.records_dir / f"{digest_key}.json"

    def _lock_path(self, digest_key: str, kind: str) -> Path:
        return self.locks_dir / f"{digest_key}.{kind}.lock"


def classify_pull_failure(result: DockerResult) -> ImageCacheError:
    detail = (result.stderr + "\n" + result.stdout)[-4000:]
    lowered = detail.lower()
    if any(
        marker in lowered
        for marker in ("unauthorized", "authentication required", "denied:")
    ):
        return ImageRegistryAuthError(detail)
    if any(
        marker in lowered
        for marker in ("manifest unknown", "no such manifest", "not found")
    ):
        return ImageNotFoundError(detail)
    if "no space left on device" in lowered:
        return ImageDiskPressureError(detail)
    return ImagePullTransientError(
        detail or f"docker pull failed rc={result.returncode}"
    )


async def run_command(
    args: list[str],
    *,
    timeout_sec: float,
    terminate_process_group: bool = False,
) -> DockerResult:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=terminate_process_group,
    )
    try:
        async with asyncio.timeout(timeout_sec):
            stdout, stderr = await process.communicate()
    except (TimeoutError, asyncio.CancelledError):
        await _terminate_process(process, process_group=terminate_process_group)
        raise
    return DockerResult(
        process.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def _terminate_process(
    process: asyncio.subprocess.Process, *, process_group: bool
) -> None:
    if process.returncode is not None:
        return
    try:
        if process_group and process.pid is not None:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=10.0)
        return
    except TimeoutError:
        pass
    try:
        if process_group and process.pid is not None:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("image cache record must be a JSON object")
    return payload


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


__all__ = [
    "AsyncFlock",
    "DockerResult",
    "ImageAdmissionInvariantError",
    "ImageCache",
    "ImageCacheError",
    "ImageDigestMismatchError",
    "ImageDiskPressureError",
    "ImageNotFoundError",
    "ImagePullTransientError",
    "ImageReferenceError",
    "ImageRegistryAuthError",
    "ImageUseHandle",
    "classify_pull_failure",
    "run_command",
]
