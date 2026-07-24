"""Persist rollout artifacts such as trajectory payloads and segment samples."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import partial
import json
import logging
import os
from pathlib import Path
import time
import traceback
from typing import Any
import uuid

import httpx

from dressage.config import trajectory_error_log_dir, trajectory_payload_log_dir
from dressage.rollout.artifacts.harbor_convert import trajectory_payload_to_harbor
from dressage.rollout.artifacts.samples import (
    copy_sample_with_metadata,
    sample_artifact_payload,
    write_sample_from_segment,
)

logger = logging.getLogger(__name__)


class RolloutArtifactWriter:
    """Writes rollout artifacts while keeping background writes drainable."""

    def __init__(self) -> None:
        self._write_tasks: set[asyncio.Future] = set()

    async def write_json(
        self,
        output_path: Path,
        payload: Any,
        *,
        failure_context: str | None = None,
    ) -> Path:
        payload = json_safe(payload)
        if _log_write_mode() == "await":
            return await asyncio.to_thread(_write_json_file_atomic, output_path, payload)

        loop = asyncio.get_running_loop()
        try:
            task = loop.run_in_executor(
                None,
                _write_json_file_atomic,
                output_path,
                payload,
            )
        except RuntimeError as exc:
            if "shutdown" not in str(exc).lower():
                raise
            _write_json_file_atomic(output_path, payload)
            return output_path

        self._write_tasks.add(task)
        task.add_done_callback(
            partial(
                self._discard_write_task,
                output_path=output_path,
                failure_context=failure_context,
            )
        )
        return output_path

    async def drain(self) -> None:
        while self._write_tasks:
            tasks = tuple(self._write_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def write_session_payload(
        self,
        trajectory_payload: dict[str, Any],
        *,
        session_id: str,
        instance_id: str,
    ) -> Path | None:
        output_dir = _rollout_log_dir(session_id=session_id, instance_id=instance_id)
        if output_dir is None:
            return None
        session_path = await self.write_json(
            output_dir / "session.json",
            trajectory_payload,
        )
        if _harbor_atif_export_enabled():
            try:
                await self.write_harbor_trajectory(
                    trajectory_payload,
                    session_id=session_id,
                    instance_id=instance_id,
                )
            except Exception as exc:
                logger.warning(
                    "failed to export Harbor ATIF trajectory for "
                    "session_id=%s instance_id=%s: %s",
                    session_id,
                    instance_id,
                    exception_summary(exc),
                    exc_info=True,
                )
        return session_path

    async def write_harbor_trajectory(
        self,
        trajectory_payload: dict[str, Any],
        *,
        session_id: str,
        instance_id: str,
        agent: dict[str, Any] | None = None,
    ) -> Path | None:
        """Convert and persist one trajectory as Harbor ATIF-v1.7."""
        output_dir = _rollout_log_dir(
            session_id=session_id,
            instance_id=instance_id,
        )
        if output_dir is None:
            return None
        harbor_payload = trajectory_payload_to_harbor(
            trajectory_payload,
            session_id=session_id,
            instance_id=instance_id,
            agent=agent,
        )
        return await self.write_json(
            output_dir / "harbor" / "trajectory.json",
            harbor_payload,
            failure_context=(
                "Harbor ATIF trajectory for "
                f"session_id={session_id} instance_id={instance_id}"
            ),
        )

    async def write_error(
        self,
        exc: BaseException,
        *,
        sample: Any,
        metadata: dict[str, Any],
        session_id: str,
        instance_id: str,
        blackbox_type: str,
        env_args: dict[str, Any],
        state: Any,
        agent_response: str,
    ) -> Path | None:
        output_dir = _rollout_error_log_dir(
            session_id=session_id,
            instance_id=instance_id,
        )
        if output_dir is None:
            return None

        payload = {
            "success": False,
            "timestamp": time.time(),
            "session_id": session_id,
            "trajectory_id": session_id,
            "instance_id": instance_id,
            "blackbox_type": blackbox_type,
            "error": exception_details(exc),
            "http": http_exception_details(exc),
            "sample": sample_error_payload(sample, metadata=metadata),
            "env_args": env_args,
            "state": state,
            "agent_response": agent_response,
        }
        return await self.write_json(output_dir / "error.json", payload)

    async def write_segment_samples(
        self,
        sample: Any,
        *,
        args: Any,
        segments: list[dict[str, Any]],
        base_metadata: dict[str, Any],
        session_id: str,
        instance_id: str,
        agent_response: str,
    ) -> None:
        if _rollout_log_dir(session_id=session_id, instance_id=instance_id) is None:
            return

        for segment in segments:
            try:
                sample_for_log = copy_sample_with_metadata(
                    sample,
                    metadata=base_metadata,
                )
                segment_sample = write_sample_from_segment(
                    sample_for_log,
                    args=args,
                    segment=segment,
                    all_segments=segments,
                    session_id=session_id,
                    instance_id=instance_id,
                    agent_response=agent_response,
                )
                await self.write_segment_sample(
                    segment_sample,
                    segment=segment,
                    all_segments=segments,
                    session_id=session_id,
                    instance_id=instance_id,
                )
            except Exception:
                logger.warning(
                    "failed to write sample log for session_id=%s, segment_index=%s",
                    session_id,
                    segment.get("segment_index", 0),
                    exc_info=True,
                )

    async def write_segment_sample(
        self,
        sample: Any,
        *,
        segment: dict[str, Any],
        all_segments: list[dict[str, Any]],
        session_id: str,
        instance_id: str,
    ) -> Path | None:
        output_dir = _rollout_log_dir(session_id=session_id, instance_id=instance_id)
        if output_dir is None:
            return None

        segment_index = segment.get("segment_index", 0)
        output_path = (
            output_dir
            / "samples"
            / f"{safe_filename_part(segment_index)}.json"
        )
        payload = sample_artifact_payload(
            sample,
            segment=segment,
            all_segments=all_segments,
            session_id=session_id,
            instance_id=instance_id,
        )
        return await self.write_json(output_path, payload)

    def _discard_write_task(
        self,
        task: asyncio.Future,
        *,
        output_path: Path,
        failure_context: str | None,
    ) -> None:
        self._write_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            context = f" ({failure_context})" if failure_context else ""
            logger.warning(
                "background log write failed for path=%s%s",
                output_path,
                context,
                exc_info=True,
            )


DEFAULT_WRITER = RolloutArtifactWriter()


async def write_json_log(output_path: Path, payload: Any) -> Path:
    return await DEFAULT_WRITER.write_json(output_path, payload)


async def drain_log_write_tasks() -> None:
    await DEFAULT_WRITER.drain()


async def write_trajectory_payload_log(
    trajectory_payload: dict[str, Any],
    *,
    session_id: str,
    instance_id: str,
) -> Path | None:
    return await DEFAULT_WRITER.write_session_payload(
        trajectory_payload,
        session_id=session_id,
        instance_id=instance_id,
    )


async def write_all_sample_logs(
    sample: Any,
    *,
    args: Any,
    segments: list[dict[str, Any]],
    base_metadata: dict[str, Any],
    session_id: str,
    instance_id: str,
    agent_response: str,
) -> None:
    await DEFAULT_WRITER.write_segment_samples(
        sample,
        args=args,
        segments=segments,
        base_metadata=base_metadata,
        session_id=session_id,
        instance_id=instance_id,
        agent_response=agent_response,
    )


def safe_filename_part(value: Any) -> str:
    text = str(value)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    safe = "".join(ch if ch in allowed else "_" for ch in text)
    return safe.strip("._") or "unknown"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def exception_summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__


def exception_details(exc: BaseException) -> dict[str, Any]:
    return {
        "type": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
        "summary": exception_summary(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def http_exception_details(exc: BaseException) -> dict[str, Any] | None:
    if not isinstance(exc, httpx.HTTPStatusError):
        return None

    request = exc.request
    response = exc.response
    return {
        "request": {
            "method": request.method,
            "url": str(request.url),
            "headers": headers_for_log(request.headers),
            "body": bytes_for_log(request.content),
        },
        "response": {
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "headers": headers_for_log(response.headers),
            "body": response.text,
        },
    }


def sample_error_payload(sample: Any, *, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": getattr(sample, "prompt", None),
        "label": getattr(sample, "label", None),
        "session_id": getattr(sample, "session_id", None),
        "group_index": getattr(sample, "group_index", None),
        "index": getattr(sample, "index", None),
        "status": getattr(sample, "status", None),
        "response": getattr(sample, "response", None),
        "metadata": metadata,
    }


def headers_for_log(headers: httpx.Headers) -> dict[str, str]:
    redacted = {}
    sensitive = {"authorization", "api-key", "x-api-key", "cookie", "set-cookie"}
    for key, value in headers.items():
        redacted[key] = "<redacted>" if key.lower() in sensitive else value
    return redacted


def bytes_for_log(data: bytes | None) -> str | None:
    if not data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _write_json_file_atomic(output_path: Path, payload: Any) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uuid_value = uuid.uuid4()
    unique = getattr(uuid_value, "hex", None) or safe_filename_part(uuid_value)
    tmp_path = output_path.parent / (
        f".{output_path.name}.{os.getpid()}.{unique}.tmp"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return output_path


def _log_write_mode() -> str:
    mode = os.environ.get("DRESSAGE_LOG_WRITE_MODE", "background").strip().lower()
    if mode in {"await", "background"}:
        return mode
    logger.warning(
        "invalid DRESSAGE_LOG_WRITE_MODE=%r; falling back to background",
        mode,
    )
    return "background"


def _harbor_atif_export_enabled() -> bool:
    value = os.environ.get("DRESSAGE_HARBOR_ATIF_EXPORT", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"", "0", "false", "no", "off"}:
        return False
    logger.warning(
        "invalid DRESSAGE_HARBOR_ATIF_EXPORT=%r; Harbor export is disabled",
        value,
    )
    return False


def _rollout_log_dir(
    *,
    session_id: str,
    instance_id: str,
) -> Path | None:
    return _rollout_log_dir_from_env(
        "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR",
        session_id=session_id,
        instance_id=instance_id,
        default_dir=trajectory_payload_log_dir(),
    )


def _rollout_error_log_dir(
    *,
    session_id: str,
    instance_id: str,
) -> Path | None:
    return _rollout_log_dir_from_env(
        "DRESSAGE_TRAJECTORY_ERROR_LOG_DIR",
        session_id=session_id,
        instance_id=instance_id,
        default_dir=trajectory_error_log_dir(),
    )


def _rollout_log_dir_from_env(
    env_name: str,
    *,
    session_id: str,
    instance_id: str,
    default_dir: Path,
) -> Path | None:
    log_dir = Path(os.environ.get(env_name) or default_dir)
    return (
        Path(log_dir)
        / safe_filename_part(instance_id)
        / safe_filename_part(session_id)
    )
