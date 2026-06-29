"""Fully asynchronous rollout entrypoint for Dressage.

The shape follows Slime's fully_async example: a global background worker keeps
pulling prompt groups and generating them continuously, while each rollout call
only drains completed groups for training.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_STALENESS_ERROR_MARKER = "partial_rollout_staleness_exceeded"

try:
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
    from slime.utils.async_utils import run
    from slime.utils.types import Sample
except ImportError:
    GenerateState = None  # type: ignore[assignment]
    generate_and_rm_group = None  # type: ignore[assignment]
    RolloutFnTrainOutput = None  # type: ignore[assignment]

    def run(coro):  # type: ignore[no-redef]
        return asyncio.run(coro)

    Sample = None  # type: ignore[assignment]

from dressage.rollout.multi_segment import (
    compute_multi_segment_metrics,
    mark_aborted_no_grad,
)
from dressage.rollout.staleness import (
    PendingGroup,
    StalenessGroupFilter,
    StalenessTracker,
    config_from_args,
)


@dataclass
class CompletedGroup:
    group_id: int
    original_group: list[Any]
    result: list[Any] | None = None
    error: BaseException | None = None

    @property
    def is_failed(self) -> bool:
        if self.error is not None:
            return True
        if self.result is None:
            return True
        return _is_aborted_group(self.result)

    @property
    def for_training(self) -> list[Any]:
        return self.result if self.result is not None else self.original_group


class _FallbackGenerateState:
    def __init__(self, args: Any) -> None:
        self.sampling_params = {
            "temperature": getattr(args, "rollout_temperature", 1.0),
            "top_p": getattr(args, "rollout_top_p", 1.0),
            "top_k": getattr(args, "rollout_top_k", -1),
            "max_new_tokens": getattr(args, "rollout_max_response_len", 4096),
        }


def _state_for(args: Any):
    if GenerateState is None:
        return _FallbackGenerateState(args)
    return GenerateState(args)


def _sample_status_name(sample: Any) -> str:
    status = getattr(sample, "status", None)
    if hasattr(status, "name"):
        return str(status.name)
    if hasattr(status, "value"):
        return str(status.value).upper()
    return str(status).upper()


def _is_aborted_group(group: list[Any]) -> bool:
    return any(_sample_status_name(sample) == "ABORTED" for sample in group)


def _flatten_multi_segment_result(result: list[Any]) -> list[Any]:
    if not any(isinstance(item, list) for item in result):
        return result
    flat: list[Any] = []
    for item in result:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _short_error(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _contains_staleness_marker(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(
            _contains_staleness_marker(key) or _contains_staleness_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return any(_contains_staleness_marker(item) for item in value)
    return _STALENESS_ERROR_MARKER in str(value)


def _sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _group_has_staleness_failure(
    group: list[Any],
    error: BaseException | None = None,
) -> bool:
    if _contains_staleness_marker(error):
        return True
    return any(_contains_staleness_marker(_sample_metadata(sample)) for sample in group)


def _group_failure_summary(
    group: list[Any],
    error: BaseException | None = None,
    *,
    max_samples: int = 3,
) -> str:
    parts: list[str] = []
    if error is not None:
        parts.append(f"task_error={type(error).__name__}: {_short_error(error)}")

    status_counts: dict[str, int] = {}
    for sample in group:
        status = _sample_status_name(sample)
        status_counts[status] = status_counts.get(status, 0) + 1
    if status_counts:
        parts.append(
            "statuses="
            + ",".join(
                f"{status}:{count}" for status, count in sorted(status_counts.items())
            )
        )

    for offset, sample in enumerate(group[:max_samples]):
        metadata = _sample_metadata(sample)
        sample_parts = []
        for key in ("blackbox_error", "rollout_error"):
            if metadata.get(key):
                sample_parts.append(f"{key}={_short_error(metadata[key])}")
        if not sample_parts:
            continue
        sample_id = getattr(sample, "index", offset)
        session_id = (
            metadata.get("session_id")
            or getattr(sample, "session_id", None)
            or metadata.get("last_failed_session_id")
        )
        instance_id = metadata.get("instance_id")
        parts.append(
            f"sample[{sample_id} session_id={session_id} instance_id={instance_id}] "
            + " ".join(sample_parts)
        )

    return "; ".join(parts) or "no detailed failure metadata"


def _sample_has_trainable_tokens(sample: Any) -> bool:
    if getattr(sample, "remove_sample", False):
        return False

    try:
        response_length = int(getattr(sample, "response_length", 0) or 0)
    except (TypeError, ValueError):
        response_length = 0
    if response_length <= 0:
        return False

    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return True
    return any(int(value) != 0 for value in loss_mask)


def _group_has_trainable_tokens(group: list[Any]) -> bool:
    return any(_sample_has_trainable_tokens(sample) for sample in group)


def _allow_empty_train_batch() -> bool:
    return os.environ.get("DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _set_status(sample: Any, name: str) -> None:
    if Sample is not None:
        sample.status = getattr(Sample.Status, name)
        return
    status_cls = getattr(sample, "Status", None)
    sample.status = getattr(status_cls, name) if status_cls is not None else name.lower()


def _mark_no_grad_failed(group: list[Any], error: BaseException | None = None) -> list[Any]:
    for sample in group:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        if error is not None:
            metadata["rollout_error"] = str(error)
        mark_aborted_no_grad(
            sample,
            session_id=metadata.get("session_id"),
            instance_id=metadata.get("instance_id"),
        )
        _set_status(sample, "FAILED")
        sample.reward = 0.0
        sample.tokens = getattr(sample, "tokens", None) or [0]
        sample.response = getattr(sample, "response", "") or ""
        sample.response_length = 0
        sample.loss_mask = []
        sample.rollout_log_probs = []
    return group


def _completed_from_task(
    group_id: int, original_group: list[Any], task: asyncio.Task
) -> CompletedGroup:
    try:
        return CompletedGroup(
            group_id=group_id,
            original_group=original_group,
            result=task.result(),
        )
    except BaseException as exc:  # noqa: BLE001
        return CompletedGroup(
            group_id=group_id,
            original_group=original_group,
            error=exc,
        )


class AsyncRolloutWorker:
    """Persistent background worker that continuously generates sample groups."""

    def __init__(self, args: Any, data_buffer: Any) -> None:
        self.args = args
        self.data_buffer = data_buffer
        self.max_active_groups = int(
            os.environ.get(
                "DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS",
                str(getattr(args, "rollout_batch_size", 1)),
            )
        )
        output_size = int(os.environ.get("DRESSAGE_ASYNC_OUTPUT_QUEUE_SIZE", "1000"))
        self.output_queue: queue.Queue[CompletedGroup] = queue.Queue(maxsize=output_size)
        self.high_watermark = max(1, int(output_size * 0.8))
        self.running = True
        self.worker_thread: threading.Thread | None = None
        self._next_group_id = 0
        self.staleness = StalenessTracker(config_from_args(args))

    def start(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        self.worker_thread = threading.Thread(
            target=lambda: asyncio.run(self.continuous_worker_loop()),
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    async def continuous_worker_loop(self) -> None:
        logger.info("Dressage fully async rollout worker started")
        state = _state_for(self.args)
        active: dict[asyncio.Task, tuple[int, list[Any]]] = {}

        while self.running:
            done_tasks = [task for task in active if task.done()]
            for task in done_tasks:
                group_id, group = active.pop(task)
                self._put_completed(_completed_from_task(group_id, group, task))

            while (
                self.running
                and len(active) < self.max_active_groups
                and self.output_queue.qsize() < self.high_watermark
            ):
                groups = self.data_buffer.get_samples(1)
                if not groups:
                    break
                group = groups[0]
                group_id = self._next_group_id
                self._next_group_id += 1
                task = asyncio.create_task(
                    self._run_group(group, state.sampling_params.copy())
                )
                active[task] = (group_id, group)

            await asyncio.sleep(0.01)

        if active:
            await asyncio.wait(active.keys())
            for task, (group_id, group) in active.items():
                self._put_completed(_completed_from_task(group_id, group, task))
        logger.info("Dressage fully async rollout worker stopped")

    async def _run_group(self, group: list[Any], sampling_params: dict[str, Any]) -> list[Any]:
        if generate_and_rm_group is None:
            raise RuntimeError("slime.rollout.sglang_rollout.generate_and_rm_group is unavailable")
        result = await generate_and_rm_group(
            self.args,
            group,
            sampling_params=sampling_params,
            evaluation=False,
        )
        return _flatten_multi_segment_result(result)

    def _put_completed(self, item: CompletedGroup) -> None:
        while True:
            try:
                self.output_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if not self.running:
                    logger.error(
                        "output_queue full during shutdown drain; dropping "
                        "completed group %d (qsize=%d)",
                        item.group_id,
                        self.output_queue.qsize(),
                    )
                    return

    def get_completed_groups(self) -> list[CompletedGroup]:
        completed: list[CompletedGroup] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                return completed


_GLOBAL_WORKER: AsyncRolloutWorker | None = None
_WORKER_LOCK = threading.Lock()


def get_global_worker(args: Any, data_buffer: Any) -> AsyncRolloutWorker:
    global _GLOBAL_WORKER
    with _WORKER_LOCK:
        if _GLOBAL_WORKER is None or _GLOBAL_WORKER.worker_thread is None or not _GLOBAL_WORKER.worker_thread.is_alive():
            _GLOBAL_WORKER = AsyncRolloutWorker(args, data_buffer)
            _GLOBAL_WORKER.start()
        return _GLOBAL_WORKER


def stop_global_worker() -> None:
    global _GLOBAL_WORKER
    with _WORKER_LOCK:
        if _GLOBAL_WORKER is not None:
            _GLOBAL_WORKER.stop()
            _GLOBAL_WORKER = None


def _retry_count(group: list[Any]) -> int:
    if not group:
        return 0
    metadata = getattr(group[0], "metadata", None)
    return int((metadata or {}).get("dressage_retry_count", 0))


def _increment_retry(group: list[Any]) -> None:
    for sample in group:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        previous_session_id = metadata.get("session_id") or getattr(sample, "session_id", None)
        if previous_session_id is not None:
            metadata["last_retry_session_id"] = previous_session_id
        metadata.pop("session_id", None)
        metadata.pop("parent_traj_id", None)
        metadata.pop("segment_index", None)
        if hasattr(sample, "session_id"):
            sample.session_id = None
        metadata["dressage_retry_count"] = int(metadata.get("dressage_retry_count", 0)) + 1
        sample.remove_sample = False


async def generate_rollout_async(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
) -> tuple[list[list[Any]], dict[str, float]]:
    del rollout_id
    worker = get_global_worker(args, data_buffer)
    target_data_size = int(getattr(args, "rollout_batch_size", 1))
    max_retries = int(os.environ.get("DRESSAGE_ROLLOUT_MAX_RETRIES", "2"))
    completed_by_id: dict[int, CompletedGroup] = {}
    data: list[PendingGroup] = []
    dropped_failed_groups = 0
    dropped_failure_summaries: list[str] = []
    last_progress_time = time.time()
    no_progress_timeout = float(os.environ.get("DRESSAGE_ASYNC_NO_PROGRESS_WARN_SEC", "600"))
    max_dropped_failed_groups = int(
        os.environ.get(
            "DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS",
            str(max(target_data_size * 10, 100)),
        )
    )
    staleness_filter = StalenessGroupFilter(
        tracker=worker.staleness,
        rollout_name="fully",
    )

    while len(data) < target_data_size:
        completed_groups = worker.get_completed_groups()
        advanced_version = staleness_filter.observe_completed(completed_groups)
        if advanced_version and data:
            previous_count = len(data)
            data = staleness_filter.filter_pending(data, logger)
            if len(data) != previous_count:
                last_progress_time = time.time()

        for completed in completed_groups:
            completed_by_id[completed.group_id] = completed

        if advanced_version:
            last_progress_time = time.time()

        for group_id in list(completed_by_id.keys()):
            if len(data) >= target_data_size:
                break
            completed = completed_by_id.pop(group_id)
            if completed.is_failed:
                summary = _group_failure_summary(
                    completed.result or completed.original_group, completed.error
                )
                if _retry_count(completed.original_group) < max_retries:
                    _increment_retry(completed.original_group)
                    data_buffer.add_samples([completed.original_group])
                    logger.warning(
                        "returned group %s to rollout buffer for retry: %s",
                        group_id,
                        summary,
                    )
                    continue
                dropped_failed_groups += 1
                if len(dropped_failure_summaries) < 3:
                    dropped_failure_summaries.append(summary)
                logger.error(
                    "rollout group %s exhausted retries and will be dropped: %s",
                    group_id,
                    summary,
                )
                if dropped_failed_groups >= max_dropped_failed_groups:
                    raise RuntimeError(
                        "Dressage fully async rollout dropped too many failed groups "
                        f"after exhausted retries ({dropped_failed_groups}); "
                        "refusing to wait forever for a trainable batch. "
                        f"First failures: {' | '.join(dropped_failure_summaries)}"
                    )
                last_progress_time = time.time()
                continue
            else:
                group = completed.result

            if not staleness_filter.keep_group(group_id, group, logger):
                last_progress_time = time.time()
                continue
            data.append(PendingGroup(group_id=group_id, samples=group))
            last_progress_time = time.time()

        now = time.time()
        if now - last_progress_time > no_progress_timeout:
            logger.warning(
                "no completed rollout group for %.1fs; collected %d/%d",
                no_progress_timeout,
                len(data),
                target_data_size,
            )
            last_progress_time = now
        if len(data) < target_data_size:
            await asyncio.sleep(0.01)

    data_groups = [group.samples for group in data]
    data_groups = sorted(data_groups, key=lambda group: getattr(group[0], "index", 0))
    if not _allow_empty_train_batch() and not any(
        _group_has_trainable_tokens(group) for group in data_groups
    ):
        summaries = [
            _group_failure_summary(group)
            for group in data_groups[: min(3, len(data_groups))]
        ]
        raise RuntimeError(
            "Dressage fully async rollout produced no trainable samples; "
            "refusing to train on failed placeholder samples. "
            f"First failures: {' | '.join(summaries)}. "
            "Set DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=1 to keep the previous behavior."
        )

    return data_groups, staleness_filter.metrics_for_groups(data_groups)


def generate_rollout_fully_async(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
):
    if evaluation:
        raise ValueError("Dressage fully async rollout does not support evaluation mode")
    data, staleness_metrics = run(generate_rollout_async(args, rollout_id, data_buffer))
    metrics: dict[str, Any] = compute_multi_segment_metrics(
        [sample for group in data for sample in group]
    )
    metrics.update(staleness_metrics)
    if RolloutFnTrainOutput is None:
        return data
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


atexit.register(stop_global_worker)
