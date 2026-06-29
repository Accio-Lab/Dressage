"""Partially asynchronous rollout entrypoint for Dressage.

This rollout mode is intentionally close to :mod:`dressage.rollout.fully_async_rollout`:
a persistent worker keeps blackbox rollouts in flight, while the Slime rollout call
returns as soon as enough completed groups are available for the next trainer step.

The important difference from the existing fully-async entrypoint is the target size.
A Slime rollout can request ``rollout_batch_size * n_samples_per_prompt`` samples, but
training may only need ``global_batch_size`` samples for one update. For example,
``rollout_batch_size=16`` and ``n_samples_per_prompt=8`` creates 128 samples. If
``global_batch_size=64``, this entrypoint can return the first 8 prompt groups while
other groups keep running in the background.

Weight updates must not happen while Dressage proxy/blackbox LLM calls are actively
emitting tokens. Use ``dressage.training.train_async_with_rollout_pause`` so trainer
weight updates pause/resume the Dressage proxy without modifying Slime source.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from dressage.rollout.fully_async_rollout import (
    _FallbackGenerateState,
    _allow_empty_train_batch,
    _flatten_multi_segment_result,
    _group_failure_summary,
    _group_has_staleness_failure,
    _group_has_trainable_tokens,
    _increment_retry,
    _is_aborted_group,
    _mark_no_grad_failed,
    _retry_count,
)
from dressage.rollout.staleness import (
    PendingGroup,
    StalenessGroupFilter,
    StalenessTracker,
    config_from_args,
)

logger = logging.getLogger(__name__)

try:
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
    from slime.utils.async_utils import run
except ImportError:
    RolloutFnTrainOutput = None  # type: ignore[assignment]
    GenerateState = None  # type: ignore[assignment]
    generate_and_rm_group = None  # type: ignore[assignment]

    def run(coro):  # type: ignore[no-redef]
        return asyncio.run(coro)


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


def _state_for(args: Any):
    if GenerateState is None:
        return _FallbackGenerateState(args)
    return GenerateState(args)


def _int_attr(args: Any, name: str, default: int) -> int:
    try:
        value = int(getattr(args, name, default) or default)
    except (TypeError, ValueError):
        value = default
    return value


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return parsed


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _is_final_rollout(args: Any, rollout_id: int) -> bool:
    try:
        num_rollout = int(getattr(args, "num_rollout"))
    except (AttributeError, TypeError, ValueError):
        return False
    return int(rollout_id) + 1 >= num_rollout


def _should_drain_worker_on_rollout(args: Any, rollout_id: int) -> bool:
    return _is_final_rollout(args, rollout_id) and _env_flag(
        "DRESSAGE_DRAIN_PARTIAL_WORKER_ON_FINAL_ROLLOUT",
        True,
    )


def _metadata_for(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    return metadata


def _annotate_submitted_group(group: list[Any], *, group_id: int, rollout_id: int) -> None:
    for sample in group:
        metadata = _metadata_for(sample)
        metadata.setdefault("dressage_start_rollout_id", rollout_id)
        metadata["dressage_async_group_id"] = group_id
        metadata["dressage_partial_rollout"] = True


def _annotate_returned_group(group: list[Any], *, group_id: int, rollout_id: int) -> None:
    for sample in group:
        metadata = _metadata_for(sample)
        metadata["dressage_return_rollout_id"] = rollout_id
        metadata["dressage_return_async_group_id"] = group_id
        metadata["dressage_partial_rollout_returned"] = True


def _partial_target_groups(args: Any) -> int:
    """Return the number of prompt groups needed before returning to Slime.

    Priority:
      1. ``DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS`` for exact prompt-group control.
      2. ``DRESSAGE_PARTIAL_ROLLOUT_TARGET_SAMPLES`` for sample-count control.
      3. ``args.global_batch_size`` when it is smaller than the full rollout sample
         count.
      4. ``args.rollout_batch_size`` as a compatibility fallback.
    """

    rollout_batch_size = max(1, _int_attr(args, "rollout_batch_size", 1))
    n_samples_per_prompt = max(1, _int_attr(args, "n_samples_per_prompt", 1))
    full_sample_count = rollout_batch_size * n_samples_per_prompt

    target_groups = _env_int("DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS")
    if target_groups is not None:
        return max(1, min(rollout_batch_size, target_groups))

    target_samples = _env_int("DRESSAGE_PARTIAL_ROLLOUT_TARGET_SAMPLES")
    if target_samples is None:
        global_batch_size = _int_attr(args, "global_batch_size", full_sample_count)
        if global_batch_size < full_sample_count:
            target_samples = global_batch_size
        else:
            target_samples = full_sample_count

    target_groups = int(math.ceil(max(1, target_samples) / n_samples_per_prompt))
    if target_samples % n_samples_per_prompt:
        logger.warning(
            "partial rollout target samples (%s) is not divisible by "
            "n_samples_per_prompt (%s); returning %s prompt groups (%s samples)",
            target_samples,
            n_samples_per_prompt,
            target_groups,
            target_groups * n_samples_per_prompt,
        )
    return max(1, min(rollout_batch_size, target_groups))


class PartialAsyncRolloutWorker:
    """Persistent worker that keeps blackbox rollout groups in flight.

    The worker deliberately continues across Slime rollout calls. This makes partial
    rollout useful: each call can return a trainer-sized subset, while already-started
    blackbox sessions finish and are consumed by a later call. The companion training
    entrypoint pauses the Dressage proxy around weight updates so those background
    sessions do not straddle an update while tokens are actively generated.
    """

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
        self._rollout_id_lock = threading.Lock()
        self._current_rollout_id = 0
        self.staleness = StalenessTracker(config_from_args(args))

    def start(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        self.worker_thread = threading.Thread(
            target=lambda: asyncio.run(self.continuous_worker_loop()),
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self, *, timeout: float | None = None) -> None:
        self.running = False
        if self.worker_thread is not None and self.worker_thread.is_alive():
            if timeout is None:
                timeout = _env_float("DRESSAGE_ASYNC_WORKER_STOP_TIMEOUT_SEC", 300.0)
            self.worker_thread.join(timeout=timeout)
            if self.worker_thread.is_alive():
                logger.warning(
                    "Dressage partial async rollout worker did not stop after %.1fs; "
                    "active blackbox sessions may continue in the background",
                    timeout,
                )

    def set_rollout_id(self, rollout_id: int) -> None:
        with self._rollout_id_lock:
            self._current_rollout_id = int(rollout_id)

    def _current_rollout(self) -> int:
        with self._rollout_id_lock:
            return self._current_rollout_id

    async def continuous_worker_loop(self) -> None:
        logger.info("Dressage partial async rollout worker started")
        state = _state_for(self.args)
        active: dict[asyncio.Task, tuple[int, list[Any]]] = {}

        while self.running:
            done_tasks = [task for task in active if task.done()]
            for task in done_tasks:
                group_id, group = active.pop(task)
                try:
                    result = _flatten_multi_segment_result(task.result())
                    self._put_completed(CompletedGroup(group_id, original_group=group, result=result))
                except BaseException as exc:
                    self._put_completed(CompletedGroup(group_id, original_group=group, error=exc))

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
                _annotate_submitted_group(
                    group,
                    group_id=group_id,
                    rollout_id=self._current_rollout(),
                )
                task = asyncio.create_task(
                    self._run_group(group, state.sampling_params.copy())
                )
                active[task] = (group_id, group)

            await asyncio.sleep(0.01)

        if active:
            await asyncio.wait(active.keys())
            for task, (group_id, group) in active.items():
                try:
                    result = _flatten_multi_segment_result(task.result())
                    self._put_completed(CompletedGroup(group_id, original_group=group, result=result))
                except BaseException as exc:
                    self._put_completed(CompletedGroup(group_id, original_group=group, error=exc))
        logger.info("Dressage partial async rollout worker stopped")

    async def _run_group(self, group: list[Any], sampling_params: dict[str, Any]) -> list[Any]:
        if generate_and_rm_group is None:
            raise RuntimeError("slime.rollout.sglang_rollout.generate_and_rm_group is unavailable")
        return await generate_and_rm_group(
            self.args,
            group,
            sampling_params=sampling_params,
            evaluation=False,
        )

    def _put_completed(self, item: CompletedGroup) -> None:
        while True:
            try:
                self.output_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if not self.running:
                    logger.warning(
                        "dropping completed partial rollout group %s because the "
                        "output queue is full while the worker is stopping",
                        item.group_id,
                    )
                    return

    def get_completed_groups(self) -> list[CompletedGroup]:
        completed: list[CompletedGroup] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                return completed

    def return_completed_groups(self, completed: list[CompletedGroup]) -> None:
        # A partial call may drain more completed groups than it needs. Put the
        # leftovers back so a later rollout call can consume them instead of
        # silently dropping already-finished blackbox work.
        for item in completed:
            self._put_completed(item)

    def queued_completed_count(self) -> int:
        return self.output_queue.qsize()


_GLOBAL_PARTIAL_WORKER: PartialAsyncRolloutWorker | None = None
_WORKER_LOCK = threading.Lock()


def get_global_partial_worker(
    args: Any,
    data_buffer: Any,
    *,
    rollout_id: int | None = None,
) -> PartialAsyncRolloutWorker:
    global _GLOBAL_PARTIAL_WORKER
    with _WORKER_LOCK:
        if (
            _GLOBAL_PARTIAL_WORKER is None
            or _GLOBAL_PARTIAL_WORKER.worker_thread is None
            or not _GLOBAL_PARTIAL_WORKER.worker_thread.is_alive()
        ):
            _GLOBAL_PARTIAL_WORKER = PartialAsyncRolloutWorker(args, data_buffer)
            if rollout_id is not None:
                _GLOBAL_PARTIAL_WORKER.set_rollout_id(rollout_id)
            _GLOBAL_PARTIAL_WORKER.start()
        elif rollout_id is not None:
            _GLOBAL_PARTIAL_WORKER.set_rollout_id(rollout_id)
        return _GLOBAL_PARTIAL_WORKER


def stop_global_partial_worker(*, timeout: float | None = None) -> list[CompletedGroup]:
    global _GLOBAL_PARTIAL_WORKER
    with _WORKER_LOCK:
        worker = _GLOBAL_PARTIAL_WORKER
        _GLOBAL_PARTIAL_WORKER = None

    if worker is None:
        return []

    worker.stop(timeout=timeout)
    return worker.get_completed_groups()


async def generate_rollout_partial_async_impl(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
):
    worker = get_global_partial_worker(args, data_buffer, rollout_id=rollout_id)

    target_groups = _partial_target_groups(args)
    full_groups = max(1, _int_attr(args, "rollout_batch_size", target_groups))
    n_samples_per_prompt = max(1, _int_attr(args, "n_samples_per_prompt", 1))
    max_retries = int(os.environ.get("DRESSAGE_ROLLOUT_MAX_RETRIES", "2"))
    completed_by_id: dict[int, CompletedGroup] = {}
    data: list[PendingGroup] = []
    dropped_failed_groups = 0
    staleness_rejected_groups = 0
    staleness_rejected_samples = 0
    dropped_failure_summaries: list[str] = []
    retried_groups = 0
    last_progress_time = time.time()
    no_progress_timeout = float(os.environ.get("DRESSAGE_ASYNC_NO_PROGRESS_WARN_SEC", "600"))
    max_dropped_failed_groups = int(
        os.environ.get(
            "DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS",
            str(max(target_groups * 10, 100)),
        )
    )
    drain_final_worker = _should_drain_worker_on_rollout(args, rollout_id)
    drained_completed_groups = 0
    staleness_filter = StalenessGroupFilter(
        tracker=worker.staleness,
        rollout_name="partial",
    )

    logger.info(
        "starting Dressage partial async rollout: rollout_id=%s target_groups=%s "
        "target_samples=%s full_groups=%s full_samples=%s",
        rollout_id,
        target_groups,
        target_groups * n_samples_per_prompt,
        full_groups,
        full_groups * n_samples_per_prompt,
    )

    while len(data) < target_groups:
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
            if len(data) >= target_groups:
                break
            completed = completed_by_id.pop(group_id)
            if completed.is_failed:
                failed_group = completed.result or completed.original_group
                staleness_failure = _group_has_staleness_failure(
                    failed_group,
                    completed.error,
                )
                if staleness_failure:
                    staleness_rejected_groups += 1
                    staleness_rejected_samples += len(completed.original_group)
                summary = _group_failure_summary(
                    failed_group, completed.error
                )
                if _retry_count(completed.original_group) < max_retries:
                    _increment_retry(completed.original_group)
                    data_buffer.add_samples([completed.original_group])
                    retried_groups += 1
                    logger.warning(
                        "returned group %s to rollout buffer for retry during partial rollout: %s",
                        group_id,
                        summary,
                    )
                    continue
                dropped_failed_groups += 1
                if len(dropped_failure_summaries) < 3:
                    dropped_failure_summaries.append(summary)
                logger.error(
                    "partial rollout group %s exhausted retries and will be dropped: %s",
                    group_id,
                    summary,
                )
                if dropped_failed_groups >= max_dropped_failed_groups:
                    raise RuntimeError(
                        "Dressage partial async rollout dropped too many failed groups "
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
            _annotate_returned_group(group, group_id=group_id, rollout_id=rollout_id)
            data.append(PendingGroup(group_id=group_id, samples=group))
            last_progress_time = time.time()

        now = time.time()
        if now - last_progress_time > no_progress_timeout:
            logger.warning(
                "no completed partial rollout group for %.1fs; collected %d/%d",
                no_progress_timeout,
                len(data),
                target_groups,
            )
            last_progress_time = now
        if len(data) < target_groups:
            await asyncio.sleep(0.01)

    if completed_by_id:
        leftovers = list(completed_by_id.values())
        if drain_final_worker:
            drained_completed_groups += len(leftovers)
            logger.info(
                "dropping %d extra completed partial rollout groups after final rollout",
                len(leftovers),
            )
        else:
            worker.return_completed_groups(leftovers)
        completed_by_id.clear()

    if drain_final_worker:
        logger.info(
            "final rollout %s completed; stopping and draining partial async worker",
            rollout_id,
        )
        drained = stop_global_partial_worker()
        drained_completed_groups += len(drained)
        if drained:
            logger.info(
                "drained %d completed partial rollout groups after final rollout",
                len(drained),
            )

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
            "Dressage partial async rollout produced no trainable samples; "
            "refusing to train on failed placeholder samples. "
            f"First failures: {' | '.join(summaries)}. "
            "Set DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=1 to keep the previous behavior."
        )

    from dressage.rollout.multi_segment import compute_multi_segment_metrics

    metrics: dict[str, Any] = compute_multi_segment_metrics(
        [sample for group in data_groups for sample in group]
    )
    metrics.update(staleness_filter.metrics_for_groups(data_groups))
    metrics.update({
        "dressage/partial_rollout_target_groups": target_groups,
        "dressage/partial_rollout_target_samples": target_groups * n_samples_per_prompt,
        "dressage/partial_rollout_full_groups": full_groups,
        "dressage/partial_rollout_full_samples": full_groups * n_samples_per_prompt,
        "dressage/partial_rollout_returned_groups": len(data_groups),
        "dressage/partial_rollout_returned_samples": sum(len(group) for group in data_groups),
        "dressage/partial_rollout_retried_groups": retried_groups,
        "dressage/partial_rollout_failed_groups": dropped_failed_groups,
        "dressage/partial_rollout_dropped_failed_groups": dropped_failed_groups,
        "staleness/partial_rollout_rejected_groups": staleness_rejected_groups,
        "staleness/partial_rollout_rejected_samples": staleness_rejected_samples,
        "dressage/partial_rollout_queued_completed_groups": worker.queued_completed_count(),
        "dressage/partial_rollout_final_worker_drain": int(drain_final_worker),
        "dressage/partial_rollout_drained_completed_groups": drained_completed_groups,
    })
    if RolloutFnTrainOutput is not None:
        return RolloutFnTrainOutput(samples=data_groups, metrics=metrics)
    return data_groups


def generate_rollout_partial_async(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
):
    if evaluation:
        raise ValueError("Dressage partial async rollout does not support evaluation mode")
    return run(generate_rollout_partial_async_impl(args, rollout_id, data_buffer))


atexit.register(stop_global_partial_worker)
