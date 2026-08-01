"""Synchronous rollout entrypoint for Dressage on colocate setups.

Used when actor and sglang share the same GPUs (e.g. qwen3.5-35B-A3B on
8xH100 with `--colocate`). Mirrors the dressage retry / empty-batch /
failure-summary semantics of `fully_async_rollout`, but runs to completion
per `rollout_id` so the framework can offload the sglang engine before
training kicks in.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from dressage.rollout.fully_async_rollout import (
    _allow_empty_train_batch,
    _flatten_multi_segment_result,
    _group_failure_summary,
    _group_has_trainable_tokens,
    _increment_retry,
    _is_aborted_group,
    _mark_no_grad_failed,
    _retry_count,
)

logger = logging.getLogger(__name__)

try:
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
    from slime.utils.async_utils import run
except ImportError:
    GenerateState = None  # type: ignore[assignment]
    generate_and_rm_group = None  # type: ignore[assignment]
    RolloutFnTrainOutput = None  # type: ignore[assignment]

    def run(coro):  # type: ignore[no-redef]
        return asyncio.run(coro)

from dressage.rollout.multi_segment import (
    compute_multi_segment_metrics,
    compute_prewarm_metrics,
)


def _max_retries() -> int:
    return int(os.environ.get("DRESSAGE_ROLLOUT_MAX_RETRIES", "2"))


async def _submit_group(
    args: Any,
    group: list[Any],
    state: Any,
    pendings: set[asyncio.Task],
    task_to_group: dict[asyncio.Task, list[Any]],
) -> None:
    if generate_and_rm_group is None:
        raise RuntimeError("slime.rollout.sglang_rollout.generate_and_rm_group is unavailable")
    task = asyncio.create_task(
        generate_and_rm_group(
            args,
            group,
            sampling_params=state.sampling_params.copy(),
            evaluation=False,
        )
    )
    pendings.add(task)
    task_to_group[task] = group


async def _run_sync_rollout(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
) -> list[list[Any]]:
    del rollout_id
    if GenerateState is None or generate_and_rm_group is None:
        raise RuntimeError(
            "Dressage sync rollout requires slime.rollout.sglang_rollout to be importable"
        )

    target = int(getattr(args, "rollout_batch_size", 1))
    max_retries = _max_retries()
    state = GenerateState(args)
    data: list[list[Any]] = []
    pendings: set[asyncio.Task] = set()
    task_to_group: dict[asyncio.Task, list[Any]] = {}

    groups = data_buffer.get_samples(target)
    if len(groups) < target:
        raise RuntimeError(
            f"data_buffer.get_samples({target}) returned {len(groups)} groups; "
            "Dressage sync rollout submits the whole batch one-shot and does not oversample."
        )
    for group in groups:
        await _submit_group(args, group, state, pendings, task_to_group)

    while pendings:
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group_for_task = task_to_group.pop(task)
            error: BaseException | None = None
            result_group: list[Any] | None = None
            try:
                result_group = task.result()
            except BaseException as exc:  # noqa: BLE001 - mirror fully_async behavior
                error = exc

            if result_group is not None:
                result_group = _flatten_multi_segment_result(result_group)

            failed = error is not None or _is_aborted_group(result_group or group_for_task)
            if not failed:
                data.append(result_group)
                continue

            summary = _group_failure_summary(
                result_group if result_group is not None else group_for_task, error
            )
            if _retry_count(group_for_task) < max_retries:
                _increment_retry(group_for_task)
                logger.warning(
                    "resubmitting rollout group for retry (attempt %d/%d): %s",
                    _retry_count(group_for_task),
                    max_retries,
                    summary,
                )
                await _submit_group(args, group_for_task, state, pendings, task_to_group)
            else:
                logger.error(
                    "rollout group exhausted retries and will be marked failed: %s",
                    summary,
                )
                data.append(_mark_no_grad_failed(group_for_task, error))

    state.reset()

    data = sorted(data, key=lambda group: getattr(group[0], "index", 0))
    if not _allow_empty_train_batch() and not any(
        _group_has_trainable_tokens(group) for group in data
    ):
        summaries = [_group_failure_summary(group) for group in data[: min(3, len(data))]]
        raise RuntimeError(
            "Dressage sync rollout produced no trainable samples; "
            "refusing to train on failed placeholder samples. "
            f"First failures: {' | '.join(summaries)}. "
            "Set DRESSAGE_ALLOW_EMPTY_TRAIN_BATCH=1 to keep the previous behavior."
        )

    return data


def generate_rollout_sync(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
):
    if evaluation:
        raise ValueError("Dressage sync rollout does not support evaluation mode")
    data = run(_run_sync_rollout(args, rollout_id, data_buffer))
    flat = [sample for group in data for sample in group]
    metrics: dict[str, Any] = compute_multi_segment_metrics(flat)
    metrics.update(compute_prewarm_metrics(flat))
    if RolloutFnTrainOutput is None:
        return data
    return RolloutFnTrainOutput(samples=data, metrics=metrics)
