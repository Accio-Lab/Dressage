"""Dressage rollout-side logging hook.

Registered via ``--custom-rollout-log-function-path``. Adds the historical
aggregate trajectory reward and, for MOPD samples, one trainable-trajectory
reward curve per routed teacher. It then returns ``False`` so slime's default
``_log_rollout_data`` continues with its standard metric collection.

The trajectory-mean metric gives a clean "fraction of trajectories that
got reward 1.0" (for binary rewards) without multi-segment trajectories
contributing N times more weight than single-segment ones.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from dressage.training.log_helpers import compute_trajectory_mean_raw_reward

_STALENESS_WANDB_METRICS_DEFINED = False


def _metric_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return component or "unnamed"


def _define_staleness_wandb_metrics(args: Any) -> None:
    global _STALENESS_WANDB_METRICS_DEFINED
    if _STALENESS_WANDB_METRICS_DEFINED or not getattr(args, "use_wandb", False):
        return

    import wandb

    if wandb.run is None:
        return
    wandb.define_metric("staleness/*", step_metric="rollout/step")
    _STALENESS_WANDB_METRICS_DEFINED = True


def _sample_has_trainable_loss(sample: Any) -> bool:
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


def _trajectory_key(sample: Any, metadata: dict[str, Any], position: int) -> str:
    """Return a stable reward-aggregation key for a routed trajectory."""
    value = (
        metadata.get("parent_traj_id")
        or metadata.get("session_id")
        or getattr(sample, "session_id", None)
        or metadata.get("rollout_id")
        or getattr(sample, "rollout_id", None)
    )
    return str(value) if value not in (None, "") else f"sample:{position}"


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list,
    extra_metrics: dict,
    rollout_time: float,
) -> bool:
    """Append trajectory-mean raw_reward to *extra_metrics*, then let slime log.

    Reads ``parent_traj_id``, ``segment_index``, and ``reward`` from each
    sample's ``metadata`` dict.  Samples without ``parent_traj_id`` are
    skipped (single-segment mode).

    Returns ``False`` so slime's default logging continues — the extra
    metric rides through ``log_dict = {**(rollout_extra_metrics or {})}``
    in slime's ``_log_rollout_data``.
    """
    _define_staleness_wandb_metrics(args)

    parent_traj_ids: list[str] = []
    segment_indices: list[int] = []
    raw_rewards: list[float] = []

    routed_parent_ids: dict[str, list[str]] = defaultdict(list)
    routed_segment_indices: dict[str, list[int]] = defaultdict(list)
    routed_rewards: dict[str, list[float]] = defaultdict(list)

    has_multi_segment = False
    for position, sample in enumerate(samples):
        meta = getattr(sample, "metadata", None) or {}
        ptid = meta.get("parent_traj_id")
        if ptid is not None:
            has_multi_segment = True

        if not _sample_has_trainable_loss(sample):
            continue
        trajectory_key = (
            str(ptid) if ptid is not None else _trajectory_key(sample, meta, position)
        )
        segment_index = int(meta.get("segment_index", 0))
        r = getattr(sample, "reward", None)
        reward = float(r) if r is not None else 0.0
        if ptid is not None:
            parent_traj_ids.append(trajectory_key)
            segment_indices.append(segment_index)
            raw_rewards.append(reward)

        teacher_id_value = meta.get("teacher_id")
        teacher_id = (
            str(teacher_id_value).strip()
            if teacher_id_value not in (None, "")
            else None
        )
        if teacher_id is not None:
            routed_parent_ids[teacher_id].append(trajectory_key)
            routed_segment_indices[teacher_id].append(segment_index)
            routed_rewards[teacher_id].append(reward)

    if has_multi_segment:
        traj_mean = compute_trajectory_mean_raw_reward(
            parent_traj_ids, raw_rewards, segment_indices
        )
        if traj_mean is not None:
            extra_metrics["rollout/raw_reward_trajectory_mean"] = traj_mean

    for teacher_id, teacher_parent_ids in routed_parent_ids.items():
        routed_mean = compute_trajectory_mean_raw_reward(
            teacher_parent_ids,
            routed_rewards[teacher_id],
            routed_segment_indices[teacher_id],
        )
        if routed_mean is not None:
            extra_metrics[
                "rollout/mopd/raw_reward_trainable_trajectory_mean/"
                f"{_metric_component(teacher_id)}"
            ] = routed_mean

    return False
