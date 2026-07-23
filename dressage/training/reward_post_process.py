"""Slime reward post-processing hook for GRPO advantage normalization.

Registered via ``--custom-reward-post-process-path``.  Handles:
  - Per-group GRPO mean-subtraction (with optional std normalization).
  - Multi-segment trajectory broadcast: the anchor segment's normalized
    reward is copied to every sibling segment of the trajectory.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _compute_parent_groups(
    samples: list,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Group sample indices by parent_traj_id and pick each trajectory's anchor.

    Returns:
      parent_groups: ptid -> list of sample indices belonging to that
        trajectory (in input order).
      parent_anchor: ptid -> the single anchor index of that trajectory.
        The anchor is the segment with the highest segment_index — by plan
        §1.3 it's the only segment that ran reward_fn, so it carries the
        trajectory's terminal reward (every other segment was pre-set to
        reward=0.0 by multi_segment.expand_segments_to_samples).
    """
    parent_groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        if getattr(s, "remove_sample", False):
            continue
        ptid = s.metadata.get("parent_traj_id") if s.metadata else None
        if ptid:
            parent_groups[ptid].append(i)

    def _segment_index_of(idx: int) -> int:
        meta = samples[idx].metadata or {}
        return int(meta.get("segment_index", 0))

    parent_anchor: dict[str, int] = {}
    for ptid, seg_indices in parent_groups.items():
        parent_anchor[ptid] = max(seg_indices, key=_segment_index_of)

    return parent_groups, parent_anchor


def _broadcast_to_segments(
    values: list[float],
    parent_groups: dict[str, list[int]],
    parent_anchor: dict[str, int],
) -> None:
    """In-place broadcast values[anchor_idx] to every segment of each parent.

    Used for ``rewards`` (= GRPO advantage) so every segment of a trajectory
    sees the same advantage. NOT used for ``raw_rewards`` — that value stays
    sparse (anchor carries the trajectory's terminal reward, every other
    segment is 0) so downstream consumers can sum within a trajectory to
    recover the terminal reward (relied on by
    ``log_rollout_data.compute_trajectory_mean_raw_reward`` for the wandb
    trajectory-level mean metric, and by ``slime`` for correct-length stats).
    """
    for ptid, seg_indices in parent_groups.items():
        anchor_idx = parent_anchor[ptid]
        if anchor_idx >= len(values):
            logger.warning(
                "parent_traj_id=%s anchor index %d out of range",
                ptid, anchor_idx,
            )
            continue
        representative = values[anchor_idx]
        for idx in seg_indices:
            values[idx] = representative


def reward_post_process(
    args: Any, samples: list
) -> tuple[list[float], list[float]]:
    """Compute raw and normalized rewards for GRPO advantage estimation.

    For standard trajectories:
      - raw_rewards = [s.reward for s in samples]
      - rewards = normalize per group_index group (GRPO advantage).
      - Samples marked remove_sample do not participate in normalization.

    For rewrite-aware segmented trajectories (samples carry
    ``metadata['parent_traj_id']``):
      - Group segments by parent_traj_id.
      - Normalize at the parent level (anchor = highest segment_index).
      - Broadcast the same advantage to every segment of the trajectory.

    ``raw_rewards`` is intentionally NOT broadcast: it stays sparse, with
    the anchor segment carrying the trajectory's terminal reward and every
    other segment carrying the placeholder 0.0 set by
    ``multi_segment.expand_segments_to_samples``. Broadcasting raw would
    make a long trajectory contribute N times more weight to wandb's
    ``rollout/raw_reward`` than a single-segment trajectory with the same
    terminal reward. The trajectory-level mean is emitted as a separate
    scalar (``raw_reward_trajectory_mean``) by
    ``dressage.rollout.log_rollout`` instead — see that module for
    the sum-per-trajectory invariant relied on here.

    Returns: (raw_rewards, processed_rewards)
    """
    raw_rewards = []
    for s in samples:
        r = s.reward
        if isinstance(r, dict):
            reward_key = getattr(args, "reward_key", None)
            r = r[reward_key] if reward_key else 0.0
        raw_rewards.append(float(r) if r is not None else 0.0)

    parent_groups, parent_anchor = _compute_parent_groups(samples)

    advantage_estimator = getattr(args, "advantage_estimator", "grpo")
    do_normalization = getattr(args, "rewards_normalization", True)

    if advantage_estimator not in ("grpo", "gspo", "reinforce_plus_plus_baseline") or not do_normalization:
        rewards = list(raw_rewards)
        if parent_groups:
            _broadcast_to_segments(rewards, parent_groups, parent_anchor)
        return raw_rewards, rewards

    grpo_std = getattr(args, "grpo_std_normalization", False)

    _NONE_GROUP = -1

    parent_representative_indices = set(parent_anchor.values())
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, s in enumerate(samples):
        if getattr(s, "remove_sample", False):
            continue
        gi = s.group_index if s.group_index is not None else _NONE_GROUP
        if i in parent_representative_indices:
            groups[gi].append((i, raw_rewards[i]))
        elif not (s.metadata and s.metadata.get("parent_traj_id")):
            groups[gi].append((i, raw_rewards[i]))

    rewards = list(raw_rewards)

    for gi, members in groups.items():
        indices = [m[0] for m in members]
        values = [m[1] for m in members]

        mean_val = sum(values) / len(values) if values else 0.0
        normalized = [v - mean_val for v in values]

        if advantage_estimator in ("grpo", "gspo") and grpo_std:
            var = sum(v ** 2 for v in normalized) / len(normalized) if normalized else 0.0
            std = var ** 0.5
            if std > 1e-6:
                normalized = [v / std for v in normalized]

        for idx, norm_val in zip(indices, normalized):
            rewards[idx] = norm_val

    _broadcast_to_segments(rewards, parent_groups, parent_anchor)

    return raw_rewards, rewards
