"""Dynamic-sampling filters for synchronous Harbor GRPO rollout."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Any

try:
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput
except ImportError:  # optional Slime dependency
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class DynamicFilterOutput:  # type: ignore[no-redef]
        keep: bool
        reason: str | None = None


def _reward_value(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("reward")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def check_harbor_complete_reward_nonzero_std(
    args: Any, samples: list[Any], **_: Any
) -> DynamicFilterOutput:
    """Keep a complete group with positive reward and non-zero variance.

    Rewards are deduplicated by ``harbor_attempt_id`` because one trajectory
    may contain multiple token segments.  "Positive reward" means at least
    one selected trajectory is positive; requiring every binary reward to be
    positive would contradict the non-zero-variance requirement.
    """

    if not samples:
        return DynamicFilterOutput(keep=False, reason="empty_group")
    by_attempt: OrderedDict[str, float] = OrderedDict()
    for sample in samples:
        if bool(getattr(sample, "remove_sample", False)):
            return DynamicFilterOutput(keep=False, reason="remove_sample")
        metadata = getattr(sample, "metadata", None) or {}
        if metadata.get("harbor_group_zero_grad"):
            return DynamicFilterOutput(keep=False, reason="zero_grad_group")
        attempt_id = str(metadata.get("harbor_attempt_id") or "")
        if not attempt_id:
            return DynamicFilterOutput(keep=False, reason="missing_attempt_id")
        candidate = _reward_value(metadata.get("harbor_rewards"))
        if candidate is None:
            candidate = _reward_value(getattr(sample, "reward", None))
        if candidate is None:
            continue
        previous = by_attempt.get(attempt_id)
        if previous is not None and abs(previous - candidate) > 1e-6:
            return DynamicFilterOutput(keep=False, reason="inconsistent_attempt_reward")
        by_attempt[attempt_id] = candidate

    expected = int(getattr(args, "n_samples_per_prompt", 1))
    if len(by_attempt) != expected:
        return DynamicFilterOutput(
            keep=False, reason=f"incomplete_group_{len(by_attempt)}_of_{expected}"
        )
    rewards = list(by_attempt.values())
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    if not math.isfinite(variance) or variance <= 1e-12:
        return DynamicFilterOutput(keep=False, reason="zero_reward_std")
    if not any(value > 0 for value in rewards):
        return DynamicFilterOutput(keep=False, reason="no_positive_reward")
    return DynamicFilterOutput(keep=True, reason=None)


def check_harbor_quorum_reward_nonzero_std(
    args: Any, samples: list[Any], **_: Any
) -> DynamicFilterOutput:
    """Keep a partially complete group only after its frozen quorum is met.

    ``remove_sample`` entries are allowed only as explicit zero-gradient slot
    placeholders.  Rewards are computed from unique successful attempts, so a
    multi-segment trajectory still contributes exactly once.
    """

    if not samples:
        return DynamicFilterOutput(keep=False, reason="empty_group")
    expected = int(getattr(args, "n_samples_per_prompt", 1))
    ratio = float(os.environ.get("ROLLOUT_GROUP_MIN_SUCCESS_RATIO", "1.0"))
    if not 0 < ratio <= 1:
        return DynamicFilterOutput(keep=False, reason="invalid_success_ratio")
    required = math.ceil(expected * ratio)
    by_attempt: OrderedDict[str, float] = OrderedDict()
    successful_slots: dict[int, str] = {}
    placeholder_slots: set[int] = set()
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        if bool(getattr(sample, "remove_sample", False)):
            if metadata.get("harbor_partial_slot_zero_grad") is not True:
                return DynamicFilterOutput(
                    keep=False, reason="unexpected_remove_sample"
                )
            slot = metadata.get("harbor_attempt_ordinal")
            if not isinstance(slot, int) or slot < 0 or slot >= expected:
                return DynamicFilterOutput(keep=False, reason="invalid_zero_grad_slot")
            placeholder_slots.add(slot)
            continue
        if metadata.get("harbor_group_zero_grad"):
            return DynamicFilterOutput(keep=False, reason="zero_grad_group")
        if metadata.get("harbor_training_committed") is not True:
            return DynamicFilterOutput(keep=False, reason="uncommitted_attempt")
        attempt_id = str(metadata.get("harbor_attempt_id") or "")
        if not attempt_id:
            return DynamicFilterOutput(keep=False, reason="missing_attempt_id")
        slot = metadata.get("harbor_attempt_ordinal")
        if not isinstance(slot, int) or slot < 0 or slot >= expected:
            return DynamicFilterOutput(keep=False, reason="invalid_successful_slot")
        previous_id = successful_slots.get(slot)
        if previous_id is not None and previous_id != attempt_id:
            return DynamicFilterOutput(keep=False, reason="duplicate_successful_slot")
        successful_slots[slot] = attempt_id
        candidate = _reward_value(metadata.get("harbor_rewards"))
        if candidate is None:
            candidate = _reward_value(getattr(sample, "reward", None))
        if candidate is None:
            continue
        previous = by_attempt.get(attempt_id)
        if previous is not None and abs(previous - candidate) > 1e-6:
            return DynamicFilterOutput(keep=False, reason="inconsistent_attempt_reward")
        by_attempt[attempt_id] = candidate

    successful = len(by_attempt)
    if successful < required:
        return DynamicFilterOutput(
            keep=False, reason=f"quorum_{successful}_of_{required}"
        )
    if successful + len(placeholder_slots) != expected:
        return DynamicFilterOutput(
            keep=False,
            reason=(
                f"incomplete_group_{successful}_plus_"
                f"{len(placeholder_slots)}_of_{expected}"
            ),
        )
    if set(successful_slots).intersection(placeholder_slots):
        return DynamicFilterOutput(keep=False, reason="overlapping_slot_outcome")
    if len(successful_slots) != successful:
        return DynamicFilterOutput(keep=False, reason="duplicate_attempt_slot")
    rewards = list(by_attempt.values())
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    if not math.isfinite(variance) or variance <= 1e-12:
        return DynamicFilterOutput(keep=False, reason="zero_reward_std")
    if not any(value > 0 for value in rewards):
        return DynamicFilterOutput(keep=False, reason="no_positive_reward")
    return DynamicFilterOutput(keep=True, reason=None)
