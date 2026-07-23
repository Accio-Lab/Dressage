"""Custom ``_convert_samples_to_train_data`` for Dressage multi-segment.

Registered via ``--custom-convert-samples-to-train-data-path``.

This file is an intentional near-verbatim copy of
``slime.ray.rollout.RolloutManager._convert_samples_to_train_data``
(current slime, ``slime/ray/rollout.py``).  Its training-semantic delta is
the ``rollout_mask_sums`` computation: when the estimator is GRPO /
Reinforce++ baseline, we substitute prompt-equal denominators
(``M_P × N_P / gbs``) for slime's default trajectory-equal per-rollout mask
totals.  MOPD also encodes its per-sample teacher route in slime's existing
train-side ``prompt`` passthrough slot.

When updating slime, diff this file against the new
``_convert_samples_to_train_data`` and carry over any additions.
"""

from __future__ import annotations

import os
from typing import Any

_PROMPT_EQUAL_ESTIMATORS = ("grpo", "reinforce_plus_plus_baseline")


def _prompt_equal_rollout_mask_sums(
    args: Any,
    samples: list,
    loss_masks: list[list[int]],
) -> list[float]:
    """Prompt-equal denominators for ``rollout_mask_sums``.

    Each sample's denom is ``M_P × N_P / gbs`` where:
      - M_P = total mask-sum across all samples belonging to prompt P
      - N_P = number of prompts with at least one live sample
      - gbs = global_batch_size

    Dead samples (``remove_sample=True``) are excluded from M_P and N_P.
    Their loss_mask is zeroed upstream, so they contribute 0 to loss
    regardless of denom.
    """
    prompt_token_counts: dict[str, int] = {}
    real_prompt_ids: set[str] = set()

    for i, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", None) or {}
        ptid = metadata.get("parent_traj_id")
        pid = metadata.get("instance_id")
        is_dead = bool(getattr(sample, "remove_sample", False))

        assert ptid is not None, (
            f"sample at index {getattr(sample, 'index', '?')} has no "
            "metadata['parent_traj_id']; multi-segment training requires this."
        )
        assert pid is not None, (
            f"sample at index {getattr(sample, 'index', '?')} has no "
            "metadata['instance_id']; prompt-equal aggregation requires this."
        )
        if not is_dead:
            assert getattr(sample, "group_index", None) is not None, (
                f"real sample at index {getattr(sample, 'index', '?')} "
                f"(instance_id={pid!r}) has group_index=None; the _NONE_GROUP "
                "sentinel reserved for dead samples would contaminate its "
                "advantage."
            )

        if not is_dead:
            real_prompt_ids.add(str(pid))
            prompt_token_counts[str(pid)] = prompt_token_counts.get(str(pid), 0) + sum(
                loss_masks[i]
            )

    prompt_count = len(real_prompt_ids)
    gbs = int(getattr(args, "global_batch_size", 0))
    scale = prompt_count / gbs if gbs > 0 and prompt_count > 0 else 0.0

    result: list[float] = []
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        pid = str(metadata.get("instance_id", ""))
        result.append(float(prompt_token_counts.get(pid, 0)) * scale)
    return result


def convert_samples_to_train_data(args: Any, samples: list) -> dict:
    """Dressage replacement for ``RolloutManager._convert_samples_to_train_data``."""
    from dressage.training.reward_post_process import reward_post_process

    raw_rewards, rewards = reward_post_process(args, samples)

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    rollout_ids = [sample.rollout_id for sample in samples]
    existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
    tmp_id = 0
    for i in range(len(rollout_ids)):
        if rollout_ids[i] is None:
            while tmp_id in existed_rollout_id_values:
                tmp_id += 1
            rollout_ids[i] = tmp_id
            existed_rollout_id_values.add(tmp_id)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [
            1 if sample.status == sample.Status.TRUNCATED else 0 for sample in samples
        ],
        "sample_indices": [sample.index for sample in samples],
        "rollout_ids": rollout_ids,
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    # Per-rollout aggregate, precomputed at the step level (where we can
    # see every sample of every rollout) and broadcast per-sample so the
    # per-mb loss reducer uses the correct whole-rollout denominator even
    # when a rollout's samples land in different micro-batches (first-fit
    # packing can split a rollout across mbs):
    #
    #   ``rollout_mask_sums[i]`` — sum of loss-mask totals over every
    #   sample in sample i's rollout. Used as the reducer's denominator
    #   so summing partial contributions across mbs yields one
    #   token-weighted mean per rollout.
    rollout_id_list = train_data["rollout_ids"]
    mask_sums_per_sample = [sum(m) for m in loss_masks]
    if getattr(args, "advantage_estimator", None) in _PROMPT_EQUAL_ESTIMATORS:
        train_data["rollout_mask_sums"] = _prompt_equal_rollout_mask_sums(
            args,
            samples,
            loss_masks,
        )
    else:
        rollout_total_mask: dict[int, int] = {}
        for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
            rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
        train_data["rollout_mask_sums"] = [
            rollout_total_mask[rid] for rid in rollout_id_list
        ]

    # Overwrite raw_reward when available. Mixed-source batches may only
    # populate this field for a subset of samples (e.g. SWE but not code).
    if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
        train_data["raw_reward"] = [
            (
                sample.metadata["raw_reward"]
                if sample.metadata and "raw_reward" in sample.metadata
                else sample.reward
            )
            for sample in samples
        ]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [
            sample.metadata["round_number"] for sample in samples
        ]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [
            sample.rollout_log_probs for sample in samples
        ]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [
            sample.rollout_routed_experts for sample in samples
        ]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [
            sample.multimodal_train_inputs for sample in samples
        ]

    config_path = getattr(args, "mopd_teacher_config", None) or os.environ.get(
        "DRESSAGE_MOPD_TEACHER_CONFIG"
    )
    if config_path:
        if (
            not bool(getattr(args, "use_opd", False))
            or getattr(args, "opd_type", None) != "megatron"
        ):
            raise ValueError(
                "MOPD train conversion requires --use-opd --opd-type megatron"
            )
        from dressage.rollout.mopd import (
            collect_mopd_teacher_ids,
            load_mopd_config,
        )

        teacher_ids = collect_mopd_teacher_ids(
            samples, load_mopd_config(str(config_path))
        )

        # Slime already DP-partitions and forwards this train-side field.  It
        # is deliberately not Sample.prompt: rollout generation and tokenization
        # are complete before this converter runs.
        train_data["prompt"] = teacher_ids
    elif samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [
            sample.teacher_log_probs for sample in samples
        ]

    return train_data
