"""Tests for dressage/rollout/convert_samples.py.

Cover the prompt-equal ``rollout_mask_sums`` computation
(``_cs._prompt_equal_rollout_mask_sums``) and the trajectory-equal
fallback, plus end-to-end ``convert_samples_to_train_data``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from dressage.rollout.multi_segment import mark_aborted_no_grad
from dressage.rollout import convert_samples as cs
from dressage.transport import TQ_SAMPLE_REF_METADATA_KEY


class _Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"


@dataclass
class SampleLike:
    group_index: int | None = 0
    index: int | None = 0
    rollout_id: int | None = None
    reward: float | None = 0.0
    metadata: dict = field(default_factory=dict)
    remove_sample: bool = False
    status: _Status = _Status.COMPLETED
    tokens: list[int] = field(default_factory=list)
    response_length: int = 0
    loss_mask: list[int] | None = None
    train_metadata: dict | None = None
    rollout_log_probs: list | None = None
    rollout_routed_experts: list | None = None
    multimodal_train_inputs: list | None = None
    teacher_log_probs: list | None = None
    Status = _Status


def _real_seg(
    *,
    ptid: str,
    mask: list[int],
    index: int = 0,
    group_index: int = 0,
    rollout_id: int | None = None,
    instance_id: str | None = None,
    reward: float = 0.5,
) -> SampleLike:
    s = SampleLike(
        group_index=group_index,
        index=index,
        rollout_id=rollout_id,
        reward=reward,
        metadata={
            "parent_traj_id": ptid,
            "instance_id": instance_id if instance_id is not None else f"prompt-{group_index}",
        },
        remove_sample=False,
        tokens=[0] * (len(mask) + 1),
        response_length=len(mask),
    )
    s.loss_mask = list(mask)
    return s


def _failed_seg(*, ptid: str, instance_id: str, group_index: int = 0, index: int = 0) -> SampleLike:
    s = SampleLike(
        group_index=group_index,
        index=index,
        rollout_id=None,
        reward=0.0,
        metadata={"parent_traj_id": ptid, "instance_id": instance_id},
        remove_sample=True,
        tokens=[0],
        response_length=0,
    )
    s.loss_mask = []
    return s


def _prompt_equal_args(*, gbs: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        reward_key=None,
        global_batch_size=gbs,
    )


def _traj_equal_args() -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator="gspo",
        rewards_normalization=False,
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        reward_key=None,
    )


# ---------- _cs._prompt_equal_rollout_mask_sums ----------


def test_prompt_equal_basic():
    """Two prompts, each with 2 tokens → M_P=2 each, N_P=2, gbs=4.
    denom = M_P × N_P / gbs = 2 × 2 / 4 = 1.0"""
    args = _prompt_equal_args(gbs=4)
    samples = [
        _real_seg(ptid="t1", instance_id="p1", mask=[1, 1], index=0),
        _real_seg(ptid="t2", instance_id="p2", mask=[1, 1], index=1),
    ]
    loss_masks = [s.loss_mask for s in samples]
    result = cs._prompt_equal_rollout_mask_sums(args, samples, loss_masks)
    assert result == [pytest.approx(1.0), pytest.approx(1.0)]


def test_prompt_equal_pools_across_trajectories():
    """All samples of the same prompt pool tokens into one M_P."""
    args = _prompt_equal_args(gbs=4)
    samples = [
        _real_seg(ptid="t1", instance_id="p1", mask=[1, 1, 1], index=0),
        _real_seg(ptid="t1", instance_id="p1", mask=[1, 0, 1], index=1),
        _real_seg(ptid="t2", instance_id="p1", mask=[1, 1], index=2),
        _real_seg(ptid="t3", instance_id="p2", mask=[1, 1, 1, 1], index=3),
    ]
    loss_masks = [s.loss_mask for s in samples]
    result = cs._prompt_equal_rollout_mask_sums(args, samples, loss_masks)
    # p1: M_P = 3+2+2 = 7, N_P = 2, scale = 2/4 = 0.5 → denom = 3.5
    # p2: M_P = 4, denom = 4 × 0.5 = 2.0
    assert result == [pytest.approx(3.5), pytest.approx(3.5), pytest.approx(3.5), pytest.approx(2.0)]


def test_prompt_equal_excludes_dead_samples():
    """Dead samples (remove_sample=True) don't contribute to M_P or N_P."""
    args = _prompt_equal_args(gbs=4)
    samples = [
        _real_seg(ptid="t1", instance_id="p1", mask=[1, 1], index=0, group_index=0),
        _failed_seg(ptid="t2", instance_id="p2", index=1, group_index=1),
        _failed_seg(ptid="t3", instance_id="p2", index=2, group_index=1),
    ]
    loss_masks = [s.loss_mask for s in samples]
    result = cs._prompt_equal_rollout_mask_sums(args, samples, loss_masks)
    # Only p1 is live → N_P = 1, M_P = 2, scale = 1/4 = 0.25
    # Live: denom = 2 × 0.25 = 0.5
    # Dead: denom = 0 (p2 has no live tokens)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.0)


def test_prompt_equal_missing_parent_traj_id_raises():
    args = _prompt_equal_args()
    samples = [SampleLike(metadata={"instance_id": "p1"}, tokens=[0], response_length=0)]
    samples[0].loss_mask = []
    with pytest.raises(AssertionError, match="parent_traj_id"):
        cs._prompt_equal_rollout_mask_sums(args, samples, [[]])


def test_prompt_equal_missing_instance_id_raises():
    args = _prompt_equal_args()
    samples = [SampleLike(metadata={"parent_traj_id": "t1"}, tokens=[0], response_length=0)]
    samples[0].loss_mask = []
    with pytest.raises(AssertionError, match="instance_id"):
        cs._prompt_equal_rollout_mask_sums(args, samples, [[]])


def test_prompt_equal_rejects_real_sample_with_none_group_index():
    args = _prompt_equal_args()
    s = _real_seg(ptid="t1", instance_id="p1", mask=[1, 1], index=0)
    s.group_index = None
    with pytest.raises(AssertionError, match="group_index"):
        cs._prompt_equal_rollout_mask_sums(args, [s], [s.loss_mask])


def test_prompt_equal_allows_none_group_index_on_dead_sample():
    args = _prompt_equal_args(gbs=4)
    dead = _failed_seg(ptid="t2", instance_id="p2", index=1)
    dead.group_index = None
    live = _real_seg(ptid="t1", instance_id="p1", mask=[1], index=0, group_index=0)
    result = cs._prompt_equal_rollout_mask_sums(
        args, [live, dead], [live.loss_mask, dead.loss_mask]
    )
    assert result[0] == pytest.approx(0.25)  # 1 × 1/4
    assert result[1] == pytest.approx(0.0)


# ---------- convert_samples_to_train_data ----------


def test_convert_samples_rollout_ids_from_rollout_id_field():
    args = _traj_equal_args()
    s1 = SampleLike(group_index=0, index=0, rollout_id=42, reward=0.5,
                     tokens=[1, 1], response_length=1)
    s1.loss_mask = [1]
    s2 = SampleLike(group_index=0, index=1, rollout_id=42, reward=0.5,
                     tokens=[1, 1, 1], response_length=2)
    s2.loss_mask = [1, 1]
    train_data = cs.convert_samples_to_train_data(args, [s1, s2])
    assert train_data["rollout_ids"] == [42, 42]


def test_convert_samples_assigns_tmp_rollout_ids_when_unset():
    args = _traj_equal_args()
    s1 = SampleLike(group_index=0, index=10, reward=0.5, tokens=[1], response_length=1)
    s1.loss_mask = [1]
    s2 = SampleLike(group_index=0, index=11, reward=0.5, tokens=[1], response_length=1)
    s2.loss_mask = [1]
    train_data = cs.convert_samples_to_train_data(args, [s1, s2])
    assert train_data["rollout_ids"] == [0, 1]


def test_convert_samples_trajectory_equal_rollout_mask_sums():
    args = _traj_equal_args()
    s1 = SampleLike(group_index=0, index=0, rollout_id=7, reward=0.5,
                     tokens=[1, 1, 1], response_length=2)
    s1.loss_mask = [1, 1]
    s2 = SampleLike(group_index=0, index=1, rollout_id=7, reward=0.5,
                     tokens=[1, 1, 1, 1], response_length=3)
    s2.loss_mask = [1, 1, 1]
    s3 = SampleLike(group_index=1, index=2, rollout_id=9, reward=0.5,
                     tokens=[1, 1], response_length=1)
    s3.loss_mask = [1]
    train_data = cs.convert_samples_to_train_data(args, [s1, s2, s3])
    assert train_data["rollout_mask_sums"] == [5, 5, 1]


def test_convert_samples_dead_sample_mask_zeroed():
    args = _traj_equal_args()
    s1 = SampleLike(group_index=0, index=0, rollout_id=7, reward=0.0,
                     remove_sample=True, tokens=[1, 1], response_length=1)
    s1.loss_mask = [1]
    train_data = cs.convert_samples_to_train_data(args, [s1])
    assert train_data["loss_masks"] == [[0]]
    assert train_data["rollout_mask_sums"] == [0]


def test_convert_samples_passes_only_remote_fields_through_prompt():
    args = _traj_equal_args()
    args.use_rollout_routing_replay = True
    args.num_layers = 2
    args.moe_router_topk = 2
    live = _real_seg(ptid="t1", instance_id="p1", mask=[1, 1])
    live.rollout_log_probs = None
    live.rollout_routed_experts = None
    layouts = {
        "full_logprobs": {"schema_version": "layout"},
        "routed_experts": {"schema_version": "layout"},
    }
    live.metadata[TQ_SAMPLE_REF_METADATA_KEY] = layouts
    failed = _failed_seg(ptid="t2", instance_id="p2", index=1)

    train_data = cs.convert_samples_to_train_data(args, [live, failed])

    assert train_data["prompt"] == [layouts, None]
    assert "rollout_log_probs" not in train_data
    assert "rollout_routed_experts" not in train_data


def test_convert_samples_keeps_unselected_r3_on_normal_path():
    args = _traj_equal_args()
    args.use_rollout_routing_replay = True
    args.num_layers = 2
    args.moe_router_topk = 2
    live = _real_seg(ptid="t1", instance_id="p1", mask=[1, 1])
    live.rollout_log_probs = None
    live.rollout_routed_experts = np.ones((2, 2, 2), dtype=np.int32)
    layouts = {"full_logprobs": {"schema_version": "layout"}}
    live.metadata[TQ_SAMPLE_REF_METADATA_KEY] = layouts
    failed = _failed_seg(ptid="t2", instance_id="p2", index=1)

    train_data = cs.convert_samples_to_train_data(args, [live, failed])

    assert train_data["prompt"] == [layouts, None]
    assert "rollout_log_probs" not in train_data
    assert train_data["rollout_routed_experts"][0] is live.rollout_routed_experts
    assert train_data["rollout_routed_experts"][1].shape == (0, 2, 2)


def test_convert_samples_rejects_normal_actual_and_remote_field_mix():
    args = _traj_equal_args()
    remote = _real_seg(ptid="t1", instance_id="p1", mask=[1, 1])
    remote.rollout_log_probs = None
    remote.metadata[TQ_SAMPLE_REF_METADATA_KEY] = {
        "full_logprobs": {"schema_version": "layout"}
    }
    local = _real_seg(ptid="t2", instance_id="p2", mask=[1, 1], index=1)

    with pytest.raises(ValueError, match="cannot mix actual values"):
        cs.convert_samples_to_train_data(args, [remote, local])


def test_convert_samples_prompt_equal_when_multi_segment_grpo():
    args = _prompt_equal_args(gbs=4)
    samples = [
        _real_seg(ptid="t1", instance_id="p1", mask=[1, 1], index=0, rollout_id=0),
        _real_seg(ptid="t2", instance_id="p1", mask=[1, 1], index=1, rollout_id=1),
        _real_seg(ptid="t3", instance_id="p2", mask=[1, 1], index=2, rollout_id=2),
        _real_seg(ptid="t4", instance_id="p2", mask=[1, 1], index=3, rollout_id=3),
    ]
    train_data = cs.convert_samples_to_train_data(args, samples)
    # M_p1 = 4, M_p2 = 4, N_P = 2, gbs = 4, scale = 0.5
    # denom = 4 × 0.5 = 2.0 for all
    assert train_data["rollout_mask_sums"] == [pytest.approx(2.0)] * 4


def test_convert_samples_grpo_always_prompt_equal():
    """GRPO always uses prompt-equal rollout_mask_sums."""
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        n_samples_per_prompt=1,
        grpo_std_normalization=False,
        reward_key=None,
        global_batch_size=1,
    )
    s = SampleLike(reward=0.5, group_index=0, index=0,
                    tokens=[0, 0], response_length=1,
                    metadata={"parent_traj_id": "t1", "instance_id": "p1"})
    s.loss_mask = [1]
    train_data = cs.convert_samples_to_train_data(args, [s])
    assert "rollout_ids" in train_data
    assert "rollout_mask_sums" in train_data
    assert train_data["rollout_mask_sums"] == [1.0]


def test_convert_samples_trajectory_equal_when_non_grpo_estimator():
    args = SimpleNamespace(
        advantage_estimator="gspo",
        rewards_normalization=False,
        n_samples_per_prompt=2,
        grpo_std_normalization=False,
        reward_key=None,
    )
    s1 = SampleLike(reward=0.5, group_index=0, index=0, rollout_id=7,
                     tokens=[0, 0, 0], response_length=2)
    s1.loss_mask = [1, 1]
    s2 = SampleLike(reward=0.5, group_index=0, index=1, rollout_id=7,
                     tokens=[0, 0], response_length=1)
    s2.loss_mask = [1]
    train_data = cs.convert_samples_to_train_data(args, [s1, s2])
    # trajectory-equal: rollout 7 total mask = 2+1 = 3
    assert train_data["rollout_mask_sums"] == [3, 3]


def test_convert_samples_routes_pure_mopd_without_additive_opd(tmp_path):
    config_path = tmp_path / "mopd.json"
    config_path.write_text(
        json.dumps({"teachers": {"a": {"load": "/checkpoint/a"}}}),
        encoding="utf-8",
    )
    args = _traj_equal_args()
    args.advantage_estimator = "mopd"
    args.n_samples_per_prompt = 1
    args.mopd_teacher_config = str(config_path)
    sample = _real_seg(ptid="t1", instance_id="p1", mask=[1, 1], index=0, rollout_id=7)
    sample.metadata["teacher_id"] = "a"

    train_data = cs.convert_samples_to_train_data(args, [sample])

    assert train_data["prompt"] == ["a"]


def test_aborted_no_grad_sample_passes_convert_samples():
    args = _prompt_equal_args(gbs=4)
    real = _real_seg(ptid="t1", instance_id="p1", mask=[1, 1], index=0, rollout_id=0)

    aborted = SampleLike(
        group_index=0, index=1, rollout_id=1, reward=None,
        metadata={}, remove_sample=False,
        tokens=[0], response_length=0,
    )
    aborted.loss_mask = []
    mark_aborted_no_grad(aborted, session_id="aborted-sess", instance_id="p1")

    train_data = cs.convert_samples_to_train_data(args, [real, aborted])
    assert train_data["loss_masks"] == [[1, 1], []]
    # Only p1 live (aborted is dead) → N_P=1, M_P=2, scale=1/4=0.25, denom=0.5
    assert train_data["rollout_mask_sums"][0] == pytest.approx(0.5)
    # Dead sample shares instance_id="p1" → same denom; harmless because
    # its loss_mask is zeroed so it contributes 0 to loss regardless.
    assert train_data["rollout_mask_sums"][1] == pytest.approx(0.5)


@pytest.mark.parametrize("failed_first", [False, True])
def test_convert_samples_fills_r3_for_failed_no_grad_sample(failed_first):
    args = _prompt_equal_args(gbs=4)
    args.use_rollout_routing_replay = True
    args.num_layers = 2
    args.moe_router_topk = 2
    real = _real_seg(
        ptid="t1",
        instance_id="p1",
        mask=[1, 1],
        index=0,
        rollout_id=0,
    )
    real_r3 = np.ones((2, 2, 2), dtype=np.int32)
    real.rollout_routed_experts = real_r3
    failed = _failed_seg(ptid="t2", instance_id="p2", index=1)
    samples = [failed, real] if failed_first else [real, failed]

    train_data = cs.convert_samples_to_train_data(args, samples)

    real_index = 1 if failed_first else 0
    failed_index = 0 if failed_first else 1
    assert train_data["rollout_routed_experts"][real_index] is real_r3
    failed_r3 = train_data["rollout_routed_experts"][failed_index]
    assert failed_r3.shape == (0, 2, 2)
    assert failed_r3.dtype == np.int32


def test_convert_samples_fills_r3_for_multiple_failed_no_grad_samples():
    args = _prompt_equal_args(gbs=4)
    args.use_rollout_routing_replay = True
    args.num_layers = 2
    args.moe_router_topk = 2
    failed_empty = _failed_seg(ptid="t1", instance_id="p1", index=0)
    failed_partial = _failed_seg(ptid="t2", instance_id="p2", index=1)
    failed_partial.tokens = [0, 0, 0]
    failed_partial.response_length = 2
    failed_partial.loss_mask = [0, 0]

    train_data = cs.convert_samples_to_train_data(
        args,
        [failed_empty, failed_partial],
    )

    first_r3, second_r3 = train_data["rollout_routed_experts"]
    assert first_r3.shape == (0, 2, 2)
    assert second_r3.shape == (2, 2, 2)
    assert np.count_nonzero(first_r3) == 0
    assert np.count_nonzero(second_r3) == 0


def test_convert_samples_does_not_fill_failed_r3_without_routing_replay():
    args = _prompt_equal_args(gbs=4)
    args.use_rollout_routing_replay = False
    failed = _failed_seg(ptid="t1", instance_id="p1", index=0)
    real = _real_seg(
        ptid="t2",
        instance_id="p2",
        mask=[1, 1],
        index=1,
        rollout_id=1,
    )
    real.rollout_routed_experts = np.ones((2, 2, 2), dtype=np.int32)

    train_data = cs.convert_samples_to_train_data(args, [failed, real])

    assert "rollout_routed_experts" not in train_data
