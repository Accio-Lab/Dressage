"""Tests for dressage.training.log_helpers.

The full log_rollout_data() function is not exercised here because it
requires a live mpu/distributed environment (it's called inside the
megatron actor's train_actor). We pin the trajectory-mean helper that
the function delegates to — that's where the multi-segment-specific
math lives.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import dressage.rollout.log_rollout as log_rollout_module
from dressage.rollout.log_rollout import log_rollout_data
from dressage.training.log_helpers import compute_trajectory_mean_raw_reward


def test_returns_none_on_empty_input():
    assert compute_trajectory_mean_raw_reward([], [], []) is None


def test_rollout_hook_filters_no_loss_samples_from_trajectory_mean():
    samples = [
        SimpleNamespace(
            metadata={"parent_traj_id": "good", "segment_index": 0},
            reward=1.0,
            response_length=1,
            loss_mask=[1],
            remove_sample=False,
        ),
        SimpleNamespace(
            metadata={"parent_traj_id": "aborted", "segment_index": 0},
            reward=0.0,
            response_length=0,
            loss_mask=[],
            remove_sample=True,
        ),
        SimpleNamespace(
            metadata={"parent_traj_id": "masked", "segment_index": 0},
            reward=0.0,
            response_length=1,
            loss_mask=[0],
            remove_sample=False,
        ),
    ]
    extra_metrics = {}

    assert log_rollout_data(0, SimpleNamespace(), samples, extra_metrics, 0.0) is False
    assert extra_metrics["rollout/raw_reward_trajectory_mean"] == pytest.approx(1.0)


def test_rollout_hook_selects_configured_reward_from_mapping():
    samples = [
        SimpleNamespace(
            metadata={"parent_traj_id": "harbor-trial", "segment_index": 0},
            reward={"reward": 0.75, "aux": 12.0},
            response_length=1,
            loss_mask=[1],
            remove_sample=False,
        )
    ]
    extra_metrics = {}

    assert (
        log_rollout_data(
            0,
            SimpleNamespace(reward_key="reward"),
            samples,
            extra_metrics,
            0.0,
        )
        is False
    )
    assert extra_metrics["rollout/raw_reward_trajectory_mean"] == pytest.approx(0.75)


def test_rollout_hook_defines_staleness_wandb_step_metric(monkeypatch):
    calls = []

    class FakeWandb:
        run = object()

        @staticmethod
        def define_metric(name, **kwargs):
            calls.append((name, kwargs))

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    monkeypatch.setattr(log_rollout_module, "_STALENESS_WANDB_METRICS_DEFINED", False)

    args = SimpleNamespace(use_wandb=True)
    assert log_rollout_data(0, args, [], {}, 0.0) is False
    assert log_rollout_data(1, args, [], {}, 0.0) is False

    assert calls == [("staleness/*", {"step_metric": "rollout/step"})]


def test_single_segment_trajectories():
    parent_traj_ids = ["t1", "t2", "t3"]
    raw_rewards = [1.0, 0.0, 1.0]
    segment_indices = [0, 0, 0]
    # 3 trajectories, 2/3 correct → 0.666...
    assert compute_trajectory_mean_raw_reward(
        parent_traj_ids, raw_rewards, segment_indices
    ) == pytest.approx(2 / 3)


def test_long_trajectory_counted_once_not_per_segment():
    """A 5-segment correct trajectory must not outweigh a 1-segment
    correct trajectory — both contribute 1.0 to the trajectory mean.
    """
    # t1 = 5 segments, anchor=last (segment_index=4), terminal reward 1.0.
    # t2 = 1 segment, reward 1.0.
    parent_traj_ids = ["t1"] * 5 + ["t2"]
    raw_rewards = [0.0, 0.0, 0.0, 0.0, 1.0] + [1.0]
    segment_indices = [0, 1, 2, 3, 4] + [0]
    assert compute_trajectory_mean_raw_reward(
        parent_traj_ids, raw_rewards, segment_indices
    ) == pytest.approx(1.0)
    # Contrast with slime's default per-sample mean (which is what
    # rollout/raw_reward shows): that would dilute t1 → (1+1)/6 ≈ 0.333.
    assert sum(raw_rewards) / len(raw_rewards) == pytest.approx(2 / 6)


def test_mixed_segment_counts():
    parent_traj_ids = ["t1", "t1", "t1", "t2", "t3", "t3"]
    # t1 correct (anchor=index 2 / segment_index=2), t2 wrong (single segment),
    # t3 correct (anchor=index 5 / segment_index=1).
    raw_rewards = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    segment_indices = [0, 1, 2, 0, 0, 1]
    # 2/3 trajectories correct.
    assert compute_trajectory_mean_raw_reward(
        parent_traj_ids, raw_rewards, segment_indices
    ) == pytest.approx(2 / 3)


def test_anchor_lookup_ignores_stray_non_anchor_reward():
    """Anchor lookup is robust to a non-anchor segment carrying a non-zero
    reward (the previous sum-within-trajectory implementation would have
    double-counted it). The anchor segment's reward is the only one used."""
    parent_traj_ids = ["t1", "t1", "t1"]
    # Non-anchor segments accidentally carry rewards; only segment_index=2 is anchor.
    raw_rewards = [0.5, 0.7, 1.0]
    segment_indices = [0, 1, 2]
    # Sum-within-trajectory (old behavior) would give 2.2; anchor lookup gives 1.0.
    assert compute_trajectory_mean_raw_reward(
        parent_traj_ids, raw_rewards, segment_indices
    ) == pytest.approx(1.0)


def test_non_binary_rewards_picked_from_anchor():
    """Partial / non-binary rewards on the anchor are preserved exactly;
    the non-anchor's placeholder 0.0 is ignored."""
    parent_traj_ids = ["t1", "t1", "t2"]
    # t1 terminal reward 0.7 on anchor (segment_index=1); t2 single-segment 0.2.
    raw_rewards = [0.0, 0.7, 0.2]
    segment_indices = [0, 1, 0]
    # (0.7 + 0.2) / 2 = 0.45.
    assert compute_trajectory_mean_raw_reward(
        parent_traj_ids, raw_rewards, segment_indices
    ) == pytest.approx(0.45)


def test_anchor_is_max_segment_index_regardless_of_input_order():
    """Sample order in the input is arbitrary — the function picks anchor
    by max segment_index per trajectory, not by list position."""
    parent_traj_ids = ["t1", "t1", "t1"]
    # Anchor (segment_index=2) appears in the middle of the list.
    raw_rewards = [0.0, 1.0, 0.0]
    segment_indices = [0, 2, 1]
    assert compute_trajectory_mean_raw_reward(
        parent_traj_ids, raw_rewards, segment_indices
    ) == pytest.approx(1.0)


def test_length_mismatch_raises():
    """Defensive: catching upstream bugs where the three lists desync."""
    with pytest.raises(ValueError):
        compute_trajectory_mean_raw_reward(["t1", "t2"], [1.0], [0, 0])
    with pytest.raises(ValueError):
        compute_trajectory_mean_raw_reward(["t1", "t2"], [1.0, 0.0], [0])
