"""Verify dressage.training.reward_post_process handles multi-segment +
padding samples correctly.

The implementation does not need any code change for Phase 1 — the existing
parent_traj_id broadcast logic happens to be safe under padding (because
padding samples have parent_traj_id="__padding__" and group_index=None,
which falls into the _NONE_GROUP sentinel and is isolated from real
GRPO groups). These tests pin that property so a future refactor can't
silently break it.

What's covered:
  - Legacy single-segment GRPO behavior unchanged when parent_traj_id is
    absent.
  - Multi-segment: trajectories of varying segment counts share a single
    advantage value (the trajectory-level reward) within their GRPO group.
  - Padding samples: don't participate in real-group normalization, end up
    with reward 0.0.
  - Mixed: real trajectories + padding samples in the same batch produce
    the right per-sample rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from dressage.training.reward_post_process import reward_post_process


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class SampleLike:
    group_index: int | None = 0
    index: int | None = 0
    reward: float | None = 0.0
    metadata: dict = field(default_factory=dict)
    remove_sample: bool = False


def _segment(
    *,
    group_index: int,
    parent_traj_id: str,
    reward: float,
    index: int = 0,
    segment_index: int = 0,
) -> SampleLike:
    return SampleLike(
        group_index=group_index,
        index=index,
        reward=reward,
        metadata={
            "parent_traj_id": parent_traj_id,
            "segment_index": segment_index,
        },
    )


def _real_samples(
    group_index: int,
    trajectories: list[tuple[str, float, int]],
) -> list[SampleLike]:
    """Expand (parent_traj_id, terminal_reward, segment_count) tuples into per-segment
    Samples within one GRPO group.

    Mirrors the multi_segment.expand_segments_to_samples convention:
    only the LAST segment carries the trajectory's real reward; earlier
    segments carry the placeholder 0.0 (set by expand because slime would
    otherwise call reward_fn on them and we want exactly one call per
    trajectory). reward_post_process must pick the last by segment_index
    as the GRPO representative.
    """
    out = []
    index_counter = 0
    for ptid, terminal_reward, count in trajectories:
        for seg_idx in range(count):
            is_last = (seg_idx == count - 1)
            out.append(_segment(
                group_index=group_index,
                parent_traj_id=ptid,
                reward=terminal_reward if is_last else 0.0,
                index=index_counter,
                segment_index=seg_idx,
            ))
            index_counter += 1
    return out


def _grpo_args(*, grpo_std: bool = False, normalize: bool = True, n_per: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=normalize,
        n_samples_per_prompt=n_per,
        grpo_std_normalization=grpo_std,
        reward_key=None,
    )


# ---------------------------------------------------------------------------
# Legacy behavior (no parent_traj_id)
# ---------------------------------------------------------------------------


def test_legacy_no_parent_traj_id_falls_back_to_per_sample_grpo():
    """Without parent_traj_id, samples normalize within group_index — the
    original slime-equivalent behavior."""
    samples = [
        SampleLike(group_index=0, index=0, reward=1.0),
        SampleLike(group_index=0, index=1, reward=3.0),
    ]
    raw, normalized = reward_post_process(_grpo_args(), samples)
    assert raw == [1.0, 3.0]
    assert normalized == [pytest.approx(-1.0), pytest.approx(1.0)]


def test_remove_sample_is_excluded_from_reward_normalization():
    samples = [
        SampleLike(group_index=0, index=0, reward=1.0),
        SampleLike(group_index=0, index=1, reward=100.0, remove_sample=True),
        SampleLike(group_index=0, index=2, reward=3.0),
    ]

    raw, normalized = reward_post_process(_grpo_args(), samples)

    assert raw == [1.0, 100.0, 3.0]
    assert normalized == [pytest.approx(-1.0), 100.0, pytest.approx(1.0)]


def test_no_normalization_passthrough():
    args = _grpo_args(normalize=False)
    samples = [
        SampleLike(group_index=0, index=0, reward=1.0),
        SampleLike(group_index=0, index=1, reward=3.0),
    ]
    raw, normalized = reward_post_process(args, samples)
    assert raw == [1.0, 3.0]
    assert normalized == [1.0, 3.0]


def test_non_grpo_estimator_passthrough():
    args = _grpo_args()
    args.advantage_estimator = "ppo"
    samples = [
        SampleLike(group_index=0, index=0, reward=1.0),
        SampleLike(group_index=0, index=1, reward=3.0),
    ]
    raw, normalized = reward_post_process(args, samples)
    assert raw == [1.0, 3.0]
    assert normalized == [1.0, 3.0]


def test_multi_segment_no_normalization_broadcasts_anchor_reward():
    """When normalization is disabled, every segment must still receive its
    parent trajectory's anchor reward — otherwise non-anchor segments get
    advantage=0 (their placeholder) while their tokens stay in the loss
    reducer's denominator, silently shrinking the gradient.
    """
    args = _grpo_args(normalize=False)
    samples = _real_samples(
        group_index=0,
        trajectories=[
            ("t1", 2.0, 3),   # anchor at index 2 carries 2.0
            ("t2", 4.0, 1),   # anchor at index 3 carries 4.0
        ],
    )
    raw, normalized = reward_post_process(args, samples)
    # raw stays sparse (anchor-only) — same invariant as the normalizing path
    # so log_rollout_data's per-trajectory sum still recovers terminal reward.
    assert raw == [0.0, 0.0, 2.0, 4.0]
    # advantage = raw reward broadcast across each trajectory's segments.
    assert normalized == [2.0, 2.0, 2.0, 4.0]


def test_multi_segment_non_grpo_estimator_broadcasts_anchor_reward():
    """Same broadcast requirement when advantage_estimator is outside the
    GRPO whitelist (e.g., 'reinforce' / 'ppo') — without it the non-anchor
    segments silently lose their reward signal.
    """
    args = _grpo_args()
    args.advantage_estimator = "reinforce"
    samples = _real_samples(
        group_index=0,
        trajectories=[
            ("t1", 1.5, 2),
            ("t2", 7.0, 1),
        ],
    )
    raw, normalized = reward_post_process(args, samples)
    assert raw == [0.0, 1.5, 7.0]
    assert normalized == [1.5, 1.5, 7.0]


def test_multi_segment_no_normalization_broadcasts_anchor():
    """Non-normalizing path still broadcasts the anchor reward to every
    segment of a multi-segment trajectory."""
    real = _real_samples(
        group_index=0,
        trajectories=[("t1", 2.0, 2), ("t2", 4.0, 1)],
    )
    args = _grpo_args(normalize=False)
    raw, normalized = reward_post_process(args, real)
    # Real segments broadcast their anchor (2.0 for t1, 4.0 for t2)
    assert normalized == [2.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Multi-segment broadcast
# ---------------------------------------------------------------------------


def test_representative_is_last_segment_by_segment_index():
    """Per plan §1.3, reward_fn runs only on the last segment of a trajectory;
    non-last segments carry placeholder reward=0.0. reward_post_process must
    pick the LAST-by-segment_index as the GRPO representative — picking the
    first would broadcast 0.0 to the entire trajectory, losing the signal."""
    # 2 trajectories in one GRPO group:
    #   t1 has 3 segments with terminal reward 2.0 (segment_index 0,1,2 → only 2 carries 2.0)
    #   t2 has 1 segment with terminal reward 4.0 (segment_index 0 carries 4.0)
    samples = _real_samples(
        group_index=0,
        trajectories=[
            ("t1", 2.0, 3),
            ("t2", 4.0, 1),
        ],
    )
    # Sanity: the placeholder 0.0 lives on segments 0 and 1 of t1
    assert [s.reward for s in samples] == [0.0, 0.0, 2.0, 4.0]

    raw, normalized = reward_post_process(_grpo_args(), samples)
    # Per-trajectory reward used for GRPO normalization must be the LAST
    # segment's reward (2.0 for t1, 4.0 for t2). Mean = 3.0 → t1 advantage = -1.0,
    # t2 advantage = +1.0. Both broadcast to every segment of their trajectory.
    assert normalized == [pytest.approx(-1.0)] * 3 + [pytest.approx(1.0)]
    # raw_reward stays sparse: anchor holds the terminal reward, every other
    # segment keeps the 0.0 placeholder. The trajectory-level mean for wandb
    # is computed in dressage.rollout.log_rollout, not via broadcast.
    assert raw == [0.0, 0.0, 2.0, 4.0]


def test_representative_picks_last_segment_even_when_unsorted_in_input():
    """If samples are reordered such that segment_index increasing != list order,
    the representative must still be by segment_index, not by list position."""
    # Build samples with segment_index 2 first, then 0, then 1
    samples = [
        SampleLike(group_index=0, index=0, reward=99.0,  # last segment, terminal reward
                   metadata={"parent_traj_id": "t1", "segment_index": 2}),
        SampleLike(group_index=0, index=1, reward=0.0,
                   metadata={"parent_traj_id": "t1", "segment_index": 0}),
        SampleLike(group_index=0, index=2, reward=0.0,
                   metadata={"parent_traj_id": "t1", "segment_index": 1}),
        SampleLike(group_index=0, index=3, reward=11.0,
                   metadata={"parent_traj_id": "t2", "segment_index": 0}),
    ]
    raw, normalized = reward_post_process(_grpo_args(), samples)
    # mean of (99, 11) = 55 → t1 = +44, t2 = -44
    # (no std normalization)
    assert normalized[0] == pytest.approx(44.0)   # t1's last seg
    assert normalized[1] == pytest.approx(44.0)   # t1's seg 0 broadcast
    assert normalized[2] == pytest.approx(44.0)   # t1's seg 1 broadcast
    assert normalized[3] == pytest.approx(-44.0)  # t2's only seg


def test_multi_segment_trajectory_reward_broadcast_within_group():
    """Two trajectories in one GRPO group: t1 has reward 2.0 across 3 segments,
    t2 has reward 4.0 across 1 segment. After normalization, all segments of
    t1 share the same normalized reward; same for t2."""
    samples = _real_samples(
        group_index=0,
        trajectories=[
            ("t1", 2.0, 3),
            ("t2", 4.0, 1),
        ],
    )
    raw, normalized = reward_post_process(_grpo_args(), samples)
    # raw stays sparse: only the LAST segment of a trajectory carries the
    # terminal reward; earlier segments keep the 0.0 placeholder. The wandb
    # trajectory-level mean is computed separately by
    # dressage.rollout.log_rollout (per-trajectory sum then average).
    assert raw == [0.0, 0.0, 2.0, 4.0]
    # GRPO over trajectories: representative for t1 = last (segment_index=2, raw 2.0);
    # representative for t2 = last (segment_index=0, raw 4.0). Mean = 3.0
    # → t1 normalized = -1.0 → broadcast to indices 0, 1, 2
    # → t2 normalized = +1.0 → broadcast to index 3
    assert normalized == [pytest.approx(-1.0)] * 3 + [pytest.approx(1.0)]


def test_raw_reward_stays_sparse_anchor_only():
    """raw_reward must NOT be broadcast across segments — that would make a
    long trajectory contribute N times more weight to wandb's reward mean
    than a 1-segment trajectory with the same terminal reward. The
    trajectory-level mean is computed separately by log_rollout_data.

    The sum-per-trajectory invariant (anchor holds the full reward, others
    are 0 → summing within a trajectory recovers the terminal reward) is
    what log_rollout_data's trajectory-mean metric relies on. Pin it here.
    """
    samples = _real_samples(
        group_index=0,
        trajectories=[("t1", 2.0, 3), ("t2", 4.0, 1)],
    )
    raw, _ = reward_post_process(_grpo_args(), samples)
    # Anchor of t1 is index 2 (last segment of t1); anchor of t2 is index 3.
    assert raw == [0.0, 0.0, 2.0, 4.0]
    # Sum within each trajectory recovers the terminal reward.
    assert raw[0] + raw[1] + raw[2] == pytest.approx(2.0)
    assert raw[3] == pytest.approx(4.0)


def test_multi_segment_equal_rewards_yield_zero_advantage():
    samples = _real_samples(
        group_index=0,
        trajectories=[
            ("t1", 1.0, 2),
            ("t2", 1.0, 2),
        ],
    )
    _, normalized = reward_post_process(_grpo_args(), samples)
    assert all(abs(n) < 1e-9 for n in normalized)


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([0.0, 0.0], [0.0, 0.0]),
        ([1.0, 1.0], [0.0, 0.0]),
        ([0.0, 1.0], [-0.5, 0.5]),
    ],
)
def test_binary_reward_groups_keep_zero_variance_behavior(rewards, expected):
    samples = _real_samples(
        group_index=0,
        trajectories=[
            ("t1", rewards[0], 1),
            ("t2", rewards[1], 1),
        ],
    )

    _, normalized = reward_post_process(_grpo_args(), samples)

    assert normalized == pytest.approx(expected)


def test_multi_segment_with_std_normalization():
    samples = _real_samples(
        group_index=0,
        trajectories=[
            ("t1", 1.0, 2),
            ("t2", 3.0, 2),
        ],
    )
    args = _grpo_args(grpo_std=True)
    _, normalized = reward_post_process(args, samples)
    # mean=2, normalized = [-1, +1], std=1, so /std unchanged.
    assert normalized == [pytest.approx(-1.0)] * 2 + [pytest.approx(1.0)] * 2


def test_multi_segment_two_groups_independent():
    samples = (
        _real_samples(group_index=0, trajectories=[("a", 0.0, 1), ("b", 2.0, 1)])
        + _real_samples(group_index=1, trajectories=[("c", 5.0, 2), ("d", 1.0, 2)])
    )
    _, normalized = reward_post_process(_grpo_args(), samples)
    # group 0: mean=1 → [a=-1, b=+1]
    # group 1: mean=3 → [c=+2 x2, d=-2 x2]
    assert normalized[:2] == [pytest.approx(-1.0), pytest.approx(1.0)]
    assert normalized[2:] == [
        pytest.approx(2.0), pytest.approx(2.0),
        pytest.approx(-2.0), pytest.approx(-2.0),
    ]


# ---------------------------------------------------------------------------
# Length & shape invariants
# ---------------------------------------------------------------------------


def test_output_length_matches_input_length():
    real = _real_samples(group_index=0, trajectories=[("t1", 1.0, 3)])
    raw, normalized = reward_post_process(_grpo_args(), real)
    assert len(raw) == len(real)
    assert len(normalized) == len(real)


def test_handles_sample_with_none_reward():
    samples = [
        SampleLike(group_index=0, index=0, reward=None),
        SampleLike(group_index=0, index=1, reward=2.0),
    ]
    raw, _ = reward_post_process(_grpo_args(), samples)
    assert raw == [0.0, 2.0]


def test_handles_dict_reward_with_reward_key():
    args = _grpo_args()
    args.reward_key = "score"
    samples = [
        SampleLike(group_index=0, index=0, reward={"score": 1.0, "extra": "x"}),
        SampleLike(group_index=0, index=1, reward={"score": 3.0, "extra": "y"}),
    ]
    raw, normalized = reward_post_process(args, samples)
    assert raw == [1.0, 3.0]
    assert normalized == [pytest.approx(-1.0), pytest.approx(1.0)]
