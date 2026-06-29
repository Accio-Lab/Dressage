from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from dressage.rollout.staleness import (
    PendingGroup,
    StalenessConfig,
    StalenessGroupFilter,
    StalenessTracker,
    config_from_args,
    real_version,
    trajectory_key,
    trajectory_version_infos,
)


@dataclass
class SampleLike:
    index: int = 0
    rollout_id: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletedLike:
    group_id: int
    result: list[SampleLike]


def test_config_from_args_defaults_disabled_and_rejects_negative():
    assert config_from_args(SimpleNamespace()).enabled is False
    assert config_from_args(SimpleNamespace(dressage_staleness_keep_versions=0)).enabled is False
    assert (
        config_from_args(SimpleNamespace(dressage_staleness_keep_versions=2))
        == StalenessConfig(keep_versions=2)
    )
    with pytest.raises(ValueError, match="integer >= 0"):
        config_from_args(SimpleNamespace(dressage_staleness_keep_versions=-1))


def test_real_version_treats_versions_as_opaque_labels():
    assert real_version("alpha") == "alpha"
    assert real_version("release-candidate") == "release-candidate"
    assert real_version(" weight-current ") == "weight-current"
    assert real_version("unknown") is None
    assert real_version("-1") is None
    assert real_version(None) is None


def test_trajectory_key_uses_parent_traj_id_only():
    sample = SampleLike(
        index=0,
        rollout_id=10,
        metadata={"parent_traj_id": "traj-a", "session_id": "sess-a"},
    )

    assert trajectory_key(sample) == "traj-a"
    assert trajectory_key(SampleLike(rollout_id=10, metadata={"session_id": "sess-a"})) == ""


def test_trajectory_version_uses_last_segment_end_version():
    group = [
        SampleLike(
            index=0,
            metadata={
                "parent_traj_id": "traj-a",
                "segment_index": 0,
                "dressage_end_token_version": "alpha",
            },
        ),
        SampleLike(
            index=1,
            metadata={
                "parent_traj_id": "traj-a",
                "segment_index": 2,
                "dressage_end_token_version": "gamma",
            },
        ),
        SampleLike(
            index=2,
            metadata={
                "parent_traj_id": "traj-a",
                "segment_index": 1,
                "dressage_end_token_version": "beta",
            },
        ),
    ]

    infos = trajectory_version_infos(group)

    assert len(infos) == 1
    assert infos[0].key == "traj-a"
    assert infos[0].version == "gamma"


def test_same_segment_index_uses_later_sample_order():
    group = [
        SampleLike(
            index=0,
            metadata={
                "parent_traj_id": "traj-a",
                "segment_index": 0,
                "dressage_end_token_version": "alpha",
            },
        ),
        SampleLike(
            index=1,
            metadata={
                "parent_traj_id": "traj-a",
                "segment_index": 0,
                "dressage_end_token_version": "beta",
            },
        ),
    ]

    assert trajectory_version_infos(group)[0].version == "beta"


def test_missing_versions_do_not_advance_or_drop():
    tracker = StalenessTracker(StalenessConfig(keep_versions=1))
    missing = [SampleLike(index=0, metadata={"parent_traj_id": "missing"})]

    assert tracker.observe_group(missing) is False
    assert tracker.current_version_index is None
    assert tracker.should_drop_group(missing) is False


def test_missing_parent_traj_id_does_not_advance_or_drop():
    tracker = StalenessTracker(StalenessConfig(keep_versions=1))
    missing_key = [
        SampleLike(
            index=0,
            rollout_id=10,
            metadata={"session_id": "sess-a", "dressage_end_token_version": "alpha"},
        )
    ]

    assert trajectory_version_infos(missing_key) == []
    assert tracker.observe_group(missing_key) is False
    assert tracker.current_version_index is None
    assert tracker.should_drop_group(missing_key) is False


def test_tracker_drops_by_observed_version_order_not_label_sort():
    tracker = StalenessTracker(StalenessConfig(keep_versions=1))
    fresh = [
        SampleLike(
            index=0,
            metadata={"parent_traj_id": "fresh", "dressage_end_token_version": "alpha"},
        )
    ]
    newer = [
        SampleLike(
            index=1,
            metadata={"parent_traj_id": "newer", "dressage_end_token_version": "aardvark"},
        )
    ]

    assert tracker.observe_group(fresh) is True
    assert tracker.observe_group(newer) is True
    assert tracker.versions == ["alpha", "aardvark"]
    assert tracker.should_drop_group(fresh) is True
    assert tracker.should_drop_group(newer) is False


def test_mixed_group_drops_when_any_versioned_trajectory_is_stale():
    tracker = StalenessTracker(StalenessConfig(keep_versions=1))
    tracker.observe_group([
        SampleLike(metadata={"parent_traj_id": "old", "dressage_end_token_version": "old"}),
        SampleLike(metadata={"parent_traj_id": "new", "dressage_end_token_version": "new"}),
    ])
    group = [
        SampleLike(metadata={"parent_traj_id": "missing"}),
        SampleLike(metadata={"parent_traj_id": "old", "dressage_end_token_version": "old"}),
    ]

    assert tracker.should_drop_group(group) is True


def test_observe_completed_then_filter_pending_when_newer_version_arrives():
    tracker = StalenessTracker(StalenessConfig(keep_versions=1), versions=["old"])
    staleness_filter = StalenessGroupFilter(
        tracker=tracker,
        rollout_name="test",
    )
    old_group = [
        SampleLike(
            index=0,
            metadata={"parent_traj_id": "old", "dressage_end_token_version": "old"},
        )
    ]
    new_group = [
        SampleLike(
            index=1,
            metadata={"parent_traj_id": "new", "dressage_end_token_version": "new"},
        )
    ]
    pending = [PendingGroup(group_id=0, samples=old_group)]
    advanced = staleness_filter.observe_completed([CompletedLike(group_id=1, result=new_group)])
    collected = staleness_filter.filter_pending(
        pending,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert advanced is True
    assert collected == []
    assert tracker.versions == ["old", "new"]
    assert staleness_filter.dropped_groups == 1


def test_metrics_are_trajectory_weighted_by_version_index_gap():
    tracker = StalenessTracker(
        StalenessConfig(keep_versions=3),
        versions=["middle", "new"],
    )
    staleness_filter = StalenessGroupFilter(
        tracker=tracker,
        rollout_name="test",
        dropped_groups=2,
    )
    groups = [[
        SampleLike(
            index=0,
            metadata={
                "parent_traj_id": "long",
                "segment_index": 0,
                "dressage_end_token_version": "old",
            },
        ),
        SampleLike(
            index=1,
            metadata={
                "parent_traj_id": "long",
                "segment_index": 1,
                "dressage_end_token_version": "middle",
            },
        ),
        SampleLike(
            index=2,
            metadata={"parent_traj_id": "short", "dressage_end_token_version": "new"},
        ),
    ]]

    metrics = staleness_filter.metrics_for_groups(groups)

    assert metrics["staleness/dropped_groups"] == 2.0
    assert metrics["staleness/current_version_index"] == 1.0
    assert metrics["staleness/cutoff_version_index"] == 0.0
    assert metrics["staleness/version_gap_min"] == 0.0
    assert metrics["staleness/version_gap_max"] == 1.0
    assert metrics["staleness/version_gap_mean"] == pytest.approx(0.5)
