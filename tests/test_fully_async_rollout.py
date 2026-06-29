from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

from dressage.rollout import fully_async_rollout


@dataclass
class SampleLike:
    index: int
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)
    reward: float | None = None
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    remove_sample: bool = False

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING


class DataBuffer:
    def __init__(self, groups):
        self.groups = list(groups)
        self.requeued = []

    def get_samples(self, count):
        out = self.groups[:count]
        del self.groups[:count]
        return out

    def add_samples(self, groups):
        self.requeued.extend(groups)
        self.groups.extend(groups)


def teardown_function():
    fully_async_rollout.stop_global_worker()


def test_increment_retry_resets_session_ids_for_whole_group():
    group = [
        SampleLike(
            index=0,
            session_id="bbs-success-old",
            metadata={
                "session_id": "bbs-success-old",
                "parent_traj_id": "bbs-success-old",
                "segment_index": 0,
            },
        ),
        SampleLike(
            index=1,
            session_id="bbs-failed-old",
            metadata={"session_id": "bbs-failed-old"},
            remove_sample=True,
        ),
    ]

    fully_async_rollout._increment_retry(group)

    assert [sample.session_id for sample in group] == [None, None]
    assert [sample.metadata["dressage_retry_count"] for sample in group] == [1, 1]
    assert [sample.metadata["last_retry_session_id"] for sample in group] == [
        "bbs-success-old",
        "bbs-failed-old",
    ]
    assert all("session_id" not in sample.metadata for sample in group)
    assert "parent_traj_id" not in group[0].metadata
    assert "segment_index" not in group[0].metadata
    assert all(sample.remove_sample is False for sample in group)


def test_fully_async_rollout_drains_completed_groups(monkeypatch):
    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        for sample in group:
            sample.status = SampleLike.Status.COMPLETED
            sample.reward = 1.0
            sample.tokens = [1, 2]
            sample.response_length = 1
            sample.loss_mask = [1]
            sample.rollout_log_probs = [-0.1]
        return group

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(fully_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "2")

    groups = [[SampleLike(index=2)], [SampleLike(index=1)]]
    data = DataBuffer(groups)
    args = SimpleNamespace(rollout_batch_size=2)

    result = fully_async_rollout.generate_rollout_fully_async(args, 0, data)

    assert [group[0].index for group in result] == [1, 2]
    assert all(group[0].reward == 1.0 for group in result)


def test_fully_async_rollout_retries_aborted_group(monkeypatch):
    attempts = {"count": 0}

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        attempts["count"] += 1
        if attempts["count"] == 1:
            assert group[0].session_id == "old-session"
            group[0].status = SampleLike.Status.ABORTED
            group[0].metadata["blackbox_error"] = "duplicate session"
            group[0].metadata["last_failed_session_id"] = "old-session"
            group[0].session_id = None
        else:
            assert group[0].session_id is None
            group[0].session_id = "new-session"
            group[0].status = SampleLike.Status.COMPLETED
            group[0].reward = 1.0
            group[0].tokens = [1, 2]
            group[0].response_length = 1
            group[0].loss_mask = [1]
            group[0].rollout_log_probs = [-0.1]
        return group

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(fully_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ROLLOUT_MAX_RETRIES", "2")

    data = DataBuffer([[SampleLike(index=0, session_id="old-session")]])
    args = SimpleNamespace(rollout_batch_size=1)

    result = fully_async_rollout.generate_rollout_fully_async(args, 0, data)

    assert attempts["count"] == 2
    assert len(data.requeued) == 1
    assert result[0][0].status == SampleLike.Status.COMPLETED
    assert result[0][0].session_id == "new-session"


def test_fully_async_rollout_fails_fast_when_all_groups_failed(monkeypatch):
    attempts = {"count": 0}

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        attempts["count"] += 1
        group[0].status = SampleLike.Status.ABORTED
        group[0].metadata["blackbox_error"] = "sandbox register timed out"
        group[0].metadata["last_failed_session_id"] = "bbs-old-session"
        group[0].session_id = None
        return group

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(fully_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ROLLOUT_MAX_RETRIES", "1")
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS", "1")

    data = DataBuffer([[SampleLike(index=0)]])
    args = SimpleNamespace(rollout_batch_size=1)

    with pytest.raises(RuntimeError, match="dropped too many failed groups") as excinfo:
        fully_async_rollout.generate_rollout_fully_async(args, 0, data)

    assert attempts["count"] == 2
    assert len(data.requeued) == 1
    assert "sandbox register timed out" in str(excinfo.value)
    assert "session_id=bbs-old-session" in str(excinfo.value)


def test_fully_async_rollout_drops_exhausted_failed_group_and_keeps_collecting(monkeypatch):
    attempts_by_index = {}

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        sample = group[0]
        attempts_by_index[sample.index] = attempts_by_index.get(sample.index, 0) + 1
        if sample.index == 0:
            sample.status = SampleLike.Status.ABORTED
            sample.metadata["blackbox_error"] = "permanent failure"
            sample.session_id = None
            return group

        sample.status = SampleLike.Status.COMPLETED
        sample.reward = 1.0
        sample.tokens = [1, 2]
        sample.response_length = 1
        sample.loss_mask = [1]
        sample.rollout_log_probs = [-0.1]
        return group

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(fully_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ROLLOUT_MAX_RETRIES", "0")
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS", "10")

    data = DataBuffer([[SampleLike(index=0)], [SampleLike(index=1)]])
    args = SimpleNamespace(rollout_batch_size=1)

    result = fully_async_rollout.generate_rollout_fully_async(args, 0, data)

    assert attempts_by_index == {0: 1, 1: 1}
    assert [group[0].index for group in result] == [1]
    assert result[0][0].status == SampleLike.Status.COMPLETED


def test_fully_async_rollout_drops_stale_group_by_trajectory(monkeypatch):
    fully_async_rollout.stop_global_worker()

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        for sample in group:
            sample.status = SampleLike.Status.COMPLETED
            sample.reward = 1.0
            sample.tokens = [1, 2]
            sample.response_length = 1
            sample.loss_mask = [1]
            sample.rollout_log_probs = [-0.1]
        return group

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(fully_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "3")

    stale_group = [
        SampleLike(index=0, metadata={"parent_traj_id": "traj-old", "dressage_end_token_version": "old"}),
    ]
    data = DataBuffer([
        stale_group,
        [SampleLike(index=1, metadata={"parent_traj_id": "traj-new-1", "dressage_end_token_version": "new"})],
        [SampleLike(index=2, metadata={"parent_traj_id": "traj-new-2", "dressage_end_token_version": "new"})],
    ])
    args = SimpleNamespace(rollout_batch_size=2, dressage_staleness_keep_versions=1)

    result = fully_async_rollout.generate_rollout_fully_async(args, 0, data)

    assert all(isinstance(group, list) for group in result)
    assert [[sample.index for sample in group] for group in result] == [[1], [2]]


def test_fully_async_rollout_staleness_metrics_are_trajectory_weighted(monkeypatch):
    fully_async_rollout.stop_global_worker()

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        for sample in group:
            sample.status = SampleLike.Status.COMPLETED
            sample.reward = 1.0
            sample.tokens = [1, 2]
            sample.response_length = 1
            sample.loss_mask = [1]
            sample.rollout_log_probs = [-0.1]
        return group

    class TrainOutput:
        def __init__(self, samples, metrics=None):
            self.samples = samples
            self.metrics = metrics or {}

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(fully_async_rollout, "GenerateState", None)
    monkeypatch.setattr(fully_async_rollout, "RolloutFnTrainOutput", TrainOutput)

    data = DataBuffer([[
        SampleLike(index=0, metadata={"parent_traj_id": "long", "segment_index": 0, "dressage_end_token_version": "old"}),
        SampleLike(index=1, metadata={"parent_traj_id": "long", "segment_index": 1, "dressage_end_token_version": "middle"}),
        SampleLike(index=2, metadata={"parent_traj_id": "short", "dressage_end_token_version": "new"}),
    ]])
    args = SimpleNamespace(rollout_batch_size=1, dressage_staleness_keep_versions=3)

    output = fully_async_rollout.generate_rollout_fully_async(args, 0, data)

    assert all(isinstance(group, list) for group in output.samples)
    assert output.metrics["staleness/current_version_index"] == 1.0
    assert output.metrics["staleness/cutoff_version_index"] == 0.0
    assert output.metrics["staleness/version_gap_min"] == 0.0
    assert output.metrics["staleness/version_gap_max"] == 1.0
    assert output.metrics["staleness/version_gap_mean"] == pytest.approx(0.5)


def test_fully_async_rollout_ignores_full_versions_and_uses_end_version(monkeypatch):
    fully_async_rollout.stop_global_worker()

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        for sample in group:
            sample.status = SampleLike.Status.COMPLETED
            sample.reward = 1.0
            sample.tokens = [1, 2, 3]
            sample.response_length = 2
            sample.rollout_log_probs = [-0.1, -0.2]
        return group

    monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(fully_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "2")

    data = DataBuffer([
        [
            SampleLike(
                index=0,
                loss_mask=[0, 1],
                metadata={
                    "parent_traj_id": "traj-partial",
                    "full_versions": ["-1", "old", "new"],
                    "dressage_end_token_version": "new",
                },
            )
        ],
        [SampleLike(index=1, metadata={"parent_traj_id": "traj-v2", "dressage_end_token_version": "new"})],
    ])
    args = SimpleNamespace(rollout_batch_size=2, dressage_staleness_keep_versions=1)

    result = fully_async_rollout.generate_rollout_fully_async(args, 0, data)

    assert all(isinstance(group, list) for group in result)
    assert [[sample.index for sample in group] for group in result] == [[0], [1]]
