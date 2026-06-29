from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

from dressage.rollout import partial_async_rollout


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


def _samples(output):
    return output.samples if hasattr(output, "samples") else output


def teardown_function():
    partial_async_rollout.stop_global_partial_worker()


def test_partial_async_rollout_returns_global_batch_sized_subset(monkeypatch):
    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        # The first two groups complete quickly; the remaining groups make it clear
        # that the rollout call returns before the full rollout batch is finished.
        if group[0].index >= 2:
            await asyncio.sleep(0.2)
        for sample in group:
            sample.status = SampleLike.Status.COMPLETED
            sample.reward = 1.0
            sample.tokens = [1, 2]
            sample.response_length = 1
            sample.loss_mask = [1]
            sample.rollout_log_probs = [-0.1]
        return group

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "4")
    monkeypatch.delenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS", raising=False)
    monkeypatch.delenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_SAMPLES", raising=False)

    groups = [[SampleLike(index=i)] for i in range(4)]
    data = DataBuffer(groups)
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=1, global_batch_size=2)

    start = time.time()
    result = _samples(partial_async_rollout.generate_rollout_partial_async(args, 7, data))

    assert time.time() - start < 0.19
    assert [group[0].index for group in result] == [0, 1]
    assert all(group[0].metadata["dressage_partial_rollout"] for group in result)
    assert all(group[0].metadata["dressage_return_rollout_id"] == 7 for group in result)


def test_partial_async_rollout_target_samples_are_converted_to_prompt_groups(monkeypatch):
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

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_SAMPLES", "4")

    groups = [
        [SampleLike(index=prompt_index * 2 + sample_index) for sample_index in range(2)]
        for prompt_index in range(4)
    ]
    data = DataBuffer(groups)
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=2, global_batch_size=8)

    result = _samples(partial_async_rollout.generate_rollout_partial_async(args, 0, data))

    assert len(result) == 2
    assert sum(len(group) for group in result) == 4


def test_partial_async_rollout_does_not_drop_completed_leftovers(monkeypatch):
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

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "4")
    monkeypatch.setenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS", "2")

    data = DataBuffer([[SampleLike(index=i)] for i in range(4)])
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=1, global_batch_size=2)

    first = _samples(partial_async_rollout.generate_rollout_partial_async(args, 0, data))
    second = _samples(partial_async_rollout.generate_rollout_partial_async(args, 1, data))

    assert [group[0].index for group in first] == [0, 1]
    assert [group[0].index for group in second] == [2, 3]


def test_partial_async_rollout_retries_aborted_group(monkeypatch):
    attempts = {"count": 0}

    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        attempts["count"] += 1
        if attempts["count"] == 1:
            group[0].status = SampleLike.Status.ABORTED
            group[0].metadata["blackbox_error"] = "duplicate session"
            group[0].session_id = None
        else:
            group[0].status = SampleLike.Status.COMPLETED
            group[0].session_id = "new-session"
            group[0].reward = 1.0
            group[0].tokens = [1, 2]
            group[0].response_length = 1
            group[0].loss_mask = [1]
            group[0].rollout_log_probs = [-0.1]
        return group

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ROLLOUT_MAX_RETRIES", "2")

    data = DataBuffer([[SampleLike(index=0, session_id="old-session")]])
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1, global_batch_size=1)

    result = _samples(partial_async_rollout.generate_rollout_partial_async(args, 0, data))

    assert attempts["count"] == 2
    assert len(data.requeued) == 1
    assert result[0][0].status == SampleLike.Status.COMPLETED
    assert result[0][0].session_id == "new-session"


def test_partial_async_rollout_fails_fast_when_all_groups_failed(monkeypatch):
    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        group[0].status = SampleLike.Status.ABORTED
        group[0].metadata["blackbox_error"] = "sandbox register timed out"
        group[0].session_id = None
        return group

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ROLLOUT_MAX_RETRIES", "1")
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS", "1")

    data = DataBuffer([[SampleLike(index=0)]])
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1, global_batch_size=1)

    with pytest.raises(RuntimeError, match="dropped too many failed groups"):
        partial_async_rollout.generate_rollout_partial_async(args, 0, data)


def test_partial_async_rollout_drops_exhausted_failed_group_and_keeps_collecting(monkeypatch):
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

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ROLLOUT_MAX_RETRIES", "0")
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS", "10")

    data = DataBuffer([[SampleLike(index=0)], [SampleLike(index=1)]])
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1, global_batch_size=1)

    result = _samples(partial_async_rollout.generate_rollout_partial_async(args, 0, data))

    assert attempts_by_index == {0: 1, 1: 1}
    assert [group[0].index for group in result] == [1]
    assert result[0][0].status == SampleLike.Status.COMPLETED


def test_partial_async_rollout_reports_staleness_rejected_groups(monkeypatch):
    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        sample = group[0]
        if sample.index == 0:
            sample.status = SampleLike.Status.ABORTED
            sample.metadata["blackbox_error"] = (
                "Dressage proxy error=partial_rollout_staleness_exceeded "
                "version_span=3 max_version_span=2"
            )
            sample.session_id = None
            return group

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

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setattr(partial_async_rollout, "RolloutFnTrainOutput", TrainOutput)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "1")
    monkeypatch.setenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS", "1")
    monkeypatch.setenv("DRESSAGE_ROLLOUT_MAX_RETRIES", "0")
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_DROPPED_FAILED_GROUPS", "10")

    data = DataBuffer([[SampleLike(index=0)], [SampleLike(index=1)]])
    args = SimpleNamespace(rollout_batch_size=1, n_samples_per_prompt=1, global_batch_size=1)

    output = partial_async_rollout.generate_rollout_partial_async(args, 0, data)

    assert _samples(output)[0][0].index == 1
    assert output.metrics["staleness/partial_rollout_rejected_groups"] == 1
    assert output.metrics["staleness/partial_rollout_rejected_samples"] == 1


def test_partial_async_rollout_drains_worker_after_final_rollout(monkeypatch):
    async def fake_generate_and_rm_group(args, group, sampling_params, evaluation=False):
        del args, sampling_params, evaluation
        if group[0].index > 0:
            await asyncio.sleep(0.05)
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

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setattr(partial_async_rollout, "RolloutFnTrainOutput", TrainOutput)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "3")
    monkeypatch.setenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS", "1")

    data = DataBuffer([[SampleLike(index=i)] for i in range(3)])
    args = SimpleNamespace(
        rollout_batch_size=3,
        n_samples_per_prompt=1,
        global_batch_size=1,
        num_rollout=1,
    )

    output = partial_async_rollout.generate_rollout_partial_async(args, 0, data)
    result = _samples(output)

    assert [group[0].index for group in result] == [0]
    assert partial_async_rollout._GLOBAL_PARTIAL_WORKER is None
    assert output.metrics["dressage/partial_rollout_final_worker_drain"] == 1
    assert output.metrics["dressage/partial_rollout_drained_completed_groups"] >= 1


def test_partial_async_rollout_drops_stale_group_by_trajectory(monkeypatch):
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

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "3")
    monkeypatch.setenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS", "2")

    stale_group = [
        SampleLike(index=0, metadata={"parent_traj_id": "traj-old", "dressage_end_token_version": "old"}),
    ]
    data = DataBuffer([
        stale_group,
        [SampleLike(index=1, metadata={"parent_traj_id": "traj-new-1", "dressage_end_token_version": "new"})],
        [SampleLike(index=2, metadata={"parent_traj_id": "traj-new-2", "dressage_end_token_version": "new"})],
    ])
    args = SimpleNamespace(
        rollout_batch_size=3,
        n_samples_per_prompt=1,
        global_batch_size=2,
        dressage_staleness_keep_versions=1,
    )

    result = _samples(partial_async_rollout.generate_rollout_partial_async(args, 0, data))

    assert all(isinstance(group, list) for group in result)
    assert [[sample.index for sample in group] for group in result] == [[1], [2]]


def test_partial_async_rollout_returns_staleness_metrics(monkeypatch):
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

    monkeypatch.setattr(partial_async_rollout, "generate_and_rm_group", fake_generate_and_rm_group)
    monkeypatch.setattr(partial_async_rollout, "GenerateState", None)
    monkeypatch.setattr(partial_async_rollout, "RolloutFnTrainOutput", TrainOutput)
    monkeypatch.setenv("DRESSAGE_PARTIAL_ROLLOUT_TARGET_GROUPS", "1")

    data = DataBuffer([[
        SampleLike(index=0, metadata={"parent_traj_id": "long", "segment_index": 0, "dressage_end_token_version": "old"}),
        SampleLike(index=1, metadata={"parent_traj_id": "long", "segment_index": 1, "dressage_end_token_version": "middle"}),
        SampleLike(index=2, metadata={"parent_traj_id": "short", "dressage_end_token_version": "new"}),
    ]])
    args = SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
        dressage_staleness_keep_versions=3,
    )

    output = partial_async_rollout.generate_rollout_partial_async(args, 0, data)

    assert all(isinstance(group, list) for group in output.samples)
    assert output.metrics["staleness/current_version_index"] == 1.0
    assert output.metrics["staleness/version_gap_mean"] == pytest.approx(0.5)
