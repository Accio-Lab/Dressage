from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dressage.transport import (
    TQ_FIELD_LAYOUT_SCHEMA,
    TQ_FIELD_REF_SCHEMA,
    TQFieldFragment,
    TQFieldLayout,
    TQFieldRef,
    TQTrajectoryRef,
    TransferQueueRuntime,
    TransferQueueStore,
    build_concatenated_logprobs,
    build_snapshot_logprobs,
    is_tq_field_fragment_dict,
    is_tq_field_layout_dict,
    is_tq_field_ref_dict,
    is_tq_trajectory_ref_dict,
    normalize_transfer_params,
    resolve_step_transfer_fields,
)


class _FakeTransferQueue:
    def __init__(self) -> None:
        self.async_kv_put = AsyncMock(return_value=object())
        self.async_kv_clear = AsyncMock(return_value=None)
        self.async_kv_list = AsyncMock(return_value={})


def test_transport_import_does_not_import_optional_dependency() -> None:
    import dressage.transport.store as store_module

    assert "transfer_queue" not in store_module.__dict__


def test_transfer_param_mapping_is_stable_and_mode_aware() -> None:
    assert normalize_transfer_params("routed_experts, logprobs logprobs") == (
        "logprobs",
        "routed_experts",
    )
    assert normalize_transfer_params(["routed_experts,logprobs"]) == (
        "logprobs",
        "routed_experts",
    )
    assert resolve_step_transfer_fields(
        ["logprobs", "routed_experts"],
        token_build_mode="snapshot",
    ) == (
        "prompt_token_logprobs",
        "response_logprobs",
        "all_logprobs",
        "response_routed_experts_chunks",
    )
    assert resolve_step_transfer_fields(
        ["routed_experts", "logprobs"],
        token_build_mode="tito",
    ) == (
        "prompt_token_logprobs",
        "response_logprobs",
        "all_logprobs",
        "concat_response_logprobs",
        "response_routed_experts_chunks",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_transfer_params(["tokens"])


def test_reference_and_layout_serialization_round_trip() -> None:
    ref = TQFieldRef(
        store_id="store",
        partition="steps",
        key="step-1",
        field="response_logprobs",
        incarnation_id="incarnation",
    )
    fragment = TQFieldFragment(
        ref=ref,
        source_start=2,
        target_start=5,
        length=3,
    )
    layout = TQFieldLayout(
        logical_field="full_logprobs",
        codec="float32",
        fragments=(fragment,),
        token_count=9,
        shape=(9,),
    )
    trajectory = TQTrajectoryRef(
        store_id="store",
        trajectory_id="trajectory",
        incarnation_id="incarnation",
        refs=(ref,),
    )

    assert is_tq_field_ref_dict(ref.to_dict())
    assert is_tq_field_fragment_dict(fragment.to_dict())
    assert is_tq_field_layout_dict(layout.to_dict())
    assert is_tq_trajectory_ref_dict(trajectory.to_dict())
    assert TQFieldRef.from_dict(ref.to_dict()) == ref
    assert TQFieldFragment.from_dict(fragment.to_dict()) == fragment
    assert TQFieldLayout.from_dict(layout.to_dict()) == layout
    assert TQTrajectoryRef.from_dict(trajectory.to_dict()) == trajectory
    assert layout.to_dict()["shape"] == [9]

    invalid = ref.to_dict() | {"schema_version": TQ_FIELD_LAYOUT_SCHEMA}
    assert not is_tq_field_ref_dict(invalid)
    with pytest.raises(ValueError, match="schema"):
        TQFieldRef.from_dict(invalid)
    assert ref.to_dict()["schema_version"] == TQ_FIELD_REF_SCHEMA


def test_logprob_layout_builders_preserve_target_offsets() -> None:
    first = TQFieldRef(
        store_id="store",
        partition="steps",
        key="step-1",
        field="response_logprobs",
        incarnation_id="incarnation",
    )
    second = TQFieldRef(
        store_id="store",
        partition="steps",
        key="step-2",
        field="concat_response_logprobs",
        incarnation_id="incarnation",
    )

    snapshot, invalid = build_snapshot_logprobs(
        first,
        prompt_length=3,
        response_length=2,
        token_count=5,
        remote_invalid=True,
    )
    concatenated = build_concatenated_logprobs(
        [first, second],
        [2, 3],
        token_count=5,
    )

    assert snapshot.fragments[0].target_start == 3
    assert invalid is True
    local_snapshot, invalid = build_snapshot_logprobs(
        [-0.1],
        prompt_length=2,
        response_length=2,
        token_count=4,
        remote_invalid=True,
    )
    assert local_snapshot == [0.0, 0.0, -0.1, 0.0]
    assert invalid is True
    _, invalid = build_snapshot_logprobs(
        [-0.1, -0.2],
        prompt_length=2,
        response_length=2,
        token_count=4,
        remote_invalid=True,
    )
    assert invalid is False
    assert [fragment.target_start for fragment in concatenated.fragments] == [
        0,
        2,
    ]
    assert build_concatenated_logprobs(
        [[0.1], [0.2, 0.3]],
        [1, 2],
        token_count=3,
    ) == [0.1, 0.2, 0.3]
    with pytest.raises(RuntimeError, match="not aligned"):
        build_concatenated_logprobs(
            [[0.1], [0.2]],
            [1, 1],
            token_count=3,
        )


@pytest.mark.asyncio
async def test_store_wraps_official_async_kv_api() -> None:
    tq = _FakeTransferQueue()
    store = TransferQueueStore(tq, store_id="run")
    key = store.step_key(
        trajectory_id="trajectory",
        incarnation_id="incarnation",
        step_index=3,
    )

    await store.put(
        key=key,
        partition=store.step_partition,
        fields={"response_logprobs": [0.1]},
        tag={"expires_at": 10.0},
    )
    await store.clear(keys=[key], partition=store.step_partition)
    await store.list(partition=store.step_partition)

    assert key == "run:trajectory:incarnation:step:3"
    tq.async_kv_put.assert_awaited_once_with(
        key=key,
        partition_id="dressage:run:steps",
        fields={"response_logprobs": [0.1]},
        tag={"expires_at": 10.0},
    )
    tq.async_kv_clear.assert_awaited_once_with(
        keys=[key],
        partition_id="dressage:run:steps",
    )
    tq.async_kv_list.assert_awaited_once_with(
        partition_id="dressage:run:steps"
    )


@pytest.mark.asyncio
async def test_runtime_writes_only_selected_columns_and_builds_trajectory_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tq = _FakeTransferQueue()
    store = TransferQueueStore(tq, store_id="run")
    runtime = TransferQueueRuntime(
        store,
        transfer_params="logprobs routed_experts",
        token_build_mode="snapshot",
        retention_seconds=30.0,
    )
    monkeypatch.setattr("dressage.transport.runtime.time.time", lambda: 100.0)

    refs = await runtime.offload_step(
        session_id="trajectory",
        incarnation_id="incarnation",
        step_index=0,
        fields={
            "prompt_token_logprobs": [0.0],
            "response_logprobs": [0.1, 0.2],
            "all_logprobs": [0.0, 0.1, 0.2],
            "response_routed_experts_chunks": [{"data": "encoded"}],
            "concat_response_logprobs": [9.0],
            "all_token_ids": [1, 2, 3],
        },
    )
    trajectory = runtime.trajectory_ref(
        SimpleNamespace(
            session_id="trajectory",
            incarnation_id="incarnation",
            steps=[SimpleNamespace(**refs)],
        )
    )

    assert tuple(refs) == (
        "prompt_token_logprobs",
        "response_logprobs",
        "all_logprobs",
        "response_routed_experts_chunks",
    )
    assert trajectory.refs == (refs["prompt_token_logprobs"],)
    tq.async_kv_put.assert_awaited_once_with(
        key="run:trajectory:incarnation:step:0",
        partition_id="dressage:run:steps",
        fields={
            "prompt_token_logprobs": [0.0],
            "response_logprobs": [0.1, 0.2],
            "all_logprobs": [0.0, 0.1, 0.2],
            "response_routed_experts_chunks": [{"data": "encoded"}],
        },
        tag={
            "expires_at": 130.0,
            "trajectory_id": "trajectory",
            "incarnation_id": "incarnation",
        },
    )


@pytest.mark.asyncio
async def test_runtime_clear_is_deduplicated_and_idempotent() -> None:
    tq = _FakeTransferQueue()
    store = TransferQueueStore(tq, store_id="run")
    runtime = TransferQueueRuntime(
        store,
        transfer_params="logprobs",
        token_build_mode="snapshot",
    )
    refs = await runtime.offload_step(
        session_id="trajectory",
        incarnation_id="incarnation",
        step_index=0,
        fields={
            "prompt_token_logprobs": [],
            "response_logprobs": [0.1],
            "all_logprobs": [0.1],
        },
    )
    ref = refs["response_logprobs"]

    await runtime.clear_refs([ref, ref])
    await runtime.clear_refs([ref])

    assert tq.async_kv_clear.await_count == 2
    assert tq.async_kv_clear.await_args_list[0].kwargs == {
        "keys": [ref.key],
        "partition_id": ref.partition,
    }


@pytest.mark.asyncio
async def test_runtime_trajectory_ref_scans_session_steps() -> None:
    tq = _FakeTransferQueue()
    runtime = TransferQueueRuntime(
        TransferQueueStore(tq, store_id="run"),
        transfer_params="logprobs",
        token_build_mode="snapshot",
    )
    session = SimpleNamespace(
        session_id="trajectory",
        incarnation_id="incarnation",
        steps=[],
    )
    first_refs = await runtime.offload_step(
        session_id=session.session_id,
        incarnation_id=session.incarnation_id,
        step_index=0,
        fields={
            "prompt_token_logprobs": [],
            "response_logprobs": [0.1],
            "all_logprobs": [0.1],
        },
    )
    second_refs = await runtime.offload_step(
        session_id=session.session_id,
        incarnation_id=session.incarnation_id,
        step_index=1,
        fields={
            "prompt_token_logprobs": [],
            "response_logprobs": [0.2],
            "all_logprobs": [0.2],
        },
    )
    session.steps = [
        SimpleNamespace(**first_refs),
        SimpleNamespace(**second_refs),
    ]

    trajectory_ref = runtime.trajectory_ref(session)

    assert trajectory_ref.refs == (
        first_refs["prompt_token_logprobs"],
        second_refs["prompt_token_logprobs"],
    )
    assert "tracked_trajectories" not in runtime.stats()


@pytest.mark.asyncio
async def test_runtime_retention_clears_only_expired_keys() -> None:
    tq = _FakeTransferQueue()
    store = TransferQueueStore(tq, store_id="run")
    runtime = TransferQueueRuntime(
        store,
        transfer_params="routed_experts",
        token_build_mode="tito",
    )
    tq.async_kv_list.return_value = {
        store.step_partition: {
            "expired": {"expires_at": 9.0},
            "live": {"expires_at": 11.0},
            "untagged": {},
        }
    }

    assert await runtime.sweep_expired(now=10.0) == 1
    tq.async_kv_clear.assert_awaited_once_with(
        keys=["expired"],
        partition_id=store.step_partition,
    )


@pytest.mark.asyncio
async def test_runtime_retention_task_can_be_closed() -> None:
    tq = _FakeTransferQueue()
    runtime = TransferQueueRuntime(
        TransferQueueStore(tq, store_id="run"),
        transfer_params="logprobs",
        token_build_mode="snapshot",
        retention_poll_seconds=0.001,
    )
    task = runtime.start_retention()
    assert runtime.start_retention() is task
    await asyncio.sleep(0)
    await runtime.close()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_runtime_retention_continues_after_external_failure() -> None:
    tq = _FakeTransferQueue()
    tq.async_kv_list.side_effect = [RuntimeError("temporary"), {}, {}]
    runtime = TransferQueueRuntime(
        TransferQueueStore(tq, store_id="run"),
        transfer_params="logprobs",
        token_build_mode="snapshot",
        retention_poll_seconds=0.001,
    )

    runtime.start_retention()
    for _ in range(100):
        if tq.async_kv_list.await_count >= 2:
            break
        await asyncio.sleep(0.001)
    await runtime.close()

    assert tq.async_kv_list.await_count >= 2
