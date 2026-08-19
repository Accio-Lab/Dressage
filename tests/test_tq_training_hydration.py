from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dressage.training.tq_hydration import (
    FULL_LOGPROBS_FIELD,
    ROUTED_EXPERTS_FIELD,
    clear_requests_from_layouts,
    hydrate_training_layouts,
    validate_tq_training_config,
)
from dressage.transport import (
    TQ_LOGPROBS_CODEC,
    TQ_ROUTED_EXPERTS_CODEC,
    TQFieldFragment,
    TQFieldLayout,
    TQFieldRef,
)


class FakeBatchGet:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, *, keys, partition_id, select_fields):
        self.calls.append((tuple(keys), partition_id, select_fields))
        return {
            select_fields: [
                self.values[(partition_id, key, select_fields)] for key in keys
            ]
        }


def test_tq_training_rejects_mopd_prompt_channel(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DRESSAGE_MOPD_TEACHER_CONFIG", "/tmp/teacher.yaml")
    with pytest.raises(ValueError, match="cannot share"):
        validate_tq_training_config(SimpleNamespace(mopd_teacher_config=None))

    monkeypatch.delenv("DRESSAGE_MOPD_TEACHER_CONFIG")
    validate_tq_training_config(SimpleNamespace(mopd_teacher_config=None))


def _field_ref(key: str, field: str) -> TQFieldRef:
    return TQFieldRef(
        store_id="store",
        partition="steps",
        key=key,
        field=field,
        incarnation_id="incarnation",
    )


def _layout(
    logical_field: str,
    codec: str,
    ref: TQFieldRef,
    *,
    source_start: int,
    target_start: int,
    length: int,
    token_count: int,
    shape: tuple[int, ...] | None = None,
) -> TQFieldLayout:
    return TQFieldLayout(
        logical_field=logical_field,
        codec=codec,
        fragments=(
            TQFieldFragment(
                ref=ref,
                source_start=source_start,
                target_start=target_start,
                length=length,
            ),
        ),
        token_count=token_count,
        shape=shape,
    )


def _routed_experts_chunks(
    values: np.ndarray,
    dtype: str = "int32",
) -> list[dict]:
    return [
        {
            "data": np.ascontiguousarray(values.astype(dtype, copy=False)),
            "row_count": values.shape[0],
            "dtype": dtype,
        }
    ]


@pytest.mark.parametrize(
    "remote_fields",
    [
        {FULL_LOGPROBS_FIELD},
        {ROUTED_EXPERTS_FIELD},
        {FULL_LOGPROBS_FIELD, ROUTED_EXPERTS_FIELD},
    ],
)
def test_hydrates_selected_field_combinations(remote_fields):
    logprob_ref = _field_ref("step-0", "response_logprobs")
    routed_ref = _field_ref("step-0", "response_routed_experts_chunks")
    routed_values = np.arange(12, dtype=np.int32).reshape(3, 4)
    layouts = {
        FULL_LOGPROBS_FIELD: _layout(
            FULL_LOGPROBS_FIELD,
            TQ_LOGPROBS_CODEC,
            logprob_ref,
            source_start=0,
            target_start=2,
            length=2,
            token_count=4,
            shape=(4,),
        ).to_dict(),
        ROUTED_EXPERTS_FIELD: _layout(
            ROUTED_EXPERTS_FIELD,
            TQ_ROUTED_EXPERTS_CODEC,
            routed_ref,
            source_start=0,
            target_start=0,
            length=3,
            token_count=4,
            shape=(3, 4),
        ).to_dict(),
    }
    layouts = {
        field: layout for field, layout in layouts.items() if field in remote_fields
    }
    batch_get = FakeBatchGet(
        {
            ("steps", "step-0", "response_logprobs"): [-0.2, -0.3],
            (
                "steps",
                "step-0",
                "response_routed_experts_chunks",
            ): _routed_experts_chunks(routed_values),
        }
    )
    rollout_data = {
        "total_lengths": [4],
        "response_lengths": [2],
    }

    hydrate_training_layouts(
        SimpleNamespace(
            use_rollout_routing_replay=True,
            num_layers=2,
            moe_router_topk=2,
        ),
        rollout_data,
        [layouts],
        remote_fields=remote_fields,
        batch_get=batch_get,
    )

    if FULL_LOGPROBS_FIELD in remote_fields:
        assert torch.equal(
            rollout_data["rollout_log_probs"][0],
            torch.tensor([-0.2, -0.3], dtype=torch.float32),
        )
    else:
        assert "rollout_log_probs" not in rollout_data
    if ROUTED_EXPERTS_FIELD in remote_fields:
        assert torch.equal(
            rollout_data["rollout_routed_experts"][0].reshape(3, 4),
            torch.from_numpy(routed_values),
        )
    else:
        assert "rollout_routed_experts" not in rollout_data


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32"])
def test_hydrates_routed_experts_from_recorded_dtype(dtype):
    routed_ref = _field_ref("step-0", "response_routed_experts_chunks")
    routed_values = np.arange(12, dtype=np.int32).reshape(3, 4)
    layout = _layout(
        ROUTED_EXPERTS_FIELD,
        TQ_ROUTED_EXPERTS_CODEC,
        routed_ref,
        source_start=0,
        target_start=0,
        length=3,
        token_count=4,
        shape=(3, 4),
    ).to_dict()
    batch_get = FakeBatchGet(
        {
            (
                "steps",
                "step-0",
                "response_routed_experts_chunks",
            ): _routed_experts_chunks(routed_values, dtype),
        }
    )
    rollout_data = {
        "total_lengths": [4],
        "response_lengths": [3],
    }

    hydrate_training_layouts(
        SimpleNamespace(
            use_rollout_routing_replay=True,
            num_layers=2,
            moe_router_topk=2,
        ),
        rollout_data,
        [{ROUTED_EXPERTS_FIELD: layout}],
        remote_fields={ROUTED_EXPERTS_FIELD},
        batch_get=batch_get,
    )

    assert rollout_data["rollout_routed_experts"][0].dtype == torch.int32
    assert torch.equal(
        rollout_data["rollout_routed_experts"][0].reshape(3, 4),
        torch.from_numpy(routed_values),
    )


def test_failed_samples_without_layout_are_zero_filled_without_tq_reads():
    rollout_data = {
        "total_lengths": [1, 3],
        "response_lengths": [0, 2],
    }

    hydrate_training_layouts(
        SimpleNamespace(
            use_rollout_routing_replay=True,
            num_layers=2,
            moe_router_topk=2,
        ),
        rollout_data,
        [None, None],
        remote_fields={FULL_LOGPROBS_FIELD, ROUTED_EXPERTS_FIELD},
        batch_get=lambda **kwargs: pytest.fail(
            "Failed samples must not read TransferQueue"
        ),
    )

    assert rollout_data["rollout_log_probs"][0].shape == (0,)
    assert rollout_data["rollout_log_probs"][1].shape == (2,)
    assert rollout_data["rollout_routed_experts"][0].shape == (0, 2, 2)
    assert rollout_data["rollout_routed_experts"][1].shape == (2, 2, 2)
    assert torch.count_nonzero(rollout_data["rollout_routed_experts"][1]) == 0


def test_routed_experts_are_not_read_when_routing_replay_is_disabled():
    routed_ref = _field_ref("step-0", "response_routed_experts_chunks")
    layout = _layout(
        ROUTED_EXPERTS_FIELD,
        TQ_ROUTED_EXPERTS_CODEC,
        routed_ref,
        source_start=0,
        target_start=0,
        length=2,
        token_count=3,
        shape=(2, 4),
    )
    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
    }

    hydrate_training_layouts(
        SimpleNamespace(use_rollout_routing_replay=False),
        rollout_data,
        [{ROUTED_EXPERTS_FIELD: layout.to_dict()}],
        remote_fields={ROUTED_EXPERTS_FIELD},
        batch_get=lambda **kwargs: pytest.fail(
            "Unused routed experts must not be read"
        ),
    )

    assert "rollout_routed_experts" not in rollout_data


def test_batch_get_deduplicates_shared_field_refs_after_dp_sharding():
    logprob_ref = _field_ref("step-0", "response_logprobs")
    layout = _layout(
        FULL_LOGPROBS_FIELD,
        TQ_LOGPROBS_CODEC,
        logprob_ref,
        source_start=0,
        target_start=1,
        length=2,
        token_count=3,
        shape=(3,),
    )
    batch_get = FakeBatchGet(
        {("steps", "step-0", "response_logprobs"): [-0.1, -0.2]}
    )
    rollout_data = {
        "total_lengths": [3, 3],
        "response_lengths": [2, 2],
    }

    hydrate_training_layouts(
        SimpleNamespace(use_rollout_routing_replay=False),
        rollout_data,
        [
            {FULL_LOGPROBS_FIELD: layout.to_dict()},
            {FULL_LOGPROBS_FIELD: layout.to_dict()},
        ],
        remote_fields={FULL_LOGPROBS_FIELD},
        batch_get=batch_get,
    )

    assert batch_get.calls == [
        (("step-0",), "steps", "response_logprobs")
    ]
    assert torch.equal(
        rollout_data["rollout_log_probs"][0],
        rollout_data["rollout_log_probs"][1],
    )


def test_clear_requests_deduplicate_keys_by_partition():
    first = _field_ref("step-0", "response_logprobs")
    second = _field_ref("step-1", "response_logprobs")
    first_layout = _layout(
        FULL_LOGPROBS_FIELD,
        TQ_LOGPROBS_CODEC,
        first,
        source_start=0,
        target_start=1,
        length=2,
        token_count=3,
    )
    second_layout = _layout(
        FULL_LOGPROBS_FIELD,
        TQ_LOGPROBS_CODEC,
        second,
        source_start=0,
        target_start=1,
        length=2,
        token_count=3,
    )

    assert clear_requests_from_layouts(
        [
            {FULL_LOGPROBS_FIELD: first_layout.to_dict()},
            {FULL_LOGPROBS_FIELD: first_layout.to_dict()},
            {FULL_LOGPROBS_FIELD: second_layout.to_dict()},
            None,
        ]
    ) == {"steps": ["step-0", "step-1"]}


def test_layout_materialization_truncates_to_the_sample_token_count():
    logprob_ref = _field_ref("step-0", "response_logprobs")
    layout = _layout(
        FULL_LOGPROBS_FIELD,
        TQ_LOGPROBS_CODEC,
        logprob_ref,
        source_start=0,
        target_start=1,
        length=4,
        token_count=5,
        shape=(5,),
    )
    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
    }

    hydrate_training_layouts(
        SimpleNamespace(use_rollout_routing_replay=False),
        rollout_data,
        [{FULL_LOGPROBS_FIELD: layout.to_dict()}],
        remote_fields={FULL_LOGPROBS_FIELD},
        batch_get=FakeBatchGet(
            {
                ("steps", "step-0", "response_logprobs"): [
                    -0.1,
                    -0.2,
                    -0.3,
                    -0.4,
                ]
            }
        ),
    )

    assert torch.equal(
        rollout_data["rollout_log_probs"][0],
        torch.tensor([-0.1, -0.2], dtype=torch.float32),
    )


def test_incremental_routed_experts_fragments_are_materialized_in_layout_order():
    first_ref = _field_ref("step-0", "response_routed_experts_chunks")
    second_ref = _field_ref("step-1", "response_routed_experts_chunks")
    first_values = np.arange(4, dtype=np.int32).reshape(1, 4)
    second_values = np.arange(8, dtype=np.int32).reshape(2, 4) + 10
    layout = TQFieldLayout(
        logical_field=ROUTED_EXPERTS_FIELD,
        codec=TQ_ROUTED_EXPERTS_CODEC,
        fragments=(
            TQFieldFragment(
                ref=first_ref,
                source_start=0,
                target_start=0,
                length=1,
            ),
            TQFieldFragment(
                ref=second_ref,
                source_start=0,
                target_start=1,
                length=2,
            ),
        ),
        token_count=4,
        shape=(3, 4),
    )
    batch_get = FakeBatchGet(
        {
            (
                "steps",
                "step-0",
                "response_routed_experts_chunks",
            ): _routed_experts_chunks(first_values),
            (
                "steps",
                "step-1",
                "response_routed_experts_chunks",
            ): _routed_experts_chunks(second_values),
        }
    )
    rollout_data = {
        "total_lengths": [4],
        "response_lengths": [3],
    }

    hydrate_training_layouts(
        SimpleNamespace(
            use_rollout_routing_replay=True,
            num_layers=2,
            moe_router_topk=2,
        ),
        rollout_data,
        [{ROUTED_EXPERTS_FIELD: layout.to_dict()}],
        remote_fields={ROUTED_EXPERTS_FIELD},
        batch_get=batch_get,
    )

    expected = np.concatenate([first_values, second_values], axis=0)
    assert torch.equal(
        rollout_data["rollout_routed_experts"][0].reshape(3, 4),
        torch.from_numpy(expected),
    )
    assert batch_get.calls == [
        (
            ("step-0", "step-1"),
            "steps",
            "response_routed_experts_chunks",
        )
    ]


def test_logprobs_use_the_training_actor_transform_after_hydration():
    logprob_ref = _field_ref("step-0", "response_logprobs")
    layout = _layout(
        FULL_LOGPROBS_FIELD,
        TQ_LOGPROBS_CODEC,
        logprob_ref,
        source_start=0,
        target_start=1,
        length=2,
        token_count=3,
        shape=(3,),
    )
    calls = []

    def transform(values, total_length, response_length):
        calls.append((values.clone(), total_length, response_length))
        return values + 1

    rollout_data = {
        "total_lengths": [3],
        "response_lengths": [2],
    }
    hydrate_training_layouts(
        SimpleNamespace(use_rollout_routing_replay=False),
        rollout_data,
        [{FULL_LOGPROBS_FIELD: layout.to_dict()}],
        remote_fields={FULL_LOGPROBS_FIELD},
        batch_get=FakeBatchGet(
            {("steps", "step-0", "response_logprobs"): [-0.1, -0.2]}
        ),
        logprob_transform=transform,
    )

    assert calls[0][1:] == (3, 2)
    assert torch.equal(
        calls[0][0],
        torch.tensor([-0.1, -0.2], dtype=torch.float32),
    )
    assert torch.equal(
        rollout_data["rollout_log_probs"][0],
        torch.tensor([0.9, 0.8], dtype=torch.float32),
    )


def test_logprobs_preserve_proxy_padding_and_invalid_value_normalization():
    logprob_ref = _field_ref("step-0", "response_logprobs")
    layout = _layout(
        FULL_LOGPROBS_FIELD,
        TQ_LOGPROBS_CODEC,
        logprob_ref,
        source_start=0,
        target_start=1,
        length=3,
        token_count=4,
        shape=(4,),
    )
    rollout_data = {
        "total_lengths": [4],
        "response_lengths": [3],
    }

    hydrate_training_layouts(
        SimpleNamespace(use_rollout_routing_replay=False),
        rollout_data,
        [{FULL_LOGPROBS_FIELD: layout.to_dict()}],
        remote_fields={FULL_LOGPROBS_FIELD},
        batch_get=FakeBatchGet(
            {("steps", "step-0", "response_logprobs"): [-0.1, None]}
        ),
    )

    assert torch.equal(
        rollout_data["rollout_log_probs"][0],
        torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float32),
    )
