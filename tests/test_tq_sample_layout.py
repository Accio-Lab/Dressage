from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

from dressage.rollout.artifacts.samples import write_sample_from_segment
from dressage.transport import (
    TQ_LOGPROBS_CODEC,
    TQ_ROUTED_EXPERTS_CODEC,
    TQ_SAMPLE_REF_METADATA_KEY,
    TQFieldFragment,
    TQFieldLayout,
    TQFieldRef,
)


class _Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"


@dataclass
class _Sample:
    metadata: dict = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    rollout_routed_experts: object | None = None
    status: _Status = _Status.PENDING
    Status = _Status


def _layout(logical_field: str, codec: str, field: str, token_count: int):
    ref = TQFieldRef(
        store_id="store",
        partition="steps",
        key="step-0",
        field=field,
        incarnation_id="incarnation",
    )
    length = token_count if logical_field == "full_logprobs" else token_count - 1
    target_start = 0 if logical_field == "routed_experts" else 2
    if logical_field == "full_logprobs":
        length -= target_start
    return TQFieldLayout(
        logical_field=logical_field,
        codec=codec,
        fragments=(
            TQFieldFragment(
                ref=ref,
                source_start=0,
                target_start=target_start,
                length=length,
            ),
        ),
        token_count=token_count,
        shape=(length,),
    ), ref


def test_segment_layout_reaches_sample_without_materializing_or_rewriting_fragments():
    logprobs, _ = _layout(
        "full_logprobs",
        TQ_LOGPROBS_CODEC,
        "response_logprobs",
        5,
    )
    routed_experts, _ = _layout(
        "routed_experts",
        TQ_ROUTED_EXPERTS_CODEC,
        "response_routed_experts_chunks",
        5,
    )
    segment = {
        "uid": "segment",
        "segment_index": 0,
        "tokens": [10, 11, 12, 13, 14],
        "full_loss_mask": [0, 0, 1, 1, 1],
        "full_logprobs": logprobs.to_dict(),
        "routed_experts_chunks": routed_experts.to_dict(),
        "messages": [{"role": "assistant", "content": "answer"}],
        "finish_reason": "stop",
    }
    sample = _Sample()

    write_sample_from_segment(
        sample,
        args=SimpleNamespace(
            max_tokens_per_gpu=4,
            context_parallel_size=1,
            use_rollout_routing_replay=True,
            num_layers=2,
            moe_router_topk=2,
        ),
        segment=segment,
        all_segments=[segment],
        session_id="trajectory",
        instance_id="instance",
        agent_response="fallback",
    )

    assert sample.tokens == [10, 11, 12, 13]
    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]
    assert sample.rollout_log_probs is None
    assert sample.rollout_routed_experts is None
    assert sample.status is _Status.COMPLETED
    assert sample.metadata["truncated"] is True
    layouts = sample.metadata[TQ_SAMPLE_REF_METADATA_KEY]
    assert layouts["full_logprobs"] == logprobs.to_dict()
    assert layouts["routed_experts"] == routed_experts.to_dict()
