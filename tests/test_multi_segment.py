"""Unit tests for dressage.rollout.multi_segment.

These tests are generate-implementation-agnostic — they exercise the public
helpers directly with synthetic segment dicts and a SampleLike fake.
Integration with `blackbox_dispatch.generate` is covered by
`tests/test_blackbox_dispatch.py`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from dressage.rollout.multi_segment import (
    compute_multi_segment_metrics,
    compute_prewarm_metrics,
    expand_segments_to_samples,
    mark_aborted_no_grad,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class SampleLike:
    prompt: str = "p"
    label: str | None = None
    group_index: int | None = 7
    index: int | None = 3
    rollout_id: int | None = None
    session_id: str | None = "sess-7"
    metadata: dict = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    reward: float | None = None
    remove_sample: bool = False

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING


_UNSET = object()


def _segment(
    *,
    segment_index: int = 0,
    tokens: list[int] | None = None,
    full_loss_mask: list[int] | None = None,
    full_logprobs: list[float] | None = None,
    full_versions: list[str] | None = None,
    messages: Any = _UNSET,
    finish_reason: str = "stop",
    uid: Any = _UNSET,
    extra_info: dict | None = None,
) -> dict:
    if messages is _UNSET:
        messages = [{"role": "assistant", "content": f"resp-{segment_index}"}]
    if uid is _UNSET:
        uid = f"seg-{segment_index}"
    segment = {
        "uid": uid,
        "segment_index": segment_index,
        "tokens": tokens if tokens is not None else [100, 200, 300],
        "full_loss_mask": full_loss_mask if full_loss_mask is not None else [0, 1, 1],
        "full_logprobs": full_logprobs if full_logprobs is not None else [0.0, -0.1, -0.2],
        "messages": messages,
        "finish_reason": finish_reason,
        "extra_info": extra_info or {},
    }
    if full_versions is not None:
        segment["full_versions"] = full_versions
    return segment


def _args(
    *,
    max_tokens_per_gpu: int | None = 64,
    context_parallel_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_tokens_per_gpu=max_tokens_per_gpu,
        context_parallel_size=context_parallel_size,
    )


def _sample_with_meta(**meta) -> SampleLike:
    return SampleLike(metadata=meta)


# ---------------------------------------------------------------------------
# expand_segments_to_samples
# ---------------------------------------------------------------------------


def test_expand_empty_segments_raises():
    with pytest.raises(ValueError, match="empty segments"):
        expand_segments_to_samples(SampleLike(), [], args=_args())


def test_expand_single_segment_keeps_reward_none():
    """Single segment IS the last segment → reward stays None for slime to compute."""
    sample = _sample_with_meta(instance_id="inst-1")
    result = expand_segments_to_samples(sample, [_segment(segment_index=0)], args=_args())
    assert len(result) == 1
    only = result[0]
    assert only.reward is None
    assert only.metadata["segment_index"] == 0
    assert only.metadata["segment_count"] == 1
    assert only.metadata["parent_traj_id"] == "sess-7"


def test_expand_multiple_segments_only_last_keeps_reward_none():
    """Non-last segments get reward=0.0 pre-set so slime skips reward_fn;
    last segment keeps reward=None so reward_fn runs once."""
    segs = [_segment(segment_index=i) for i in (1, 0, 2)]  # unsorted on purpose
    result = expand_segments_to_samples(SampleLike(), segs, args=_args())
    assert [s.metadata["segment_index"] for s in result] == [0, 1, 2]
    assert [s.reward for s in result] == [0.0, 0.0, None]


def test_expand_sets_rollout_id_per_trajectory():
    """All segments of one trajectory share rollout_id (= template.index) so
    slime's build_dp_schedule keeps them together in one training step."""
    sample = SampleLike(index=42)
    segs = [_segment(segment_index=i) for i in range(3)]
    result = expand_segments_to_samples(sample, segs, args=_args())
    assert all(s.rollout_id == 42 for s in result)


def test_expand_does_not_mutate_template():
    sample = _sample_with_meta(instance_id="inst-1", custom_key="keep me")
    snapshot = copy.deepcopy(sample)
    expand_segments_to_samples(sample, [_segment(), _segment(segment_index=1)], args=_args())
    assert sample.tokens == snapshot.tokens
    assert sample.response_length == snapshot.response_length
    assert sample.metadata == snapshot.metadata


def test_expand_preserves_template_metadata():
    sample = _sample_with_meta(instance_id="inst-1", custom_key="keep me")
    result = expand_segments_to_samples(sample, [_segment()], args=_args())
    assert result[0].metadata["custom_key"] == "keep me"
    assert result[0].metadata["instance_id"] == "inst-1"


def test_expand_writes_tokens_loss_mask_logprobs_correctly():
    seg = _segment(
        tokens=[1, 2, 3, 4, 5],
        full_loss_mask=[0, 0, 1, 1, 1],
        full_logprobs=[0.0, 0.0, -0.5, -0.4, -0.3],
    )
    sample = expand_segments_to_samples(SampleLike(), [seg], args=_args())[0]
    assert sample.tokens == [1, 2, 3, 4, 5]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.5, -0.4, -0.3]


def test_expand_masks_nonlast_version_tokens_when_proxy_flagged():
    seg = _segment(
        tokens=[10, 11, 12, 13, 14],
        full_loss_mask=[0, 1, 1, 0, 1],
        full_logprobs=[0.0, -0.1, -0.2, 0.0, -0.3],
        full_versions=["-1", "v0", "v1", "-1", "v1"],
        extra_info={"mask_nonlast_version_tokens": True},
    )

    sample = expand_segments_to_samples(
        SampleLike(),
        [seg],
        args=_args(),
    )[0]

    assert sample.response_length == 4
    assert sample.loss_mask == [0, 1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, -0.3]
    assert sample.metadata["full_versions"] == ["-1", "v0", "v1", "-1", "v1"]
    assert sample.metadata["dressage_start_token_version"] == "v0"
    assert sample.metadata["dressage_end_token_version"] == "v1"
    assert sample.metadata["dressage_partial_rollout"] is True


def test_expand_response_falls_back_to_agent_response_for_all_segments():
    segs = [
        _segment(segment_index=0, messages=[]),
        _segment(segment_index=1, messages=[]),
    ]
    result = expand_segments_to_samples(
        SampleLike(), segs, args=_args(), agent_response="FINAL"
    )
    assert [s.response for s in result] == ["FINAL", "FINAL"]


def test_expand_response_prefers_assistant_message():
    segs = [
        _segment(segment_index=0, messages=[{"role": "assistant", "content": "hi"}]),
    ]
    result = expand_segments_to_samples(SampleLike(), segs, args=_args(), agent_response="FALLBACK")
    assert result[0].response == "hi"


def test_expand_truncates_when_segment_exceeds_token_cap():
    seg = _segment(
        tokens=list(range(20)),
        full_loss_mask=[0] * 10 + [1] * 10,
        full_logprobs=[0.0] * 20,
    )
    args = _args(max_tokens_per_gpu=4, context_parallel_size=2)  # cap = 8
    result = expand_segments_to_samples(SampleLike(), [seg], args=args)[0]
    assert len(result.tokens) == 8
    assert result.metadata["truncated"] is True
    assert result.response_length == 0
    assert result.loss_mask == []


def test_expand_no_truncation_when_cap_disabled():
    seg = _segment(tokens=list(range(20)), full_loss_mask=[0] * 10 + [1] * 10, full_logprobs=[0.0] * 20)
    args = _args(max_tokens_per_gpu=None)
    result = expand_segments_to_samples(SampleLike(), [seg], args=args)[0]
    assert len(result.tokens) == 20
    assert "truncated" not in result.metadata


def test_expand_status_completed_when_finish_reason_stop():
    seg = _segment(finish_reason="stop")
    result = expand_segments_to_samples(SampleLike(), [seg], args=_args())[0]
    assert result.status == SampleLike.Status.COMPLETED


def test_expand_status_truncated_when_finish_reason_length():
    seg = _segment(finish_reason="length")
    result = expand_segments_to_samples(SampleLike(), [seg], args=_args())[0]
    assert result.status == SampleLike.Status.TRUNCATED


def test_expand_records_all_segment_uids():
    segs = [
        _segment(segment_index=0, uid="seg-a"),
        _segment(segment_index=1, uid="seg-b"),
        _segment(segment_index=2, uid=None),
    ]
    result = expand_segments_to_samples(SampleLike(), segs, args=_args())
    expected = ["seg-a", "seg-b"]
    for sample in result:
        assert sample.metadata["all_segment_uids"] == expected


def test_expand_explicit_session_and_instance_id_override_template():
    sample = SampleLike(session_id="from-template")
    sample.metadata["instance_id"] = "from-template"
    result = expand_segments_to_samples(
        sample,
        [_segment()],
        args=_args(),
        session_id="explicit-sess",
        instance_id="explicit-inst",
    )[0]
    assert result.metadata["session_id"] == "explicit-sess"
    assert result.metadata["instance_id"] == "explicit-inst"
    assert result.metadata["parent_traj_id"] == "explicit-sess"


def test_expand_parent_traj_id_equals_session_id_by_default():
    sample = SampleLike(session_id="my-session")
    result = expand_segments_to_samples(sample, [_segment()], args=_args())[0]
    assert result.metadata["parent_traj_id"] == "my-session"


@pytest.mark.parametrize("bad_mask", [
    [0, 0.5, 1],
    [0, "abc", 0],
    [0, 2, 1],
])
def test_expand_rejects_invalid_loss_mask_values(bad_mask):
    seg = _segment(full_loss_mask=bad_mask)
    with pytest.raises(ValueError, match="full_loss_mask"):
        expand_segments_to_samples(SampleLike(), [seg], args=_args())


def test_expand_rejects_length_mismatch_between_tokens_and_mask():
    seg = _segment(tokens=[1, 2, 3], full_loss_mask=[0, 1], full_logprobs=[0.0, -0.1, -0.2])
    with pytest.raises(ValueError, match="full_loss_mask length"):
        expand_segments_to_samples(SampleLike(), [seg], args=_args())


def test_expand_rejects_empty_tokens():
    seg = _segment(tokens=[], full_loss_mask=[], full_logprobs=[])
    with pytest.raises(ValueError, match="empty tokens"):
        expand_segments_to_samples(SampleLike(), [seg], args=_args())


def test_expand_rejects_missing_required_segment_field():
    bad = {"segment_index": 0, "tokens": [1], "full_loss_mask": [1]}
    with pytest.raises(ValueError, match="full_logprobs"):
        expand_segments_to_samples(SampleLike(), [bad], args=_args())


def test_expand_rejects_duplicate_segment_index():
    seg_a = _segment(segment_index=0, uid="seg-a")
    seg_b = _segment(segment_index=0, uid="seg-b")
    with pytest.raises(ValueError, match="duplicate segment_index"):
        expand_segments_to_samples(SampleLike(), [seg_a, seg_b], args=_args())


# ---------------------------------------------------------------------------
# compute_multi_segment_metrics
# ---------------------------------------------------------------------------


def _make_metric_sample(*, parent_traj_id=None, full_versions=None, remove_sample=False):
    meta: dict[str, Any] = {}
    if parent_traj_id is not None:
        meta["parent_traj_id"] = parent_traj_id
    if full_versions is not None:
        meta["full_versions"] = full_versions
    return SimpleNamespace(metadata=meta, remove_sample=remove_sample)


def test_compute_multi_segment_metrics_basic():
    samples = [
        _make_metric_sample(parent_traj_id="t1"),
        _make_metric_sample(parent_traj_id="t1"),
        _make_metric_sample(parent_traj_id="t1"),
        _make_metric_sample(parent_traj_id="t2"),
        _make_metric_sample(parent_traj_id="t2"),
        _make_metric_sample(parent_traj_id="t3"),
    ]
    metrics = compute_multi_segment_metrics(samples)
    assert metrics["rollout/num_trajectories"] == 3
    assert metrics["rollout/num_segments"] == 6
    assert metrics["rollout/segments_per_trajectory_mean"] == pytest.approx(2.0)
    assert metrics["rollout/segments_per_trajectory_max"] == 3
    assert metrics["rollout/segments_per_trajectory_min"] == 1


def test_compute_multi_segment_metrics_excludes_failures():
    samples = [
        _make_metric_sample(parent_traj_id="t1"),
        _make_metric_sample(parent_traj_id="t2", remove_sample=True),
        _make_metric_sample(parent_traj_id="t1"),
    ]
    metrics = compute_multi_segment_metrics(samples)
    assert metrics["rollout/num_trajectories"] == 1
    assert metrics["rollout/num_segments"] == 2


def test_compute_multi_segment_metrics_reports_version_span():
    samples = [
        _make_metric_sample(parent_traj_id="t1", full_versions=["-1", "1", "1"]),
        _make_metric_sample(parent_traj_id="t1", full_versions=["1", "2"]),
        _make_metric_sample(parent_traj_id="t2", full_versions=["-1", "3"]),
        _make_metric_sample(parent_traj_id="t3", full_versions=["4"], remove_sample=True),
    ]

    metrics = compute_multi_segment_metrics(samples)

    assert metrics["staleness/version_span_mean"] == pytest.approx(1.5)
    assert metrics["staleness/version_span_max"] == 2
    assert metrics["staleness/version_span_min"] == 1


def test_compute_multi_segment_metrics_empty():
    assert compute_multi_segment_metrics([]) == {}
    samples = [
        _make_metric_sample(parent_traj_id=None),
    ]
    assert compute_multi_segment_metrics(samples) == {}


# ---------------------------------------------------------------------------
# compute_prewarm_metrics
# ---------------------------------------------------------------------------


def _prewarm_sample(
    *,
    parent_traj_id=None,
    prewarm_requested=False,
    prewarm_hit=None,
    prewarm_wait_seconds=None,
    remove_sample=False,
):
    meta: dict[str, Any] = {}
    if parent_traj_id is not None:
        meta["parent_traj_id"] = parent_traj_id
    if prewarm_requested:
        meta["prewarm_requested"] = True
    if prewarm_hit is not None:
        meta["prewarm_hit"] = prewarm_hit
    if prewarm_wait_seconds is not None:
        meta["prewarm_wait_seconds"] = prewarm_wait_seconds
    return SimpleNamespace(metadata=meta, remove_sample=remove_sample)


def test_compute_prewarm_metrics_hit_rate_and_wait():
    samples = [
        # trajectory t1: prewarm hit, 3 segments share the same metadata
        _prewarm_sample(parent_traj_id="t1", prewarm_requested=True, prewarm_hit=1,
                        prewarm_wait_seconds=0.5),
        _prewarm_sample(parent_traj_id="t1", prewarm_requested=True, prewarm_hit=1,
                        prewarm_wait_seconds=0.5),
        _prewarm_sample(parent_traj_id="t1", prewarm_requested=True, prewarm_hit=1,
                        prewarm_wait_seconds=0.5),
        # trajectory t2: prewarm hit with a longer wait
        _prewarm_sample(parent_traj_id="t2", prewarm_requested=True, prewarm_hit=1,
                        prewarm_wait_seconds=1.5),
        # trajectory t3: prewarm miss (fell back to inline init)
        _prewarm_sample(parent_traj_id="t3", prewarm_requested=True, prewarm_hit=0),
    ]
    metrics = compute_prewarm_metrics(samples)
    assert metrics["prewarm/num_requested"] == 3
    assert metrics["prewarm/num_hits"] == 2
    assert metrics["prewarm/num_misses"] == 1
    assert metrics["prewarm/hit_rate"] == pytest.approx(2 / 3)
    assert metrics["prewarm/wait_seconds_mean"] == pytest.approx(1.0)
    assert metrics["prewarm/wait_seconds_max"] == pytest.approx(1.5)
    assert metrics["prewarm/wait_seconds_min"] == pytest.approx(0.5)


def test_compute_prewarm_metrics_ignores_non_requested_trajectories():
    """Rollout modes that never prewarm (e.g. sync) leave no prewarm_requested
    marker, so no prewarm metrics are emitted at all."""
    samples = [
        _prewarm_sample(parent_traj_id="t1"),
        _prewarm_sample(parent_traj_id="t2"),
    ]
    assert compute_prewarm_metrics(samples) == {}
    assert compute_prewarm_metrics([]) == {}


def test_compute_prewarm_metrics_counts_aborted_trajectories():
    """A prewarm hit reflects sandbox reuse and must count even when the
    rollout later aborted (remove_sample=True)."""
    samples = [
        _prewarm_sample(parent_traj_id="t1", prewarm_requested=True, prewarm_hit=1,
                        prewarm_wait_seconds=0.2, remove_sample=True),
        _prewarm_sample(parent_traj_id="t2", prewarm_requested=True, prewarm_hit=0),
    ]
    metrics = compute_prewarm_metrics(samples)
    assert metrics["prewarm/num_requested"] == 2
    assert metrics["prewarm/num_hits"] == 1
    assert metrics["prewarm/hit_rate"] == pytest.approx(0.5)


def test_compute_prewarm_metrics_all_miss_reports_zero_hit_rate():
    samples = [
        _prewarm_sample(parent_traj_id="t1", prewarm_requested=True, prewarm_hit=0),
        _prewarm_sample(parent_traj_id="t2", prewarm_requested=True, prewarm_hit=0),
    ]
    metrics = compute_prewarm_metrics(samples)
    assert metrics["prewarm/num_requested"] == 2
    assert metrics["prewarm/num_hits"] == 0
    assert metrics["prewarm/hit_rate"] == pytest.approx(0.0)
    # No hits → no wait-time distribution.
    assert "prewarm/wait_seconds_mean" not in metrics


# ---------------------------------------------------------------------------
# mark_aborted_no_grad
# ---------------------------------------------------------------------------


def test_mark_aborted_no_grad_sets_required_metadata():
    """A sample stamped by mark_aborted_no_grad must satisfy the contracts
    convert_samples / reward_post_process rely on (parent_traj_id,
    instance_id, remove_sample=True)."""
    sample = SampleLike(metadata={"existing_key": "preserved"})
    mark_aborted_no_grad(sample, session_id="sess-123", instance_id="inst-456")

    assert sample.metadata["parent_traj_id"] == "sess-123"
    assert sample.metadata["instance_id"] == "inst-456"
    assert sample.remove_sample is True
    assert sample.metadata["existing_key"] == "preserved"


def test_mark_aborted_no_grad_initializes_metadata_when_missing():
    sample = SampleLike()
    sample.metadata = None  # type: ignore[assignment]
    mark_aborted_no_grad(sample, session_id="sess-1", instance_id="inst-1")
    assert sample.metadata["parent_traj_id"] == "sess-1"
    assert sample.metadata["instance_id"] == "inst-1"
    assert sample.remove_sample is True


def test_mark_aborted_no_grad_falls_back_to_instance_id_when_no_session():
    sample = SampleLike(metadata={})
    mark_aborted_no_grad(sample, session_id=None, instance_id="inst-only")
    assert sample.metadata["parent_traj_id"] == "inst-only"
    assert sample.metadata["instance_id"] == "inst-only"


def test_mark_aborted_no_grad_synthesizes_id_when_both_missing():
    sample = SampleLike(metadata={})
    mark_aborted_no_grad(sample, session_id=None, instance_id=None)
    ptid = sample.metadata["parent_traj_id"]
    assert isinstance(ptid, str) and ptid.startswith("aborted-")
    assert sample.metadata["instance_id"] == ptid


def test_mark_aborted_no_grad_does_not_clobber_existing_ids():
    sample = SampleLike(
        metadata={"parent_traj_id": "real-traj", "instance_id": "real-inst"}
    )
    mark_aborted_no_grad(sample, session_id="other-sess", instance_id="other-inst")
    assert sample.metadata["parent_traj_id"] == "real-traj"
    assert sample.metadata["instance_id"] == "real-inst"


def test_mark_aborted_no_grad_clears_session_id_for_retry():
    """The retry path re-submits the same Sample to slime. slime's
    generate_and_rm_group only mints a fresh session UUID when
    sample.session_id is None; otherwise the retry reuses the dead session
    id. mark_aborted_no_grad is the single chokepoint that guarantees the
    cleanup."""
    sample = SampleLike(
        session_id="dead-sess-123",
        metadata={"session_id": "dead-sess-123", "other": "kept"},
    )
    mark_aborted_no_grad(sample, session_id="dead-sess-123", instance_id="inst-1")

    assert sample.session_id is None
    assert "session_id" not in sample.metadata
    assert sample.metadata["last_failed_session_id"] == "dead-sess-123"
    assert sample.metadata["other"] == "kept"


def test_mark_aborted_no_grad_prefers_explicit_session_id_for_audit():
    sample = SampleLike(
        session_id="stale-attr",
        metadata={"session_id": "stale-meta"},
    )
    mark_aborted_no_grad(sample, session_id="real-dead-sess", instance_id="inst-1")
    assert sample.metadata["last_failed_session_id"] == "real-dead-sess"
    assert sample.session_id is None


def test_mark_aborted_no_grad_no_session_to_record():
    sample = SampleLike(session_id=None, metadata={})
    mark_aborted_no_grad(sample, session_id=None, instance_id="inst-1")
    assert "last_failed_session_id" not in sample.metadata
    assert sample.session_id is None
