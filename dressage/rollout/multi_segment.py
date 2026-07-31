"""Multi-segment trajectory expansion utilities.

When ``--dressage-multi-segment`` is on, a single trajectory may produce
multiple proxy-finalized segments (boundary on history-rewrite / tools-change
/ prefix-mismatch). This module turns those segment dicts into training-ready
``Sample`` objects by reusing ``rollout.artifacts.samples.write_sample_from_segment``
for the per-segment Sample construction, then applying multi-segment-specific
fields (``parent_traj_id``, ``segment_index``, ``rollout_id``, reward routing).

The DP / micro-batch alignment is handled by slime's rollout-aware
``build_dp_schedule`` (slime v0.3.0+): every segment of one trajectory shares
``Sample.rollout_id`` so the scheduler keeps them in one training step.

Segment dict shape (matches proxy ``trajectory_payload["data"]`` entries)::

    {
        "uid":            str,
        "segment_index":  int,
        "tokens":         list[int],
        "full_loss_mask": list[int],   # 0/1 per token
        "full_logprobs":  list[float],
        "messages":       list[dict],
        "extra_info":     dict,
        "finish_reason":  str,
    }
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

_NON_REAL_TOKEN_VERSIONS = {"", "-1", "none", "unknown"}


def _real_token_version(value: Any) -> str | None:
    if value is None:
        return None
    version = str(value).strip()
    if version.lower() in _NON_REAL_TOKEN_VERSIONS:
        return None
    return version


def _metric_trajectory_id(sample: Any) -> str | None:
    meta = getattr(sample, "metadata", None) or {}
    value = meta.get("parent_traj_id") or meta.get("session_id")
    if value is not None:
        return str(value)

    rollout_id = getattr(sample, "rollout_id", None)
    if rollout_id is not None:
        return f"rollout:{rollout_id}"

    index = getattr(sample, "index", None)
    if index is not None:
        return f"sample:{index}"

    return None


def _append_ordered_unique(target: list[str], values: list[Any]) -> None:
    for value in values:
        version = _real_token_version(value)
        if version is not None and version not in target:
            target.append(version)


def mark_aborted_no_grad(
    sample: Any,
    *,
    session_id: str | None,
    instance_id: str | None,
) -> Any:
    """Stamp a rollout sample as failed-no-grad so downstream multi-segment
    training keeps it without crashing on missing metadata, AND clear its
    proxy session so a retry gets a fresh one.

    Contracts ``convert_samples`` / ``reward_post_process`` rely on:
      * ``metadata['parent_traj_id']`` is set.
      * ``metadata['instance_id']`` is set.
      * ``remove_sample = True`` so the loss reducer multiplies the sample
        by zero.

    Session contract: clear ``sample.session_id`` and ``metadata['session_id']``,
    preserve the dead id in ``metadata['last_failed_session_id']`` so retries
    get fresh sessions but audit logs can correlate.
    """
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    if metadata.get("parent_traj_id") is None:
        metadata["parent_traj_id"] = (
            session_id
            or instance_id
            or f"aborted-{id(sample):x}"
        )
    if metadata.get("instance_id") is None:
        metadata["instance_id"] = instance_id or metadata["parent_traj_id"]
    sample.remove_sample = True

    dead_session = session_id or metadata.get("session_id") or getattr(
        sample, "session_id", None
    )
    if dead_session is not None:
        metadata["last_failed_session_id"] = dead_session
    metadata.pop("session_id", None)
    if hasattr(sample, "session_id"):
        sample.session_id = None

    return sample


def expand_segments_to_samples(
    template_sample: Any,
    segments: list[dict],
    *,
    args: Any,
    agent_response: str = "",
    session_id: str | None = None,
    instance_id: str | None = None,
) -> list[Any]:
    """Emit one Sample per segment, all deep-copied from ``template_sample``.

    Reuses ``rollout.artifacts.samples.write_sample_from_segment`` for per-segment
    Sample construction (tokens, loss_mask, logprobs, metadata, status),
    then applies multi-segment-specific fields:

      * ``rollout_id``: all segments share ``template_sample.index`` so slime's
        ``build_dp_schedule`` keeps them in one training step.
      * ``parent_traj_id``: set to ``session_id`` for reward_post_process
        grouping and log_rollout_data trajectory-mean computation.
      * ``segment_index``: position within the trajectory.
      * ``reward``: only the last segment (anchor) keeps ``reward=None`` so
        slime runs reward_fn; earlier segments get ``reward=0.0``.
        ``reward_post_process`` broadcasts the anchor's reward back.

    Segments are sorted by ``segment_index`` ascending. Duplicate indices
    raise ``ValueError`` (would cause reward_post_process to pick wrong anchor).

    Returns:
      list[Sample] of length ``len(segments)``, sorted by segment_index.
    """
    from dressage.rollout.artifacts.samples import write_sample_from_segment

    if not segments:
        raise ValueError("expand_segments_to_samples: empty segments list")

    sid = session_id if session_id is not None else getattr(template_sample, "session_id", None)
    template_metadata = getattr(template_sample, "metadata", None)
    if instance_id is not None:
        iid = instance_id
    elif isinstance(template_metadata, dict):
        iid = template_metadata.get("instance_id")
    else:
        iid = None

    sorted_segments = sorted(segments, key=lambda s: int(s.get("segment_index", 0)))
    seg_indices = [int(s.get("segment_index", 0)) for s in sorted_segments]
    if len(set(seg_indices)) != len(seg_indices):
        raise ValueError(f"duplicate segment_index in segments: {seg_indices}")
    last_idx = len(sorted_segments) - 1

    rollout_id = getattr(template_sample, "index", None)

    out: list[Any] = []
    for i, segment in enumerate(sorted_segments):
        sample = copy.deepcopy(template_sample)
        sample.rollout_id = rollout_id

        write_sample_from_segment(
            sample,
            args=args,
            segment=segment,
            all_segments=sorted_segments,
            session_id=sid,
            instance_id=iid,
            agent_response=agent_response,
        )

        sample.metadata["parent_traj_id"] = sid
        sample.metadata["segment_index"] = int(segment.get("segment_index", 0))

        if i != last_idx:
            sample.reward = 0.0

        out.append(sample)
    return out


def compute_multi_segment_metrics(samples: list[Any]) -> dict[str, float]:
    """Per-rollout segment-count stats keyed under ``rollout/`` for wandb.

    Buckets samples by ``metadata['parent_traj_id']`` (= rollout session id)
    and reports the segment-count distribution across REAL trajectories.
    ``remove_sample=True`` failures are excluded.

    Called from the slime-facing rollout entrypoints (sync + fully_async)
    so the metrics ride through ``RolloutFnTrainOutput.metrics``.
    """
    counts: dict[str, int] = {}
    versions_by_traj: dict[str, list[str]] = {}
    for s in samples:
        if getattr(s, "remove_sample", False):
            continue
        meta = getattr(s, "metadata", None) or {}
        traj_id = meta.get("parent_traj_id")
        metric_traj_id = _metric_trajectory_id(s)

        if traj_id is None:
            pass
        else:
            counts[traj_id] = counts.get(traj_id, 0) + 1

        if metric_traj_id is not None and isinstance(meta.get("full_versions"), list):
            versions = versions_by_traj.setdefault(metric_traj_id, [])
            _append_ordered_unique(versions, meta["full_versions"])

    metrics: dict[str, float] = {}

    if counts:
        values = list(counts.values())
        n_traj = len(values)
        total = sum(values)
        metrics.update({
            "rollout/segments_per_trajectory_mean": total / n_traj,
            "rollout/segments_per_trajectory_max": float(max(values)),
            "rollout/segments_per_trajectory_min": float(min(values)),
            "rollout/num_trajectories": float(n_traj),
            "rollout/num_segments": float(total),
        })

    version_spans = [
        len(versions)
        for versions in versions_by_traj.values()
        if versions
    ]
    if version_spans:
        metrics.update({
            "staleness/version_span_mean": sum(version_spans) / len(version_spans),
            "staleness/version_span_max": float(max(version_spans)),
            "staleness/version_span_min": float(min(version_spans)),
        })

    return metrics


def compute_prewarm_metrics(samples: list[Any]) -> dict[str, float]:
    """Sandbox-prewarm hit-rate stats keyed under ``prewarm/`` for wandb.

    Only trajectories that actually requested a prewarm (stamped by the
    scheduler via ``metadata['prewarm_requested']``) are counted, so
    rollout modes that never prewarm (e.g. sync) emit no prewarm metrics at
    all instead of a misleading 0% hit rate.

    Buckets samples by trajectory (``metadata['parent_traj_id']``, falling
    back to ``_metric_trajectory_id``) so a multi-segment trajectory counts
    once. ``remove_sample=True`` (aborted/no-grad) samples are intentionally
    NOT excluded here: a prewarm *hit* reflects sandbox reuse and is
    independent of whether the rollout later succeeded.

    Called from the slime-facing rollout entrypoints (sync + fully_async +
    partial_async) so the metrics ride through ``RolloutFnTrainOutput.metrics``.
    """
    hits_by_traj: dict[str, bool] = {}
    waits: list[float] = []
    for s in samples:
        meta = getattr(s, "metadata", None) or {}
        if not meta.get("prewarm_requested"):
            continue
        traj_id = meta.get("parent_traj_id") or _metric_trajectory_id(s)
        if traj_id is None or traj_id in hits_by_traj:
            continue
        hit = int(meta.get("prewarm_hit", 0)) == 1
        hits_by_traj[traj_id] = hit
        if hit:
            wait = meta.get("prewarm_wait_seconds")
            if wait is not None:
                waits.append(float(wait))

    if not hits_by_traj:
        return {}

    num_requested = len(hits_by_traj)
    num_hits = sum(1 for hit in hits_by_traj.values() if hit)
    metrics: dict[str, float] = {
        "prewarm/num_requested": float(num_requested),
        "prewarm/num_hits": float(num_hits),
        "prewarm/num_misses": float(num_requested - num_hits),
        "prewarm/hit_rate": num_hits / num_requested,
    }
    if waits:
        metrics.update({
            "prewarm/wait_seconds_mean": sum(waits) / len(waits),
            "prewarm/wait_seconds_max": float(max(waits)),
            "prewarm/wait_seconds_min": float(min(waits)),
        })
    return metrics
