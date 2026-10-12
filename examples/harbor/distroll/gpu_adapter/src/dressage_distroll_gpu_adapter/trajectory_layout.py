"""Validate native TQ references without materializing tensor values."""

import logging

from dressage.transport.fields import (
    TQFieldLayout,
    TQTrajectoryRef,
    is_tq_field_layout_dict,
    resolve_step_transfer_fields,
)

logger = logging.getLogger(__name__)


def inspect_trajectory_layouts(segments, finalized, *, trial_id, instance_id):
    requests, layouts = {}, {}
    trajectory = None
    for index, segment in enumerate(segments):
        extra = segment.get("extra_info") or {}
        invalid_flags = [
            flag
            for flag in (
                "snapshot_logprobs_invalid",
                "tito_logprobs_invalid",
                "tito_incremental_tokenization_failed",
            )
            if extra.get(flag)
        ]
        if invalid_flags:
            logger.error(
                "trajectory_invalid trial=%s instance=%s segment=%d flags=%s "
                "token_count=%d step_count=%s",
                trial_id,
                instance_id,
                index,
                invalid_flags,
                len(segment.get("tokens") or []),
                extra.get("num_steps"),
            )
            raise ValueError(
                f"segment {index}: Proxy marked token/logprobs invalid ({','.join(invalid_flags)})"
            )
        if is_tq_field_layout_dict(segment.get("routed_experts_chunks")):
            raise ValueError(
                "DistRoll TQ materialization currently supports logprobs only"
            )
        raw = segment.get("full_logprobs")
        if not is_tq_field_layout_dict(raw):
            continue  # Existing array path is checked by validate_attempt().
        if trajectory is None:
            if (
                finalized.get("success") is not True
                or finalized.get("trajectory_id") != trial_id
                or finalized.get("instance_id") != instance_id
            ):
                raise ValueError("TQ finalization identity mismatch")
            trajectory = TQTrajectoryRef.from_dict(finalized["tq_trajectory_ref"])
            if trajectory.trajectory_id != trial_id:
                raise ValueError("TQ trajectory identity mismatch")
        if extra.get("finalization_id") != finalized.get("finalization_id"):
            raise ValueError("TQ segment belongs to another finalization")
        # Reject coercible floats/bools before the upstream dataclass casts.
        counts = [raw["token_count"]] + [
            fragment[name]
            for fragment in raw["fragments"]
            for name in ("source_start", "target_start", "length")
        ]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("TQ layout offsets must be nonnegative integers")
        layout = TQFieldLayout.from_dict(raw)
        if (
            layout.logical_field != "full_logprobs"
            or layout.codec != "float32"
            or layout.token_count != len(segment["tokens"])
            or layout.shape not in (None, (layout.token_count,))
        ):
            raise ValueError("TQ logprob layout type/length mismatch")
        allowed = {
            (ref.store_id, ref.incarnation_id, ref.partition, ref.key)
            for ref in trajectory.refs
        }
        fields = resolve_step_transfer_fields(
            "logprobs", token_build_mode=extra.get("token_build_mode")
        )
        for fragment in layout.fragments:
            ref = fragment.ref
            if (
                ref.store_id != trajectory.store_id
                or ref.incarnation_id != trajectory.incarnation_id
                or (ref.store_id, ref.incarnation_id, ref.partition, ref.key)
                not in allowed
                or ref.field not in fields
            ):
                raise ValueError("TQ fragment identity/field mismatch")
            keys = requests.setdefault((ref.partition, ref.field), [])
            if ref.key not in keys:
                keys.append(ref.key)
        layouts[index] = layout
    return requests, layouts
