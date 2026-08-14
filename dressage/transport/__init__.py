"""Field-level TransferQueue integration primitives."""

from .fields import (
    SUPPORTED_TRANSFER_PARAMS,
    TQ_FIELD_FRAGMENT_SCHEMA,
    TQ_FIELD_LAYOUT_SCHEMA,
    TQ_FIELD_REF_SCHEMA,
    TQ_LOGPROBS_CODEC,
    TQ_ROUTED_EXPERTS_CODEC,
    TQ_SAMPLE_REF_METADATA_KEY,
    TQ_TRAJECTORY_REF_SCHEMA,
    TQFieldFragment,
    TQFieldLayout,
    TQFieldRef,
    TQTrajectoryRef,
    is_tq_field_fragment_dict,
    is_tq_field_layout_dict,
    is_tq_field_ref_dict,
    is_tq_trajectory_ref_dict,
    normalize_transfer_params,
    resolve_step_transfer_fields,
)
from .layouts import (
    build_concatenated_logprobs,
    build_concatenated_routed_experts,
    build_snapshot_logprobs,
    normalize_logprobs_to_length,
)
from .runtime import TransferQueueRuntime
from .store import TransferQueueStore

__all__ = [
    "SUPPORTED_TRANSFER_PARAMS",
    "TQ_FIELD_FRAGMENT_SCHEMA",
    "TQ_FIELD_LAYOUT_SCHEMA",
    "TQ_FIELD_REF_SCHEMA",
    "TQ_LOGPROBS_CODEC",
    "TQ_ROUTED_EXPERTS_CODEC",
    "TQ_SAMPLE_REF_METADATA_KEY",
    "TQ_TRAJECTORY_REF_SCHEMA",
    "TQFieldFragment",
    "TQFieldLayout",
    "TQFieldRef",
    "TQTrajectoryRef",
    "TransferQueueRuntime",
    "TransferQueueStore",
    "build_concatenated_logprobs",
    "build_concatenated_routed_experts",
    "build_snapshot_logprobs",
    "is_tq_field_fragment_dict",
    "is_tq_field_layout_dict",
    "is_tq_field_ref_dict",
    "is_tq_trajectory_ref_dict",
    "normalize_logprobs_to_length",
    "normalize_transfer_params",
    "resolve_step_transfer_fields",
]
