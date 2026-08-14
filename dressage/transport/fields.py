"""Serializable references and layouts for field-level offload."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

TQ_FIELD_REF_SCHEMA = "dressage.tq.field-ref.v1"
TQ_FIELD_FRAGMENT_SCHEMA = "dressage.tq.field-fragment.v1"
TQ_FIELD_LAYOUT_SCHEMA = "dressage.tq.field-layout.v1"
TQ_TRAJECTORY_REF_SCHEMA = "dressage.tq.trajectory-ref.v1"
TQ_SAMPLE_REF_METADATA_KEY = "dressage_tq_sample_ref"

TQ_LOGPROBS_CODEC = "float32"
TQ_ROUTED_EXPERTS_CODEC = "routed_experts_chunks"

SUPPORTED_TRANSFER_PARAMS = ("logprobs", "routed_experts")

_STEP_FIELDS_BY_TRANSFER_PARAM = {
    "logprobs": (
        "prompt_token_logprobs",
        "response_logprobs",
        "all_logprobs",
        "concat_response_logprobs",
    ),
    "routed_experts": ("response_routed_experts_chunks",),
}


def normalize_transfer_params(
    values: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize user-facing transfer parameters in a stable order."""

    if values is None:
        return ()
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    requested = {
        item
        for value in values
        for item in str(value).replace(",", " ").split()
        if item
    }
    unsupported = requested.difference(SUPPORTED_TRANSFER_PARAMS)
    if unsupported:
        raise ValueError(
            "Unsupported TransferQueue parameters: "
            + ", ".join(sorted(unsupported))
        )
    return tuple(name for name in SUPPORTED_TRANSFER_PARAMS if name in requested)


def resolve_step_transfer_fields(
    transfer_params: str | Iterable[str] | None,
    *,
    token_build_mode: str,
) -> tuple[str, ...]:
    """Map public transfer parameters to physical StepRecord fields."""

    if token_build_mode not in {"snapshot", "tito"}:
        raise ValueError(f"Unsupported token build mode: {token_build_mode}")

    params = normalize_transfer_params(transfer_params)
    fields: list[str] = []
    if "logprobs" in params:
        fields.extend(_STEP_FIELDS_BY_TRANSFER_PARAM["logprobs"][:3])
        if token_build_mode == "tito":
            fields.append(_STEP_FIELDS_BY_TRANSFER_PARAM["logprobs"][3])
    if "routed_experts" in params:
        fields.extend(_STEP_FIELDS_BY_TRANSFER_PARAM["routed_experts"])
    return tuple(fields)


def _has_schema(value: object, schema: str) -> bool:
    return isinstance(value, Mapping) and value.get("schema_version") == schema


@dataclass(frozen=True)
class TQFieldRef:
    store_id: str
    partition: str
    key: str
    field: str
    incarnation_id: str
    schema_version: str = TQ_FIELD_REF_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "store_id": self.store_id,
            "partition": self.partition,
            "key": self.key,
            "field": self.field,
            "incarnation_id": self.incarnation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TQFieldRef:
        if not is_tq_field_ref_dict(value):
            raise ValueError("TransferQueue field reference schema does not match")
        return cls(
            store_id=str(value["store_id"]),
            partition=str(value["partition"]),
            key=str(value["key"]),
            field=str(value["field"]),
            incarnation_id=str(value["incarnation_id"]),
            schema_version=str(value["schema_version"]),
        )


def is_tq_field_ref_dict(value: object) -> bool:
    return _has_schema(value, TQ_FIELD_REF_SCHEMA)


@dataclass(frozen=True)
class TQFieldFragment:
    ref: TQFieldRef
    source_start: int
    target_start: int
    length: int
    schema_version: str = TQ_FIELD_FRAGMENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ref": self.ref.to_dict(),
            "source_start": self.source_start,
            "target_start": self.target_start,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TQFieldFragment:
        if not is_tq_field_fragment_dict(value):
            raise ValueError("TransferQueue field fragment schema does not match")
        return cls(
            ref=TQFieldRef.from_dict(value["ref"]),
            source_start=int(value["source_start"]),
            target_start=int(value["target_start"]),
            length=int(value["length"]),
            schema_version=str(value["schema_version"]),
        )


def is_tq_field_fragment_dict(value: object) -> bool:
    return _has_schema(value, TQ_FIELD_FRAGMENT_SCHEMA)


@dataclass(frozen=True)
class TQFieldLayout:
    logical_field: str
    codec: str
    fragments: tuple[TQFieldFragment, ...]
    token_count: int
    shape: tuple[int, ...] | None = None
    schema_version: str = TQ_FIELD_LAYOUT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_field": self.logical_field,
            "codec": self.codec,
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "token_count": self.token_count,
            "shape": None if self.shape is None else list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TQFieldLayout:
        if not is_tq_field_layout_dict(value):
            raise ValueError("TransferQueue field layout schema does not match")
        return cls(
            logical_field=str(value["logical_field"]),
            codec=str(value["codec"]),
            fragments=tuple(
                TQFieldFragment.from_dict(fragment)
                for fragment in value["fragments"]
            ),
            token_count=int(value["token_count"]),
            shape=(
                None
                if value.get("shape") is None
                else tuple(int(item) for item in value["shape"])
            ),
            schema_version=str(value["schema_version"]),
        )


def is_tq_field_layout_dict(value: object) -> bool:
    return _has_schema(value, TQ_FIELD_LAYOUT_SCHEMA)


@dataclass(frozen=True)
class TQTrajectoryRef:
    store_id: str
    trajectory_id: str
    incarnation_id: str
    refs: tuple[TQFieldRef, ...]
    schema_version: str = TQ_TRAJECTORY_REF_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "store_id": self.store_id,
            "trajectory_id": self.trajectory_id,
            "incarnation_id": self.incarnation_id,
            "refs": [ref.to_dict() for ref in self.refs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TQTrajectoryRef:
        if not is_tq_trajectory_ref_dict(value):
            raise ValueError("TransferQueue trajectory reference schema does not match")
        return cls(
            store_id=str(value["store_id"]),
            trajectory_id=str(value["trajectory_id"]),
            incarnation_id=str(value["incarnation_id"]),
            refs=tuple(TQFieldRef.from_dict(ref) for ref in value["refs"]),
            schema_version=str(value["schema_version"]),
        )


def is_tq_trajectory_ref_dict(value: object) -> bool:
    return _has_schema(value, TQ_TRAJECTORY_REF_SCHEMA)
