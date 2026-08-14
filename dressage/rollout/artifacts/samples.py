"""Rollout sample artifact construction helpers."""

from __future__ import annotations

import copy
import logging
import uuid
from typing import Any

from dressage.transport import (
    TQ_SAMPLE_REF_METADATA_KEY,
    TQFieldLayout,
    is_tq_field_layout_dict,
)

logger = logging.getLogger(__name__)


def _load_slime_sample():
    try:
        from slime.utils.types import Sample

        return Sample
    except ImportError:
        return None


def _status(sample: Any, name: str):
    sample_cls = _load_slime_sample()
    if sample_cls is not None:
        return getattr(sample_cls.Status, name)
    status_cls = getattr(sample, "Status", None)
    if status_cls is not None:
        return getattr(status_cls, name)
    return name.lower()


def set_status(sample: Any, name: str) -> None:
    sample.status = _status(sample, name)


def instance_id(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    value = metadata.get("instance_id")
    if value is not None:
        return str(value)
    group_index = getattr(sample, "group_index", None)
    return str(group_index if group_index is not None else uuid.uuid4())


def select_last_segment(data: list[dict[str, Any]]) -> dict[str, Any]:
    if not data:
        raise ValueError("proxy returned no trajectory segments")
    return sorted(
        data,
        key=lambda item: (
            int(item.get("segment_index", 0)),
            float(item.get("timestamp") or 0.0),
        ),
    )[-1]


def copy_sample_with_metadata(sample: Any, *, metadata: dict[str, Any]) -> Any:
    sample_copy = copy.copy(sample)
    sample_copy.metadata = dict(metadata)
    return sample_copy


def sample_artifact_payload(
    sample: Any,
    *,
    segment: dict[str, Any],
    all_segments: list[dict[str, Any]],
    session_id: str,
    instance_id: str,
) -> dict[str, Any]:
    segment_index = segment.get("segment_index", 0)
    metadata = getattr(sample, "metadata", None)
    return {
        "session_id": session_id,
        "trajectory_id": session_id,
        "instance_id": instance_id,
        "segment_index": segment_index,
        "segment_uid": segment.get("uid"),
        "segment_count": len(all_segments),
        "prompt": getattr(sample, "prompt", None),
        "label": getattr(sample, "label", None),
        "response": getattr(sample, "response", None),
        "tokens": getattr(sample, "tokens", None),
        "response_length": getattr(sample, "response_length", None),
        "loss_mask": getattr(sample, "loss_mask", None),
        "rollout_log_probs": getattr(sample, "rollout_log_probs", None),
        "reward": getattr(sample, "reward", None),
        "status": getattr(sample, "status", None),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def _last_assistant_content(messages: list[dict[str, Any]], fallback: str = "") -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            if content is not None:
                return str(content)
    return fallback


def _required_segment_list(segment: dict[str, Any], key: str) -> list[Any]:
    if key not in segment:
        raise ValueError(f"selected segment missing required field: {key}")
    value = segment[key]
    if value is None:
        raise ValueError(f"selected segment field is null: {key}")
    try:
        return list(value)
    except TypeError as exc:
        raise ValueError(f"selected segment field is not a list: {key}") from exc


def _normalize_segment_loss_mask(values: list[Any]) -> list[int]:
    normalized: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"full_loss_mask[{index}] is not 0 or 1: {value!r}")
        try:
            mask_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"full_loss_mask[{index}] cannot be converted to int: {value!r}"
            ) from exc
        if mask_value not in (0, 1):
            raise ValueError(f"full_loss_mask[{index}] is not 0 or 1: {value!r}")
        normalized.append(mask_value)
    return normalized


def _normalize_segment_logprobs(values: list[Any]) -> list[float]:
    normalized: list[float] = []
    for index, value in enumerate(values):
        try:
            normalized.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"full_logprobs[{index}] cannot be converted to float: {value!r}"
            ) from exc
    return normalized


def _normalize_segment_versions(values: list[Any]) -> list[str]:
    return ["unknown" if value is None else str(value) for value in values]


def _compress_version_spans(versions: list[str]) -> list[dict[str, Any]]:
    if not versions:
        return []

    spans: list[dict[str, Any]] = []
    start = 0
    current = versions[0]
    for index, version in enumerate(versions[1:], start=1):
        if version == current:
            continue
        spans.append({"start": start, "end": index, "version": current})
        start = index
        current = version
    spans.append({"start": start, "end": len(versions), "version": current})
    return spans


def _is_real_output_version(version: str) -> bool:
    return version.strip().lower() not in {"", "-1", "unknown", "none"}


def _trainable_output_versions(
    full_loss_mask: list[int],
    full_versions: list[str],
) -> list[str]:
    return [
        version
        for loss_mask, version in zip(full_loss_mask, full_versions)
        if int(loss_mask) == 1 and _is_real_output_version(version)
    ]


def _trainable_output_version_bounds(
    full_loss_mask: list[int],
    full_versions: list[str],
) -> tuple[str, str] | None:
    versions = _trainable_output_versions(full_loss_mask, full_versions)
    if not versions:
        return None
    return versions[0], versions[-1]


def _has_token_level_partial_rollout(
    full_loss_mask: list[int],
    full_versions: list[str],
) -> bool:
    return len(set(_trainable_output_versions(full_loss_mask, full_versions))) > 1


def _mask_nonlast_version_tokens(
    full_loss_mask: list[int],
    full_versions: list[str],
) -> list[int]:
    trainable_versions = _trainable_output_versions(full_loss_mask, full_versions)
    if len(set(trainable_versions)) <= 1:
        return list(full_loss_mask)

    last_version = trainable_versions[-1]
    return [
        1
        if int(mask_value) == 1
        and _is_real_output_version(version)
        and version == last_version
        else 0
        for mask_value, version in zip(full_loss_mask, full_versions)
    ]


def _segment_masks_nonlast_version_tokens(segment: dict[str, Any]) -> bool:
    extra_info = segment.get("extra_info") or {}
    return bool(extra_info.get("mask_nonlast_version_tokens"))


def _segment_arrays(
    segment: dict[str, Any],
) -> tuple[
    list[Any],
    list[int],
    list[float] | TQFieldLayout,
    list[str] | None,
]:
    tokens = _required_segment_list(segment, "tokens")
    full_loss_mask = _normalize_segment_loss_mask(
        _required_segment_list(segment, "full_loss_mask")
    )
    raw_logprobs = segment.get("full_logprobs")
    full_logprobs = (
        TQFieldLayout.from_dict(raw_logprobs)
        if is_tq_field_layout_dict(raw_logprobs)
        else _normalize_segment_logprobs(
            _required_segment_list(segment, "full_logprobs")
        )
    )
    raw_versions = segment.get("full_versions")
    full_versions = (
        None
        if raw_versions is None
        else _normalize_segment_versions(_required_segment_list(segment, "full_versions"))
    )
    if not tokens:
        raise ValueError("selected segment has empty tokens")
    if len(tokens) != len(full_loss_mask):
        raise ValueError(
            f"tokens length {len(tokens)} != full_loss_mask length {len(full_loss_mask)}"
        )
    if (
        isinstance(full_logprobs, TQFieldLayout)
        and len(tokens) != full_logprobs.token_count
    ):
        raise ValueError(
            "tokens length "
            f"{len(tokens)} != full_logprobs layout length "
            f"{full_logprobs.token_count}"
        )
    if isinstance(full_logprobs, list) and len(tokens) != len(full_logprobs):
        raise ValueError(
            "tokens length "
            f"{len(tokens)} != full_logprobs length {len(full_logprobs)}"
        )
    if full_versions is not None and len(tokens) != len(full_versions):
        raise ValueError(
            "tokens length "
            f"{len(tokens)} != full_versions length {len(full_versions)}"
        )
    return tokens, full_loss_mask, full_logprobs, full_versions


def _segment_token_cap(args: Any) -> int | None:
    max_tokens_per_gpu = getattr(args, "max_tokens_per_gpu", None)
    if max_tokens_per_gpu is None:
        return None
    cp_size = getattr(args, "context_parallel_size", None)
    if cp_size is None:
        cp_size = getattr(args, "cp_size", 1)
    return int(max_tokens_per_gpu) * int(cp_size)


def write_sample_from_segment(
    sample: Any,
    *,
    args: Any,
    segment: dict[str, Any],
    all_segments: list[dict[str, Any]],
    session_id: str,
    instance_id: str,
    agent_response: str,
) -> Any:
    tokens, full_loss_mask, full_logprobs, full_versions = _segment_arrays(segment)
    origin_tokens_len = len(tokens)
    token_cap = _segment_token_cap(args)
    truncated = token_cap is not None and origin_tokens_len > token_cap
    if truncated:
        tokens = tokens[:token_cap]
        full_loss_mask = full_loss_mask[:token_cap]
        if isinstance(full_logprobs, list):
            full_logprobs = full_logprobs[:token_cap]
        if full_versions is not None:
            full_versions = full_versions[:token_cap]
        logger.warning(
            "segment truncated for session_id=%s, instance_id=%s, segment_index=%s: %s > %s",
            session_id,
            instance_id,
            segment.get("segment_index", 0),
            origin_tokens_len,
            token_cap,
        )

    train_full_loss_mask = full_loss_mask
    if full_versions is not None and _segment_masks_nonlast_version_tokens(segment):
        train_full_loss_mask = _mask_nonlast_version_tokens(full_loss_mask, full_versions)

    response_start = next(
        (idx for idx, value in enumerate(full_loss_mask) if value == 1),
        len(tokens),
    )
    response_length = len(tokens) - response_start

    sample.tokens = tokens
    sample.response_length = response_length
    sample.loss_mask = train_full_loss_mask[response_start:]
    sample.rollout_log_probs = (
        None
        if isinstance(full_logprobs, TQFieldLayout)
        else full_logprobs[response_start:]
    )
    if len(sample.loss_mask) != response_length:
        raise ValueError(
            f"loss_mask length {len(sample.loss_mask)} != response_length {response_length}"
        )
    if (
        sample.rollout_log_probs is not None
        and len(sample.rollout_log_probs) != response_length
    ):
        raise ValueError(
            "rollout_log_probs length "
            f"{len(sample.rollout_log_probs)} != response_length {response_length}"
        )

    messages = segment.get("messages") or []
    sample.response = _last_assistant_content(messages, fallback=agent_response)
    sample.metadata["session_id"] = session_id
    sample.metadata["instance_id"] = instance_id
    sample.metadata["messages"] = messages
    sample.metadata["proxy_extra_info"] = segment.get("extra_info") or {}
    sample.metadata.pop("dressage_partial_rollout", None)
    sample.metadata.pop("dressage_async_group_id", None)
    sample.metadata.pop("response_versions", None)
    sample.metadata.pop("response_version_spans", None)
    sample.metadata.pop("dressage_start_token_version", None)
    sample.metadata.pop("dressage_end_token_version", None)
    sample.metadata.pop("full_versions", None)
    sample.metadata.pop("version_spans", None)
    if full_versions is not None:
        sample.metadata["full_versions"] = list(full_versions)
        sample.metadata["version_spans"] = _compress_version_spans(list(full_versions))
        version_bounds = _trainable_output_version_bounds(full_loss_mask, full_versions)
        if version_bounds is not None:
            start_token_version, end_token_version = version_bounds
            sample.metadata["dressage_start_token_version"] = start_token_version
            sample.metadata["dressage_end_token_version"] = end_token_version
        if _has_token_level_partial_rollout(full_loss_mask, full_versions):
            sample.metadata["dressage_partial_rollout"] = True
    sample.metadata["segment_count"] = len(all_segments)
    sample.metadata["selected_segment_index"] = segment.get("segment_index", 0)
    sample.metadata["all_segment_uids"] = [
        item.get("uid") for item in all_segments if item.get("uid") is not None
    ]
    if truncated:
        sample.metadata["truncated"] = True
    layouts: dict[str, dict[str, Any]] = {}
    if isinstance(full_logprobs, TQFieldLayout):
        layouts[full_logprobs.logical_field] = full_logprobs.to_dict()
    raw_routed_experts = segment.get("routed_experts_chunks")
    routed_experts_layout = (
        TQFieldLayout.from_dict(raw_routed_experts)
        if is_tq_field_layout_dict(raw_routed_experts)
        else None
    )
    if routed_experts_layout is not None:
        layouts[routed_experts_layout.logical_field] = (
            routed_experts_layout.to_dict()
        )
        routed_experts = None
    else:
        routed_experts = extract_routed_experts(
            segment,
            args,
            expected_token_count=len(tokens),
        )
    if layouts:
        sample.metadata[TQ_SAMPLE_REF_METADATA_KEY] = layouts
    if routed_experts is not None:
        sample.rollout_routed_experts = routed_experts
    elif (
        routed_experts_layout is None
        and getattr(args, "use_rollout_routing_replay", False)
    ):
        raise ValueError(
            "use_rollout_routing_replay is enabled but segment contains no routed_experts. "
            "Pass --use-rollout-routing-replay when starting the Dressage proxy."
        )

    finish_reason = str(segment.get("finish_reason") or "stop")
    set_status(sample, "TRUNCATED" if finish_reason == "length" else "COMPLETED")
    return sample


def extract_routed_experts(
    segment: dict[str, Any], args: Any, *, expected_token_count: int = 0,
) -> Any:
    num_layers = getattr(args, "num_layers", None)
    moe_router_topk = getattr(args, "moe_router_topk", None)
    if num_layers is None or moe_router_topk is None:
        return None

    import numpy as np

    try:
        import pybase64
    except ImportError:
        import base64 as pybase64

    def decode(chunk: dict[str, Any]) -> Any:
        dtype_name = str(chunk.get("dtype", "int32"))
        if dtype_name not in {"uint8", "uint16", "int32"}:
            raise ValueError(f"unsupported routed_experts dtype: {dtype_name}")
        return (
            np.frombuffer(
                pybase64.b64decode(str(chunk["data"]).encode("ascii")),
                dtype=np.dtype(dtype_name),
            )
            .astype(np.int32, copy=False)
            .reshape(-1, num_layers, moe_router_topk)
        )

    chunks_info = segment.get("routed_experts_chunks")
    if not chunks_info:
        return None

    arrays = []
    for chunk in chunks_info:
        decoded = decode(chunk)
        row_count = int(chunk["row_count"])
        if decoded.shape[0] != row_count:
            raise ValueError(
                "routed_experts chunk row count does not match its payload: "
                f"expected={row_count}, actual={decoded.shape[0]}"
            )
        arrays.append(decoded)
    result = np.concatenate(arrays, axis=0)
    if expected_token_count > 0:
        expected_rows = expected_token_count - 1
        if result.shape[0] < expected_rows:
            raise ValueError(
                "routed_experts length does not match trajectory tokens: "
                f"expected_at_least={expected_rows}, actual={result.shape[0]}"
            )
        result = result[:expected_rows]
    return result
