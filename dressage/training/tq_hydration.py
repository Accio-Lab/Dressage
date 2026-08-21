"""Materialize TransferQueue field layouts into Slime rollout tensors."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from dressage.transport.fields import (
    TQ_LOGPROBS_CODEC,
    TQ_ROUTED_EXPERTS_CODEC,
    TQFieldLayout,
)

FULL_LOGPROBS_FIELD = "full_logprobs"
ROUTED_EXPERTS_FIELD = "routed_experts"

LayoutBundle = Mapping[str, dict[str, Any]]
LayoutValueKey = tuple[str, str, str, str]
BatchGet = Callable[..., Any]

logger = logging.getLogger(__name__)


def validate_tq_training_config(args: Any) -> None:
    if bool(getattr(args, "debug_rollout_only", False)):
        raise ValueError("TransferQueue does not support --debug-rollout-only")
    config_path = getattr(args, "mopd_teacher_config", None) or os.environ.get(
        "DRESSAGE_MOPD_TEACHER_CONFIG"
    )
    if config_path:
        raise ValueError("TransferQueue and MOPD cannot share train_data['prompt']")


def decode_layout_bundle(value: LayoutBundle | None) -> dict[str, TQFieldLayout]:
    if value is None:
        return {}
    layouts = {
        logical_field: TQFieldLayout.from_dict(layout)
        for logical_field, layout in value.items()
    }
    for logical_field, layout in layouts.items():
        if layout.logical_field != logical_field:
            raise ValueError("TransferQueue layout logical field does not match")
    return layouts


def remote_fields_from_layouts(
    values: Sequence[LayoutBundle | None],
) -> set[str]:
    return {
        logical_field
        for value in values
        for logical_field in decode_layout_bundle(value)
    }


def clear_requests_from_layouts(
    values: Sequence[LayoutBundle | None],
    logical_fields: set[str] | None = None,
) -> dict[str, list[str]]:
    requests: dict[str, list[str]] = {}
    for value in values:
        for logical_field, layout in decode_layout_bundle(value).items():
            if logical_fields is not None and logical_field not in logical_fields:
                continue
            for fragment in layout.fragments:
                keys = requests.setdefault(fragment.ref.partition, [])
                if fragment.ref.key not in keys:
                    keys.append(fragment.ref.key)
    return requests


def clear_tq_rollout_data(rollout_data_ref: Sequence[Any]) -> None:
    import ray
    import transfer_queue as tq

    requests: dict[str, set[str]] = {}
    for boxed_ref in rollout_data_ref:
        layouts = ray.get(boxed_ref.inner).get("prompt") or []
        for partition, keys in clear_requests_from_layouts(layouts).items():
            requests.setdefault(partition, set()).update(keys)
    if not requests:
        return

    try:
        tq.init()
        for partition, keys in requests.items():
            tq.kv_clear(keys=list(keys), partition_id=partition)
    except Exception:
        logger.exception(
            "TransferQueue cleanup failed; retention will reclaim the data"
        )


def _value_key(ref: Any) -> LayoutValueKey:
    return (ref.store_id, ref.partition, ref.key, ref.field)


def _column_item(column: Any, index: int) -> Any:
    item = column[index]
    return getattr(item, "data", item)


def read_layout_values(
    layouts: Sequence[dict[str, TQFieldLayout]],
    logical_fields: set[str],
    batch_get: BatchGet,
) -> dict[LayoutValueKey, Any]:
    requests: dict[tuple[str, str, str], list[str]] = {}
    for bundle in layouts:
        for logical_field in logical_fields:
            layout = bundle.get(logical_field)
            if layout is None:
                continue
            for fragment in layout.fragments:
                ref = fragment.ref
                request_key = (ref.store_id, ref.partition, ref.field)
                keys = requests.setdefault(request_key, [])
                if ref.key not in keys:
                    keys.append(ref.key)

    values: dict[LayoutValueKey, Any] = {}
    for (store_id, partition, field), keys in requests.items():
        data = batch_get(
            keys=keys,
            partition_id=partition,
            select_fields=field,
        )
        column = data[field]
        try:
            actual_size = int(data.batch_size[0])
        except (AttributeError, IndexError, TypeError):
            actual_size = len(column)
        if actual_size != len(keys):
            raise RuntimeError(
                "TransferQueue returned an unexpected field value count"
            )
        for index, key in enumerate(keys):
            values[(store_id, partition, key, field)] = _column_item(column, index)
    return values


def _coerce_float32_tensor(source: Any) -> torch.Tensor:
    if isinstance(source, np.ndarray):
        if source.dtype == np.float32:
            return torch.from_numpy(source)
        if source.dtype != object:
            return torch.from_numpy(source.astype(np.float32, copy=False))
    else:
        try:
            return torch.as_tensor(source, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError):
            pass
    normalized = []
    for value in source:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            normalized.append(0.0)
    return torch.tensor(normalized, dtype=torch.float32)


def _materialize_logprobs(
    layout: TQFieldLayout,
    values: Mapping[LayoutValueKey, Any],
    token_count: int,
) -> torch.Tensor:
    if layout.codec != TQ_LOGPROBS_CODEC:
        raise ValueError("TransferQueue logprob layout codec does not match")
    if layout.token_count < token_count:
        raise ValueError("TransferQueue logprob layout is shorter than the sample")

    output = torch.zeros(token_count, dtype=torch.float32)
    for fragment in layout.fragments:
        source_start = int(fragment.source_start)
        target_start = int(fragment.target_start)
        length = int(fragment.length)
        copy_length = min(length, max(0, token_count - target_start))
        if copy_length <= 0:
            continue
        source = _coerce_float32_tensor(
            values[_value_key(fragment.ref)][
                source_start : source_start + copy_length
            ]
        )
        if source.numel() > 0:
            output[target_start : target_start + source.numel()] = source
    return output


def _decode_routed_experts_chunks(value: Any, width: int) -> torch.Tensor:
    arrays = []
    for chunk in value or []:
        row_count = int(chunk["row_count"])
        dtype_name = str(chunk.get("dtype", "int32"))
        if dtype_name not in {"uint8", "uint16", "int32"}:
            raise ValueError(f"unsupported routed experts dtype: {dtype_name}")
        # chunk["data"] is a numpy array stored zero-copy by TransferQueue
        # (previously a base64 string); reinterpret to int32 for replay.
        decoded = np.asarray(chunk["data"]).astype(np.int32, copy=False)
        if decoded.size != row_count * width:
            raise ValueError("TransferQueue routed experts chunk shape does not match")
        arrays.append(decoded.reshape(row_count, width).copy())
    if not arrays:
        return torch.empty((0, width), dtype=torch.int32)
    return torch.from_numpy(np.concatenate(arrays, axis=0))


def _materialize_routed_experts(
    layout: TQFieldLayout,
    values: Mapping[LayoutValueKey, Any],
    token_count: int,
    num_layers: int,
    topk: int,
) -> torch.Tensor:
    if layout.codec != TQ_ROUTED_EXPERTS_CODEC:
        raise ValueError("TransferQueue routed experts layout codec does not match")
    if layout.token_count < token_count:
        raise ValueError("TransferQueue routed experts layout is shorter than the sample")

    width = num_layers * topk
    target_rows = max(0, token_count - 1)
    output = torch.zeros((target_rows, width), dtype=torch.int32)
    for fragment in layout.fragments:
        source = _decode_routed_experts_chunks(
            values[_value_key(fragment.ref)],
            width,
        )
        source_start = int(fragment.source_start)
        target_start = int(fragment.target_start)
        length = int(fragment.length)
        if source_start + length > source.shape[0]:
            raise ValueError(
                "TransferQueue routed experts fragment is shorter than its layout"
            )
        copy_length = min(length, max(0, target_rows - target_start))
        if copy_length:
            output[target_start : target_start + copy_length] = source[
                source_start : source_start + copy_length
            ]
    return output.reshape(target_rows, num_layers, topk)


def hydrate_training_layouts(
    args: Any,
    rollout_data: dict[str, Any],
    raw_layouts: Sequence[LayoutBundle | None],
    *,
    remote_fields: set[str],
    batch_get: BatchGet,
    logprob_transform: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
) -> None:
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    if len(raw_layouts) != len(total_lengths):
        raise ValueError("TransferQueue layout count does not match the DP shard")

    layouts = [decode_layout_bundle(value) for value in raw_layouts]
    fields_to_read = set(remote_fields)
    if not getattr(args, "use_rollout_routing_replay", False):
        fields_to_read.discard(ROUTED_EXPERTS_FIELD)
    values = read_layout_values(layouts, fields_to_read, batch_get)

    if FULL_LOGPROBS_FIELD in fields_to_read:
        rollout_logprobs = []
        for layout_bundle, total_length, response_length in zip(
            layouts,
            total_lengths,
            response_lengths,
            strict=True,
        ):
            total_length = int(total_length)
            response_length = int(response_length)
            layout = layout_bundle.get(FULL_LOGPROBS_FIELD)
            if layout is None:
                response = torch.zeros(response_length, dtype=torch.float32)
            else:
                full = _materialize_logprobs(layout, values, total_length)
                response_start = total_length - response_length
                response = full[response_start:total_length]
            if logprob_transform is not None:
                response = logprob_transform(
                    response,
                    total_length,
                    response_length,
                )
            rollout_logprobs.append(response)
        rollout_data["rollout_log_probs"] = rollout_logprobs

    if ROUTED_EXPERTS_FIELD in fields_to_read:
        num_layers = int(args.num_layers)
        topk = int(args.moe_router_topk)
        rollout_data["rollout_routed_experts"] = [
            (
                torch.zeros(
                    (max(0, int(total_length) - 1), num_layers, topk),
                    dtype=torch.int32,
                )
                if layout_bundle.get(ROUTED_EXPERTS_FIELD) is None
                else _materialize_routed_experts(
                    layout_bundle[ROUTED_EXPERTS_FIELD],
                    values,
                    int(total_length),
                    num_layers,
                    topk,
                )
            )
            for layout_bundle, total_length in zip(
                layouts,
                total_lengths,
                strict=True,
            )
        ]
