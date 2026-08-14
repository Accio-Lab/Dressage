"""Routed-expert payload normalization."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pybase64 as _base64
except ImportError:
    import base64 as _base64


def canonicalize_routed_experts(
    raw_chunks: list[dict[str, Any]],
    *,
    target_start: int,
    target_count: int,
    expert_id_dtype: str = "int32",
) -> list[dict[str, Any]]:
    """Keep only the routed-expert rows required by the recorded step."""

    if target_start < 0 or target_count < 0:
        raise ValueError("routed-expert target range must be non-negative")
    if expert_id_dtype not in {"uint8", "uint16", "int32"}:
        raise ValueError(f"unsupported routed-expert dtype: {expert_id_dtype}")
    if not raw_chunks or target_count == 0:
        return []

    storage_dtype = np.dtype(expert_id_dtype)
    target_end = target_start + target_count
    row_width: int | None = None
    emitted_rows = 0
    result: list[dict[str, Any]] = []

    for chunk in raw_chunks:
        prefix_count = int(chunk["prefix_token_count"])
        output_count = int(chunk["output_token_count"])
        raw_row_count = prefix_count + output_count - 1
        valid_start = 0 if chunk.get("is_first_chunk") else prefix_count - 1
        overlap_start = max(valid_start, target_start)
        overlap_end = min(raw_row_count, target_end)
        if overlap_start >= overlap_end:
            continue

        raw = _base64.b64decode(str(chunk["data"]).encode("ascii"))
        if raw_row_count <= 0 or len(raw) % raw_row_count != 0:
            raise ValueError(
                "routed-expert payload size does not match its expected row width"
            )
        current_row_width = len(raw) // raw_row_count
        if current_row_width == 0 or current_row_width % np.dtype(np.int32).itemsize:
            raise ValueError("routed-expert row width is not int32-aligned")
        if row_width is None:
            row_width = current_row_width
        elif current_row_width != row_width:
            raise ValueError("routed-expert chunks have inconsistent row widths")

        selected = raw[
            overlap_start * current_row_width : overlap_end * current_row_width
        ]
        selected_values = np.frombuffer(selected, dtype=np.int32)
        if storage_dtype != np.dtype(np.int32):
            dtype_info = np.iinfo(storage_dtype)
            if (
                selected_values.min() < dtype_info.min
                or selected_values.max() > dtype_info.max
            ):
                raise ValueError(
                    f"routed-expert id does not fit in {expert_id_dtype}"
                )
            selected = selected_values.astype(storage_dtype).tobytes()
        selected_rows = overlap_end - overlap_start
        result.append(
            {
                "data": _base64.b64encode(selected).decode("ascii"),
                "row_count": selected_rows,
                "dtype": expert_id_dtype,
            }
        )
        emitted_rows += selected_rows

    if emitted_rows != target_count:
        raise ValueError(
            "routed-expert chunks do not cover the requested target range: "
            f"expected={target_count}, actual={emitted_rows}"
        )
    return result
