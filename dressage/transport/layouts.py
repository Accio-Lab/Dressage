from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .fields import (
    TQ_LOGPROBS_CODEC,
    TQ_ROUTED_EXPERTS_CODEC,
    TQFieldFragment,
    TQFieldLayout,
    TQFieldRef,
)


def _build_linear_layout(
    refs: Sequence[TQFieldRef],
    lengths: Sequence[int],
    *,
    logical_field: str,
    codec: str,
    token_count: int,
    target_start: int = 0,
    shape: tuple[int, ...] | None = None,
) -> TQFieldLayout:
    fragments: list[TQFieldFragment] = []
    offset = target_start
    for ref, length in zip(refs, lengths, strict=True):
        fragments.append(
            TQFieldFragment(
                ref=ref,
                source_start=0,
                target_start=offset,
                length=length,
            )
        )
        offset += length
    return TQFieldLayout(
        logical_field=logical_field,
        codec=codec,
        fragments=tuple(fragments),
        token_count=token_count,
        shape=shape,
    )


def normalize_logprobs_to_length(
    values: Sequence[float],
    token_count: int,
) -> tuple[list[float], bool]:
    invalid = len(values) != token_count
    normalized: list[float] = []
    for index in range(token_count):
        if index >= len(values):
            normalized.append(0.0)
            continue
        try:
            normalized.append(float(values[index]))
        except (TypeError, ValueError):
            normalized.append(0.0)
            invalid = True
    return normalized, invalid


def build_snapshot_logprobs(
    value: Sequence[float] | TQFieldRef,
    *,
    prompt_length: int,
    response_length: int,
    token_count: int,
    remote_invalid: bool,
) -> tuple[list[float] | TQFieldLayout, bool]:
    if prompt_length + response_length != token_count:
        raise RuntimeError("snapshot logprobs are not aligned with trajectory tokens")
    if isinstance(value, TQFieldRef):
        return (
            _build_linear_layout(
                (value,),
                (response_length,),
                logical_field="full_logprobs",
                codec=TQ_LOGPROBS_CODEC,
                token_count=token_count,
                target_start=prompt_length,
                shape=(token_count,),
            ),
            remote_invalid,
        )
    normalized, invalid = normalize_logprobs_to_length(value, response_length)
    return [0.0] * prompt_length + normalized, invalid


def build_concatenated_logprobs(
    values: Sequence[list[float] | TQFieldRef],
    lengths: Sequence[int],
    *,
    token_count: int,
) -> list[float] | TQFieldLayout:
    if sum(lengths) != token_count:
        raise RuntimeError("TITO logprobs are not aligned with trajectory tokens")
    refs = [value for value in values if isinstance(value, TQFieldRef)]
    if not refs:
        result = [item for value in values for item in value]
        if len(result) != token_count:
            raise RuntimeError("TITO logprobs are not aligned with trajectory tokens")
        return result
    if len(refs) != len(values):
        raise RuntimeError(
            "TITO logprobs cannot mix local values and TransferQueue refs"
        )
    return _build_linear_layout(
        refs,
        lengths,
        logical_field="full_logprobs",
        codec=TQ_LOGPROBS_CODEC,
        token_count=token_count,
        shape=(token_count,),
    )


def build_concatenated_routed_experts(
    values: Sequence[list[dict[str, Any]] | TQFieldRef],
    lengths: Sequence[int],
    *,
    token_count: int,
) -> list[dict[str, Any]] | TQFieldLayout | None:
    refs = [value for value in values if isinstance(value, TQFieldRef)]
    if not refs:
        chunks = [dict(chunk) for value in values for chunk in value]
        return chunks or None
    if len(refs) != len(values):
        raise RuntimeError(
            "routed experts cannot mix local values and TransferQueue refs"
        )
    expected_rows = max(0, token_count - 1)
    if sum(lengths) != expected_rows:
        raise RuntimeError(
            "routed experts layout is not aligned with trajectory tokens: "
            f"rows={sum(lengths)}, expected={expected_rows}"
        )
    return _build_linear_layout(
        refs,
        lengths,
        logical_field="routed_experts",
        codec=TQ_ROUTED_EXPERTS_CODEC,
        token_count=token_count,
        shape=(expected_rows,),
    )
