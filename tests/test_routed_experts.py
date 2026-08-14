from __future__ import annotations

import base64
from array import array

import numpy as np
import pytest

from dressage.proxy.routed_experts import canonicalize_routed_experts


def _encode(values: list[int]) -> str:
    encoded = array("i", values)
    assert encoded.itemsize == 4
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode(chunks: list[dict]) -> list[int]:
    values: list[int] = []
    for chunk in chunks:
        decoded = np.frombuffer(
            base64.b64decode(chunk["data"]),
            dtype=np.dtype(chunk.get("dtype", "int32")),
        )
        values.extend(decoded.tolist())
    return values


def _raw_partial_chunks() -> list[dict]:
    return [
        {
            "data": _encode([10, 11, 12, 13]),
            "prefix_token_count": 3,
            "output_token_count": 2,
            "is_first_chunk": True,
        },
        {
            "data": _encode([20, 21, 22, 23, 24, 25]),
            "prefix_token_count": 5,
            "output_token_count": 2,
            "is_first_chunk": False,
        },
    ]


def test_canonicalize_routed_experts_removes_partial_prefix() -> None:
    chunks = canonicalize_routed_experts(
        _raw_partial_chunks(),
        target_start=0,
        target_count=6,
    )

    assert [chunk["row_count"] for chunk in chunks] == [4, 2]
    assert _decode(chunks) == [10, 11, 12, 13, 24, 25]


def test_canonicalize_routed_experts_intersects_step_delta() -> None:
    chunks = canonicalize_routed_experts(
        _raw_partial_chunks(),
        target_start=3,
        target_count=3,
    )

    assert [chunk["row_count"] for chunk in chunks] == [1, 2]
    assert _decode(chunks) == [13, 24, 25]


def test_canonicalize_routed_experts_preserves_complete_rows() -> None:
    chunks = canonicalize_routed_experts(
        [
            {
                "data": _encode([10, 110, 11, 111, 12, 112, 13, 113]),
                "prefix_token_count": 3,
                "output_token_count": 2,
                "is_first_chunk": True,
            },
            {
                "data": _encode(
                    [20, 120, 21, 121, 22, 122, 23, 123, 24, 124, 25, 125]
                ),
                "prefix_token_count": 5,
                "output_token_count": 2,
                "is_first_chunk": False,
            },
        ],
        target_start=3,
        target_count=3,
    )

    assert [chunk["row_count"] for chunk in chunks] == [1, 2]
    assert _decode(chunks) == [13, 113, 24, 124, 25, 125]


@pytest.mark.parametrize(
    ("expert_id_dtype", "expected_bytes"),
    [("uint8", 6), ("uint16", 12), ("int32", 24)],
)
def test_canonicalize_routed_experts_converts_expert_id_dtype(
    expert_id_dtype: str,
    expected_bytes: int,
) -> None:
    chunks = canonicalize_routed_experts(
        [
            {
                "data": _encode([1, 2, 3, 4, 5, 6]),
                "prefix_token_count": 5,
                "output_token_count": 2,
                "is_first_chunk": True,
            }
        ],
        target_start=0,
        target_count=6,
        expert_id_dtype=expert_id_dtype,
    )

    assert chunks[0]["dtype"] == expert_id_dtype
    assert len(base64.b64decode(chunks[0]["data"])) == expected_bytes
    assert _decode(chunks) == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    ("expert_id_dtype", "expert_id"),
    [("uint8", -1), ("uint8", 256), ("uint16", 65536)],
)
def test_canonicalize_routed_experts_rejects_dtype_overflow(
    expert_id_dtype: str,
    expert_id: int,
) -> None:
    chunks = [
        {
            "data": _encode([expert_id]),
            "prefix_token_count": 1,
            "output_token_count": 1,
            "is_first_chunk": True,
        }
    ]

    with pytest.raises(ValueError, match=f"does not fit in {expert_id_dtype}"):
        canonicalize_routed_experts(
            chunks,
            target_start=0,
            target_count=1,
            expert_id_dtype=expert_id_dtype,
        )


def test_canonicalize_routed_experts_rejects_invalid_row_width() -> None:
    chunks = [
        {
            "data": base64.b64encode(b"abc").decode("ascii"),
            "prefix_token_count": 2,
            "output_token_count": 1,
            "is_first_chunk": True,
        }
    ]

    with pytest.raises(ValueError, match="row width"):
        canonicalize_routed_experts(chunks, target_start=0, target_count=2)
