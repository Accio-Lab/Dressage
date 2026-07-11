"""Incremental tokenization helpers for TITO trajectory building."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tito_tokenizer import Qwen35TITOTokenizer


_TEMPLATE_FILES = {
    "qwen3_5": "qwen3_5_fixed.jinja",
}


def load_fixed_template(model_type: str) -> str:
    """Load the fixed chat template for a supported TITO model type."""

    try:
        filename = _TEMPLATE_FILES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported TITO model type: {model_type!r}") from exc

    template_path = Path(__file__).with_name("templates") / filename
    template = template_path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"TITO chat template is empty: {template_path}")
    return template


def create_tito_tokenizer(tokenizer: Any, *, model_type: str) -> Qwen35TITOTokenizer:
    """Create the TITO tokenizer for a supported model type."""

    if model_type != "qwen3_5":
        raise ValueError(f"Unsupported TITO model type: {model_type!r}")
    return Qwen35TITOTokenizer(tokenizer)


__all__ = [
    "Qwen35TITOTokenizer",
    "create_tito_tokenizer",
    "load_fixed_template",
]
