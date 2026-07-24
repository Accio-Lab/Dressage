"""Configuration policy for sandbox prewarming."""

from __future__ import annotations

import os


DEFAULT_PREWARM_AHEAD = 8


def prewarm_enabled() -> bool:
    """Return whether sandbox prewarming is enabled for this provider."""
    value = os.environ.get("DRESSAGE_SANDBOX_PREWARM")
    if value is None or not value.strip():
        provider = os.environ.get("DRESSAGE_SANDBOX_PROVIDER", "").strip().lower()
        return provider == "e2b"
    normalized = value.strip()
    if normalized == "1":
        return True
    if normalized == "0":
        return False
    raise ValueError(
        f"DRESSAGE_SANDBOX_PREWARM must be 0 or 1, got {value!r}"
    )


def prewarm_ahead() -> int:
    """Return the configured number of groups to prefetch."""
    return _positive_int_env(
        "DRESSAGE_SANDBOX_PREWARM_AHEAD",
        DEFAULT_PREWARM_AHEAD,
    )


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed
