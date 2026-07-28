"""Explicit ownership APIs for sandbox prewarming."""

from dressage.rollout.prewarm.config import prewarm_enabled
from dressage.rollout.prewarm.store import (
    PrewarmHandle,
    claim_prewarm,
    cleanup_all,
    cleanup_group,
    ensure_blackbox_session_id,
    start_prewarm,
)

__all__ = [
    "PrewarmHandle",
    "claim_prewarm",
    "cleanup_all",
    "cleanup_group",
    "ensure_blackbox_session_id",
    "prewarm_enabled",
    "start_prewarm",
]
