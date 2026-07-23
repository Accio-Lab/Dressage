"""Group prefetch policy for sandbox prewarming.

The scheduler decides which groups to prefetch and asks the prewarm store to
start sandbox initialization.  It never owns initialization tasks directly;
the store owns every unclaimed prewarm until dispatch claims it.
"""

from __future__ import annotations

import logging
from typing import Any

from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_BLACKBOX_TYPE,
    normalize_blackbox_type,
)
from dressage.rollout.generate.runtime import (
    get_paddock_from_env,
    paddock_env_args_from_metadata,
)
from dressage.rollout.prewarm.config import (
    prewarm_ahead,
    prewarm_enabled,
)
from dressage.rollout.prewarm.store import (
    cleanup_all,
    cleanup_group as cleanup_prewarm_group,
    start_prewarm,
)

logger = logging.getLogger(__name__)


class PrewarmScheduler:
    """Prefetch groups and initiate prewarms without owning their tasks."""

    def __init__(self) -> None:
        self.enabled = prewarm_enabled()
        self.ahead = prewarm_ahead() if self.enabled else 0
        self._prefetched_groups: list[tuple[int, list[Any]]] = []
        self._next_group_id = 0

    def allocate_group_id(self) -> int:
        """Allocate the next worker-local group ID."""
        group_id = self._next_group_id
        self._next_group_id += 1
        return group_id

    def do_prefetch(self, data_buffer: Any) -> None:
        """Fill the prefetched group queue up to the configured lookahead."""
        if not self.enabled:
            return
        while len(self._prefetched_groups) < self.ahead:
            groups = data_buffer.get_samples(1)
            if not groups:
                break
            group = groups[0]
            group_id = self.allocate_group_id()
            self._prefetched_groups.append((group_id, group))
            try:
                started = self._prefetch_and_prewarm(
                    group,
                    group_id,
                )
            except Exception:
                logger.warning(
                    "prewarm failed for group %d; dispatching without full prewarm",
                    group_id,
                    exc_info=True,
                )
                continue
            if started:
                logger.debug(
                    "prewarm started for %d/%d samples in group %d",
                    started,
                    len(group),
                    group_id,
                )

    def _prefetch_and_prewarm(
        self,
        group: list[Any],
        group_id: int,
    ) -> int:
        try:
            paddock = get_paddock_from_env(allow_whitebox_mode=False)
        except Exception:
            logger.debug(
                "prewarm skipped for group %d: paddock unavailable",
                group_id,
                exc_info=True,
            )
            return 0

        started = 0
        for sample in group:
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                sample.metadata = metadata
            blackbox_type = normalize_blackbox_type(
                metadata.get("blackbox_type") or DEFAULT_BLACKBOX_TYPE
            )
            extra_env_args = None
            if "blackbox_type" in metadata or blackbox_type != DEFAULT_BLACKBOX_TYPE:
                extra_env_args = {"blackbox_type": blackbox_type}
            env_args = paddock_env_args_from_metadata(
                metadata,
                extra_env_args=extra_env_args,
            )
            if start_prewarm(
                sample,
                group_id=group_id,
                paddock=paddock,
                env_args=env_args,
            ) is not None:
                started += 1
        return started

    def pop_next_group(self, data_buffer: Any) -> tuple[int, list[Any]] | None:
        """Return a prefetched group first, then fall back to fresh input."""
        if self._prefetched_groups:
            return self._prefetched_groups.pop(0)
        groups = data_buffer.get_samples(1)
        if not groups:
            return None
        return self.allocate_group_id(), groups[0]

    async def cleanup_group(self, group_id: int) -> None:
        """Release this group's prewarms that dispatch never claimed."""
        await cleanup_prewarm_group(group_id)

    async def cleanup(self) -> None:
        """Release every unclaimed prewarm at worker shutdown."""
        self._prefetched_groups.clear()
        await cleanup_all()
