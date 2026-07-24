"""Store for unclaimed sandbox prewarms and explicit ownership handoff."""

from __future__ import annotations

import asyncio
import copy
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from dressage.paddock.lifecycle import (
    exception_summary,
    maybe_await,
    terminate_paddock_best_effort,
)

logger = logging.getLogger(__name__)


def ensure_blackbox_session_id(sample: Any) -> str:
    """Ensure *sample* has a ``bbs-`` prefixed session ID and return it."""
    session_id = getattr(sample, "session_id", None)
    if session_id is None:
        session_id = str(uuid.uuid4())
    session_id = str(session_id)
    if not session_id.startswith("bbs-"):
        session_id = f"bbs-{session_id}"
    sample.session_id = session_id
    return session_id


@dataclass(frozen=True, slots=True)
class PrewarmHandle:
    """A ready sandbox whose ownership has transferred to rollout dispatch."""

    session_id: str
    group_id: int
    paddock: Any
    state: Any
    env_args: dict[str, Any]


@dataclass(slots=True)
class _PrewarmRecord:
    session_id: str
    group_id: int
    paddock: Any
    env_args: dict[str, Any]
    task: asyncio.Task[Any]


class PrewarmStore:
    """Own sandbox initialization tasks until they are claimed or cleaned up."""

    def __init__(self) -> None:
        self._records: dict[str, _PrewarmRecord] = {}
        self._group_sessions: dict[int, set[str]] = {}

    @property
    def unclaimed_count(self) -> int:
        """Return the number of prewarms still owned by this store."""
        return len(self._records)

    def start(
        self,
        sample: Any,
        *,
        group_id: int,
        paddock: Any,
        env_args: dict[str, Any],
    ) -> str | None:
        """Start and own one sandbox prewarm, or reject a duplicate session."""
        session_id = ensure_blackbox_session_id(sample)
        if session_id in self._records:
            return None
        metadata = getattr(sample, "metadata", None)
        env_type = metadata.get("env_type") if isinstance(metadata, dict) else None
        owned_env_args = copy.deepcopy(env_args)

        async def _initialize() -> Any:
            return await maybe_await(
                paddock.init(session_id, env_type, owned_env_args)
            )

        task = asyncio.create_task(_initialize(), name=f"prewarm:{session_id}")
        self._records[session_id] = _PrewarmRecord(
            session_id=session_id,
            group_id=group_id,
            paddock=paddock,
            env_args=owned_env_args,
            task=task,
        )
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        logger.debug(
            "sandbox prewarm started for session_id=%s group_id=%d",
            session_id,
            group_id,
        )
        return session_id

    async def claim(self, session_id: str | None) -> PrewarmHandle | None:
        """Atomically transfer one ready prewarm to active rollout ownership."""
        if session_id is None:
            return None
        record = self._take(session_id)
        if record is None:
            return None
        try:
            state = await record.task
        except asyncio.CancelledError:
            if not record.task.done():
                record.task.cancel()
            await asyncio.gather(record.task, return_exceptions=True)
            await terminate_paddock_best_effort(
                record.paddock,
                session_id=record.session_id,
                env_args=record.env_args,
            )
            raise
        except Exception as exc:
            logger.warning(
                "sandbox prewarm failed for session_id=%s: %s; "
                "falling back to inline init",
                session_id,
                exception_summary(exc),
            )
            await terminate_paddock_best_effort(
                record.paddock,
                session_id=record.session_id,
                env_args=record.env_args,
            )
            return None
        return PrewarmHandle(
            session_id=record.session_id,
            group_id=record.group_id,
            paddock=record.paddock,
            state=state,
            env_args=copy.deepcopy(record.env_args),
        )

    async def cleanup_group(self, group_id: int) -> None:
        """Release every unclaimed prewarm owned by *group_id*."""
        session_ids = tuple(self._group_sessions.pop(group_id, ()))
        records = [
            record
            for session_id in session_ids
            if (record := self._records.pop(session_id, None)) is not None
        ]
        await self._cleanup_records(records)

    async def cleanup_all(self) -> None:
        """Release all prewarms that have not transferred to dispatch."""
        records = list(self._records.values())
        self._records.clear()
        self._group_sessions.clear()
        await self._cleanup_records(records)

    async def _cleanup_records(self, records: list[_PrewarmRecord]) -> None:
        if not records:
            return
        await asyncio.gather(
            *(self._dispose(record) for record in records),
            return_exceptions=True,
        )

    async def _dispose(self, record: _PrewarmRecord) -> None:
        if not record.task.done():
            record.task.cancel()
        await asyncio.gather(record.task, return_exceptions=True)
        await terminate_paddock_best_effort(
            record.paddock,
            session_id=record.session_id,
            env_args=record.env_args,
        )
        logger.debug(
            "unused sandbox prewarm terminated for session_id=%s group_id=%d",
            record.session_id,
            record.group_id,
        )

    def _take(self, session_id: str) -> _PrewarmRecord | None:
        record = self._records.pop(session_id, None)
        if record is None:
            return None
        sessions = self._group_sessions.get(record.group_id)
        if sessions is not None:
            sessions.discard(session_id)
            if not sessions:
                self._group_sessions.pop(record.group_id, None)
        return record


_DEFAULT_STORE = PrewarmStore()


def start_prewarm(
    sample: Any,
    *,
    group_id: int,
    paddock: Any,
    env_args: dict[str, Any],
) -> str | None:
    """Start a prewarm owned by the default store."""
    return _DEFAULT_STORE.start(
        sample,
        group_id=group_id,
        paddock=paddock,
        env_args=env_args,
    )


async def claim_prewarm(session_id: str) -> PrewarmHandle | None:
    """Claim a prewarm from the default store for active rollout use."""
    return await _DEFAULT_STORE.claim(session_id)


async def cleanup_group(group_id: int) -> None:
    """Release unclaimed prewarms belonging to one group."""
    await _DEFAULT_STORE.cleanup_group(group_id)


async def cleanup_all() -> None:
    """Release every unclaimed prewarm in the default store."""
    await _DEFAULT_STORE.cleanup_all()
