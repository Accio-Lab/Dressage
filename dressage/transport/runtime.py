"""Proxy-side field offload and retention bookkeeping."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .fields import (
    TQFieldRef,
    TQTrajectoryRef,
    normalize_transfer_params,
    resolve_step_transfer_fields,
)
from .store import TransferQueueStore

logger = logging.getLogger(__name__)


class TransferQueueRuntime:
    """Write selected Step fields without owning trajectory construction."""

    def __init__(
        self,
        store: TransferQueueStore,
        *,
        transfer_params: str | Iterable[str],
        token_build_mode: str,
        retention_seconds: float = 86400.0,
        retention_poll_seconds: float = 60.0,
    ):
        self.store = store
        self.transfer_params = normalize_transfer_params(transfer_params)
        self.transfer_fields = resolve_step_transfer_fields(
            self.transfer_params,
            token_build_mode=token_build_mode,
        )
        if not self.transfer_fields:
            raise ValueError(
                "TransferQueue requires at least one transfer parameter"
            )
        self.retention_seconds = float(retention_seconds)
        self.retention_poll_seconds = float(retention_poll_seconds)
        self._retention_task: asyncio.Task[None] | None = None

    @classmethod
    async def from_config(
        cls,
        config_path: str,
        *,
        store_id: str,
        transfer_params: str | Iterable[str],
        token_build_mode: str,
        retention_seconds: float = 86400.0,
        retention_poll_seconds: float = 60.0,
    ) -> TransferQueueRuntime:
        store = await TransferQueueStore.from_config(
            config_path,
            store_id=store_id,
        )
        return cls(
            store,
            transfer_params=transfer_params,
            token_build_mode=token_build_mode,
            retention_seconds=retention_seconds,
            retention_poll_seconds=retention_poll_seconds,
        )

    async def offload_step(
        self,
        *,
        session_id: str,
        incarnation_id: str,
        step_index: int,
        fields: dict[str, Any],
    ) -> dict[str, TQFieldRef]:
        """Write one Step key and return a typed reference per stored column."""

        selected = {name: fields[name] for name in self.transfer_fields}
        key = self.store.step_key(
            trajectory_id=session_id,
            incarnation_id=incarnation_id,
            step_index=step_index,
        )
        expires_at = time.time() + self.retention_seconds
        await self.store.put(
            key=key,
            partition=self.store.step_partition,
            fields=selected,
            tag={
                "expires_at": expires_at,
                "trajectory_id": session_id,
                "incarnation_id": incarnation_id,
            },
        )

        refs = {
            field: TQFieldRef(
                store_id=self.store.store_id,
                partition=self.store.step_partition,
                key=key,
                field=field,
                incarnation_id=incarnation_id,
            )
            for field in selected
        }
        return refs

    def trajectory_ref(self, session: Any) -> TQTrajectoryRef:
        refs: list[TQFieldRef] = []
        seen: set[tuple[str, str]] = set()
        for step in session.steps:
            for field in self.transfer_fields:
                ref = getattr(step, field)
                if not isinstance(ref, TQFieldRef):
                    continue
                identity = (ref.partition, ref.key)
                if identity in seen:
                    continue
                seen.add(identity)
                refs.append(ref)
        return TQTrajectoryRef(
            store_id=self.store.store_id,
            trajectory_id=session.session_id,
            incarnation_id=session.incarnation_id,
            refs=tuple(refs),
        )

    async def clear_refs(self, refs: Iterable[TQFieldRef]) -> None:
        by_partition: dict[str, set[str]] = defaultdict(set)
        for ref in refs:
            by_partition[ref.partition].add(ref.key)
        for partition, keys in by_partition.items():
            await self.store.clear(keys=sorted(keys), partition=partition)

    async def sweep_expired(self, *, now: float | None = None) -> int:
        partitions = await self.store.list(partition=self.store.step_partition)
        entries = partitions.get(self.store.step_partition, {})
        deadline = time.time() if now is None else now
        expired = sorted(
            key
            for key, tag in entries.items()
            if float(tag.get("expires_at", float("inf"))) <= deadline
        )
        if expired:
            await self.store.clear(
                keys=expired,
                partition=self.store.step_partition,
            )
        return len(expired)

    def start_retention(self) -> asyncio.Task[None]:
        if self._retention_task is None or self._retention_task.done():
            self._retention_task = asyncio.create_task(self._retention_loop())
        return self._retention_task

    def stats(self) -> dict[str, Any]:
        return {
            **self.store.stats(),
            "transfer_params": self.transfer_params,
            "retention_seconds": self.retention_seconds,
        }

    async def close(self) -> None:
        task = self._retention_task
        self._retention_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _retention_loop(self) -> None:
        while True:
            await asyncio.sleep(self.retention_poll_seconds)
            try:
                await self.sweep_expired()
            except Exception:
                logger.exception("TransferQueue retention sweep failed")
