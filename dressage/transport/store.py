"""Thin asynchronous wrapper around the official TransferQueue KV API."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any


def _load_transfer_queue() -> Any:
    try:
        import transfer_queue as tq
    except ImportError as exc:
        raise RuntimeError(
            "TransferQueue is not installed; install Dressage with the "
            "transfer-queue optional dependency"
        ) from exc
    return tq


class TransferQueueStore:
    """Expose only the KV operations needed by field-level offload."""

    def __init__(self, tq_api: Any, *, store_id: str):
        self._tq = tq_api
        self.store_id = store_id
        self.step_partition = f"dressage:{store_id}:steps"
        self._stats = {
            "write_count": 0,
            "write_bytes": 0,
            "write_seconds": 0.0,
            "clear_count": 0,
        }

    @staticmethod
    def estimate_bytes(value: Any) -> int:
        value = getattr(value, "data", value)
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            return int(value.numel()) * int(value.element_size())
        if hasattr(value, "nbytes"):
            return int(value.nbytes)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, dict):
            return sum(
                TransferQueueStore.estimate_bytes(key)
                + TransferQueueStore.estimate_bytes(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return sum(TransferQueueStore.estimate_bytes(item) for item in value)
        if isinstance(value, (int, float, bool)):
            return 8
        return 0

    def stats(self) -> dict[str, int | float | str]:
        return {"store_id": self.store_id, **self._stats}

    @classmethod
    async def from_config(
        cls,
        config_path: str,
        *,
        store_id: str,
    ) -> TransferQueueStore:
        tq = _load_transfer_queue()
        try:
            import ray
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "TransferQueue dependencies are incomplete; install Dressage with "
                "the transfer-queue optional dependency"
            ) from exc

        if not ray.is_initialized():
            await asyncio.to_thread(ray.init, address="auto")
        config = OmegaConf.load(Path(config_path))
        await asyncio.to_thread(tq.init, config)
        return cls(tq, store_id=store_id)

    def step_key(
        self,
        *,
        trajectory_id: str,
        incarnation_id: str,
        step_index: int,
    ) -> str:
        return (
            f"{self.store_id}:{trajectory_id}:{incarnation_id}:step:{step_index}"
        )

    async def put(
        self,
        *,
        key: str,
        partition: str,
        fields: dict[str, Any],
        tag: dict[str, Any],
    ) -> Any:
        started_at = time.perf_counter()
        result = await self._tq.async_kv_put(
            key=key,
            partition_id=partition,
            fields=fields,
            tag=tag,
        )
        self._stats["write_count"] += 1
        self._stats["write_bytes"] += self.estimate_bytes(fields)
        self._stats["write_seconds"] += time.perf_counter() - started_at
        return result

    async def clear(self, *, keys: list[str], partition: str) -> None:
        if keys:
            await self._tq.async_kv_clear(
                keys=keys,
                partition_id=partition,
            )
            self._stats["clear_count"] += len(keys)

    async def list(self, *, partition: str) -> dict[str, dict[str, Any]]:
        return await self._tq.async_kv_list(partition_id=partition)
