"""Lifecycle invariants shared by asynchronous rollout workers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dressage.paddock import lifecycle as paddock_lifecycle
from dressage.rollout import fully_async_rollout, partial_async_rollout


class EmptyDataBuffer:
    def get_samples(self, _count: int) -> list:
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "worker_type"),
    [
        (fully_async_rollout, fully_async_rollout.AsyncRolloutWorker),
        (partial_async_rollout, partial_async_rollout.PartialAsyncRolloutWorker),
    ],
)
async def test_worker_shutdown_drains_background_terminations(
    monkeypatch,
    module,
    worker_type,
) -> None:
    terminated = asyncio.Event()

    class Paddock:
        async def terminate(self, _session_id, _env_args) -> None:
            await asyncio.sleep(0.01)
            terminated.set()

    monkeypatch.setattr(
        module,
        "_state_for",
        lambda _args: SimpleNamespace(sampling_params={}),
    )
    worker = worker_type(
        SimpleNamespace(rollout_batch_size=1),
        EmptyDataBuffer(),
    )
    worker.running = False
    worker._scheduler = SimpleNamespace(
        enabled=False,
        ahead=0,
        cleanup=AsyncMock(),
    )
    paddock_lifecycle.schedule_terminate_paddock(
        Paddock(),
        session_id="bbs-shutdown",
        env_args={},
    )

    try:
        await worker.continuous_worker_loop()
        assert terminated.is_set()
    finally:
        await paddock_lifecycle.drain_terminate_tasks()
