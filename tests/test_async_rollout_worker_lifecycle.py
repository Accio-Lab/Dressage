"""Lifecycle invariants shared by asynchronous rollout workers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dressage.paddock import lifecycle as paddock_lifecycle
from dressage.paddock.blackbox.paddock import BlackboxAgentPaddock
from dressage.rollout import fully_async_rollout, partial_async_rollout
from dressage.rollout.prewarm import scheduler as prewarm_scheduler
from dressage.rollout.prewarm import store as prewarm_store
from dressage.sandbox.remote.e2b.provider import E2BSandboxProvider


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


@pytest.mark.asyncio
async def test_slow_unclaimed_prewarm_does_not_block_completed_groups(
    monkeypatch,
) -> None:
    create_started = asyncio.Event()
    finish_create = asyncio.Event()
    events: list[tuple[str, str]] = []

    class Sandbox:
        def __init__(self, trajectory_id: str) -> None:
            self.sandbox_id = f"sandbox-{trajectory_id}"
            self.trajectory_id = trajectory_id

        async def get_host(self, _port: int) -> str:
            return f"{self.trajectory_id}.e2b.test"

        async def kill(self) -> bool:
            events.append(("kill", self.trajectory_id))
            return True

    async def sandbox_factory(**kwargs):
        trajectory_id = kwargs["metadata"]["trajectory_id"]
        if trajectory_id == "bbs-slow":
            create_started.set()
            await finish_create.wait()
        return Sandbox(trajectory_id)

    provider = E2BSandboxProvider(
        template="blackbox-template",
        sandbox_factory=sandbox_factory,
    )
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    isolated_store = prewarm_store.PrewarmStore()
    monkeypatch.setattr(prewarm_store, "_DEFAULT_STORE", isolated_store)
    monkeypatch.setattr(
        prewarm_scheduler,
        "get_paddock_from_env",
        lambda **_kwargs: paddock,
    )
    monkeypatch.setattr(
        fully_async_rollout,
        "_state_for",
        lambda _args: SimpleNamespace(sampling_params={}),
    )
    monkeypatch.setenv("DRESSAGE_SANDBOX_PROVIDER", "e2b")
    monkeypatch.setenv("DRESSAGE_SANDBOX_PREWARM", "1")
    monkeypatch.setenv("DRESSAGE_SANDBOX_PREWARM_AHEAD", "2")
    monkeypatch.setenv("DRESSAGE_ASYNC_MAX_ACTIVE_GROUPS", "2")

    groups = [
        [
            SimpleNamespace(
                session_id="bbs-slow",
                metadata={"env_type": "repo", "sandbox_image": "blackbox-template"},
            )
        ],
        [
            SimpleNamespace(
                session_id="bbs-fast",
                metadata={"env_type": "repo", "sandbox_image": "blackbox-template"},
            )
        ],
    ]

    class DataBuffer:
        def get_samples(self, _count: int) -> list:
            if not groups:
                return []
            return [groups.pop(0)]

    worker = fully_async_rollout.AsyncRolloutWorker(
        SimpleNamespace(rollout_batch_size=2, n_samples_per_prompt=1),
        DataBuffer(),
    )

    async def run_group(group, _sampling_params):
        return group

    worker._run_group = run_group
    worker_task = asyncio.create_task(worker.continuous_worker_loop())

    async def wait_for_completed_groups() -> None:
        while worker.output_queue.qsize() < 2:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(create_started.wait(), timeout=1)
        await asyncio.wait_for(wait_for_completed_groups(), timeout=1)
        assert not finish_create.is_set()
    finally:
        worker.running = False
        finish_create.set()
        await asyncio.wait_for(worker_task, timeout=1)
        await paddock_lifecycle.drain_lifecycle_tasks()
        await paddock.close()

    assert sorted(events) == [
        ("kill", "bbs-fast"),
        ("kill", "bbs-slow"),
    ]
