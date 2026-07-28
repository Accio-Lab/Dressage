"""Prewarm handoff coverage for the public E2B and local_bwrap providers."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dressage.paddock.blackbox.paddock import BlackboxAgentPaddock
from dressage.paddock.lifecycle import drain_lifecycle_tasks
from dressage.rollout.prewarm.store import PrewarmStore
from dressage.sandbox import SandboxSpec
from dressage.sandbox.local.bwrap.provider import LocalBwrapSandboxProvider
from dressage.sandbox.remote.e2b.provider import E2BSandboxProvider


class FakeE2BSandbox:
    sandbox_id = "e2b-sandbox-1"

    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events

    async def get_host(self, port: int) -> str:
        self.events.append(("get_host", port))
        return "sandbox.e2b.test"

    async def kill(self) -> bool:
        self.events.append(("kill",))
        return True


class FakeBwrapManager:
    pool_mode = "blackbox"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def acquire(self, trajectory_id, env_type=None, env_args=None):
        self.calls.append(("acquire", trajectory_id, env_type, env_args))
        return {
            "lease_id": f"lease-{trajectory_id}",
            "node_id": "node-1",
            "node_ip": "127.0.0.1",
            "slot_id": 1,
            "port": 31001,
            "generation": 1,
            "sandbox_url": "http://127.0.0.1:31001",
        }

    async def release(self, trajectory_id=None, lease_id=None, reason=None):
        self.calls.append(("release", trajectory_id, lease_id, reason))
        return {"released": True}


@pytest.mark.asyncio
async def test_e2b_prewarm_claim_transfers_live_lease() -> None:
    events: list[tuple[Any, ...]] = []

    async def sandbox_factory(**_kwargs):
        return FakeE2BSandbox(events)

    provider = E2BSandboxProvider(
        template="blackbox-template",
        sandbox_factory=sandbox_factory,
    )
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    store = PrewarmStore()
    sample = type(
        "Sample",
        (),
        {"session_id": "bbs-e2b", "metadata": {"env_type": "repo"}},
    )()

    store.start(
        sample,
        group_id=1,
        paddock=paddock,
        env_args={"sandbox_image": "blackbox-template"},
    )
    handle = await store.claim("bbs-e2b")

    assert handle is not None
    assert handle.state.sandbox_id == "e2b-sandbox-1"
    assert handle.state.sandbox_url == "https://sandbox.e2b.test"
    assert events == [("get_host", 31000)]

    await paddock.terminate(handle.session_id, handle.env_args)
    await paddock.close()
    assert events == [("get_host", 31000), ("kill",)]


@pytest.mark.asyncio
async def test_local_bwrap_prewarm_claim_transfers_pool_slot() -> None:
    manager = FakeBwrapManager()
    provider = LocalBwrapSandboxProvider(manager=manager)
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    store = PrewarmStore()
    sample = type(
        "Sample",
        (),
        {"session_id": "bbs-bwrap", "metadata": {"env_type": "repo"}},
    )()

    store.start(
        sample,
        group_id=2,
        paddock=paddock,
        env_args={},
    )
    handle = await store.claim("bbs-bwrap")

    assert handle is not None
    assert handle.state.sandbox_id == "lease-bbs-bwrap"
    assert handle.state.sandbox_url == "http://127.0.0.1:31001"
    assert manager.calls == [("acquire", "bbs-bwrap", "repo", {})]

    await paddock.terminate(handle.session_id, handle.env_args)
    await paddock.close()
    assert manager.calls[-1] == (
        "release",
        "bbs-bwrap",
        "lease-bbs-bwrap",
        "paddock_terminate",
    )


@pytest.mark.asyncio
async def test_e2b_cleanup_returns_before_create_then_kills_sandbox() -> None:
    events: list[tuple[Any, ...]] = []
    create_started = asyncio.Event()
    finish_create = asyncio.Event()

    async def sandbox_factory(**_kwargs):
        create_started.set()
        await finish_create.wait()
        return FakeE2BSandbox(events)

    provider = E2BSandboxProvider(
        template="blackbox-template",
        sandbox_factory=sandbox_factory,
    )
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    store = PrewarmStore()
    sample = type(
        "Sample",
        (),
        {"session_id": "bbs-e2b-cleanup", "metadata": {"env_type": "repo"}},
    )()

    store.start(
        sample,
        group_id=3,
        paddock=paddock,
        env_args={"sandbox_image": "blackbox-template"},
    )
    await create_started.wait()
    cleanup_task = asyncio.create_task(store.cleanup_group(3))
    await asyncio.sleep(0)

    assert cleanup_task.done()
    assert events == []

    finish_create.set()
    await cleanup_task
    await drain_lifecycle_tasks()
    await paddock.close()
    assert events == [("get_host", 31000), ("kill",)]


@pytest.mark.asyncio
async def test_paddock_create_cleanup_survives_repeated_caller_cancellation() -> None:
    events: list[tuple[Any, ...]] = []
    create_started = asyncio.Event()
    finish_create = asyncio.Event()

    async def sandbox_factory(**_kwargs):
        create_started.set()
        await finish_create.wait()
        return FakeE2BSandbox(events)

    provider = E2BSandboxProvider(
        template="blackbox-template",
        sandbox_factory=sandbox_factory,
    )
    original_terminate = provider.terminate
    provider.terminate = AsyncMock(side_effect=original_terminate)
    paddock = BlackboxAgentPaddock(
        provider=provider,
        proxy_public_url="http://proxy.test",
        wait_health=False,
    )
    init_task = asyncio.create_task(
        paddock.init(
            "bbs-e2b-double-cancel",
            "repo",
            {"sandbox_image": "blackbox-template"},
        )
    )
    await create_started.wait()

    init_task.cancel()
    await asyncio.sleep(0)
    init_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await init_task

    assert not finish_create.is_set()
    provider.terminate.assert_not_awaited()

    finish_create.set()
    await drain_lifecycle_tasks()
    provider.terminate.assert_awaited_once()
    assert events == [("get_host", 31000), ("kill",)]
    await paddock.close()


@pytest.mark.asyncio
async def test_e2b_create_cleanup_survives_repeated_cancellation() -> None:
    events: list[tuple[Any, ...]] = []
    create_started = asyncio.Event()
    finish_create = asyncio.Event()

    async def sandbox_factory(**_kwargs):
        create_started.set()
        await finish_create.wait()
        return FakeE2BSandbox(events)

    provider = E2BSandboxProvider(
        template="blackbox-template",
        sandbox_factory=sandbox_factory,
    )
    create_task = asyncio.create_task(
        provider.create(SandboxSpec(trajectory_id="e2b-repeated-cancel"))
    )
    await create_started.wait()

    create_task.cancel()
    await asyncio.sleep(0)
    create_task.cancel()
    await asyncio.sleep(0)

    assert not create_task.done()
    finish_create.set()
    with pytest.raises(asyncio.CancelledError):
        await create_task
    assert events == [("kill",)]


@pytest.mark.asyncio
async def test_e2b_terminate_is_idempotent_for_the_same_lease() -> None:
    events: list[tuple[Any, ...]] = []

    async def sandbox_factory(**_kwargs):
        return FakeE2BSandbox(events)

    provider = E2BSandboxProvider(
        template="blackbox-template",
        sandbox_factory=sandbox_factory,
    )
    lease = await provider.create(SandboxSpec(trajectory_id="e2b-idempotent"))

    first, second = await asyncio.gather(
        provider.terminate(lease),
        provider.terminate(lease),
    )

    assert events == [("kill",)]
    assert first["terminated"] is True
    assert second["missing"] is True
