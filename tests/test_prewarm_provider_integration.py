"""Prewarm handoff coverage for the public E2B and local_bwrap providers."""

from __future__ import annotations

from typing import Any

import pytest

from dressage.paddock.blackbox.paddock import BlackboxAgentPaddock
from dressage.rollout.prewarm.store import PrewarmStore
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
