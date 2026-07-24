"""Behavior tests for explicit prewarm ownership handoff."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_start_then_claim_returns_handle() -> None:
    from dressage.rollout.prewarm.store import PrewarmHandle, PrewarmStore

    state = object()
    paddock = SimpleNamespace(init=AsyncMock(return_value=state))
    sample = SimpleNamespace(session_id="bbs-one", metadata={"env_type": "repo"})
    store = PrewarmStore()

    session_id = store.start(
        sample,
        group_id=7,
        paddock=paddock,
        env_args={"sandbox_image": "image:v1"},
    )
    handle = await store.claim(session_id)

    assert handle == PrewarmHandle(
        session_id="bbs-one",
        group_id=7,
        paddock=paddock,
        state=state,
        env_args={"sandbox_image": "image:v1"},
    )
    paddock.init.assert_awaited_once_with(
        "bbs-one",
        "repo",
        {"sandbox_image": "image:v1"},
    )


@pytest.mark.asyncio
async def test_cleanup_group_terminates_only_unclaimed_ready_records() -> None:
    from dressage.rollout.prewarm.store import PrewarmStore

    paddock = SimpleNamespace(init=AsyncMock(return_value=object()))
    store = PrewarmStore()
    claimed = SimpleNamespace(session_id="bbs-claimed", metadata={})
    unused = SimpleNamespace(session_id="bbs-unused", metadata={})
    store.start(claimed, group_id=3, paddock=paddock, env_args={"kind": "claimed"})
    store.start(unused, group_id=3, paddock=paddock, env_args={"kind": "unused"})
    await asyncio.sleep(0)
    assert await store.claim("bbs-claimed") is not None

    with patch(
        "dressage.rollout.prewarm.store.terminate_paddock_best_effort",
        new_callable=AsyncMock,
    ) as terminate:
        await store.cleanup_group(3)

    terminate.assert_awaited_once_with(
        paddock,
        session_id="bbs-unused",
        env_args={"kind": "unused"},
    )
    assert await store.claim("bbs-unused") is None


@pytest.mark.asyncio
async def test_start_snapshots_env_args_and_rejects_duplicate_session() -> None:
    from dressage.rollout.prewarm.store import PrewarmStore

    paddock = SimpleNamespace(init=AsyncMock(return_value=object()))
    sample = SimpleNamespace(session_id="bbs-snapshot", metadata={})
    env_args = {"nested": {"value": 1}}
    store = PrewarmStore()

    assert store.start(sample, group_id=1, paddock=paddock, env_args=env_args)
    assert store.start(sample, group_id=2, paddock=paddock, env_args={}) is None
    env_args["new"] = "caller mutation"
    env_args["nested"]["value"] = 2
    handle = await store.claim("bbs-snapshot")

    assert handle is not None
    assert handle.group_id == 1
    assert handle.env_args == {"nested": {"value": 1}}
    assert await store.claim("bbs-snapshot") is None


@pytest.mark.asyncio
async def test_failed_prewarm_is_removed_and_terminated_before_fallback() -> None:
    from dressage.rollout.prewarm.store import PrewarmStore

    paddock = SimpleNamespace(init=AsyncMock(side_effect=RuntimeError("init failed")))
    sample = SimpleNamespace(session_id="bbs-failed", metadata={})
    store = PrewarmStore()
    store.start(sample, group_id=4, paddock=paddock, env_args={})

    with patch(
        "dressage.rollout.prewarm.store.terminate_paddock_best_effort",
        new_callable=AsyncMock,
    ) as terminate:
        assert await store.claim("bbs-failed") is None
        await store.cleanup_group(4)

    terminate.assert_awaited_once_with(
        paddock,
        session_id="bbs-failed",
        env_args={},
    )


@pytest.mark.asyncio
async def test_cleanup_group_cancels_pending_init_before_terminate() -> None:
    from dressage.rollout.prewarm.store import PrewarmStore

    started = asyncio.Event()

    async def hanging_init(*_args) -> object:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    paddock = SimpleNamespace(init=hanging_init)
    sample = SimpleNamespace(session_id="bbs-pending", metadata={})
    store = PrewarmStore()
    store.start(sample, group_id=5, paddock=paddock, env_args={"x": 1})
    await started.wait()

    with patch(
        "dressage.rollout.prewarm.store.terminate_paddock_best_effort",
        new_callable=AsyncMock,
    ) as terminate:
        await store.cleanup_group(5)

    terminate.assert_awaited_once()
    assert await store.claim("bbs-pending") is None


@pytest.mark.asyncio
async def test_cleanup_all_releases_every_unclaimed_group() -> None:
    from dressage.rollout.prewarm.store import PrewarmStore

    paddock = SimpleNamespace(init=AsyncMock(return_value=object()))
    store = PrewarmStore()
    for group_id in range(3):
        store.start(
            SimpleNamespace(session_id=f"bbs-{group_id}", metadata={}),
            group_id=group_id,
            paddock=paddock,
            env_args={},
        )
    await asyncio.sleep(0)

    with patch(
        "dressage.rollout.prewarm.store.terminate_paddock_best_effort",
        new_callable=AsyncMock,
    ) as terminate:
        await store.cleanup_all()
        await store.cleanup_all()

    assert terminate.await_count == 3


@pytest.mark.asyncio
async def test_cancelled_claim_cleans_up_and_propagates_cancellation() -> None:
    from dressage.rollout.prewarm.store import PrewarmStore

    started = asyncio.Event()

    async def hanging_init(*_args) -> object:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    paddock = SimpleNamespace(init=hanging_init)
    store = PrewarmStore()
    store.start(
        SimpleNamespace(session_id="bbs-cancelled", metadata={}),
        group_id=9,
        paddock=paddock,
        env_args={"cancel": True},
    )
    claim_task = asyncio.create_task(store.claim("bbs-cancelled"))
    await started.wait()

    with patch(
        "dressage.rollout.prewarm.store.terminate_paddock_best_effort",
        new_callable=AsyncMock,
    ) as terminate:
        claim_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await claim_task

    terminate.assert_awaited_once_with(
        paddock,
        session_id="bbs-cancelled",
        env_args={"cancel": True},
    )


def test_session_id_helper_owns_uuid_generation(monkeypatch) -> None:
    from dressage.rollout.prewarm import store as prewarm_store

    monkeypatch.setattr(prewarm_store.uuid, "uuid4", lambda: "generated")
    sample = SimpleNamespace(session_id=None)

    assert prewarm_store.ensure_blackbox_session_id(sample) == "bbs-generated"
    assert sample.session_id == "bbs-generated"


def test_prewarm_config_defaults_to_e2b_only(monkeypatch) -> None:
    from dressage.rollout.prewarm.config import (
        prewarm_ahead,
        prewarm_enabled,
    )

    monkeypatch.delenv("DRESSAGE_SANDBOX_PREWARM", raising=False)
    monkeypatch.setenv("DRESSAGE_SANDBOX_PROVIDER", "e2b")
    monkeypatch.setenv("DRESSAGE_SANDBOX_PREWARM_AHEAD", "3")
    assert prewarm_enabled() is True
    assert prewarm_ahead() == 3

    monkeypatch.setenv("DRESSAGE_SANDBOX_PROVIDER", "local_bwrap")
    assert prewarm_enabled() is False
    monkeypatch.setenv("DRESSAGE_SANDBOX_PREWARM", "1")
    assert prewarm_enabled() is True
    monkeypatch.setenv("DRESSAGE_SANDBOX_PREWARM", "0")
    assert prewarm_enabled() is False
    for invalid in ("true", "false", "yes", "no", "on", "off", "sometimes", "2"):
        monkeypatch.setenv("DRESSAGE_SANDBOX_PREWARM", invalid)
        with pytest.raises(
            ValueError,
            match="DRESSAGE_SANDBOX_PREWARM must be 0 or 1",
        ):
            prewarm_enabled()


@pytest.mark.asyncio
async def test_module_api_delegates_to_default_store(monkeypatch) -> None:
    from dressage.rollout.prewarm import store as store_module

    default_store = SimpleNamespace(
        start=MagicMock(return_value="bbs-public"),
        claim=AsyncMock(return_value=None),
        cleanup_group=AsyncMock(),
        cleanup_all=AsyncMock(),
    )
    monkeypatch.setattr(store_module, "_DEFAULT_STORE", default_store)
    sample = SimpleNamespace(session_id=None, metadata={})
    paddock = object()

    assert (
        store_module.start_prewarm(
            sample,
            group_id=12,
            paddock=paddock,
            env_args={"a": 1},
        )
        == "bbs-public"
    )
    assert await store_module.claim_prewarm("bbs-public") is None
    await store_module.cleanup_group(12)
    await store_module.cleanup_all()

    default_store.start.assert_called_once_with(
        sample,
        group_id=12,
        paddock=paddock,
        env_args={"a": 1},
    )
    default_store.claim.assert_awaited_once_with("bbs-public")
    default_store.cleanup_group.assert_awaited_once_with(12)
    default_store.cleanup_all.assert_awaited_once_with()
