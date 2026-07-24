"""Behavior tests for the prewarm scheduler."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["transformers"] = transformers

from dressage.rollout.prewarm import scheduler as scheduler_module  # noqa: F401


class FakeDataBuffer:
    def __init__(self, groups: list[list[Any]]):
        self.groups = list(groups)

    def get_samples(self, count: int) -> list[list[Any]]:
        assert count == 1
        if not self.groups:
            return []
        return [self.groups.pop(0)]


def sample(*, session_id: str | None = None, metadata: dict | None = None):
    return SimpleNamespace(session_id=session_id, metadata=metadata or {})


def scheduler(*, enabled: bool, ahead: int = 2):
    with patch(
        "dressage.rollout.prewarm.scheduler.prewarm_enabled",
        return_value=enabled,
    ), patch(
        "dressage.rollout.prewarm.scheduler.prewarm_ahead",
        return_value=ahead,
    ):
        from dressage.rollout.prewarm.scheduler import PrewarmScheduler

        return PrewarmScheduler()


def test_allocate_group_id_is_monotonic() -> None:
    subject = scheduler(enabled=False)
    assert [subject.allocate_group_id() for _ in range(4)] == [0, 1, 2, 3]


def test_provider_defaults_enable_e2b_and_skip_local_bwrap(monkeypatch) -> None:
    from dressage.rollout.prewarm.scheduler import PrewarmScheduler

    monkeypatch.delenv("DRESSAGE_SANDBOX_PREWARM", raising=False)
    monkeypatch.setenv("DRESSAGE_SANDBOX_PREWARM_AHEAD", "3")

    monkeypatch.setenv("DRESSAGE_SANDBOX_PROVIDER", "e2b")
    e2b = PrewarmScheduler()
    assert e2b.enabled is True
    assert e2b.ahead == 3

    monkeypatch.setenv("DRESSAGE_SANDBOX_PROVIDER", "local_bwrap")
    local_bwrap = PrewarmScheduler()
    assert local_bwrap.enabled is False
    assert local_bwrap.ahead == 0


def test_prefetch_passes_explicit_group_ownership_and_queues_groups() -> None:
    group_a = [sample(metadata={"blackbox_type": "opencode"}), sample()]
    group_b = [sample()]
    buffer = FakeDataBuffer([group_a, group_b])
    paddock = object()

    with patch(
        "dressage.rollout.prewarm.scheduler.get_paddock_from_env",
        return_value=paddock,
    ), patch(
        "dressage.rollout.prewarm.scheduler.paddock_env_args_from_metadata",
        side_effect=lambda metadata, **_: {"marker": id(metadata)},
    ), patch(
        "dressage.rollout.prewarm.scheduler.start_prewarm",
        side_effect=["bbs-a", "bbs-b", "bbs-c"],
    ) as start:
        subject = scheduler(enabled=True, ahead=2)
        subject.do_prefetch(buffer)

    assert subject._prefetched_groups == [(0, group_a), (1, group_b)]
    assert start.call_args_list == [
        call(
            group_a[0],
            group_id=0,
            paddock=paddock,
            env_args={"marker": id(group_a[0].metadata)},
        ),
        call(
            group_a[1],
            group_id=0,
            paddock=paddock,
            env_args={"marker": id(group_a[1].metadata)},
        ),
        call(
            group_b[0],
            group_id=1,
            paddock=paddock,
            env_args={"marker": id(group_b[0].metadata)},
        ),
    ]
    assert not hasattr(subject, "_group_sids")


def test_prefetch_keeps_group_when_paddock_or_start_is_unavailable() -> None:
    paddock_failure_group = [sample()]
    start_skip_group = [sample()]
    buffer = FakeDataBuffer([paddock_failure_group, start_skip_group])

    with patch(
        "dressage.rollout.prewarm.scheduler.get_paddock_from_env",
        side_effect=[RuntimeError("unavailable"), MagicMock()],
    ), patch(
        "dressage.rollout.prewarm.scheduler.paddock_env_args_from_metadata",
        return_value={},
    ), patch(
        "dressage.rollout.prewarm.scheduler.start_prewarm",
        return_value=None,
    ):
        subject = scheduler(enabled=True, ahead=2)
        subject.do_prefetch(buffer)

    assert subject._prefetched_groups == [
        (0, paddock_failure_group),
        (1, start_skip_group),
    ]


def test_prefetch_failure_keeps_group_and_continues_lookahead() -> None:
    failed_group = [sample()]
    next_group = [sample()]
    buffer = FakeDataBuffer([failed_group, next_group])

    with patch(
        "dressage.rollout.prewarm.scheduler.get_paddock_from_env",
        return_value=MagicMock(),
    ), patch(
        "dressage.rollout.prewarm.scheduler.paddock_env_args_from_metadata",
        return_value={},
    ), patch(
        "dressage.rollout.prewarm.scheduler.start_prewarm",
        side_effect=[RuntimeError("prewarm failed"), "bbs-next"],
    ):
        subject = scheduler(enabled=True, ahead=2)
        subject.do_prefetch(buffer)

    assert subject._prefetched_groups == [
        (0, failed_group),
        (1, next_group),
    ]


def test_disabled_prefetch_does_not_consume_buffer() -> None:
    buffer = FakeDataBuffer([[sample()]])
    subject = scheduler(enabled=False)
    subject.do_prefetch(buffer)
    assert len(buffer.groups) == 1
    assert subject._prefetched_groups == []


def test_pop_next_group_prefers_queue_then_falls_back_to_buffer() -> None:
    queued = [sample()]
    fresh = [sample()]
    buffer = FakeDataBuffer([fresh])
    subject = scheduler(enabled=False)
    subject._prefetched_groups.append((12, queued))

    assert subject.pop_next_group(buffer) == (12, queued)
    assert subject.pop_next_group(buffer) == (0, fresh)
    assert subject.pop_next_group(buffer) is None


@pytest.mark.asyncio
async def test_cleanup_group_delegates_group_id() -> None:
    with patch(
        "dressage.rollout.prewarm.scheduler.cleanup_prewarm_group",
        new_callable=AsyncMock,
    ) as cleanup:
        subject = scheduler(enabled=True)
        await subject.cleanup_group(42)

    cleanup.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_cleanup_clears_queue_and_delegates_all() -> None:
    with patch(
        "dressage.rollout.prewarm.scheduler.cleanup_all",
        new_callable=AsyncMock,
    ) as cleanup:
        subject = scheduler(enabled=True)
        subject._prefetched_groups.append((1, [sample()]))
        await subject.cleanup()

    cleanup.assert_awaited_once_with()
    assert subject._prefetched_groups == []
