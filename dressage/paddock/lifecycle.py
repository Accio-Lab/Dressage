"""Sandbox lifecycle helpers shared by paddock and rollout code."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Awaitable
from typing import Any

logger = logging.getLogger(__name__)

_LIFECYCLE_TASKS: dict[
    asyncio.AbstractEventLoop,
    set[asyncio.Task[Any]],
] = {}


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def exception_summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__


def terminate_timeout_sec() -> float:
    value = os.environ.get("DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC", "30")
    try:
        timeout = float(value)
    except ValueError:
        logger.warning(
            "invalid DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC=%r; falling back to 30",
            value,
        )
        return 30.0
    if timeout <= 0:
        logger.warning(
            "invalid DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC=%r; falling back to 30",
            value,
        )
        return 30.0
    return timeout


async def terminate_paddock_best_effort(
    paddock: Any,
    *,
    session_id: str,
    env_args: dict[str, Any],
) -> None:
    timeout = terminate_timeout_sec()
    release_task = asyncio.create_task(
        _call_paddock_terminate(paddock, session_id, env_args)
    )
    try:
        await asyncio.wait_for(asyncio.shield(release_task), timeout=timeout)
    except asyncio.CancelledError:
        track_lifecycle_task(release_task)
        raise
    except asyncio.TimeoutError:
        logger.warning(
            "timed out waiting for sandbox release RPC for session_id=%s "
            "after %.3fs; release continues in background",
            session_id,
            timeout,
        )
        track_lifecycle_task(release_task)
    except Exception as exc:
        logger.warning(
            "failed to terminate sandbox for session_id=%s: %s",
            session_id,
            exception_summary(exc),
        )


async def terminate_provider_lease_best_effort(
    provider: Any,
    lease: Any,
    *,
    trajectory_id: str,
) -> None:
    """Terminate one provider lease without abandoning a timed-out RPC."""
    timeout = terminate_timeout_sec()
    release_task = asyncio.create_task(
        maybe_await(provider.terminate(lease)),
        name=f"sandbox-terminate:{trajectory_id}",
    )
    try:
        await asyncio.wait_for(asyncio.shield(release_task), timeout=timeout)
    except asyncio.CancelledError:
        track_lifecycle_task(release_task)
        raise
    except asyncio.TimeoutError:
        logger.warning(
            "timed out waiting for sandbox provider termination "
            "for trajectory_id=%s after %.3fs; termination continues "
            "in background",
            trajectory_id,
            timeout,
        )
        track_lifecycle_task(release_task)
    except Exception as exc:
        logger.warning(
            "failed to terminate sandbox provider lease "
            "for trajectory_id=%s: %s",
            trajectory_id,
            exception_summary(exc),
        )


async def terminate_when_provider_create_finishes(
    provider: Any,
    create_task: asyncio.Task[Any],
    *,
    trajectory_id: str,
) -> None:
    """Take ownership of a cancelled caller's in-flight provider create."""
    while True:
        try:
            lease = await asyncio.shield(create_task)
            break
        except asyncio.CancelledError:
            if create_task.cancelled():
                logger.debug(
                    "sandbox provider create was cancelled before producing "
                    "a lease for trajectory_id=%s",
                    trajectory_id,
                )
                return
        except Exception as exc:
            logger.warning(
                "sandbox provider create failed after its caller was cancelled "
                "for trajectory_id=%s: %s",
                trajectory_id,
                exception_summary(exc),
            )
            return
    await terminate_provider_lease_best_effort(
        provider,
        lease,
        trajectory_id=trajectory_id,
    )


def track_lifecycle_task(task: asyncio.Task[Any]) -> asyncio.Task[Any]:
    """Keep a background cleanup task alive until completion or shutdown."""
    tasks = _LIFECYCLE_TASKS.setdefault(task.get_loop(), set())
    tasks.add(task)
    task.add_done_callback(_discard_lifecycle_task)
    return task


def pending_lifecycle_task_count() -> int:
    """Return background lifecycle work that still owns sandbox resources."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return sum(len(tasks) for tasks in tuple(_LIFECYCLE_TASKS.values()))
    return len(_LIFECYCLE_TASKS.get(loop, ()))


def schedule_lifecycle_task(
    awaitable: Awaitable[Any],
    *,
    name: str,
) -> asyncio.Task[Any]:
    """Create and strongly track one background lifecycle operation."""
    return track_lifecycle_task(asyncio.create_task(awaitable, name=name))


def schedule_terminate_paddock(
    paddock: Any,
    *,
    session_id: str,
    env_args: dict[str, Any],
) -> None:
    schedule_lifecycle_task(
        terminate_paddock_best_effort(
            paddock,
            session_id=session_id,
            env_args=dict(env_args),
        ),
        name=f"paddock-terminate:{session_id}",
    )


async def drain_lifecycle_tasks() -> None:
    """Wait for all tracked cleanup, including tasks spawned while draining."""
    loop = asyncio.get_running_loop()
    while tracked := _LIFECYCLE_TASKS.get(loop):
        tasks = tuple(tracked)
        await asyncio.gather(*tasks, return_exceptions=True)


async def drain_terminate_tasks() -> None:
    """Backward-compatible alias for draining all lifecycle cleanup tasks."""
    await drain_lifecycle_tasks()


async def _call_paddock_terminate(
    paddock: Any,
    session_id: str,
    env_args: dict[str, Any],
) -> Any:
    terminate = paddock.terminate
    if inspect.iscoroutinefunction(terminate):
        return await terminate(session_id, env_args)
    return await maybe_await(await asyncio.to_thread(terminate, session_id, env_args))


def _discard_lifecycle_task(task: asyncio.Task[Any]) -> None:
    loop = task.get_loop()
    tasks = _LIFECYCLE_TASKS.get(loop)
    if tasks is not None:
        tasks.discard(task)
        if not tasks:
            _LIFECYCLE_TASKS.pop(loop, None)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning("background lifecycle task failed: %s", exception_summary(exc))
