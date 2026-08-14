"""Shared runtime glue for Dressage generate hooks."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from dressage.config import proxy_url

if TYPE_CHECKING:
    from dressage.proxy.proxy_client import ProxyClient as ProxyClientType
else:
    ProxyClientType = Any

# Kept as an injection point for tests and embedders.  The real class is
# imported lazily so scheduler-only processes do not load proxy/model deps.
ProxyClient: Any = None

logger = logging.getLogger(__name__)

_PADDOCK = None
_PADDOCK_BY_MODE: dict[tuple[str, str], Any] = {}
_PROXY_CLIENT: ProxyClientType | None = None

_PADDOCK_ENV_ARG_KEYS = (
    "sandbox_timeout_sec",
    "sandbox_image",
    "sandbox_cmd",
    "sandbox_extra_params",
)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def get_proxy_client() -> ProxyClientType:
    global _PROXY_CLIENT, ProxyClient
    if _PROXY_CLIENT is None:
        if ProxyClient is None:
            from dressage.proxy.proxy_client import ProxyClient as ProxyClientClass

            ProxyClient = ProxyClientClass

        _PROXY_CLIENT = ProxyClient(proxy_url())
    return _PROXY_CLIENT


async def discard_proxy_session_best_effort(
    session_id: str | None,
    *,
    proxy_client: ProxyClientType | None = None,
) -> bool:
    """Discard one rejected attempt without turning cleanup into rollout failure."""

    if not session_id:
        return True

    try:
        client = proxy_client or get_proxy_client()
        result = await asyncio.wait_for(
            client.discard_session(str(session_id)),
            timeout=10.0,
        )
        if isinstance(result, dict) and result.get("success") is False:
            logger.warning(
                "proxy refused to discard failed rollout session_id=%s: %r",
                session_id,
                result,
            )
            return False
    except Exception:
        logger.warning(
            "failed to discard proxy state for aborted rollout session_id=%s",
            session_id,
            exc_info=True,
        )
        return False

    logger.debug("discarded proxy state for aborted rollout session_id=%s", session_id)
    return True


async def discard_rollout_groups_best_effort(
    groups: Iterable[Iterable[Any]],
    *,
    proxy_client: ProxyClientType | None = None,
) -> bool:
    trajectory_ids: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for sample in group:
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
            trajectory_id = (
                metadata.get("parent_traj_id")
                or metadata.get("last_failed_session_id")
                or metadata.get("session_id")
                or getattr(sample, "session_id", None)
            )
            if trajectory_id is None:
                continue
            trajectory_id = str(trajectory_id)
            if trajectory_id in seen:
                continue
            seen.add(trajectory_id)
            trajectory_ids.append(trajectory_id)

    results = await asyncio.gather(
        *(
            discard_proxy_session_best_effort(
                trajectory_id,
                proxy_client=proxy_client,
            )
            for trajectory_id in trajectory_ids
        )
    )
    return all(results)


async def generate_group(
    generate: Any,
    args: Any,
    group: list[Any],
    sampling_params: dict[str, Any],
) -> list[Any]:
    result = None
    try:
        result = await generate(
            args,
            group,
            sampling_params=sampling_params,
            evaluation=False,
        )
        return result
    finally:
        generated = [
            sample
            for item in (result or [])
            for sample in (item if isinstance(item, list) else [item])
        ]
        if result is None or any(
            getattr(getattr(sample, "status", None), "name", None) == "ABORTED"
            for sample in generated
        ):
            await discard_rollout_groups_best_effort([generated, group])


def get_paddock_from_env(
    *, allow_whitebox_mode: bool, mode: str | None = None
) -> Any:
    global _PADDOCK
    # _PADDOCK remains the explicit test/embedder override and the legacy
    # cache for callers that do not request a mode.
    if _PADDOCK is not None:
        return _PADDOCK

    paddock_class_path = os.environ.get("DRESSAGE_PADDOCK_CLASS")
    paddock_mode = (
        mode or os.environ.get("DRESSAGE_PADDOCK_MODE") or "blackbox"
    ).strip().lower()
    if not paddock_class_path and not allow_whitebox_mode and paddock_mode == "whitebox":
        raise ValueError(
            "blackbox_dispatch does not support whitebox mode; set "
            "DRESSAGE_PADDOCK_MODE=blackbox for this rollout hook, or use "
            "the Paddock API for whitebox tool execution"
        )

    from dressage.paddock import factory as paddock_factory

    if mode is None:
        _PADDOCK = paddock_factory.create_paddock_from_env()
        paddock = _PADDOCK
    else:
        cache_key = (paddock_class_path or "", paddock_mode)
        paddock = _PADDOCK_BY_MODE.get(cache_key)
        if paddock is None:
            paddock = paddock_factory.create_paddock_from_env(mode=paddock_mode)
            _PADDOCK_BY_MODE[cache_key] = paddock
    if paddock_class_path:
        logger.info("initialized paddock class override: %s", paddock_class_path)
    else:
        logger.info("initialized paddock from mode/provider env: %s", type(paddock).__name__)
    return paddock


def paddock_env_args_from_metadata(
    metadata: dict[str, Any],
    *,
    extra_env_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_args = {key: metadata[key] for key in _PADDOCK_ENV_ARG_KEYS if key in metadata}
    if extra_env_args:
        env_args.update(extra_env_args)
    return env_args
