"""Slime async training entrypoint with Dressage proxy pause/resume.

Mirrors ``slime/train_async.py`` and wraps actor weight updates with Dressage
proxy pause/resume calls. The pause tells Dressage's generation controller to
abort active upstream SGLang requests at a safe boundary, keep partial output,
and wait until resume before continuing.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import ray

from dressage.config import proxy_url
from dressage.proxy.proxy_client import ProxyClient
from slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import (
    configure_logger,
    finish_tracking,
    init_tracking,
    update_tracking_open_metrics,
)
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _proxy_url() -> str | None:
    return proxy_url()


async def _pause_proxy(reason: str) -> bool:
    if not _env_flag("DRESSAGE_PROXY_PAUSE_AROUND_WEIGHT_UPDATE", True):
        return False
    proxy_url = _proxy_url()
    if not proxy_url:
        if _env_flag("DRESSAGE_PROXY_PAUSE_REQUIRED", True):
            raise RuntimeError(
                "DRESSAGE_PROXY_URL is not set; cannot safely pause Dressage rollout "
                "around actor weight update"
            )
        logger.warning("DRESSAGE_PROXY_URL is not set; skipping rollout pause before weight update")
        return False

    timeout_seconds = float(os.environ.get("DRESSAGE_PROXY_PAUSE_TIMEOUT_SEC", "300"))
    client = ProxyClient(proxy_url)
    try:
        response = await client.pause_rollout(reason=reason, timeout_seconds=timeout_seconds)
        logger.info("paused Dressage rollout before weight update: %s", response)
        return True
    finally:
        await client.close()


async def _resume_proxy(reason: str) -> None:
    if not _env_flag("DRESSAGE_PROXY_PAUSE_AROUND_WEIGHT_UPDATE", True):
        return
    proxy_url = _proxy_url()
    if not proxy_url:
        return
    client = ProxyClient(proxy_url)
    try:
        response = await client.resume_rollout(reason=reason)
        logger.info("resumed Dressage rollout after weight update: %s", response)
    finally:
        await client.close()


def _run_async(coro):
    """Run a coroutine from the synchronous Ray driver process."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("train_async_with_rollout_pause must run from a synchronous driver")


def _safe_update_weights(actor_model: Any, *, reason: str) -> Any:
    paused = False
    try:
        paused = bool(_run_async(_pause_proxy(reason)))
    except Exception:
        if _env_flag("DRESSAGE_PROXY_PAUSE_REQUIRED", True):
            raise
        logger.exception("failed to pause Dressage rollout; continuing because pause is not required")

    try:
        return actor_model.update_weights()
    finally:
        if paused:
            try:
                _run_async(_resume_proxy(reason))
            except Exception:
                if _env_flag("DRESSAGE_PROXY_PAUSE_REQUIRED", True):
                    raise
                logger.exception("failed to resume Dressage rollout after weight update")


def train(args):
    assert not args.colocate, "Colocation is not supported for async training."

    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    actor_cls = None
    if _env_flag("DRESSAGE_ENABLE_TRANSFER_QUEUE", False):
        from dressage.training.tq_hydration import (
            clear_tq_rollout_data,
            validate_tq_training_config,
        )
        from dressage.training.tq_megatron_actor import TQMegatronTrainRayActor

        validate_tq_training_config(args)
        actor_cls = TQMegatronTrainRayActor
    actor_model, critic_model = create_training_models(
        args,
        pgs,
        rollout_manager,
        actor_cls=actor_cls,
    )

    # Always push actor weights to rollout once weights are loaded. No rollout should
    # be active yet, but keeping the wrapper here makes the update path consistent.
    _safe_update_weights(actor_model, reason="initial_weight_update")

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
                if _env_flag("DRESSAGE_ENABLE_TRANSFER_QUEUE", False):
                    clear_tq_rollout_data(rollout_data_curr_ref)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        has_future_rollout = rollout_id + 1 < args.num_rollout
        if has_future_rollout and (rollout_id + 1) % args.update_weights_interval == 0:
            # Slime waits for the visible rollout future. Dressage partial async may
            # still have hidden background blackbox/proxy work, so pause the proxy too.
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            _safe_update_weights(actor_model, reason=f"weight_update_after_rollout_{rollout_id}")
        elif (rollout_id + 1) % args.update_weights_interval == 0:
            logger.info(
                "skipping actor weight update after final rollout %s; "
                "no future rollout will consume these weights",
                rollout_id,
            )

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    train(parse_args())
