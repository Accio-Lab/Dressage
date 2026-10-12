"""Native trajectory references, validation and exact source-key release."""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

from dressage.integrations.harbor.artifacts import (
    _trajectory_finalization_failures,
    validate_attempt,
)

from .execution import identity, write_artifact
from .trajectory_layout import inspect_trajectory_layouts


def validate_trajectory(
    segments, finalized, *, trial_id, instance_id, rewards, version
):
    """Validate resident tokens and reference structure without downloading TQ data.

    Real source bounds and numeric checks run at the training hydration boundary.
    Do not fabricate zero arrays to trick the upstream array-only validator.
    """
    _, layouts = inspect_trajectory_layouts(
        segments, finalized, trial_id=trial_id, instance_id=instance_id
    )
    if not layouts:
        failures, _, _, versions = validate_attempt(
            rewards=rewards,
            reward_key="reward",
            segments=segments,
            require_token_versions=True,
            expected_weight_version=version,
            require_trainable_tokens=True,
            expected_trajectory_id=trial_id,
            expected_instance_id=instance_id,
        )
        if failures:
            raise ValueError(
                "trajectory validation failed: " + ",".join(x.code for x in failures)
            )
        return versions
    if len(layouts) != len(segments):
        raise ValueError("mixed array/reference trajectory is not supported")
    failures = _trajectory_finalization_failures(
        segments, expected_trajectory_id=trial_id, expected_instance_id=instance_id
    )
    reward = rewards.get("reward")
    if failures or type(reward) not in (float, int) or not math.isfinite(reward):
        raise ValueError("invalid finalization or reward")
    trainable = 0
    for index, segment in enumerate(segments):
        tokens, mask, versions = (
            segment.get(k) for k in ("tokens", "full_loss_mask", "full_versions")
        )
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or not isinstance(mask, list)
            or len(mask) != len(tokens)
            or any(m not in (0, 1) for m in mask)
            or not isinstance(versions, list)
            or len(versions) != len(tokens)
        ):
            raise ValueError("invalid token/mask/version arrays")
        covered = bytearray(len(tokens))
        for f in layouts[index].fragments:
            end = f.target_start + f.length
            if end > len(tokens) or any(covered[f.target_start : end]):
                raise ValueError("overlapping/out-of-bounds TQ layout")
            covered[f.target_start : end] = b"\1" * f.length
        tito = segment["extra_info"].get("token_build_mode") == "tito"
        if any(not covered[i] and (tito or m) for i, m in enumerate(mask)):
            raise ValueError("TQ layout has uncovered required positions")
        if any(m and versions[i] != version for i, m in enumerate(mask)):
            raise ValueError("trainable token weight version mismatch")
        trainable += sum(mask)
    if not trainable:
        raise ValueError("trajectory has no trainable tokens")
    return [version]


async def release_contexts(policy, records, *, policy_instance, call):
    """Idempotent exact cleanup, after consumers have stopped using references."""
    limit = asyncio.Semaphore(8)

    async def release(record):
        if not (
            record.get("trajectory_retained")
            or record.get("trajectory_retention_intent")
        ):
            return
        context, execution = record["context_id"], record["execution_id"]
        async with limit:
            result = await call(
                lambda: policy.revoke(context, "release-trajectory", execution)
            )
            if (
                result.state != "CLOSED"
                or result.context_id != context
                or result.execution_id != execution
                or result.policy_instance_id != policy_instance
                or result.trajectory_retained
            ):
                raise RuntimeError("Policy did not prove trajectory release")

    results = await asyncio.gather(
        *(release(r) for r in records), return_exceptions=True
    )
    errors = [x for x in results if isinstance(x, BaseException)]
    if errors:
        raise RuntimeError(
            "trajectory cleanup incomplete; retain delivery manifest"
        ) from errors[0]
    refs = [ref for r in records for ref in (r.get("tq_refs") or [])]
    if refs:
        # Proxy discard logs some backend clear failures instead of propagating.
        # Verify exact owned keys; HTTP 200 alone is not a release receipt.
        await asyncio.to_thread(clear_source_keys, refs)


def clear_source_keys(refs):
    import ray
    import transfer_queue as tq

    ray.get_actor("TransferQueueController", namespace="transfer_queue")
    partitions = {}
    for ref in refs:
        partitions.setdefault(ref["partition"], set()).add(ref["key"])
    for partition, keys in partitions.items():
        remaining = set(tq.kv_list(partition_id=partition).get(partition, {})) & keys
        if remaining:
            tq.kv_clear(keys=sorted(remaining), partition_id=partition)
        if set(tq.kv_list(partition_id=partition).get(partition, {})) & keys:
            raise RuntimeError("owned TQ keys remain after release")


def delivery_path(step):
    return (
        Path(os.environ["ROLLOUT_DIAGNOSTICS_DIR"])
        / "rollout"
        / identity(os.environ["DRESSAGE_DISTROLL_RUN_ID"])
        / f"step-{step}"
        / "delivery.json"
    )


async def _release_training(step):
    from .config import remote_rollout_settings_from_env
    from .execution import ExecutionQueue

    path = delivery_path(step)
    manifest = json.loads(path.read_text())
    settings = remote_rollout_settings_from_env()
    if manifest["run_id"] != settings.run_id or manifest["step"] != step:
        raise RuntimeError("delivery identity mismatch")
    if manifest["state"] == "released":
        return
    if manifest["state"] != "handed_off":
        raise RuntimeError("delivery not handed to training")
    # Reuse exactly the transport retry budget, without acquiring a CPU owner.
    queue = ExecutionQueue(settings, root=path.parent, proxy=None, materialize=None)
    try:
        await release_contexts(
            queue.policy,
            manifest["contexts"],
            policy_instance=manifest["policy_instance"],
            call=queue.call,
        )
        manifest["state"] = "released"
        write_artifact(path, json.dumps(manifest).encode())
    finally:
        await queue.close()


def release_training_trajectories(step):
    """Driver calls only after every training rank returns successfully."""
    asyncio.run(_release_training(step))
