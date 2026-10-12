from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any

import httpx
from agentic_rollout_contract.canonical import canonical_digest
from agentic_rollout_contract.models import ExecutionSpec

from .config import remote_rollout_settings_from_env
from .execution import ExecutionQueue, identity, write_artifact
from .trajectory_delivery import release_contexts, validate_trajectory

logger = logging.getLogger(__name__)
_active_runs: set[str] = set()
_active_runs_lock = threading.Lock()

try:
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.async_utils import run
except ImportError:
    RolloutFnTrainOutput = None
    run = asyncio.run


@dataclass(frozen=True)
class LiveGroupAssessment:
    group_count: int
    live_groups: int
    zero_grad_groups: int
    live_ratio: float


def generate_remote_harbor_sync(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("remote Harbor evaluation is not implemented")
    run_id = os.environ["DRESSAGE_DISTROLL_RUN_ID"]
    with _active_runs_lock:
        if run_id in _active_runs:
            raise RuntimeError(
                "a rollout owner is already executing this run in this process"
            )
        _active_runs.add(run_id)
    try:
        groups, metrics = run(_run_remote_sync(args, rollout_id, data_buffer))
    finally:
        with _active_runs_lock:
            _active_runs.remove(run_id)
    if RolloutFnTrainOutput is None:
        return groups
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)


def _source_state(source):
    # In-memory rollback of a failed candidate fetch, not a restart checkpoint.
    # Include shuffle order and buffered prompts, not only sample_offset.
    if not hasattr(source, "sample_offset") or not hasattr(source, "buffer"):
        raise TypeError(
            "DistRoll requires the Dressage text-first source cursor and buffer"
        )
    return copy.deepcopy(
        {key: value for key, value in vars(source).items() if key != "args"}
    )


def _restore_source(source, state):
    for key in tuple(vars(source)):
        if key != "args":
            delattr(source, key)
    vars(source).update(copy.deepcopy(state))


def _placeholder(template, slot, group_id):
    from dressage.rollout.fully_async_rollout import _mark_no_grad_failed

    sample = copy.deepcopy(template)
    _metadata(sample).update(
        instance_id=group_id, parent_traj_id=f"{group_id}:zero-grad:{slot}"
    )
    sample.rollout_id = getattr(template, "index", None)
    sample = _mark_no_grad_failed([sample], RuntimeError("failed logical slot"))[0]
    _metadata(sample).update(
        harbor_partial_slot_zero_grad=True,
        harbor_attempt_ordinal=slot,
        harbor_group_id=group_id,
        harbor_training_committed=False,
    )
    sample.reward = {"reward": 0.0}
    return sample


def _qualified(outcomes, minimum):
    successes = [item for item in outcomes if item is not None]
    if len(successes) < minimum:
        return False
    rewards = [item["reward"] for item in successes]
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    return (
        math.isfinite(variance)
        and variance > 1e-12
        and any(value > 0 for value in rewards)
    )


async def _run_remote_sync(args, rollout_id, data_buffer):
    from dressage.proxy.proxy_client import ProxyClient
    from dressage.rollout.multi_segment import (
        compute_multi_segment_metrics,
        expand_segments_to_samples,
    )

    settings = remote_rollout_settings_from_env()
    if settings.diagnostics_dir is None:
        raise ValueError(
            "DistRoll requires ROLLOUT_DIAGNOSTICS_DIR for durable RL artifacts"
        )
    target, width = int(args.rollout_batch_size), int(args.n_samples_per_prompt)
    dynamic = bool(getattr(args, "dynamic_sampling_filter_path", None))
    filter_path = getattr(args, "dynamic_sampling_filter_path", None)
    supported_filters = {
        "dressage_distroll_gpu_adapter.dynamic_sampling.check_harbor_complete_reward_nonzero_std",
        "dressage_distroll_gpu_adapter.dynamic_sampling.check_harbor_quorum_reward_nonzero_std",
    }
    if dynamic and filter_path not in supported_filters:
        raise ValueError(
            "DistRoll currently supports the project complete/quorum reward filters only"
        )
    if (
        filter_path
        and "check_harbor_complete_" in filter_path
        and settings.group_min_success_ratio != 1
    ):
        raise ValueError("complete-group filter requires a full-width quorum")
    candidates = (
        int(os.environ.get("DYNAMIC_SAMPLING_MAX_GROUPS", target))
        if dynamic
        else target
    )
    if candidates < target or target < 1 or width < 1:
        raise ValueError("candidate budget must cover a positive target")
    if settings.dynamic_sampling_early_stop and not dynamic:
        raise ValueError("early stop requires dynamic sampling")
    minimum = math.ceil(width * settings.group_min_success_ratio)
    root = (
        settings.diagnostics_dir
        / "rollout"
        / identity(settings.run_id)
        / f"step-{rollout_id}"
    )
    proxy = ProxyClient(settings.dressage_proxy_url)

    async def materialize(view, context):
        receipt = view.result
        finalized = await executor.call(
            lambda: proxy.finalize_session(
                receipt.trial_id, instance_id=context["instance_id"]
            ),
            attempts=settings.proxy_retry_attempts,
        )
        payload = await executor.call(
            lambda: proxy.read_trajectory(
                trajectory_id=receipt.trial_id,
                instance_id=context["instance_id"],
                drain=False,
            ),
            attempts=settings.proxy_retry_attempts,
        )
        segments = list(payload.get("data") or [])
        versions = validate_trajectory(
            segments,
            finalized,
            trial_id=receipt.trial_id,
            instance_id=context["instance_id"],
            rewards=receipt.rewards,
            version=context["expected_weight_version"],
        )
        return {
            "execution_id": view.execution_id,
            "task_ref": view.spec.task_ref,
            "result_digest": view.result_digest,
            "trial_id": receipt.trial_id,
            "trial_name": receipt.trial_name,
            "rewards": receipt.rewards,
            "versions": versions,
            "context": context,
            "segments": segments,
            "tq_refs": (finalized.get("tq_trajectory_ref") or {}).get("refs", []),
        }

    executor = ExecutionQueue(settings, root=root, proxy=proxy, materialize=materialize)
    handed_off = False
    try:
        await executor.open()
        capabilities = await executor.call(
            proxy.capabilities, attempts=settings.proxy_retry_attempts
        )
        version = str(capabilities.get("current_weight_version") or "")
        if not version:
            raise RuntimeError("Proxy weight version is missing")
        manifest = {
            "run_id": settings.run_id,
            "step": rollout_id,
            "version": version,
            "target": target,
            "width": width,
            "candidates": candidates,
            "minimum": minimum,
            "settings": settings.model_dump(mode="json"),
            "temperature": float(getattr(args, "rollout_temperature", 1)),
            "top_p": float(getattr(args, "rollout_top_p", 1)),
            "token_cap": int(args.max_tokens_per_gpu)
            * int(getattr(args, "context_parallel_size", 1)),
            "context_cap": int(os.environ["ROLLOUT_MAX_CONTEXT_LEN"]),
        }
        if manifest["context_cap"] > manifest["token_cap"]:
            raise RuntimeError("context exceeds trajectory token cap")
        # The source cursor lives in Slime. Do not recover a previous rollout batch.
        before = _source_state(data_buffer)
        try:
            templates = data_buffer.get_samples(candidates)
            if len(templates) != candidates or any(len(g) != width for g in templates):
                raise RuntimeError("data source returned incomplete candidate groups")
            plan_digest = canonical_digest(
                {
                    "manifest": manifest,
                    "tasks": [
                        [_metadata(t).get("harbor_task_ref") for t in g]
                        for g in templates
                    ],
                }
            )
        except BaseException:
            _restore_source(data_buffer, before)
            raise
        await executor.initialize_step(rollout_id, manifest, plan_digest)

        async def operation(group, slot):
            task_ref = str(
                _metadata(templates[group][slot]).get("harbor_task_ref") or ""
            )
            if not task_ref:
                raise ValueError("sample metadata.harbor_task_ref is required")
            for revision in range(settings.group_max_retries + 1):
                if executor.stop.is_set():
                    return None
                execution_id = identity(
                    settings.run_id, rollout_id, group, slot, revision
                )
                kwargs = {
                    "max_turns": settings.agent_max_turns,
                    "accept_max_turns": settings.accept_max_turns,
                    "version": settings.claude_code_version,
                    "permission_mode": "bypassPermissions",
                    "claude_context_window_tokens": manifest["context_cap"],
                    "claude_max_output_tokens": args.rollout_max_response_len,
                    "claude_file_read_max_output_tokens": settings.claude_file_read_max_output_tokens,
                }
                if settings.agent_timeout_sec:
                    kwargs["agent_timeout_sec"] = settings.agent_timeout_sec
                if settings.claude_auto_compact_pct:
                    kwargs["claude_auto_compact_pct"] = settings.claude_auto_compact_pct
                spec = ExecutionSpec(
                    execution_id=execution_id,
                    task_ref=task_ref,
                    model_name=settings.model_name,
                    agent_kwargs=kwargs,
                )
                context = {
                    "instance_id": identity(settings.run_id, rollout_id, group),
                    "expected_weight_version": version,
                    "model_override": settings.model_name,
                    "sampling_temperature": manifest["temperature"],
                    "sampling_top_p": manifest["top_p"],
                    "sampling_seed_base": int(identity(execution_id, "seed")[:8], 16)
                    % (2**31 - 1),
                }
                artifact = await executor.execute(
                    group=group,
                    slot=slot,
                    revision=revision,
                    spec=spec,
                    context=context,
                )
                if artifact is not None:
                    data = executor.read_trajectory(artifact)
                    return {
                        **artifact,
                        "execution_id": execution_id,
                        "revision": revision,
                        "reward": data["rewards"]["reward"],
                    }
                record = executor.payload["executions"].get(execution_id, {})
                if record.get("failure_class") in {
                    "ADMISSION_INVARIANT",
                    "INFRA_CONFIGURATION",
                }:
                    raise RuntimeError(
                        "execution failed an infrastructure integrity invariant"
                    )
                if record.get("cancelled") or record.get("retryable") is False:
                    return None
            return None

        def group_ready(group, finished):
            outcomes = [finished[f"{group}:{slot}"] for slot in range(width)]
            if (
                settings.failed_group_policy == "abort_batch"
                and sum(item is not None for item in outcomes) < minimum
            ):
                raise RuntimeError("group failed the configured success quorum")
            if not dynamic:
                return True
            return _qualified(outcomes, minimum)

        accepted = await executor.run_slots(
            [(group, slot) for group in range(candidates) for slot in range(width)],
            operation,
            group_ready,
            width=width,
            target=target,
            early_stop=settings.dynamic_sampling_early_stop,
        )
        outcomes = executor.payload["completed_slots"]

        def successes(group):
            return sum(
                outcomes.get(f"{group}:{slot}") is not None for slot in range(width)
            )

        if not settings.dynamic_sampling_early_stop:
            accepted.sort(key=lambda group: (successes(group) != width, group))
        selected = accepted[:target]
        if not selected:
            raise RuntimeError("candidate budget produced no qualified training group")
        await _wait_proxy_generations_idle(settings)
        completed = []
        for group in selected:
            group_id = identity(settings.run_id, rollout_id, group)
            samples, receipts = [], []
            group_valid = successes(group) >= minimum
            for slot in range(width):
                artifact = outcomes.get(f"{group}:{slot}")
                if artifact is None or not group_valid:
                    samples.append(_placeholder(templates[group][slot], slot, group_id))
                    continue
                data = executor.read_trajectory(artifact)
                receipts.append(
                    {
                        "slot": slot,
                        "execution_id": data["execution_id"],
                        "result_digest": data["result_digest"],
                        "artifact_digest": artifact["digest"],
                    }
                )
                expanded = expand_segments_to_samples(
                    templates[group][slot],
                    data["segments"],
                    args=args,
                    session_id=data["trial_id"],
                    instance_id=data["context"]["instance_id"],
                )
                if not any(
                    sample.response_length > 0 and any(sample.loss_mask or [])
                    for sample in expanded
                ):
                    raise RuntimeError(
                        "successful trajectory lost all trainable response tokens"
                    )
                for index, sample in enumerate(expanded):
                    _metadata(sample).update(
                        instance_id=data["context"]["instance_id"],
                        harbor_group_id=group_id,
                        harbor_trial_id=data["trial_id"],
                        harbor_trial_name=data["trial_name"],
                        harbor_task_ref=data["task_ref"],
                        harbor_rewards=data["rewards"],
                        harbor_weight_versions=data["versions"],
                        harbor_attempt_ordinal=slot,
                        harbor_attempt_id=data["execution_id"],
                        harbor_attempt_revision=artifact["revision"],
                        harbor_execution_result_digest=data["result_digest"],
                    )
                    sample.reward = (
                        data["rewards"]
                        if index == len(expanded) - 1
                        else {"reward": 0.0}
                    )
                samples.extend(expanded)
            commit_digest = canonical_digest(
                {
                    "group_id": group_id,
                    "width": width,
                    "minimum": minimum,
                    "receipts": receipts,
                }
            )
            for sample in samples:
                removed = bool(getattr(sample, "remove_sample", False))
                _metadata(sample).update(
                    harbor_group_id=group_id,
                    harbor_group_result_digest=commit_digest,
                    harbor_selected_attempt_ids=[
                        item["execution_id"] for item in receipts
                    ],
                    harbor_target_width=width,
                    harbor_minimum_successful_attempts=minimum,
                    harbor_group_zero_grad=not group_valid,
                    harbor_training_committed=not removed,
                    harbor_group_state="COMMITTED" if group_valid else "ZERO_GRAD",
                    harbor_training_eligible=not removed,
                )
            completed.append(samples)
        started_groups = {
            item["group"]
            for item in executor.payload["executions"].values()
            if item.get("activated")
        }
        # The barrier has proved every intent terminal. A permanent CPU cancel
        # with no activation context proves late activation cannot happen.
        data_buffer.add_samples(
            [
                templates[group]
                for group in range(candidates)
                if group not in started_groups
            ]
        )
        flat = [sample for group in completed for sample in group]
        metrics = compute_multi_segment_metrics(flat)
        metrics.update(
            {
                "dynamic_sampling/candidates": float(candidates),
                "dynamic_sampling/started_candidates": float(len(started_groups)),
                "dynamic_sampling/accepted": float(len(completed)),
                "dynamic_sampling/target": float(target),
                "dynamic_sampling/accepted_full_groups": float(
                    sum(successes(group) == width for group in selected)
                ),
                "dynamic_sampling/accepted_partial_groups": float(
                    sum(successes(group) < width for group in selected)
                ),
                "remote_harbor/global_execution_limit": float(
                    settings.max_concurrent_executions
                ),
                "remote_harbor/executions": float(len(executor.payload["executions"])),
                "remote_harbor/slot_retries": float(
                    sum(
                        item["revision"] > 0
                        for item in executor.payload["executions"].values()
                    )
                ),
            }
        )
        assessment = _assess_live_groups(completed)
        metrics.update(
            {
                "remote_harbor/groups": float(assessment.group_count),
                "remote_harbor/live_groups": float(assessment.live_groups),
                "remote_harbor/live_group_ratio": assessment.live_ratio,
                "remote_harbor/zero_grad_groups": float(assessment.zero_grad_groups),
                "dynamic_sampling/qualified_groups": float(len(accepted)),
                "dynamic_sampling/rejected": float(candidates - len(accepted)),
            }
        )
        diagnostics = {
            "manifest": manifest,
            "selected_groups": selected,
            "metrics": metrics,
            "live_ratio": assessment.live_ratio,
            "executions": executor.payload["executions"],
        }
        write_artifact(
            root / "diagnostic.json",
            json.dumps(diagnostics, allow_nan=False, indent=2).encode(),
        )
        _apply_live_group_gate(
            assessment,
            min_live_group_ratio=settings.min_live_group_ratio,
        )
        batch_digest = canonical_digest(
            [group[0].metadata["harbor_group_result_digest"] for group in completed]
        )
        records = [
            {"execution_id": key, **record}
            for key, record in executor.payload["executions"].items()
        ]
        selected_ids = {
            sample.metadata["harbor_attempt_id"]
            for group in completed
            for sample in group
            if not sample.remove_sample
        }
        await release_contexts(
            executor.policy,
            [r for r in records if r["execution_id"] not in selected_ids],
            policy_instance=executor.policy_instance,
            call=executor.call,
        )
        delivery = {
            "run_id": settings.run_id,
            "step": rollout_id,
            "policy_instance": executor.policy_instance,
            "contexts": [r for r in records if r["execution_id"] in selected_ids],
            "state": "prepared",
        }
        # No tensor values in this journal; it is for exact cleanup, not recovery.
        delivery["contexts"] = [
            {
                k: r.get(k)
                for k in (
                    "context_id",
                    "execution_id",
                    "trajectory_retained",
                    "tq_refs",
                )
            }
            for r in delivery["contexts"]
        ]
        write_artifact(root / "delivery.json", json.dumps(delivery).encode())
        with executor.commit_guard():
            if await executor._check_stop_request():
                raise RuntimeError("operator stopped rollout before commit")
            async with executor.lock:
                executor.payload.update(
                    committed=True, selected_groups=selected, batch_digest=batch_digest
                )
                await executor.save()
            write_artifact(
                root / "commit.json",
                json.dumps(
                    {
                        "step": rollout_id,
                        "batch_digest": batch_digest,
                        "plan_digest": plan_digest,
                        "selected_groups": selected,
                        "recoverable": False,
                    }
                ).encode(),
            )
        delivery["state"] = "handed_off"
        write_artifact(root / "delivery.json", json.dumps(delivery).encode())
        handed_off = True
        args.global_batch_size = len(completed) * width
        args.v3_committed_step = rollout_id
        return completed, metrics
    except BaseException as exc:
        try:
            await asyncio.to_thread(
                write_artifact,
                root / "diagnostic-failure.json",
                json.dumps(
                    {
                        "run_id": settings.run_id,
                        "step": rollout_id,
                        "error_type": type(exc).__name__,
                        "state": executor.payload,
                    },
                    allow_nan=False,
                ).encode(),
            )
        except (OSError, TypeError, ValueError):
            logger.error(
                "failed to persist RL failure diagnostic; retain Control journal and Ray logs"
            )
        raise
    finally:
        try:
            if not handed_off and executor.policy_instance:
                await release_contexts(
                    executor.policy,
                    [
                        {"execution_id": key, **record}
                        for key, record in executor.payload.get(
                            "executions", {}
                        ).items()
                    ],
                    policy_instance=executor.policy_instance,
                    call=executor.call,
                )
        finally:
            executor.artifacts.clear()
            await executor.close()
            await proxy.close()


async def _wait_proxy_generations_idle(
    settings: Any, *, client: httpx.AsyncClient | None = None
) -> None:
    """Do not cross the optimizer barrier while Proxy still owns generations."""

    owned_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=5.0, trust_env=False)
    try:
        deadline = asyncio.get_running_loop().time() + 10.0
        while True:
            response = await client.get(f"{settings.dressage_proxy_url}/health")
            response.raise_for_status()
            status = response.json().get("rollout_pause")
            if not isinstance(status, dict):
                raise TypeError("Dressage Proxy generation status is unavailable")
            active = status.get("active_sglang_generations")
            request_ids = status.get("active_request_ids")
            if not isinstance(active, int) or not isinstance(request_ids, list):
                raise TypeError("Dressage Proxy generation status is malformed")
            if active == 0 and not request_ids:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"Dressage Proxy still owns {active} generation(s) "
                    "after speculative cancellation"
                )
            await asyncio.sleep(0.5)
    finally:
        if owned_client:
            await client.aclose()


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    return metadata


def _group_is_live(group: list[Any]) -> bool:
    return any(not bool(getattr(sample, "remove_sample", False)) for sample in group)


def _assess_live_groups(groups: list[list[Any]]) -> LiveGroupAssessment:
    if not groups:
        raise RuntimeError("remote Harbor rollout returned no groups")
    live_groups = sum(1 for group in groups if _group_is_live(group))
    zero_grad_groups = sum(
        1
        for group in groups
        if group and bool(_metadata(group[0]).get("harbor_group_zero_grad"))
    )
    return LiveGroupAssessment(
        group_count=len(groups),
        live_groups=live_groups,
        zero_grad_groups=zero_grad_groups,
        live_ratio=live_groups / len(groups),
    )


def _apply_live_group_gate(
    assessment: LiveGroupAssessment,
    *,
    min_live_group_ratio: float,
) -> None:
    if assessment.live_ratio < min_live_group_ratio:
        raise RuntimeError(
            f"remote Harbor live group ratio {assessment.live_ratio:.3f} is below "
            f"configured minimum {min_live_group_ratio:.3f}"
        )
    if assessment.live_groups == 0:
        raise RuntimeError(
            "remote Harbor rollout produced no trainable groups; "
            "refusing an empty optimizer update"
        )
