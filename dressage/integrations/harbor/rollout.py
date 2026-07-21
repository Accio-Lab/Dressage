"""Synchronous Harbor rollout entry point for slime online training.

Harbor owns task, environment, agent, verifier, retry, and trial scheduling.
This module only batches immutable data-source specs into temporary jobs,
attaches :class:`DressageHarborPlugin`, and converts committed trajectory
bundles into slime Samples.  Harbor imports stay lazy so the base Dressage
package remains importable on Python 3.10.
"""

from __future__ import annotations

import atexit
import asyncio
import copy
from dataclasses import dataclass, field, replace
import hashlib
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, Mapping, Sequence
import uuid

from dressage.integrations.harbor.artifacts import (
    HarborArtifactStore,
    HarborTrajectoryBundle,
)
from dressage.integrations.harbor.plugin import (
    DressageHarborPlugin,
    TrialBinding,
)
from dressage.proxy.proxy_client import ProxyClient
from dressage.rollout.fully_async_rollout import (
    _allow_empty_train_batch,
    _group_has_trainable_tokens,
    _mark_no_grad_failed,
)
from dressage.rollout.multi_segment import (
    compute_multi_segment_metrics,
    expand_segments_to_samples,
)


logger = logging.getLogger(__name__)

_REQUIRED_REWARD_HOOK = (
    "dressage.training.reward_post_process.reward_post_process"
)
_REQUIRED_CONVERT_HOOK = (
    "dressage.rollout.convert_samples.convert_samples_to_train_data"
)
_BWRAP_ENVIRONMENT_IMPORT_PATH = (
    "dressage.integrations.harbor.environment:DressageEnvironment"
)


try:
    from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
    from slime.utils.async_utils import run
except ImportError:  # pragma: no cover - import-only compatibility path
    RolloutFnEvalOutput = None  # type: ignore[assignment]
    RolloutFnTrainOutput = None  # type: ignore[assignment]

    def run(awaitable: Awaitable[Any]) -> Any:  # type: ignore[no-redef]
        return asyncio.run(awaitable)


class HarborRolloutError(RuntimeError):
    """A Harbor batch cannot safely be used for training."""


@dataclass(frozen=True)
class _GroupWork:
    position: int
    templates: list[Any]
    spec: Any
    spec_id: str
    group_index: int
    instance_id: str


@dataclass
class _GroupOutcome:
    work: _GroupWork
    samples: list[Any] | None = None
    error: BaseException | None = None
    versions: tuple[str, ...] = ()
    bundles: list[HarborTrajectoryBundle] = field(default_factory=list)
    final_keys: list[tuple[str, str]] = field(default_factory=list)
    routing_tasks: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.samples is not None


_ROOT_LEASES: dict[str, Any] = {}


async def _default_create_job(config: Any) -> Any:
    from dressage.integrations.harbor.compat import require_harbor_runtime

    require_harbor_runtime()
    from harbor.job import Job

    return await Job.create(config)


_CREATE_JOB: Callable[[Any], Awaitable[Any]] = _default_create_job
_PLUGIN_FACTORY: Callable[..., DressageHarborPlugin] = DressageHarborPlugin


def _plugin_routing_summary(plugin: Any) -> Mapping[str, int | str]:
    return plugin.routing_summary


def _plugin_routing_tasks(plugin: Any) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in plugin.resolved_task_network_classes.items()
    }


def _gateway_runtime() -> Any:
    from dressage.integrations.harbor.gateway import GatewayRuntime

    return GatewayRuntime.get()


async def _ensure_root_gateway_lease(config: Any) -> tuple[Any, Any]:
    """Hold one process-level lease so an ephemeral port survives rounds."""

    runtime = _gateway_runtime()
    key = str(config.gateway_fingerprint())
    lease = _ROOT_LEASES.get(key)
    if lease is None:
        lease = await runtime.acquire(config)
        _ROOT_LEASES[key] = lease
    return runtime, lease


async def close_harbor_rollout_runtime() -> None:
    """Release process leases; intended for orderly shutdown and tests."""

    leases = list(_ROOT_LEASES.values())
    _ROOT_LEASES.clear()
    for lease in leases:
        release = getattr(lease, "release", None) or getattr(lease, "aclose", None)
        if release is not None:
            try:
                await release()
            except Exception:
                logger.warning("failed to release Harbor root Gateway lease", exc_info=True)


def _close_harbor_rollout_runtime_at_exit() -> None:
    if not _ROOT_LEASES:
        return
    try:
        asyncio.run(close_harbor_rollout_runtime())
    except Exception:
        logger.warning("failed to stop Harbor rollout runtime at process exit", exc_info=True)


def _backend_headers(config: Any) -> dict[str, str]:
    required = getattr(config, "execution_mode", None) == "training"
    return config.backend.service_headers(os.environ, required=required)


async def _read_proxy_capabilities(config: Any) -> dict[str, Any]:
    client = ProxyClient(
        str(config.backend.dressage_proxy_url),
        default_headers=_backend_headers(config),
        verify=bool(config.backend.verify_tls),
    )
    try:
        return await client.capabilities()
    finally:
        await client.close()


def _same_model_identity(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return True
    left_text = str(left).rstrip("/")
    right_text = str(right).rstrip("/")
    if left_text == right_text:
        return True
    left_path = Path(left_text).expanduser()
    right_path = Path(right_text).expanduser()
    if left_path.exists() and right_path.exists():
        return left_path.resolve() == right_path.resolve()
    return False


def _validate_proxy_capabilities(
    args: Any,
    config: Any,
    capabilities: Mapping[str, Any],
) -> None:
    if capabilities.get("schema_version") != "dressage.proxy.integration/v1":
        raise HarborRolloutError(
            "Dressage Proxy does not expose the supported integration capability schema"
        )
    training = config.training
    if training.require_single_weight_version:
        if not capabilities.get("record_token_versions"):
            raise HarborRolloutError(
                "Harbor training requires Dressage Proxy record_token_versions=true"
            )
        if not capabilities.get("supports_expected_version"):
            raise HarborRolloutError(
                "Dressage Proxy does not support expected weight-version routing"
            )
        if not capabilities.get("weight_version_authoritative"):
            raise HarborRolloutError(
                "Harbor training requires an authoritative SGLang worker "
                "weight-version snapshot"
            )
        if not capabilities.get("weight_versions_consistent"):
            raise HarborRolloutError(
                "SGLang rollout workers do not report one consistent weight version"
            )
        current_version = capabilities.get("current_weight_version")
        if current_version is None or not str(current_version).strip():
            raise HarborRolloutError(
                "Dressage Proxy returned an empty authoritative weight version"
            )
    if capabilities.get("partial_rollout"):
        raise HarborRolloutError(
            "Harbor v1 is strictly synchronous and requires partial_rollout=false"
        )
    if not capabilities.get("chat_template_fingerprint"):
        raise HarborRolloutError(
            "Dressage Proxy must report a tokenizer chat-template fingerprint"
        )
    expected_tokenizer = (
        getattr(args, "tokenizer_path", None)
        or getattr(args, "hf_checkpoint", None)
    )
    actual_tokenizer = capabilities.get("tokenizer_id")
    if expected_tokenizer and not _same_model_identity(
        expected_tokenizer, actual_tokenizer
    ):
        raise HarborRolloutError(
            "Dressage Proxy tokenizer_id does not match slime's training tokenizer: "
            f"{actual_tokenizer!r} != {expected_tokenizer!r}"
        )


def _validate_slime_contract(args: Any, config: Any) -> None:
    configured_reward_key = str(config.training.reward_key)
    slime_reward_key = getattr(args, "reward_key", None)
    if slime_reward_key != configured_reward_key:
        raise HarborRolloutError(
            "slime reward_key must exactly match Harbor training.reward_key: "
            f"{slime_reward_key!r} != {configured_reward_key!r}"
        )
    reward_hook = getattr(args, "custom_reward_post_process_path", None)
    if reward_hook != _REQUIRED_REWARD_HOOK:
        raise HarborRolloutError(
            "Harbor multi-segment training requires "
            f"custom_reward_post_process_path={_REQUIRED_REWARD_HOOK!r}"
        )
    convert_hook = getattr(args, "custom_convert_samples_to_train_data_path", None)
    if convert_hook != _REQUIRED_CONVERT_HOOK:
        raise HarborRolloutError(
            "Harbor multi-segment training requires "
            f"custom_convert_samples_to_train_data_path={_REQUIRED_CONVERT_HOOK!r}"
        )


def _runtime_incarnation(capabilities: Mapping[str, Any]) -> str:
    stable = {
        key: capabilities.get(key)
        for key in (
            "schema_version",
            "token_build_mode",
            "token_build_model",
            "tokenizer_id",
            "chat_template_fingerprint",
            "record_token_versions",
            "partial_rollout",
        )
    }
    encoded = repr(sorted(stable.items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata(sample: Any) -> dict[str, Any]:
    value = getattr(sample, "metadata", None)
    if not isinstance(value, dict):
        value = {}
        sample.metadata = value
    return value


def _reward_distribution_metrics(
    groups: Sequence[list[Any]],
    *,
    reward_key: str,
) -> dict[str, float]:
    """Summarize terminal trajectory rewards without changing the batch.

    Multi-segment trajectories store the verifier reward only on their final
    segment.  Select the highest ``segment_index`` for each trajectory so the
    success and group-variance metrics count physical attempts, not segments.
    Failed samples marked for removal are excluded from both denominators.
    """

    grouped_rewards: list[list[float]] = []
    for group in groups:
        anchors: dict[tuple[str, object], Any] = {}
        for position, sample in enumerate(group):
            if getattr(sample, "remove_sample", False):
                continue
            metadata = _metadata(sample)
            parent = metadata.get("parent_traj_id")
            identity: tuple[str, object]
            if parent is None:
                identity = ("sample", position)
            else:
                identity = ("trajectory", str(parent))
            current = anchors.get(identity)
            if current is None or int(metadata.get("segment_index", 0)) > int(
                _metadata(current).get("segment_index", 0)
            ):
                anchors[identity] = sample

        rewards: list[float] = []
        for sample in anchors.values():
            reward = getattr(sample, "reward", None)
            if isinstance(reward, Mapping):
                reward = reward.get(reward_key)
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                continue
            value = float(reward)
            if math.isfinite(value):
                rewards.append(value)
        if rewards:
            grouped_rewards.append(rewards)

    rewards = [reward for group in grouped_rewards for reward in group]
    zero_std_groups = sum(
        1 for group in grouped_rewards if len(set(group)) == 1
    )
    return {
        "harbor/reward_success_rate": (
            sum(reward > 0.0 for reward in rewards) / len(rewards)
            if rewards
            else 0.0
        ),
        "harbor/zero_std_groups": float(zero_std_groups),
        "harbor/zero_std_ratio": (
            zero_std_groups / len(grouped_rewards) if grouped_rewards else 0.0
        ),
    }


def _resolve_work(data_source: Any, groups: Sequence[list[Any]]) -> list[_GroupWork]:
    work: list[_GroupWork] = []
    for position, group in enumerate(groups):
        if not group:
            raise HarborRolloutError(f"Harbor source group {position} is empty")
        first_meta = _metadata(group[0])
        spec_id = str(first_meta.get("harbor_spec_id") or "")
        if not spec_id:
            raise HarborRolloutError(
                f"Harbor source group {position} has no harbor_spec_id"
            )
        if any(str(_metadata(sample).get("harbor_spec_id") or "") != spec_id for sample in group):
            raise HarborRolloutError(
                f"Harbor source group {position} mixes prompt specs"
            )
        slots = [
            _metadata(sample).get("harbor_attempt_slot") for sample in group
        ]
        if slots != list(range(len(group))):
            raise HarborRolloutError(
                f"Harbor source group {position} has invalid attempt slots {slots!r}"
            )
        group_indices = {getattr(sample, "group_index", None) for sample in group}
        if len(group_indices) != 1 or None in group_indices:
            raise HarborRolloutError(
                f"Harbor source group {position} has inconsistent group_index"
            )
        instance_ids = {
            str(_metadata(sample).get("harbor_instance_id") or "")
            for sample in group
        }
        if len(instance_ids) != 1 or "" in instance_ids:
            raise HarborRolloutError(
                f"Harbor source group {position} has inconsistent instance identity"
            )
        resolver = getattr(data_source, "resolve_spec", None)
        if not callable(resolver):
            raise HarborRolloutError("data source does not expose resolve_spec(spec_id)")
        spec = resolver(spec_id)
        work.append(
            _GroupWork(
                position=position,
                templates=list(group),
                spec=spec,
                spec_id=spec_id,
                group_index=int(next(iter(group_indices))),
                instance_id=next(iter(instance_ids)),
            )
        )
    return work


def _partition_work(work: Sequence[_GroupWork]) -> list[list[_GroupWork]]:
    partitions: dict[str, list[_GroupWork]] = {}
    for item in work:
        fingerprint = str(getattr(item.spec, "runtime_fingerprint", ""))
        if not fingerprint:
            raise HarborRolloutError(
                f"Harbor spec {item.spec_id!r} has no runtime fingerprint"
            )
        partitions.setdefault(fingerprint, []).append(item)
    return list(partitions.values())


def _copy_model(value: Any) -> Any:
    copier = getattr(value, "model_copy", None)
    return copier(deep=True) if callable(copier) else copy.deepcopy(value)


def _validated_model_update(model: Any, **updates: Any) -> Any:
    fields = getattr(type(model), "model_fields", None)
    validator = getattr(type(model), "model_validate", None)
    if isinstance(fields, Mapping) and callable(validator):
        values = {
            name: _copy_model(getattr(model, name))
            for name in fields
            if hasattr(model, name)
        }
        values.update(updates)
        return validator(values)
    clone = _copy_model(model)
    for name, value in updates.items():
        setattr(clone, name, value)
    return clone


def _temporary_task_name(task: Any, index: int) -> str:
    name = getattr(task, "name", None)
    if name:
        return str(name)
    path = getattr(task, "path", None)
    if path:
        return str(path)
    package = getattr(task, "task", None)
    name = getattr(package, "name", None)
    if name:
        return str(name)
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("instance_id"):
        return str(metadata["instance_id"])
    return f"task[{index}]"


def _validate_temporary_job_task_sources(job_config: Any) -> None:
    """Reject Dataset identities on a Job whose tasks are now direct inputs.

    Harbor resolves metrics by Dataset source.  The synchronous executor clears
    ``datasets`` after resolving its selected tasks, so each runtime copy must
    use Harbor's ``adhoc`` namespace.  Retaining the old source makes Harbor
    create an empty metric bucket and fail its END progress hook.
    """

    if list(getattr(job_config, "datasets", None) or []):
        return
    for index, task in enumerate(getattr(job_config, "tasks", None) or []):
        source = getattr(task, "source", None)
        if source is not None:
            name = _temporary_task_name(task, index)
            raise HarborRolloutError(
                f"temporary direct Harbor task {name!r} retains Dataset "
                f"source {source!r}; expected source=None for adhoc metrics"
            )


def _safe_job_component(value: Any, *, limit: int = 28) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-_")
    return (text or "harbor")[:limit]


def _partition_route_demand(
    partition: Sequence[_GroupWork],
    *,
    max_active_routes: int,
) -> int:
    """Maximum simultaneous physical Trials this temporary Job may start."""

    if max_active_routes <= 0:
        raise HarborRolloutError("gateway max_active_routes must be positive")
    total_trials = sum(len(item.templates) for item in partition)
    if total_trials <= 0:
        raise HarborRolloutError("Harbor partition has no physical trials")
    return min(total_trials, max_active_routes)


def _temporary_job_config(
    *,
    source_job_config: Any,
    partition: Sequence[_GroupWork],
    config: Any,
    run_id: str,
    rollout_id: int,
    retry_round: int,
) -> Any:
    if not partition:
        raise ValueError("cannot create a Harbor Job for an empty partition")
    attempts = len(partition[0].templates)
    if any(len(item.templates) != attempts for item in partition):
        raise HarborRolloutError("Harbor groups in one partition have different sizes")
    agent = _copy_model(partition[0].spec.agent_config)
    if any(
        str(item.spec.runtime_fingerprint)
        != str(partition[0].spec.runtime_fingerprint)
        for item in partition
    ):
        raise HarborRolloutError("Harbor partition mixes runtime agent identities")
    # Runtime fingerprints include unredacted AgentConfig material and must
    # remain process-local.  Job paths use only the public, redacted identity.
    fingerprint = str(
        getattr(partition[0].spec, "agent_id", None)
        or getattr(partition[0].spec, "public_fingerprint", "agent")
    )[:18]
    nonce = uuid.uuid4().hex[:8]
    job_name = "__".join(
        (
            "dressage",
            _safe_job_component(run_id),
            f"r{rollout_id}",
            f"retry{retry_round}",
            fingerprint,
            nonce,
        )
    )
    jobs_dir = Path(config.artifacts.root).expanduser().resolve() / "harbor-jobs"
    route_demand = _partition_route_demand(
        partition,
        max_active_routes=int(config.gateway.limits.max_active_routes),
    )
    agent = _validated_model_update(agent, n_concurrent=None)
    temporary_tasks = [
        _validated_model_update(
            _copy_model(item.spec.task_config),
            source=None,
        )
        for item in partition
    ]
    updates = {
        "job_name": job_name,
        "jobs_dir": jobs_dir,
        "n_attempts": attempts,
        "n_concurrent_trials": route_demand,
        "quiet": True,
        "datasets": [],
        "tasks": temporary_tasks,
        "agents": [agent],
    }
    if config.environment.mode == "bwrap":
        source_environment = getattr(source_job_config, "environment", None)
        if source_environment is None:
            raise HarborRolloutError("Harbor JobConfig has no environment config")
        updates["environment"] = _validated_model_update(
            source_environment,
            type=None,
            import_path=_BWRAP_ENVIRONMENT_IMPORT_PATH,
        )
    return _validated_model_update(
        source_job_config,
        **updates,
    )


def _assign_partition_trials(
    job: Any,
    partition: Sequence[_GroupWork],
    *,
    rollout_id: int,
    retry_round: int,
    expected_weight_version: str | None,
) -> tuple[dict[str, Any], dict[str, TrialBinding]]:
    from dressage.integrations.harbor.compat import assign_trial_names

    nonce = uuid.uuid4().hex[:8]

    def name(entry: Any) -> str:
        item = partition[entry.task_index]
        return "__".join(
            (
                "dr",
                f"r{rollout_id}",
                f"q{retry_round}",
                f"g{item.group_index}",
                f"a{entry.attempt_index}",
                nonce,
            )
        )

    plan = assign_trial_names(job, name)
    templates: dict[str, Any] = {}
    bindings: dict[str, TrialBinding] = {}
    for entry in plan:
        if entry.agent_index != 0 or entry.task_index >= len(partition):
            raise HarborRolloutError("Harbor v0.18 trial plan no longer matches batching")
        item = partition[entry.task_index]
        if entry.attempt_index >= len(item.templates):
            raise HarborRolloutError("Harbor trial attempt exceeds source group size")
        templates[entry.trial_name] = item.templates[entry.attempt_index]
        bindings[entry.trial_name] = TrialBinding(
            instance_id=item.instance_id,
            attempt_ordinal=entry.attempt_index,
            expected_weight_version=expected_weight_version,
        )
    return templates, bindings


def _bundle_samples(
    *,
    template: Any,
    bundle: HarborTrajectoryBundle,
    args: Any,
    reward_key: str,
) -> list[Any]:
    if not bundle.trainable:
        failures = ", ".join(failure.code for failure in bundle.failures) or "invalid bundle"
        raise HarborRolloutError(
            f"Harbor attempt {bundle.trial_name}/{bundle.trial_id} is not trainable: "
            f"{failures}"
        )
    expanded = expand_segments_to_samples(
        template,
        list(bundle.segments),
        args=args,
        session_id=bundle.session_id,
        instance_id=bundle.instance_id,
    )
    rewards = dict(bundle.rewards)
    selected = rewards.get(reward_key)
    if isinstance(selected, bool) or not isinstance(selected, (int, float)):
        raise HarborRolloutError(f"Harbor reward {reward_key!r} is not numeric")
    if not math.isfinite(float(selected)):
        raise HarborRolloutError(f"Harbor reward {reward_key!r} is not finite")
    for index, sample in enumerate(expanded):
        metadata = _metadata(sample)
        metadata.update(
            {
                "harbor_job_id": bundle.job_id,
                "harbor_trial_name": bundle.trial_name,
                "harbor_trial_id": bundle.trial_id,
                "harbor_task_name": bundle.task_name,
                "harbor_task_checksum": bundle.task_checksum,
                "harbor_agent_name": bundle.agent_name,
                "harbor_routing_guarantee": bundle.routing_guarantee,
                "harbor_task_network_class": bundle.task_network_class,
                "harbor_rewards": rewards,
                "harbor_reward_key": reward_key,
                "harbor_attempt_ordinal": bundle.attempt_ordinal,
                "harbor_weight_versions": list(bundle.observed_weight_versions),
                "instance_id": bundle.instance_id,
            }
        )
        sample.reward = (
            rewards if index == len(expanded) - 1 else {reward_key: 0.0}
        )
    return expanded


async def _run_partition(
    *,
    args: Any,
    config: Any,
    data_source: Any,
    partition: Sequence[_GroupWork],
    runtime: Any,
    rollout_id: int,
    retry_round: int,
    evaluation: bool,
    expected_weight_version: str | None,
) -> list[_GroupOutcome]:
    source_job_config = (
        getattr(data_source, "eval_job_config", None)
        if evaluation
        else getattr(data_source, "job_config", None)
    )
    if source_job_config is None:
        raise HarborRolloutError("Harbor data source has no job configuration")
    run_id = str(getattr(data_source, "run_id", "harbor"))
    job_config = _temporary_job_config(
        source_job_config=source_job_config,
        partition=partition,
        config=config,
        run_id=run_id,
        rollout_id=rollout_id,
        retry_round=retry_round,
    )
    _validate_temporary_job_task_sources(job_config)
    job = await _CREATE_JOB(job_config)
    templates, bindings = _assign_partition_trials(
        job,
        partition,
        rollout_id=rollout_id,
        retry_round=retry_round,
        expected_weight_version=expected_weight_version,
    )
    reward_key = str(config.training.reward_key)
    artifact_store = HarborArtifactStore(
        config.artifacts.root,
        run_id=run_id,
        reward_key=reward_key,
        mode=str(config.artifacts.mode),
        require_token_versions=bool(config.training.require_single_weight_version),
        require_trainable_tokens=bool(config.trajectory.require_trainable_tokens),
        file_mode=int(config.artifacts.file_mode),
        dir_mode=int(config.artifacts.dir_mode),
        fsync=bool(config.artifacts.fsync),
    )
    plugin = _PLUGIN_FACTORY(
        config=config,
        gateway_runtime=runtime,
        artifact_store=artifact_store,
        trial_bindings=bindings,
        defer_final_selection=True,
    )

    def failed_outcomes(
        error: BaseException,
        routing_tasks: Mapping[str, str],
    ) -> list[_GroupOutcome]:
        return [
            _GroupOutcome(
                work=item,
                error=error,
                routing_tasks=dict(routing_tasks),
            )
            for item in partition
        ]

    async def abort_artifacts() -> None:
        # Wait for shielded END commits before reconciling every physical
        # attempt to an explicitly aborted, non-trainable state.
        await plugin.aclose()
        routing = _plugin_routing_summary(plugin)
        await artifact_store.write_job_manifest(
            job,
            final_keys=[],
            state="aborted",
            routing_guarantee=str(routing["routing_guarantee"]),
            public_network_tasks=int(routing["public_network_tasks"]),
            restricted_network_tasks=int(routing["restricted_network_tasks"]),
        )

    try:
        await plugin.on_job_start(job)
        result = await job.run()
        await plugin.on_job_end(result)

        by_trial: dict[str, Any] = {
            str(item.trial_name): item for item in (result.trial_results or [])
        }
        outcomes_by_position: dict[int, _GroupOutcome] = {
            item.position: _GroupOutcome(work=item, samples=[])
            for item in partition
        }
        trial_to_work: dict[str, _GroupWork] = {}
        for trial_name, template in templates.items():
            del template
            match = next(
                (
                    item
                    for item in partition
                    if any(
                        sample is templates[trial_name] for sample in item.templates
                    )
                ),
                None,
            )
            if match is None:
                raise HarborRolloutError(
                    f"lost source mapping for Harbor trial {trial_name}"
                )
            trial_to_work[trial_name] = match

        for trial_name, template in templates.items():
            work = trial_to_work[trial_name]
            outcome = outcomes_by_position[work.position]
            if outcome.error is not None:
                continue
            final_result = by_trial.get(trial_name)
            if final_result is None:
                outcome.error = HarborRolloutError(
                    f"Harbor JobResult is missing final trial {trial_name}"
                )
                outcome.samples = None
                continue
            bundle = plugin.get_result(trial_name, str(final_result.id))
            if bundle is None:
                outcome.error = HarborRolloutError(
                    "Dressage Plugin did not commit final trial "
                    f"{trial_name}/{final_result.id}"
                )
                outcome.samples = None
                continue
            outcome.bundles.append(bundle)
            outcome.final_keys.append((bundle.trial_name, bundle.trial_id))
            try:
                converted = _bundle_samples(
                    template=template,
                    bundle=bundle,
                    args=args,
                    reward_key=reward_key,
                )
            except Exception as exc:  # group failure is retried atomically
                outcome.error = exc
                outcome.samples = None
                continue
            assert outcome.samples is not None
            outcome.samples.extend(converted)
            outcome.versions = tuple(
                dict.fromkeys((*outcome.versions, *bundle.observed_weight_versions))
            )

        expected_attempts = len(partition[0].templates)
        for item in partition:
            outcome = outcomes_by_position[item.position]
            if outcome.succeeded:
                parents = {
                    _metadata(sample).get("parent_traj_id")
                    for sample in outcome.samples or []
                }
                if len(parents) != expected_attempts:
                    outcome.error = HarborRolloutError(
                        f"Harbor group {item.group_index} returned {len(parents)} "
                        f"trajectories; expected {expected_attempts}"
                    )
                    outcome.samples = None
                elif (
                    config.training.require_single_weight_version
                    and len(outcome.versions) != 1
                ):
                    outcome.error = HarborRolloutError(
                        f"Harbor group {item.group_index} observed weight versions "
                        f"{list(outcome.versions)!r}; expected exactly one"
                    )
                    outcome.samples = None

        selected_keys = [
            key
            for outcome in outcomes_by_position.values()
            if outcome.succeeded
            for key in outcome.final_keys
        ]
        manifest_state = (
            "completed"
            if all(outcome.succeeded for outcome in outcomes_by_position.values())
            else "partial"
        )
        routing = _plugin_routing_summary(plugin)
        await artifact_store.write_job_manifest(
            result,
            final_keys=selected_keys,
            state=manifest_state,
            routing_guarantee=str(routing["routing_guarantee"]),
            public_network_tasks=int(routing["public_network_tasks"]),
            restricted_network_tasks=int(routing["restricted_network_tasks"]),
        )

        routing_tasks = _plugin_routing_tasks(plugin)
        for outcome in outcomes_by_position.values():
            outcome.routing_tasks = dict(routing_tasks)
        return [outcomes_by_position[item.position] for item in partition]
    except asyncio.CancelledError:
        try:
            await abort_artifacts()
        except Exception:
            logger.exception("failed to reconcile cancelled Harbor Job artifacts")
        raise
    except Exception as exc:
        routing_tasks = _plugin_routing_tasks(plugin)
        error: BaseException = exc
        try:
            await abort_artifacts()
        except Exception as abort_exc:
            error = HarborRolloutError(
                "Harbor Job failed and artifact abort reconciliation also failed: "
                f"job={type(exc).__name__}: {exc}; "
                f"abort={type(abort_exc).__name__}: {abort_exc}"
            )
        return failed_outcomes(error, routing_tasks)
    finally:
        try:
            await plugin.aclose()
        except Exception:
            logger.exception("failed to close Harbor Plugin after partition execution")


async def _run_round(
    *,
    args: Any,
    config: Any,
    data_source: Any,
    pending: Sequence[_GroupWork],
    runtime: Any,
    rollout_id: int,
    retry_round: int,
    evaluation: bool,
    expected_weight_version: str | None,
) -> list[_GroupOutcome]:
    partitions = _partition_work(pending)
    max_routes = int(config.gateway.limits.max_active_routes)

    class RouteCapacity:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity
            self.available = capacity
            self.condition = asyncio.Condition()

        async def acquire(self, weight: int) -> None:
            async with self.condition:
                await self.condition.wait_for(lambda: self.available >= weight)
                self.available -= weight

        async def release(self, weight: int) -> None:
            async with self.condition:
                self.available += weight
                self.condition.notify_all()

    limiter = RouteCapacity(max_routes)
    demands = [
        _partition_route_demand(
            partition,
            max_active_routes=max_routes,
        )
        for partition in partitions
    ]

    async def run_partition(
        partition: Sequence[_GroupWork],
        demand: int,
    ) -> list[_GroupOutcome]:
        await limiter.acquire(demand)
        try:
            try:
                return await _run_partition(
                    args=args,
                    config=config,
                    data_source=data_source,
                    partition=partition,
                    runtime=runtime,
                    rollout_id=rollout_id,
                    retry_round=retry_round,
                    evaluation=evaluation,
                    expected_weight_version=expected_weight_version,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # retry every group in the failed job
                return [
                    _GroupOutcome(
                        work=item,
                        error=exc,
                    )
                    for item in partition
                ]
        finally:
            await limiter.release(demand)

    nested = await asyncio.gather(
        *(
            run_partition(partition, demand)
            for partition, demand in zip(partitions, demands)
        )
    )
    return [outcome for partition in nested for outcome in partition]


def _max_group_retries(config: Any) -> int:
    value = os.environ.get("DRESSAGE_ROLLOUT_MAX_RETRIES")
    raw = value if value is not None else config.training.group_max_retries
    try:
        retries = int(raw)
    except (TypeError, ValueError) as exc:
        raise HarborRolloutError(
            f"invalid Harbor rollout retry count {raw!r}"
        ) from exc
    if retries < 0:
        raise HarborRolloutError("Harbor rollout retry count must be non-negative")
    return retries


def _failed_group(
    work: _GroupWork,
    error: BaseException,
    *,
    reward_key: str,
) -> list[Any]:
    group = _mark_no_grad_failed(copy.deepcopy(work.templates), error)
    for sample in group:
        metadata = _metadata(sample)
        metadata["instance_id"] = work.instance_id
        metadata["harbor_group_failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        sample.reward = {reward_key: 0.0}
    return group


def _batch_weight_version(groups: Sequence[list[Any]]) -> str | None:
    versions: list[str] = []
    for group in groups:
        for sample in group:
            for version in _metadata(sample).get("harbor_weight_versions", []) or []:
                text = str(version)
                if text and text not in versions:
                    versions.append(text)
    return versions[0] if len(versions) == 1 else None


async def _run_harbor_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
    *,
    evaluation: bool,
) -> tuple[list[list[Any]], dict[str, Any]]:
    config = getattr(data_source, "integration_config", None)
    if config is None or getattr(config, "execution_mode", None) != "training":
        raise HarborRolloutError(
            "generate_rollout_harbor_sync requires HarborDataSource in training mode"
        )
    if config.training is None:
        raise HarborRolloutError("Harbor training configuration is missing")
    _validate_slime_contract(args, config)
    capabilities = await _read_proxy_capabilities(config)
    _validate_proxy_capabilities(args, config, capabilities)
    expected_weight_version = (
        str(capabilities["current_weight_version"])
        if config.training.require_single_weight_version
        else None
    )
    runtime, _root_lease = await _ensure_root_gateway_lease(config)

    if evaluation:
        getter = getattr(data_source, "get_eval_samples", None)
        if not callable(getter):
            raise HarborRolloutError("HarborDataSource does not support evaluation")
        source_groups = getter()
    else:
        target = int(getattr(args, "rollout_batch_size", 1))
        source_groups = data_source.get_samples(target)
        if len(source_groups) != target:
            raise HarborRolloutError(
                f"HarborDataSource returned {len(source_groups)} groups; expected {target}"
            )
    work = _resolve_work(data_source, source_groups)
    if not work:
        return [], {
            "harbor/groups": 0.0,
            "harbor/routing_enforced": float(
                config.security.routing_guarantee == "enforced"
            ),
            "harbor/public_network_tasks": 0.0,
            "harbor/restricted_network_tasks": 0.0,
        }

    max_retries = _max_group_retries(config)
    failed_group_policy = str(config.training.failed_group_policy)
    max_replacements = int(config.training.max_replacement_groups)
    pending = list(work)
    completed: dict[int, list[Any]] = {}
    final_errors: dict[int, BaseException] = {}
    retry_count = 0
    replacement_count = 0
    execution_round = 0
    routing_tasks: dict[str, str] = {}
    while pending:
        exhausted: list[_GroupWork] = []
        for group_retry in range(max_retries + 1):
            outcomes = await _run_round(
                args=args,
                config=config,
                data_source=data_source,
                pending=pending,
                runtime=runtime,
                rollout_id=rollout_id,
                retry_round=execution_round,
                evaluation=evaluation,
                expected_weight_version=expected_weight_version,
            )
            execution_round += 1
            exhausted = []
            next_pending: list[_GroupWork] = []
            for outcome in outcomes:
                for digest, network_class in outcome.routing_tasks.items():
                    existing = routing_tasks.setdefault(
                        digest,
                        network_class,
                    )
                    if existing != network_class:
                        raise HarborRolloutError(
                            "resolved Harbor task changed network class within one batch"
                        )
                if outcome.succeeded:
                    completed[outcome.work.position] = outcome.samples or []
                    final_errors.pop(outcome.work.position, None)
                else:
                    error = outcome.error or HarborRolloutError(
                        "unknown Harbor group failure"
                    )
                    final_errors[outcome.work.position] = error
                    exhausted.append(outcome.work)
                    if group_retry < max_retries:
                        next_pending.append(outcome.work)
                        retry_count += 1
                        logger.warning(
                            "retrying complete Harbor group %s (%d/%d): %s",
                            outcome.work.group_index,
                            group_retry + 1,
                            max_retries,
                            error,
                        )
            pending = next_pending
            if not pending:
                break

        if not exhausted or failed_group_policy != "replace" or evaluation:
            break
        if replacement_count + len(exhausted) > max_replacements:
            first = min(item.position for item in exhausted)
            raise HarborRolloutError(
                f"Harbor replacement budget exhausted ({max_replacements} groups)"
            ) from final_errors[first]

        replacement_groups = data_source.get_samples(len(exhausted))
        replacement_work = _resolve_work(data_source, replacement_groups)
        pending = []
        for failed, replacement_item in zip(exhausted, replacement_work):
            item = replace(replacement_item, position=failed.position)
            work[failed.position] = item
            final_errors.pop(failed.position, None)
            pending.append(item)
        replacement_count += len(pending)

    if final_errors and (
        failed_group_policy == "abort_batch"
        or (evaluation and failed_group_policy == "replace")
    ):
        first = min(final_errors)
        raise HarborRolloutError(
            f"Harbor group {work[first].group_index} exhausted retries"
        ) from final_errors[first]

    reward_key = str(config.training.reward_key)
    for position, error in final_errors.items():
        completed[position] = _failed_group(
            work[position], error, reward_key=reward_key
        )
    ordered = [completed[position] for position in range(len(work))]
    live_groups = sum(
        1 for group in ordered if _group_has_trainable_tokens(group)
    )
    live_ratio = live_groups / len(ordered)
    if live_ratio < float(config.training.min_live_group_ratio):
        raise HarborRolloutError(
            f"Harbor live group ratio {live_ratio:.3f} is below configured minimum "
            f"{config.training.min_live_group_ratio:.3f}"
        )
    if not evaluation and not _allow_empty_train_batch() and live_groups == 0:
        raise HarborRolloutError(
            "Harbor rollout produced no trainable groups; refusing an empty update"
        )

    versions = {
        version
        for group in ordered
        for sample in group
        if not getattr(sample, "remove_sample", False)
        for version in (_metadata(sample).get("harbor_weight_versions", []) or [])
    }
    if config.training.require_single_weight_version and len(versions) != 1:
        raise HarborRolloutError(
            f"Harbor batch observed weight versions {sorted(map(str, versions))!r}; "
            "expected exactly one"
        )

    record_state = getattr(data_source, "record_batch_state", None)
    if callable(record_state) and not evaluation:
        record_state(
            weight_version=_batch_weight_version(ordered),
            runtime_incarnation=_runtime_incarnation(capabilities),
        )

    flat = [sample for group in ordered for sample in group]
    metrics: dict[str, Any] = compute_multi_segment_metrics(flat)
    metrics.update(
        _reward_distribution_metrics(ordered, reward_key=reward_key)
    )
    metrics.update(
        {
            "harbor/groups": float(len(ordered)),
            "harbor/live_groups": float(live_groups),
            "harbor/failed_groups": float(len(final_errors)),
            "harbor/group_retries": float(retry_count),
            "harbor/replacement_groups": float(replacement_count),
            "harbor/live_group_ratio": live_ratio,
            "harbor/routing_enforced": float(
                config.security.routing_guarantee == "enforced"
            ),
            "harbor/public_network_tasks": float(
                sum(value == "public" for value in routing_tasks.values())
            ),
            "harbor/restricted_network_tasks": float(
                sum(value == "restricted" for value in routing_tasks.values())
            ),
        }
    )
    return ordered, metrics


def _eval_output(groups: Sequence[list[Any]], metrics: dict[str, Any]) -> Any:
    # Eval statistics are trajectory-level.  Keep only each trajectory's
    # highest segment so a long, split trajectory is not over-weighted.
    anchors: dict[str, Any] = {}
    for group in groups:
        for sample in group:
            metadata = _metadata(sample)
            parent = str(metadata.get("parent_traj_id") or getattr(sample, "rollout_id", ""))
            current = anchors.get(parent)
            if current is None or int(metadata.get("segment_index", 0)) > int(
                _metadata(current).get("segment_index", 0)
            ):
                anchors[parent] = sample
    samples = sorted(anchors.values(), key=lambda sample: getattr(sample, "index", 0))
    rewards: list[float] = []
    truncated: list[bool] = []
    for sample in samples:
        reward = getattr(sample, "reward", None)
        key = _metadata(sample).get("harbor_reward_key")
        if isinstance(reward, Mapping):
            reward = reward.get(key)
        rewards.append(float(reward or 0.0))
        status = getattr(getattr(sample, "status", None), "name", "")
        truncated.append(status == "TRUNCATED")
    data = {"harbor": {"rewards": rewards, "truncated": truncated, "samples": samples}}
    if RolloutFnEvalOutput is None:
        return data
    return RolloutFnEvalOutput(data=data, metrics=metrics)


def generate_rollout_harbor_sync(
    args: Any,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
) -> Any:
    """Run one strict Harbor rollout batch and return slime's public output."""

    groups, metrics = run(
        _run_harbor_rollout(
            args,
            rollout_id,
            data_buffer,
            evaluation=evaluation,
        )
    )
    if evaluation:
        return _eval_output(groups, metrics)
    if RolloutFnTrainOutput is None:
        return groups
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)


atexit.register(_close_harbor_rollout_runtime_at_exit)


__all__ = [
    "HarborRolloutError",
    "close_harbor_rollout_runtime",
    "generate_rollout_harbor_sync",
]
