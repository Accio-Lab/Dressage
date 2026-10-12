from __future__ import annotations

import hashlib
import logging
import math
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agentic_rollout_contract.client import PolicyServiceClient, RolloutServiceClient
from agentic_rollout_contract.models import ExecutionClaim

from .config import WorkerNode
from .harbor_hooks import HarborRemotePolicyHooks
from .image_cache import ImageCache, ImageReferenceError, ImageUseHandle
from .policy_relay import PolicyRelayRegistry, RelayRoute
from .task_resource_index import TaskResourceIndex

logger = logging.getLogger(__name__)
CLAUDE_CODE_VERSION = "2.1.207"
CLAUDE_CODE_IMPORT_PATH = (
    "rollout_service.preinstalled_claude_code:PreinstalledClaudeCode"
)
_CLAUDE_CODE_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_RESERVED_AGENT_KWARGS = frozenset(
    {
        "preinstalled_only",
        "agent_timeout_sec",
        "claude_auto_compact_pct",
        "claude_context_window_tokens",
        "claude_max_output_tokens",
        "claude_file_read_max_output_tokens",
        "context_budget_env",
        "sampling_seed_base",
        "sampling_temperature",
        "sampling_top_p",
        "version",
    }
)


@dataclass(frozen=True)
class HarborOutcome:
    succeeded: bool
    trial_id: str
    trial_name: str
    rewards: dict[str, float]
    reward: float
    agent_terminal_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class HarborRunner:
    def __init__(
        self,
        node: WorkerNode,
        *,
        control_client: RolloutServiceClient,
        relay_registry: PolicyRelayRegistry,
        resource_index: TaskResourceIndex,
        policy_url: str,
    ) -> None:
        self.node = node
        self.control_client = control_client
        self.relay_registry = relay_registry
        self.resource_index = resource_index
        self.policy_url = policy_url.rstrip("/")
        self.image_cache = (
            ImageCache(
                node.image_cache,
                resource_index=resource_index,
                worker_id=node.worker_id,
            )
            if node.image_cache.enabled
            else None
        )

    async def run(self, claim: ExecutionClaim) -> HarborOutcome:
        from harbor import (
            RetryConfig,
            TrialAgentConfig,
            TrialQueue,
            TrialTaskConfig,
        )
        from harbor.models.trial.config import TrialConfig

        if claim.resource_catalog_digest != self.resource_index.digest:
            raise ValueError("claim resource catalog differs from Worker frozen index")
        local_resources = self.resource_index.resolve(claim.execution.task_ref)
        if local_resources != claim.resolved_resources:
            raise ValueError(
                "worker task resource index disagrees with frozen attempt resources"
            )
        task_path = self._resolve_task(claim.execution.task_ref)
        task_image, task_config_digest = self._task_image(task_path)
        admitted_image = self.resource_index.admitted_image(claim.execution.task_ref)
        execution_image = self.resource_index.execution_image(claim.execution.task_ref)
        if task_image != admitted_image:
            raise ValueError(
                "worker task image alias disagrees with frozen admission entry"
            )
        if task_config_digest != local_resources.task_config_digest:
            raise ValueError(
                "worker task config digest disagrees with frozen attempt resources"
            )
        image_use: ImageUseHandle | None = None
        if self.image_cache is not None:
            image_use = await self.image_cache.acquire(
                attempt_id=claim.execution.execution_id,
                task_ref=claim.execution.task_ref,
                declared_image_ref=task_image,
                execution_image_ref=execution_image,
                task_config_digest=task_config_digest,
                worker_incarnation=claim.worker_incarnation,
            )
            logger.info(
                "attempt image ready attempt_id=%s image_ref=%s cache_hit=%s "
                "use_lock_wait_seconds=%.3f pull_lock_wait_seconds=%.3f "
                "pull_attempts=%d pull_seconds=%.3f",
                claim.execution.execution_id,
                image_use.image_ref,
                image_use.cache_hit,
                image_use.lock_wait_seconds,
                image_use.pull_lock_wait_seconds,
                image_use.pull_attempts,
                image_use.pull_seconds,
            )
        local_token = secrets.token_urlsafe(32)
        relay_url = f"{self.node.relay_base_url.rstrip('/')}/relay/{claim.execution.execution_id}"
        succeeded = False
        policy_client: PolicyServiceClient | None = None
        hooks: HarborRemotePolicyHooks | None = None
        relay_added = False
        try:
            self.relay_registry.add(
                RelayRoute(
                    attempt_id=claim.execution.execution_id,
                    route_id=claim.context_id,
                    local_token=local_token,
                )
            )
            relay_added = True
            policy_client = PolicyServiceClient(self.policy_url)
            hooks = HarborRemotePolicyHooks(
                claim,
                control_client=self.control_client,
                policy_client=policy_client,
                retry_attempts=self.relay_registry.transport_retry_attempts,
                retry_base_sec=self.relay_registry.transport_retry_base_sec,
                retry_max_sec=self.relay_registry.transport_retry_max_sec,
            )
            trial_name = f"remote-{claim.execution.execution_id}"
            agent_kwargs = _claude_agent_kwargs(claim.execution.agent_kwargs)
            agent_timeout_sec = _agent_timeout_sec(claim.execution.agent_kwargs)
            auto_compact_pct = _claude_auto_compact_pct(claim.execution.agent_kwargs)
            relay_host = urlsplit(relay_url).hostname
            agent_env = {
                "ANTHROPIC_API_KEY": local_token,
                "ANTHROPIC_AUTH_TOKEN": local_token,
                "ANTHROPIC_BASE_URL": relay_url,
                "ANTHROPIC_MODEL": claim.execution.model_name,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": claim.execution.model_name,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": claim.execution.model_name,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": claim.execution.model_name,
                "CLAUDE_CODE_SUBAGENT_MODEL": claim.execution.model_name,
                "DISABLE_TELEMETRY": "1",
            }
            if auto_compact_pct is not None:
                agent_env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(auto_compact_pct)
            context_env = _claude_context_env(claim.execution.agent_kwargs)
            agent_kwargs["context_budget_env"] = context_env
            logger.info(
                "agent_context_budget execution_id=%s context=%s output=%s read=%s compact_pct=%s",
                claim.execution.execution_id,
                context_env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS"),
                context_env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
                context_env.get("CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"),
                auto_compact_pct,
            )
            agent = TrialAgentConfig(
                import_path=CLAUDE_CODE_IMPORT_PATH,
                model_name=f"anthropic/{claim.execution.model_name}",
                extra_allowed_hosts=[relay_host] if relay_host else [],
                kwargs=agent_kwargs,
                env=agent_env,
                override_timeout_sec=agent_timeout_sec,
            )
            environment = self._environment_config(
                claim, execution_image=execution_image
            )
            config = TrialConfig(
                task=TrialTaskConfig(path=task_path),
                trial_name=trial_name,
                trials_dir=self.node.state_dir / "trials",
                agent=agent,
                environment=environment,
                job_id=None,
            )
            queue = TrialQueue(
                n_concurrent=1,
                retry_config=RetryConfig(max_retries=0),
            )
            hooks.bind(queue)
            result = await queue.submit(config)
            trial_id = str(result.id)
            rewards = _numeric_rewards(
                getattr(getattr(result, "verifier_result", None), "rewards", None)
            )
            reward = rewards.get("reward")
            exception = getattr(result, "exception_info", None)
            succeeded = exception is None and reward is not None
            metadata = (
                getattr(getattr(result, "agent_result", None), "metadata", None) or {}
            )
            terminal_reason = metadata.get("agent_terminal_reason")
            logger.info(
                "agent_terminal execution_id=%s terminal_reason=%s succeeded=%s reward=%s error_code=%s",
                claim.execution.execution_id,
                terminal_reason or "other",
                succeeded,
                reward,
                getattr(exception, "exception_type", None),
            )
            return HarborOutcome(
                succeeded=succeeded,
                trial_id=trial_id,
                trial_name=str(result.trial_name),
                rewards=rewards,
                reward=float(reward) if reward is not None else 0.0,
                agent_terminal_reason=terminal_reason,
                error_code=(
                    (None if succeeded else "VerifierRewardMissingError")
                    if exception is None
                    else str(exception.exception_type)
                ),
                error_message=(
                    (None if succeeded else "verifier did not produce a finite reward")
                    if exception is None
                    else str(exception.exception_message)
                ),
            )
        finally:
            if not succeeded and hooks is not None:
                try:
                    await hooks.revoke_after_failure()
                except Exception:
                    logger.warning(
                        "failed to revoke GPU route while unwinding %s",
                        claim.execution.execution_id,
                        exc_info=True,
                    )
            try:
                if policy_client is not None:
                    await policy_client.close()
                if relay_added:
                    await self.relay_registry.remove(claim.execution.execution_id)
            finally:
                if image_use is not None:
                    await image_use.release(
                        outcome="succeeded" if succeeded else "failed"
                    )

    def _environment_config(
        self, claim: ExecutionClaim, *, execution_image: str
    ) -> Any:
        from harbor import TrialEnvironmentConfig
        from harbor.models.trial.config import ResourceMode

        resources = claim.resolved_resources.limit
        safety = self.node.runtime_safety
        return TrialEnvironmentConfig(
            import_path="rollout_service.guarded_docker:GuardedDockerEnvironment",
            override_cpus=resources.cpus,
            override_memory_mb=resources.memory_mb,
            override_storage_mb=safety.storage_mb_per_sandbox,
            cpu_enforcement_policy=ResourceMode.LIMIT,
            memory_enforcement_policy=ResourceMode.LIMIT,
            kwargs={
                "guard_pids_limit": safety.pids_per_sandbox,
                "guard_cgroup_parent": safety.cgroup_parent,
                "guard_storage_mb": safety.storage_mb_per_sandbox,
                "guard_storage_size_enforced": safety.storage_size_enforced,
                "guard_execution_image": execution_image,
                "guard_worker_id": self.node.worker_id,
                "guard_attempt_id": claim.execution.execution_id,
                "guard_client_scope": claim.client_scope,
            },
        )

    def _resolve_task(self, task_ref: str) -> Path:
        parts = Path(task_ref).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"invalid task_ref {task_ref!r}")
        root = self.node.task_catalog_root.expanduser().resolve()
        path = root.joinpath(*parts).resolve()
        if root not in path.parents:
            raise ValueError("task_ref escapes task catalog")
        if not (path / "task.toml").is_file():
            raise FileNotFoundError(f"Harbor task is missing task.toml: {path}")
        task_toml = (path / "task.toml").read_text(encoding="utf-8")
        if "[[steps]]" in task_toml:
            raise ValueError(
                "agentic-rollout/v4 supports single-step Harbor tasks only; "
                f"multi-step task rejected: {task_ref}"
            )
        return path

    @staticmethod
    def _task_image(task_path: Path) -> tuple[str, str]:
        from harbor.models.task.config import TaskConfig

        raw = (task_path / "task.toml").read_text(encoding="utf-8")
        config = TaskConfig.model_validate_toml(raw)
        image = config.environment.docker_image
        if not image:
            raise ImageReferenceError("admitted Harbor task has no docker_image")
        digest = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return image, digest


def _numeric_rewards(value: object) -> dict[str, float]:
    if value is None:
        return {}
    raw = dict(value) if not isinstance(value, dict) else value
    result: dict[str, float] = {}
    for key, item in raw.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        number = float(item)
        if math.isfinite(number):
            result[str(key)] = number
    return result


def _claude_agent_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    requested_version = raw.get("version", CLAUDE_CODE_VERSION)
    if not isinstance(
        requested_version, str
    ) or not _CLAUDE_CODE_VERSION_PATTERN.fullmatch(requested_version):
        raise ValueError(
            "Claude Code version must use an explicit numeric X.Y.Z version"
        )
    forwarded = {
        key: value for key, value in raw.items() if key not in _RESERVED_AGENT_KWARGS
    }
    return {
        "max_turns": 80,
        "permission_mode": "bypassPermissions",
        **forwarded,
        "version": requested_version,
    }


def _agent_timeout_sec(raw: dict[str, Any]) -> float | None:
    value = raw.get("agent_timeout_sec")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("agent_timeout_sec must be a positive number")
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0:
        raise ValueError("agent_timeout_sec must be a positive finite number")
    return selected


def _claude_auto_compact_pct(raw: dict[str, Any]) -> int | None:
    value = raw.get("claude_auto_compact_pct")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("claude_auto_compact_pct must be an integer in [1, 100]")
    return value


def _claude_context_env(raw: dict[str, Any]) -> dict[str, str]:
    context = raw.get("claude_context_window_tokens")
    output = raw.get("claude_max_output_tokens")
    if context is None and output is None:
        return {}
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0
        for v in (context, output)
    ):
        raise ValueError("Claude context/output budgets must be positive integers")
    if output >= context:
        raise ValueError("Claude output budget must be smaller than context")
    # Pinned 2.1.207 needs both: correcting the custom-model window alone
    # does not enable proactive compaction in the isolated runtime probe.
    env = {
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context),
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(context),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(output),
    }
    read = raw.get("claude_file_read_max_output_tokens")
    if read is not None:
        if isinstance(read, bool) or not isinstance(read, int) or read <= 0:
            raise ValueError("Claude file-read budget must be a positive integer")
        env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] = str(read)
    return env
