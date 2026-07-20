"""The only module allowed to depend on Harbor's private v0.18 contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable


SUPPORTED_HARBOR_VERSION = "0.18.0"
MINIMUM_PYTHON = (3, 12)


class HarborCompatibilityError(RuntimeError):
    """Raised before execution when the pinned Harbor contract is unavailable."""


@dataclass(frozen=True)
class HarborRuntimeInfo:
    python_version: tuple[int, int, int]
    harbor_version: str


@dataclass(frozen=True)
class TrialPlanEntry:
    position: int
    attempt_index: int
    task_index: int
    agent_index: int
    trial_name: str
    config: Any


def require_harbor_runtime(
    *,
    python_version: tuple[int, int, int] | None = None,
    harbor_version: str | None = None,
) -> HarborRuntimeInfo:
    """Validate runtime requirements without importing Harbor at module load."""

    selected_python = python_version or tuple(sys.version_info[:3])
    if selected_python[:2] < MINIMUM_PYTHON:
        raise HarborCompatibilityError(
            "the Harbor integration requires Python >=3.12; "
            f"current interpreter is {selected_python[0]}.{selected_python[1]}.{selected_python[2]}"
        )
    if harbor_version is None:
        try:
            harbor_version = importlib.metadata.version("harbor")
        except importlib.metadata.PackageNotFoundError as exc:
            raise HarborCompatibilityError(
                "Harbor is not installed; install the Dressage Harbor optional dependency"
            ) from exc
    if harbor_version != SUPPORTED_HARBOR_VERSION:
        raise HarborCompatibilityError(
            f"unsupported Harbor version {harbor_version!r}; expected exactly {SUPPORTED_HARBOR_VERSION!r}"
        )
    return HarborRuntimeInfo(
        python_version=(selected_python[0], selected_python[1], selected_python[2]),
        harbor_version=harbor_version,
    )


def pending_trial_configs(job: Any) -> tuple[Any, ...]:
    """Return Harbor's pending TrialConfigs with strict v0.18 shape checks."""

    require_harbor_runtime()
    if not hasattr(job, "_remaining_trial_configs"):
        raise HarborCompatibilityError(
            "Harbor Job no longer exposes _remaining_trial_configs; the v0.18 compat contract changed"
        )
    configs = job._remaining_trial_configs
    if not isinstance(configs, list):
        raise HarborCompatibilityError(
            "Harbor Job._remaining_trial_configs must be a list under the v0.18 contract"
        )
    for index, config in enumerate(configs):
        if (
            not hasattr(config, "trial_name")
            or not hasattr(config, "agent")
            or not hasattr(config, "task")
        ):
            raise HarborCompatibilityError(
                f"pending TrialConfig at position {index} does not match the Harbor v0.18 shape"
            )
    return tuple(configs)


def resolved_task_configs(job: Any) -> tuple[Any, ...]:
    if not hasattr(job, "_task_configs") or not isinstance(job._task_configs, list):
        raise HarborCompatibilityError(
            "Harbor Job no longer exposes list _task_configs; the v0.18 compat contract changed"
        )
    return tuple(job._task_configs)


async def resolve_task_configs(job_config: Any) -> tuple[Any, ...]:
    """Resolve explicit and dataset tasks through Harbor's pinned private API."""

    require_harbor_runtime()
    try:
        from harbor.job import Job
    except ImportError as exc:  # pragma: no cover - version check normally catches this
        raise HarborCompatibilityError("failed to import harbor.job.Job") from exc
    resolver = getattr(Job, "_resolve_task_configs", None)
    if not callable(resolver):
        raise HarborCompatibilityError(
            "Harbor Job._resolve_task_configs is unavailable; the v0.18 compat contract changed"
        )
    configs = await resolver(job_config)
    if not isinstance(configs, list) or not configs:
        raise HarborCompatibilityError(
            "Harbor Job._resolve_task_configs must return a non-empty list under v0.18"
        )
    return tuple(configs)


def build_trial_plan(job: Any) -> tuple[TrialPlanEntry, ...]:
    """Decode Harbor v0.18's attempt-major task-agent TrialConfig order."""

    if getattr(job, "is_resuming", False):
        raise HarborCompatibilityError(
            "Harbor Job resume is not supported by the Dressage integration"
        )
    configs = pending_trial_configs(job)
    tasks = resolved_task_configs(job)
    job_config = getattr(job, "config", None)
    agents = getattr(job_config, "agents", None)
    attempts = getattr(job_config, "n_attempts", None)
    if not isinstance(agents, list) or not agents:
        raise HarborCompatibilityError(
            "Harbor Job.config.agents must be a non-empty list"
        )
    if not isinstance(attempts, int) or attempts < 1:
        raise HarborCompatibilityError(
            "Harbor Job.config.n_attempts must be a positive integer"
        )
    if not tasks:
        raise HarborCompatibilityError("Harbor Job._task_configs must be non-empty")

    expected = attempts * len(tasks) * len(agents)
    if len(configs) != expected:
        raise HarborCompatibilityError(
            "pending TrialConfig count does not match n_attempts * tasks * agents: "
            f"{len(configs)} != {attempts} * {len(tasks)} * {len(agents)}"
        )

    result: list[TrialPlanEntry] = []
    per_attempt = len(tasks) * len(agents)
    for position, config in enumerate(configs):
        attempt_index = position // per_attempt
        within_attempt = position % per_attempt
        task_index = within_attempt // len(agents)
        agent_index = within_attempt % len(agents)
        if config.task is not tasks[task_index]:
            raise HarborCompatibilityError(
                f"TrialConfig position {position} violates Harbor v0.18 task-major ordering"
            )
        if config.agent is not agents[agent_index]:
            raise HarborCompatibilityError(
                f"TrialConfig position {position} violates Harbor v0.18 agent-minor ordering"
            )
        trial_name = str(config.trial_name)
        if not trial_name:
            raise HarborCompatibilityError(
                f"TrialConfig position {position} has an empty trial_name"
            )
        result.append(
            TrialPlanEntry(
                position=position,
                attempt_index=attempt_index,
                task_index=task_index,
                agent_index=agent_index,
                trial_name=trial_name,
                config=config,
            )
        )
    return tuple(result)


def assign_trial_names(
    job: Any,
    name_factory: Callable[[TrialPlanEntry], str],
) -> tuple[TrialPlanEntry, ...]:
    """Assign deterministic unique names after Job.create and return a fresh plan."""

    plan = build_trial_plan(job)
    names: set[str] = set()
    for entry in plan:
        name = name_factory(entry)
        if not isinstance(name, str) or not name:
            raise ValueError("trial name factory must return a non-empty string")
        if name in names:
            raise ValueError(f"trial name factory returned duplicate name {name!r}")
        names.add(name)
        entry.config.trial_name = name
    return build_trial_plan(job)


def stable_concurrency_group(agent: Any) -> str | None:
    """Return a stable group only when Harbor's n_concurrent is configured."""

    n_concurrent = getattr(agent, "n_concurrent", None)
    if n_concurrent is None:
        return None
    existing = getattr(agent, "concurrency_group", None)
    if existing:
        return str(existing)
    concurrency_key = getattr(agent, "concurrency_key", None)
    if isinstance(concurrency_key, str) and concurrency_key:
        identity = concurrency_key
    else:
        dump = getattr(agent, "model_dump", None)
        if not callable(dump):
            raise HarborCompatibilityError(
                "AgentConfig lacks concurrency_key/model_dump"
            )
        payload = dump(
            mode="json",
            exclude={"concurrency_group", "n_concurrent"},
            exclude_none=True,
            context={"redact_sensitive_env": False},
        )
        identity = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "dressage:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


__all__ = [
    "HarborCompatibilityError",
    "HarborRuntimeInfo",
    "MINIMUM_PYTHON",
    "SUPPORTED_HARBOR_VERSION",
    "TrialPlanEntry",
    "assign_trial_names",
    "build_trial_plan",
    "pending_trial_configs",
    "require_harbor_runtime",
    "resolve_task_configs",
    "resolved_task_configs",
    "stable_concurrency_group",
]
