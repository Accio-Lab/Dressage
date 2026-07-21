"""Harbor-backed prompt source for synchronous online training.

The public slime ``Sample`` is intentionally only a lightweight handle.  It
contains safe, opaque identifiers; resolved Harbor TaskConfig/AgentConfig
objects (and especially agent environment variables) stay in the process-local
registry and are looked up by :meth:`HarborDataSource.resolve_spec`.

Harbor 0.18 requires Python 3.12, but Dressage itself supports Python 3.10.
Consequently this module has no eager Harbor import and remains importable in a
base Dressage environment.  Runtime compatibility is checked when the data
source is instantiated.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import tempfile
import threading
from typing import Any, Awaitable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit
import uuid

from dressage.integrations.harbor.config import (
    HarborIntegrationConfig,
    load_config,
)


try:  # slime is optional for the import-only compatibility path.
    from slime.rollout.data_source import DataSource as _SlimeDataSource
    from slime.utils.types import Sample
except (ImportError, ModuleNotFoundError):

    class _SlimeDataSource:  # pragma: no cover - interface-only fallback
        pass

    @dataclass
    class Sample:  # pragma: no cover - exercised when slime is not installed
        group_index: int | None = None
        index: int | None = None
        prompt: str | list[dict[str, str]] = ""
        metadata: dict[str, Any] = field(default_factory=dict)


CHECKPOINT_SCHEMA_VERSION = "dressage.harbor.data-source/v1"
_CHECKPOINT_FILENAME = "harbor_data_source_state_{rollout_id}.json"
_SAFE_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SENSITIVE_KEY_PARTS = (
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "password",
    "privatekey",
    "secret",
    "sessionkey",
    "token",
)


class HarborDataSourceError(RuntimeError):
    """Base error for Harbor source configuration and state failures."""


class HarborDataSourceConfigurationError(HarborDataSourceError):
    """The source cannot start with the supplied configuration."""


class HarborDataSourceCheckpointError(HarborDataSourceError):
    """A checkpoint is missing, malformed, or belongs to another source."""


@dataclass(frozen=True, slots=True)
class HarborPromptSpec:
    """One immutable task-agent source entry.

    ``runtime_fingerprint`` deliberately never leaves this object.  It includes
    the unredacted runtime identity so the rollout executor can partition work
    correctly when two agents differ only by environment.  All persisted and
    Sample-visible identities derive from ``public_fingerprint`` instead.
    """

    spec_id: str
    task_config: Any = field(repr=False, compare=False)
    agent_config: Any = field(repr=False, compare=False)
    task_id: str
    agent_id: str
    task_uri: str
    public_fingerprint: str
    runtime_fingerprint: str = field(repr=False)
    integration_fingerprint: str
    occurrence_index: int
    scope: str = "train"


class HarborDataSource(_SlimeDataSource):
    """Read-only task-agent registry used by the Harbor synchronous rollout.

    Args are the normal slime argument namespace.  Configuration paths are
    resolved in this order:

    * ``args.harbor_job_config`` / ``args.dressage_harbor_job_config`` /
      ``DRESSAGE_HARBOR_JOB_CONFIG``;
    * ``args.harbor_integration_config`` /
      ``args.dressage_harbor_integration_config`` /
      ``DRESSAGE_HARBOR_INTEGRATION_CONFIG``;
    * optional eval equivalents or ``DRESSAGE_HARBOR_EVAL_JOB_CONFIG``.

    Argument values may be paths, already-validated model objects, or mappings.
    The environment variables are always interpreted as paths.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        job_config_type = _harbor_job_config_type()

        integration_source = _first_config_value(
            args,
            (
                "harbor_integration_config",
                "harbor_integration_config_path",
                "dressage_harbor_integration_config",
                "dressage_harbor_integration_config_path",
            ),
            "DRESSAGE_HARBOR_INTEGRATION_CONFIG",
        )
        if integration_source is None:
            raise HarborDataSourceConfigurationError(
                "Harbor integration config is required; set "
                "DRESSAGE_HARBOR_INTEGRATION_CONFIG or args.harbor_integration_config"
            )
        self.integration_config = _load_integration_config(integration_source)
        if self.integration_config.execution_mode != "training":
            raise HarborDataSourceConfigurationError(
                "HarborDataSource requires integration execution_mode='training'"
            )
        self.integration_fingerprint = self.integration_config.fingerprint()

        self.n_samples_per_prompt = _positive_int_arg(
            args, "n_samples_per_prompt", default=1
        )
        estimator = str(getattr(args, "advantage_estimator", "") or "").lower()
        if estimator in {"grpo", "gspo"} and self.n_samples_per_prompt < 2:
            raise HarborDataSourceConfigurationError(
                f"{estimator.upper()} Harbor training requires "
                "n_samples_per_prompt >= 2"
            )
        self.n_samples_per_eval_prompt = _positive_int_arg(
            args,
            "n_samples_per_eval_prompt",
            default=self.n_samples_per_prompt,
        )

        self.rollout_seed = int(getattr(args, "rollout_seed", 42))
        self.rollout_shuffle = bool(getattr(args, "rollout_shuffle", False))

        job_source = _first_config_value(
            args,
            (
                "harbor_job_config",
                "harbor_job_config_path",
                "dressage_harbor_job_config",
                "dressage_harbor_job_config_path",
            ),
            "DRESSAGE_HARBOR_JOB_CONFIG",
        )
        if job_source is None:
            raise HarborDataSourceConfigurationError(
                "Harbor job config is required; set DRESSAGE_HARBOR_JOB_CONFIG "
                "or args.harbor_job_config"
            )
        self.job_config = _load_job_config(job_source, job_config_type)
        tasks = _resolve_tasks(self.job_config)
        self._specs = _build_registry(
            tasks=tasks,
            agents=getattr(self.job_config, "agents", None),
            integration_fingerprint=self.integration_fingerprint,
            scope="train",
        )

        eval_source = _first_config_value(
            args,
            (
                "harbor_eval_job_config",
                "harbor_eval_job_config_path",
                "dressage_harbor_eval_job_config",
                "dressage_harbor_eval_job_config_path",
            ),
            "DRESSAGE_HARBOR_EVAL_JOB_CONFIG",
        )
        self.eval_job_config: Any | None = None
        self._eval_specs: tuple[HarborPromptSpec, ...] = ()
        if eval_source is not None:
            self.eval_job_config = _load_job_config(eval_source, job_config_type)
            eval_tasks = _resolve_tasks(self.eval_job_config)
            self._eval_specs = _build_registry(
                tasks=eval_tasks,
                agents=getattr(self.eval_job_config, "agents", None),
                integration_fingerprint=self.integration_fingerprint,
                scope="eval",
            )

        self._spec_by_id = {
            spec.spec_id: spec for spec in (*self._specs, *self._eval_specs)
        }
        if len(self._spec_by_id) != len(self._specs) + len(self._eval_specs):
            raise HarborDataSourceConfigurationError(
                "resolved Harbor registry contains duplicate spec IDs"
            )

        run_id = _first_nonempty_attr(
            args, ("harbor_run_id", "training_run_id", "run_id")
        )
        if run_id is None:
            run_id = os.environ.get("DRESSAGE_HARBOR_RUN_ID")
        self._run_id_explicit = run_id is not None and run_id != ""
        self.run_id = str(run_id) if self._run_id_explicit else uuid.uuid4().hex

        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.last_batch_weight_version: str | None = None
        self.last_runtime_incarnation: str | None = None

        self._source_digest = _source_digest(
            self.job_config,
            n_samples=self.n_samples_per_prompt,
            seed=self.rollout_seed,
            shuffle=self.rollout_shuffle,
        )
        self._resolved_digest = _registry_digest(self._specs)
        self._eval_source_digest = (
            _source_digest(
                self.eval_job_config,
                n_samples=self.n_samples_per_eval_prompt,
                seed=self.rollout_seed,
                shuffle=False,
            )
            if self.eval_job_config is not None
            else None
        )
        self._eval_resolved_digest = (
            _registry_digest(self._eval_specs) if self._eval_specs else None
        )

    @property
    def specs(self) -> tuple[HarborPromptSpec, ...]:
        """The immutable training registry in resolved source order."""

        return self._specs

    @property
    def eval_specs(self) -> tuple[HarborPromptSpec, ...]:
        """The immutable eval registry, empty when eval was not configured."""

        return self._eval_specs

    def resolve_spec(self, spec_id: str) -> HarborPromptSpec:
        """Resolve an opaque Sample identifier to its in-memory Harbor config."""

        try:
            return self._spec_by_id[str(spec_id)]
        except KeyError as exc:
            raise KeyError(f"unknown Harbor prompt spec {spec_id!r}") from exc

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """Return training groups and advance only the training cursor."""

        count = _nonnegative_count(num_samples)
        groups: list[list[Sample]] = []
        for _ in range(count):
            spec = self._next_training_spec()
            group_index = self.sample_group_index
            instance_id = _instance_id(self.run_id, "train", group_index, spec.spec_id)
            group = self._sample_group(
                spec,
                n_attempts=self.n_samples_per_prompt,
                group_index=group_index,
                first_sample_index=self.sample_index,
                instance_id=instance_id,
            )
            self.sample_group_index += 1
            self.sample_index += len(group)
            groups.append(group)
        return groups

    def get_eval_samples(self, num_samples: int | None = None) -> list[list[Sample]]:
        """Return stable eval groups without mutating any training state.

        With no explicit count every resolved eval task-agent pair is returned
        once.  Repeated calls return equivalent identifiers and indices.
        """

        if not self._eval_specs:
            raise HarborDataSourceConfigurationError(
                "evaluation requested but DRESSAGE_HARBOR_EVAL_JOB_CONFIG is not configured"
            )
        count = (
            len(self._eval_specs)
            if num_samples is None
            else _nonnegative_count(num_samples)
        )
        groups: list[list[Sample]] = []
        for position in range(count):
            spec = self._eval_specs[position % len(self._eval_specs)]
            # Negative indices form a stable namespace independent of training.
            group_index = -(position + 1)
            first_index = -(position * self.n_samples_per_eval_prompt + 1)
            instance_id = _instance_id(self.run_id, "eval", position, spec.spec_id)
            groups.append(
                self._sample_group(
                    spec,
                    n_attempts=self.n_samples_per_eval_prompt,
                    group_index=group_index,
                    first_sample_index=first_index,
                    instance_id=instance_id,
                    descending_indices=True,
                )
            )
        return groups

    def add_samples(self, samples: list[list[Sample]]) -> None:
        """Reject retry insertion; group retries belong to the Harbor executor."""

        raise RuntimeError(
            "HarborDataSource is read-only; retry complete Harbor groups in the "
            "rollout executor instead of adding Samples back to the source"
        )

    def record_batch_state(
        self,
        weight_version: str | None,
        runtime_incarnation: str | None,
    ) -> None:
        """Record the backend identity observed by the last committed batch."""

        validated_weight_version = _optional_nonempty_string(
            weight_version, "weight_version"
        )
        validated_runtime_incarnation = _optional_nonempty_string(
            runtime_incarnation, "runtime_incarnation"
        )
        self.last_batch_weight_version = validated_weight_version
        self.last_runtime_incarnation = validated_runtime_incarnation

    def save(self, rollout_id: int | str) -> None:
        """Atomically persist the safe training cursor state as JSON."""

        path = self._checkpoint_path(rollout_id, for_load=False)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "source_digest": self._source_digest,
            "integration_fingerprint": self.integration_fingerprint,
            "resolved_digest": self._resolved_digest,
            "eval_source_digest": self._eval_source_digest,
            "eval_resolved_digest": self._eval_resolved_digest,
            "last_batch": {
                "weight_version": self.last_batch_weight_version,
                "runtime_incarnation": self.last_runtime_incarnation,
            },
            "state": {
                "epoch": self.epoch_id,
                "cursor": self.sample_offset,
                "group_index": self.sample_group_index,
                "sample_index": self.sample_index,
            },
        }
        _atomic_json_write(path, payload)

    def load(self, rollout_id: int | str | None = None) -> None:
        """Restore state, failing closed on missing files or source drift."""

        if rollout_id is None:
            raise HarborDataSourceCheckpointError(
                "rollout_id is required to load a HarborDataSource checkpoint"
            )
        # slime calls ``load(start_rollout_id - 1)`` even for a brand-new run.
        # Its fresh-run sentinel is -1, for which there cannot be prior source
        # state.  Every non-negative resume point remains fail-closed below.
        if str(rollout_id) == "-1":
            return
        path = self._checkpoint_path(rollout_id, for_load=True)
        if not path.is_file():
            raise HarborDataSourceCheckpointError(
                f"HarborDataSource checkpoint does not exist: {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarborDataSourceCheckpointError(
                f"failed to read HarborDataSource checkpoint {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise HarborDataSourceCheckpointError(
                "HarborDataSource checkpoint root must be an object"
            )

        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "source_digest": self._source_digest,
            "integration_fingerprint": self.integration_fingerprint,
            "resolved_digest": self._resolved_digest,
            "eval_source_digest": self._eval_source_digest,
            "eval_resolved_digest": self._eval_resolved_digest,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise HarborDataSourceCheckpointError(
                    f"HarborDataSource checkpoint {key} mismatch; source/config drift detected"
                )

        checkpoint_run_id = payload.get("run_id")
        if not isinstance(checkpoint_run_id, str) or not checkpoint_run_id:
            raise HarborDataSourceCheckpointError(
                "checkpoint run_id must be a non-empty string"
            )
        if self._run_id_explicit and checkpoint_run_id != self.run_id:
            raise HarborDataSourceCheckpointError(
                "HarborDataSource checkpoint run_id does not match the configured run"
            )

        last_batch = payload.get("last_batch")
        if not isinstance(last_batch, dict):
            raise HarborDataSourceCheckpointError(
                "checkpoint last_batch must be an object"
            )
        weight_version = _checkpoint_optional_string(last_batch, "weight_version")
        runtime_incarnation = _checkpoint_optional_string(
            last_batch, "runtime_incarnation"
        )

        state = payload.get("state")
        if not isinstance(state, dict):
            raise HarborDataSourceCheckpointError("checkpoint state must be an object")
        epoch = _checkpoint_int(state, "epoch")
        cursor = _checkpoint_int(state, "cursor")
        group_index = _checkpoint_int(state, "group_index")
        sample_index = _checkpoint_int(state, "sample_index")
        if cursor > len(self._specs):
            raise HarborDataSourceCheckpointError(
                "checkpoint cursor exceeds the resolved Harbor registry length"
            )

        self.run_id = checkpoint_run_id
        self.epoch_id = epoch
        self.sample_offset = cursor
        self.sample_group_index = group_index
        self.sample_index = sample_index
        self.last_batch_weight_version = weight_version
        self.last_runtime_incarnation = runtime_incarnation

    def __len__(self) -> int:
        return len(self._specs)

    def _next_training_spec(self) -> HarborPromptSpec:
        if self.sample_offset >= len(self._specs):
            self.epoch_id += 1
            self.sample_offset = 0
        order = list(range(len(self._specs)))
        if self.rollout_shuffle:
            random.Random(self.rollout_seed + self.epoch_id).shuffle(order)
        spec = self._specs[order[self.sample_offset]]
        self.sample_offset += 1
        return spec

    def _sample_group(
        self,
        spec: HarborPromptSpec,
        *,
        n_attempts: int,
        group_index: int,
        first_sample_index: int,
        instance_id: str,
        descending_indices: bool = False,
    ) -> list[Sample]:
        group: list[Sample] = []
        for attempt_slot in range(n_attempts):
            index = (
                first_sample_index - attempt_slot
                if descending_indices
                else first_sample_index + attempt_slot
            )
            group.append(
                Sample(
                    group_index=group_index,
                    index=index,
                    prompt=spec.task_uri,
                    metadata={
                        "harbor_spec_id": spec.spec_id,
                        "harbor_task_id": spec.task_id,
                        "harbor_agent_id": spec.agent_id,
                        "harbor_attempt_slot": attempt_slot,
                        "harbor_instance_id": instance_id,
                    },
                )
            )
        return group

    def _checkpoint_path(self, rollout_id: int | str, *, for_load: bool) -> Path:
        label = str(rollout_id)
        if not _SAFE_CHECKPOINT_ID.fullmatch(label) or label in {".", ".."}:
            raise HarborDataSourceCheckpointError(f"invalid rollout_id {rollout_id!r}")
        explicit_root = getattr(self.args, "harbor_data_source_checkpoint_dir", None)
        root = explicit_root or getattr(self.args, "load" if for_load else "save", None)
        if not root:
            operation = "load" if for_load else "save"
            raise HarborDataSourceCheckpointError(
                f"args.{operation} (or args.harbor_data_source_checkpoint_dir) "
                f"is required to {operation} HarborDataSource state"
            )
        base = Path(root).expanduser()
        if explicit_root:
            directory = base
        else:
            directory = base / "rollout"
        return directory / _CHECKPOINT_FILENAME.format(rollout_id=label)


def _harbor_job_config_type() -> type[Any]:
    """Preflight the optional runtime, then import the public Harbor model."""

    from dressage.integrations.harbor.compat import require_harbor_runtime

    require_harbor_runtime()
    try:
        from harbor.models.job.config import JobConfig
    except (
        ImportError,
        ModuleNotFoundError,
    ) as exc:  # pragma: no cover - guarded by version check
        raise HarborDataSourceConfigurationError(
            "Harbor 0.18 is installed but harbor.models.job.config.JobConfig is unavailable"
        ) from exc
    return JobConfig


def _load_integration_config(source: Any) -> HarborIntegrationConfig:
    if isinstance(source, HarborIntegrationConfig):
        return source
    if isinstance(source, Mapping):
        return HarborIntegrationConfig.model_validate(dict(source))
    try:
        return load_config(Path(source))
    except (OSError, TypeError, ValueError) as exc:
        raise HarborDataSourceConfigurationError(
            f"failed to load Harbor integration config: {exc}"
        ) from exc


def _load_job_config(source: Any, job_config_type: type[Any]) -> Any:
    if isinstance(source, job_config_type):
        copier = getattr(source, "model_copy", None)
        return copier(deep=True) if callable(copier) else copy.deepcopy(source)
    if isinstance(source, Mapping):
        payload: Any = dict(source)
    else:
        try:
            path = Path(source).expanduser()
            raw = path.read_text(encoding="utf-8")
        except (OSError, TypeError) as exc:
            raise HarborDataSourceConfigurationError(
                f"failed to read Harbor job config: {exc}"
            ) from exc
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                payload = json.loads(raw)
            elif suffix in {".yaml", ".yml"}:
                try:
                    import yaml
                except (
                    ImportError
                ) as exc:  # pragma: no cover - optional dependency path
                    raise HarborDataSourceConfigurationError(
                        "PyYAML is required to load a Harbor YAML job config"
                    ) from exc
                payload = yaml.safe_load(raw)
            else:
                raise HarborDataSourceConfigurationError(
                    "Harbor job config must use .json, .yaml, or .yml"
                )
        except (json.JSONDecodeError, ValueError) as exc:
            raise HarborDataSourceConfigurationError(
                f"failed to parse Harbor job config: {exc}"
            ) from exc
    if not isinstance(payload, Mapping):
        raise HarborDataSourceConfigurationError(
            "Harbor job config must contain a mapping at its root"
        )
    try:
        return job_config_type.model_validate(dict(payload))
    except Exception as exc:
        raise HarborDataSourceConfigurationError(
            f"invalid Harbor job config: {exc}"
        ) from exc


def _resolve_tasks(job_config: Any) -> tuple[Any, ...]:
    """Resolve JobConfig tasks through the version-pinned compat boundary."""

    from dressage.integrations.harbor import compat

    resolver = getattr(compat, "resolve_task_configs", None)
    if not callable(resolver):
        raise HarborDataSourceConfigurationError(
            "Harbor compat layer does not expose resolve_task_configs; "
            "the pinned Harbor contract is incomplete"
        )
    try:
        result = resolver(job_config)
        if inspect.isawaitable(result):
            result = _run_awaitable(result)
    except Exception as exc:
        raise HarborDataSourceConfigurationError(
            f"failed to resolve Harbor tasks and datasets: {exc}"
        ) from exc
    if (
        not isinstance(result, Sequence)
        or isinstance(result, (str, bytes))
        or not result
    ):
        raise HarborDataSourceConfigurationError(
            "Harbor task resolver returned no TaskConfig entries"
        )
    return tuple(result)


def _run_awaitable(awaitable: Awaitable[Any]) -> Any:
    """Run Harbor's async resolver from sync construction, even under a loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: list[Any] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:  # surfaced on the constructing thread
            errors.append(exc)

    thread = threading.Thread(target=target, name="harbor-task-resolver", daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


def _build_registry(
    *,
    tasks: Sequence[Any],
    agents: Any,
    integration_fingerprint: str,
    scope: str,
) -> tuple[HarborPromptSpec, ...]:
    if (
        not isinstance(agents, Sequence)
        or isinstance(agents, (str, bytes))
        or not agents
    ):
        raise HarborDataSourceConfigurationError(
            "Harbor JobConfig.agents must contain at least one AgentConfig"
        )
    occurrences: dict[str, int] = {}
    specs: list[HarborPromptSpec] = []
    for task in tasks:
        task_runtime = _dump_model(task, reveal_sensitive=True)
        task_public = _redact_public(task_runtime)
        task_public_fp = _fingerprint(task_public)
        for agent in agents:
            agent_runtime = _dump_model(agent, reveal_sensitive=True)
            agent_public = _redact_public(agent_runtime)
            agent_public_fp = _fingerprint(agent_public)
            public_fp = _fingerprint({"task": task_public, "agent": agent_public})
            # Runtime partitioning is agent-scoped: one temporary Harbor Job can
            # safely contain many tasks as long as their AgentConfig identity is
            # identical.  The task remains part of the public/spec identity.
            runtime_fp = _fingerprint(agent_runtime)
            occurrence = occurrences.get(public_fp, 0)
            occurrences[public_fp] = occurrence + 1
            spec_id = (
                "harbor-"
                + _fingerprint(
                    {
                        "scope": scope,
                        "public_fingerprint": public_fp,
                        "occurrence": occurrence,
                    }
                )[:24]
            )
            task_id = "task-" + task_public_fp[:16]
            agent_id = "agent-" + agent_public_fp[:16]
            specs.append(
                HarborPromptSpec(
                    spec_id=spec_id,
                    task_config=_deepcopy_model(task),
                    agent_config=_deepcopy_model(agent),
                    task_id=task_id,
                    agent_id=agent_id,
                    task_uri=f"harbor://task/{task_id}",
                    public_fingerprint=public_fp,
                    runtime_fingerprint=runtime_fp,
                    integration_fingerprint=integration_fingerprint,
                    occurrence_index=occurrence,
                    scope=scope,
                )
            )
    if not specs:
        raise HarborDataSourceConfigurationError("resolved Harbor registry is empty")
    return tuple(specs)


def _deepcopy_model(value: Any) -> Any:
    copier = getattr(value, "model_copy", None)
    return copier(deep=True) if callable(copier) else copy.deepcopy(value)


def _dump_model(value: Any, *, reveal_sensitive: bool) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        kwargs: dict[str, Any] = {"mode": "json", "exclude_none": True}
        if reveal_sensitive:
            kwargs["context"] = {"redact_sensitive_env": False}
        try:
            return _jsonable(dump(**kwargs))
        except TypeError:
            kwargs.pop("context", None)
            try:
                return _jsonable(dump(**kwargs))
            except TypeError:
                return _jsonable(dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return _jsonable(value)
    if hasattr(value, "__dict__"):
        return _jsonable(
            {key: item for key, item in vars(value).items() if not key.startswith("_")}
        )
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda p: str(p[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)) or enum_value is None:
        if enum_value is not None:
            return enum_value
    return str(value)


def _redact_public(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized == "env" and isinstance(item, Mapping):
                result[key] = {
                    str(env_key): "<redacted>" for env_key in sorted(item, key=str)
                }
            elif any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                result[key] = "<redacted>"
            else:
                result[key] = _redact_public(item, parent_key=key)
        return result
    if isinstance(value, list):
        return [_redact_public(item, parent_key=parent_key) for item in value]
    if isinstance(value, str):
        return _strip_url_userinfo(value)
    return value


def _strip_url_userinfo(value: str) -> str:
    if "://" not in value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.username is None and parsed.password is None:
        return value
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_digest(job_config: Any, *, n_samples: int, seed: int, shuffle: bool) -> str:
    return _fingerprint(
        {
            "job": _redact_public(_dump_model(job_config, reveal_sensitive=True)),
            "n_samples_per_prompt": n_samples,
            "rollout_seed": seed,
            "rollout_shuffle": shuffle,
        }
    )


def _registry_digest(specs: Sequence[HarborPromptSpec]) -> str:
    return _fingerprint(
        [
            {
                "spec_id": spec.spec_id,
                "task_id": spec.task_id,
                "agent_id": spec.agent_id,
                "public_fingerprint": spec.public_fingerprint,
                "integration_fingerprint": spec.integration_fingerprint,
                "occurrence_index": spec.occurrence_index,
                "scope": spec.scope,
            }
            for spec in specs
        ]
    )


def _instance_id(run_id: str, scope: str, group_index: int, spec_id: str) -> str:
    return (
        "instance-"
        + hashlib.sha256(
            f"{run_id}\0{scope}\0{group_index}\0{spec_id}".encode("utf-8")
        ).hexdigest()[:24]
    )


def _first_config_value(args: Any, attrs: Sequence[str], env_name: str) -> Any | None:
    value = _first_nonempty_attr(args, attrs)
    if value is not None:
        return value
    environment_value = os.environ.get(env_name)
    return environment_value if environment_value else None


def _first_nonempty_attr(args: Any, attrs: Sequence[str]) -> Any | None:
    for attr in attrs:
        value = getattr(args, attr, None)
        if value is not None and value != "":
            return value
    return None


def _positive_int_arg(args: Any, name: str, *, default: int) -> int:
    value = getattr(args, name, None)
    if value is None:
        value = default
    if isinstance(value, bool):
        raise HarborDataSourceConfigurationError(
            f"args.{name} must be a positive integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HarborDataSourceConfigurationError(
            f"args.{name} must be a positive integer"
        ) from exc
    if parsed < 1 or parsed != value:
        raise HarborDataSourceConfigurationError(
            f"args.{name} must be a positive integer"
        )
    return parsed


def _nonnegative_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("sample group count must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample group count must be a non-negative integer") from exc
    if parsed < 0 or parsed != value:
        raise ValueError("sample group count must be a non-negative integer")
    return parsed


def _checkpoint_int(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarborDataSourceCheckpointError(
            f"checkpoint state.{key} must be a non-negative integer"
        )
    return value


def _optional_nonempty_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or None")
    return value


def _checkpoint_optional_string(state: Mapping[str, Any], key: str) -> str | None:
    try:
        return _optional_nonempty_string(state.get(key), f"checkpoint last_batch.{key}")
    except ValueError as exc:
        raise HarborDataSourceCheckpointError(str(exc)) from exc


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "HarborDataSource",
    "HarborDataSourceCheckpointError",
    "HarborDataSourceConfigurationError",
    "HarborDataSourceError",
    "HarborPromptSpec",
]
