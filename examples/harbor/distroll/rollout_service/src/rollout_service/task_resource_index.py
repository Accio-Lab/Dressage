from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from agentic_rollout_contract.canonical import canonical_digest
from agentic_rollout_contract.models import (
    DockerTaskLimit,
    ResolvedTaskResources,
    SchedulingCharge,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

from .config import Inventory, TaskResourceDefaults

ValidationProfile = Literal["standard", "trusted-security"]


class TaskResourceIndexError(ValueError):
    pass


class IndexModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskResourceEntry(IndexModel):
    cpus: int = Field(ge=1)
    memory_mb: int = Field(ge=256)
    cpu_source: Literal["task", "default"]
    memory_source: Literal["task", "default"]
    task_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    admitted_image: str | None = None
    execution_image: str | None = None
    admission_entry_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_digest_only_entry(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        admitted_image = str(value.get("admitted_image") or "")
        if value.get("execution_image") is None and "@sha256:" in admitted_image:
            return {**value, "execution_image": admitted_image}
        return value

    def resolved(self) -> ResolvedTaskResources:
        return ResolvedTaskResources(
            charge=SchedulingCharge(
                cpu_units=self.cpus,
                memory_mb=self.memory_mb,
            ),
            limit=DockerTaskLimit(cpus=self.cpus, memory_mb=self.memory_mb),
            cpu_source=self.cpu_source,
            memory_source=self.memory_source,
            task_config_digest=self.task_config_digest,
        )


class TaskResourceIndex(IndexModel):
    schema_version: Literal["task-resource-index/v1", "task-resource-index/v2"] = (
        "task-resource-index/v1"
    )
    defaults: TaskResourceDefaults
    tasks: dict[str, TaskResourceEntry] = Field(min_length=1)
    admission_manifest_digest: str | None = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    validation_profile: ValidationProfile = "standard"
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    _source_canonical_payload: dict[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_admission_binding(self) -> TaskResourceIndex:
        if self.schema_version == "task-resource-index/v2":
            if self.admission_manifest_digest is None:
                raise ValueError(
                    "v2 task resource index requires admission manifest digest"
                )
            incomplete = [
                task_ref
                for task_ref, entry in self.tasks.items()
                if len(
                    {
                        entry.admitted_image is None,
                        entry.execution_image is None,
                        entry.admission_entry_digest is None,
                    }
                )
                != 1
            ]
            if incomplete:
                raise ValueError(
                    "v2 task resource index has incomplete image admission entries: "
                    + ", ".join(incomplete[:5])
                )
        return self

    @property
    def supports_runtime_pull(self) -> bool:
        return self.schema_version == "task-resource-index/v2"

    def admitted_image(self, task_ref: str) -> str:
        try:
            entry = self.tasks[task_ref]
        except KeyError as exc:
            raise TaskResourceIndexError(
                f"task is missing from resource index: {task_ref}"
            ) from exc
        if (
            not self.supports_runtime_pull
            or self.admission_manifest_digest is None
            or entry.admitted_image is None
            or entry.execution_image is None
            or entry.admission_entry_digest is None
        ):
            raise TaskResourceIndexError(
                f"task is not bound to a v2 image admission entry: {task_ref}"
            )
        return entry.admitted_image

    def execution_image(self, task_ref: str) -> str:
        self.admitted_image(task_ref)
        execution_image = self.tasks[task_ref].execution_image
        assert execution_image is not None
        return execution_image

    def canonical_payload(self) -> dict:
        if self._source_canonical_payload is not None:
            return self._source_canonical_payload
        return self.model_dump(mode="json", exclude={"digest"}, exclude_none=True)

    def verify_digest(self) -> None:
        actual = canonical_digest(self.canonical_payload())
        if actual != self.digest:
            raise TaskResourceIndexError(
                f"task resource index digest mismatch: expected={self.digest} "
                f"actual={actual}"
            )

    def resolve(self, task_ref: str) -> ResolvedTaskResources:
        try:
            return self.tasks[task_ref].resolved()
        except KeyError as exc:
            raise TaskResourceIndexError(
                f"task is missing from resource index: {task_ref}"
            ) from exc


def build_task_resource_index(
    catalog_root: str | Path,
    defaults: TaskResourceDefaults,
    admission_manifest: str | Path | None = None,
) -> TaskResourceIndex:
    from harbor.models.task.config import TaskConfig, VerifierEnvironmentMode

    root = Path(catalog_root).expanduser().resolve()
    if not root.is_dir():
        raise TaskResourceIndexError(f"task catalog does not exist: {root}")

    admission_payload: dict[str, Any] | None = None
    admission_by_task: dict[str, dict[str, Any]] = {}
    admission_entries: list[dict[str, Any]] = []
    matched_admission_entries: set[int] = set()
    admission_manifest_digest: str | None = None
    validation_profile: ValidationProfile = "standard"
    if admission_manifest is not None:
        selected = Path(admission_manifest).expanduser().resolve()
        try:
            admission_payload = json.loads(selected.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskResourceIndexError(
                f"cannot read image admission manifest {selected}: {exc}"
            ) from exc
        if admission_payload.get("training_image_ready") is not True:
            raise TaskResourceIndexError("image admission manifest is not ready")
        raw_profile = admission_payload.get("validation_profile", "standard")
        if raw_profile not in {"standard", "trusted-security"}:
            raise TaskResourceIndexError(
                f"unsupported validation profile: {raw_profile!r}"
            )
        validation_profile = raw_profile
        if validation_profile == "trusted-security":
            required_flags = {
                "verifier_checked": False,
                "gold_checked": False,
                "lifecycle_checked": False,
                "user_accepted_unvalidated_tasks": True,
            }
            invalid_flags = [
                key
                for key, expected in required_flags.items()
                if admission_payload.get(key) is not expected
            ]
            if invalid_flags:
                raise TaskResourceIndexError(
                    "unvalidated publication has invalid acknowledgement flags: "
                    + ", ".join(invalid_flags)
                )
        raw_tasks = admission_payload.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise TaskResourceIndexError("image admission manifest has no tasks")
        for raw_entry in raw_tasks:
            if not isinstance(raw_entry, dict):
                raise TaskResourceIndexError(
                    "image admission task entry is not an object"
                )
            if validation_profile == "standard":
                expected_rewards = {
                    "unresolved_reward": 0,
                    "gold_reward": 1,
                    "lifecycle_reward": 1,
                }
                missing_or_invalid = [
                    key
                    for key, expected in expected_rewards.items()
                    if isinstance(raw_entry.get(key), bool)
                    or raw_entry.get(key) != expected
                ]
                if missing_or_invalid:
                    raise TaskResourceIndexError(
                        "standard image admission entry lacks successful validation "
                        f"evidence for {raw_entry.get('task_ref') or raw_entry.get('task_id') or '<unknown>'}: "
                        + ", ".join(missing_or_invalid)
                    )
            else:
                forbidden_rewards = sorted(
                    {
                        "unresolved_reward",
                        "gold_reward",
                        "lifecycle_reward",
                    }
                    & raw_entry.keys()
                )
                if forbidden_rewards:
                    raise TaskResourceIndexError(
                        "unvalidated publication must not fabricate validation "
                        "rewards: " + ", ".join(forbidden_rewards)
                    )
            task_id = str(raw_entry.get("task_id") or "").strip()
            task_ref = str(raw_entry.get("task_ref") or "").strip()
            for key in {task_id, task_ref} - {""}:
                if key in admission_by_task:
                    raise TaskResourceIndexError(
                        f"duplicate image admission task key: {key}"
                    )
                admission_by_task[key] = raw_entry
            admission_entries.append(raw_entry)
        admission_manifest_digest = canonical_digest(admission_payload)

    tasks: dict[str, TaskResourceEntry] = {}
    for config_path in sorted(root.rglob("task.toml")):
        task_dir = config_path.parent.resolve()
        try:
            task_ref = task_dir.relative_to(root).as_posix()
        except ValueError as exc:  # pragma: no cover - resolve/rglob invariant
            raise TaskResourceIndexError(
                f"task escaped catalog root: {config_path}"
            ) from exc
        if not task_ref or task_ref == ".":
            raise TaskResourceIndexError(
                f"task.toml must be inside a task directory: {config_path}"
            )
        raw = config_path.read_text(encoding="utf-8")
        try:
            config = TaskConfig.model_validate_toml(raw)
        except Exception as exc:
            raise TaskResourceIndexError(
                f"invalid Harbor task config: {config_path}: {exc}"
            ) from exc
        if config.steps:
            raise TaskResourceIndexError(
                f"multi-step task is unsupported by rollout v1: {task_ref}"
            )
        verifier = config.verifier
        if (
            verifier.environment is not None
            or verifier.environment_mode == VerifierEnvironmentMode.SEPARATE
        ):
            raise TaskResourceIndexError(
                f"separate verifier is unsupported by rollout v1: {task_ref}"
            )
        compose_path = task_dir / "environment" / "docker-compose.yaml"
        if compose_path.exists():
            raise TaskResourceIndexError(
                f"task-authored docker-compose is unsupported by rollout v1: {task_ref}"
            )
        if task_ref in tasks:
            raise TaskResourceIndexError(f"duplicate task ref: {task_ref}")

        admitted_image: str | None = None
        execution_image: str | None = None
        admission_entry_digest: str | None = None
        if admission_payload is not None:
            admission_entry = admission_by_task.get(task_ref)
            if admission_entry is None:
                admission_entry = admission_by_task.get(task_dir.name)
            if admission_entry is not None:
                admitted_image = str(
                    admission_entry.get("admitted_image") or ""
                ).strip()
                if not admitted_image:
                    raise TaskResourceIndexError(
                        f"admission image reference is empty: {task_ref}"
                    )
                if config.environment.docker_image != admitted_image:
                    raise TaskResourceIndexError(
                        f"task image disagrees with image admission manifest: {task_ref}"
                    )
                execution_image = str(
                    admission_entry.get("execution_image") or admitted_image
                ).strip()
                if "@sha256:" not in execution_image:
                    raise TaskResourceIndexError(
                        f"admission execution image is not immutable: {task_ref}"
                    )
                admission_entry_digest = canonical_digest(admission_entry)
                matched_admission_entries.add(id(admission_entry))

        declared_cpus = config.environment.cpus
        declared_memory = config.environment.memory_mb
        try:
            tasks[task_ref] = TaskResourceEntry(
                cpus=declared_cpus if declared_cpus is not None else defaults.cpus,
                memory_mb=(
                    declared_memory
                    if declared_memory is not None
                    else defaults.memory_mb
                ),
                cpu_source="task" if declared_cpus is not None else "default",
                memory_source="task" if declared_memory is not None else "default",
                task_config_digest="sha256:"
                + hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                admitted_image=admitted_image,
                execution_image=execution_image,
                admission_entry_digest=admission_entry_digest,
            )
        except ValidationError as exc:
            raise TaskResourceIndexError(
                f"invalid task resource declaration: {task_ref}: {exc}"
            ) from exc
    if not tasks:
        raise TaskResourceIndexError(f"task catalog contains no tasks: {root}")
    unmatched_admissions = [
        str(entry.get("task_ref") or entry.get("task_id") or "<unknown>")
        for entry in admission_entries
        if id(entry) not in matched_admission_entries
    ]
    if unmatched_admissions:
        raise TaskResourceIndexError(
            "image admission entries are missing from task catalog: "
            + ", ".join(unmatched_admissions[:5])
        )

    payload = {
        "schema_version": (
            "task-resource-index/v2"
            if admission_payload is not None
            else "task-resource-index/v1"
        ),
        "validation_profile": validation_profile,
        "defaults": defaults.model_dump(mode="json"),
        "tasks": {
            task_ref: entry.model_dump(mode="json", exclude_none=True)
            for task_ref, entry in tasks.items()
        },
    }
    if admission_manifest_digest is not None:
        payload["admission_manifest_digest"] = admission_manifest_digest
    return TaskResourceIndex(
        **payload,
        digest=canonical_digest(payload),
    )


def load_task_resource_index(path: str | Path) -> TaskResourceIndex:
    selected = Path(path).expanduser().resolve()
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskResourceIndexError(
            f"cannot read task resource index {selected}: {exc}"
        ) from exc
    index = TaskResourceIndex.model_validate(payload)
    raw_tasks = payload.get("tasks") or {}
    if "validation_profile" not in payload or any(
        isinstance(entry, dict)
        and entry.get("execution_image") is None
        and "@sha256:" in str(entry.get("admitted_image") or "")
        for entry in raw_tasks.values()
    ):
        index._source_canonical_payload = {
            key: value for key, value in payload.items() if key != "digest"
        }
    index.verify_digest()
    return index


def load_inventory_task_resource_index(inventory: Inventory) -> TaskResourceIndex:
    index = load_task_resource_index(inventory.task_resources.index_path)
    if index.defaults != inventory.task_resources.defaults:
        raise TaskResourceIndexError(
            "task resource index defaults do not match cluster inventory"
        )
    return index


def catalog_requires_egress_control(
    index: TaskResourceIndex, catalog_root: str | Path
) -> bool:
    from harbor.models.task.config import NetworkMode, TaskConfig

    root = Path(catalog_root).expanduser().resolve()
    for task_ref in sorted(index.tasks):
        config_path = (root / task_ref / "task.toml").resolve()
        if not config_path.is_relative_to(root) or not config_path.is_file():
            raise TaskResourceIndexError(
                f"task config is missing or escapes catalog root: {task_ref}"
            )
        config = TaskConfig.model_validate_toml(config_path.read_text(encoding="utf-8"))
        baseline = config.environment.resolve_baseline()
        policies = (
            baseline,
            config.agent.explicit_phase_policy() or baseline,
            config.verifier.explicit_phase_policy() or baseline,
        )
        if any(policy.network_mode != NetworkMode.PUBLIC for policy in policies):
            return True
    return False


def write_task_resource_index(index: TaskResourceIndex, path: str | Path) -> None:
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_name(f".{selected.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {**index.canonical_payload(), "digest": index.digest},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)


def task_resource_index_main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable Harbor task resource index"
    )
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--admission-manifest")
    parser.add_argument("--default-cpus", type=int, default=1)
    parser.add_argument("--default-memory-mb", type=int, default=4096)
    args = parser.parse_args()
    defaults = TaskResourceDefaults(
        cpus=args.default_cpus,
        memory_mb=args.default_memory_mb,
    )
    index = build_task_resource_index(
        args.catalog_root,
        defaults,
        admission_manifest=args.admission_manifest,
    )
    write_task_resource_index(index, args.output)
    print(f"task resource index written: {args.output} digest={index.digest}")
