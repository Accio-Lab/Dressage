from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from agentic_rollout_contract.models import VirtualCapacity
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeSafety(ConfigModel):
    cgroup_parent: str = "dressage-distroll-rollout.slice"
    pids_per_sandbox: int = Field(default=4096, ge=64)
    storage_mb_per_sandbox: int = Field(default=30_000, ge=1024)
    storage_size_enforced: bool = False


class ImageCacheSettings(ConfigModel):
    enabled: bool = False
    cache_root: Path = Path("/var/lib/dressage-distroll-worker/image-cache")
    docker_root: Path = Path("/var/lib/docker")
    pull_timeout_sec: float = Field(default=900.0, gt=0)
    pull_max_attempts: int = Field(default=3, ge=1)
    pull_retry_base_sec: float = Field(default=1.0, ge=0)
    max_concurrent_pulls: int = Field(default=2, ge=1)
    min_free_bytes_before_pull: int = Field(default=50 * 1024**3, ge=0)
    cleaner_enabled: bool = True
    cleaner_scan_interval_sec: float = Field(default=300.0, gt=0)
    idle_ttl_sec: float = Field(default=7200.0, ge=0)
    high_used_percent: float = Field(default=85.0, gt=0, lt=100)
    low_used_percent: float = Field(default=75.0, ge=0, lt=100)
    pressure_min_idle_sec: float = Field(default=1800.0, ge=0)
    max_delete_per_scan: int = Field(default=20, ge=1)
    max_scan_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_cache(self) -> ImageCacheSettings:
        if self.low_used_percent >= self.high_used_percent:
            raise ValueError("image cache low watermark must be below high watermark")
        return self


class TaskResourceDefaults(ConfigModel):
    cpus: int = Field(default=1, ge=1)
    memory_mb: int = Field(default=4096, ge=256)


class TaskResourceSettings(ConfigModel):
    index_path: Path = Path(
        "/opt/dressage-distroll/configs/task-resource-index-v1.json"
    )
    defaults: TaskResourceDefaults = Field(default_factory=TaskResourceDefaults)


class PolicyEndpointSettings(ConfigModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("policy URL contains an invalid port") from exc
        if parsed.scheme != "http" or not parsed.hostname or port is None:
            raise ValueError("policy URL must use http://host:port")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("policy URL must not contain credentials")
        if parsed.path or parsed.query or parsed.fragment:
            raise ValueError("policy URL must not contain path, query, or fragment")
        return normalized


class WorkerNode(ConfigModel):
    worker_id: str
    address: str = Field(min_length=1)
    relay_base_url: str = Field(min_length=1)
    labels: dict[str, str] = Field(default_factory=dict)
    virtual_capacity: VirtualCapacity
    runtime_safety: RuntimeSafety = Field(default_factory=RuntimeSafety)
    image_cache: ImageCacheSettings = Field(default_factory=ImageCacheSettings)
    task_catalog_root: Path = Path("/opt/dressage-distroll/tasks")
    state_dir: Path = Path("/var/lib/dressage-distroll-worker")


class Inventory(ConfigModel):
    schema_version: Literal["rollout-cluster/v2"] = "rollout-cluster/v2"
    cluster_id: str
    execution_timeout_sec: int = Field(default=3900, ge=60)
    settlement_timeout_sec: int = Field(default=3900, ge=60)
    heartbeat_interval_sec: int = Field(default=30, ge=1)
    heartbeat_timeout_sec: int = Field(default=90, ge=5)
    assignment_ttl_sec: int = Field(default=300, ge=60)
    policy: PolicyEndpointSettings
    task_resources: TaskResourceSettings = Field(default_factory=TaskResourceSettings)
    workers: list[WorkerNode]

    @model_validator(mode="after")
    def validate_inventory(self) -> Inventory:
        if self.heartbeat_interval_sec >= self.heartbeat_timeout_sec:
            raise ValueError("heartbeat interval must be smaller than timeout")
        if self.assignment_ttl_sec <= self.heartbeat_timeout_sec:
            raise ValueError(
                "assignment_ttl_sec must be greater than heartbeat_timeout_sec"
            )
        ids = [worker.worker_id for worker in self.workers]
        addresses = [worker.address for worker in self.workers]
        if len(ids) != len(set(ids)):
            raise ValueError("worker_id values must be unique")
        if len(addresses) != len(set(addresses)):
            raise ValueError("worker addresses must be unique")
        if not self.workers:
            raise ValueError("inventory must contain at least one worker")
        return self

    def worker(self, worker_id: str) -> WorkerNode:
        for worker in self.workers:
            if worker.worker_id == worker_id:
                return worker
        raise KeyError(worker_id)

    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


class AdmissionSettings(ConfigModel):
    reservation_ttl_sec: float = Field(default=120, gt=0)
    sweep_sec: float = Field(default=1, gt=0)
    retry_after_sec: float = Field(default=2, gt=0)
    long_poll_max_sec: float = Field(default=30, gt=0, le=30)


class ControlSettings(ConfigModel):
    inventory_path: Path
    database_url: str
    listen_host: str = Field(default="0.0.0.0", min_length=1)
    listen_port: int = 19443
    database_pool_min_size: int = Field(default=1, ge=1)
    database_pool_max_size: int = Field(default=16, ge=1)
    admission: AdmissionSettings = Field(default_factory=AdmissionSettings)

    @model_validator(mode="after")
    def validate_database_pool(self) -> ControlSettings:
        if self.database_pool_max_size < self.database_pool_min_size:
            raise ValueError("database pool max size must be >= min size")
        return self


class SandboxGuardSettings(ConfigModel):
    model_config = ConfigDict(allow_inf_nan=False)
    mode: Literal["off", "observe", "enforce"] = "observe"
    sample_interval_sec: float = Field(default=2.0, gt=0)
    oom_window_sec: float = Field(default=10.0, gt=0)
    oom_kill_threshold: int = Field(default=3, ge=1)
    memory_full_ratio: float = Field(default=0.2, gt=0, le=1)
    terminate_grace_sec: int = Field(default=3, ge=0, le=10)
    local_cleanup_timeout_sec: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_window(self):
        if self.oom_window_sec < self.sample_interval_sec:
            raise ValueError("guard window must cover at least one sample interval")
        if self.local_cleanup_timeout_sec <= self.terminate_grace_sec:
            raise ValueError("guard cleanup budget must exceed terminate grace")
        return self


class WorkerSettings(ConfigModel):
    inventory_path: Path
    worker_id: str
    control_plane_url: str
    listen_host: str = Field(default="0.0.0.0", min_length=1)
    listen_port: int = 19090
    poll_interval_sec: float = Field(default=1.0, gt=0)
    retry_max_sec: float = Field(default=30.0, gt=0)
    control_request_timeout_sec: float = Field(default=15.0, gt=0)
    policy_transport_retry_attempts: int = Field(default=20, ge=1, le=32)
    policy_transport_retry_base_sec: float = Field(default=0.5, ge=0)
    policy_transport_retry_max_sec: float = Field(default=10.0, ge=0)
    sandbox_guard: SandboxGuardSettings = Field(default_factory=SandboxGuardSettings)

    @model_validator(mode="after")
    def validate_control_url(self) -> WorkerSettings:
        _http_service_url(self.control_plane_url)
        if self.policy_transport_retry_max_sec < self.policy_transport_retry_base_sec:
            raise ValueError("policy transport retry max delay must be >= base delay")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve()
    data = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"configuration must be a YAML object: {selected}")
    return data


def load_inventory(path: str | Path) -> Inventory:
    return Inventory.model_validate(load_yaml(path))


def control_settings_from_env() -> ControlSettings:
    return ControlSettings(
        inventory_path=Path(os.environ["ROLLOUT_INVENTORY"]),
        database_url=os.environ["ROLLOUT_DATABASE_URL"],
        listen_host=os.environ.get("ROLLOUT_CONTROL_HOST", "0.0.0.0"),
        listen_port=int(os.environ.get("ROLLOUT_CONTROL_PORT", "19443")),
        database_pool_min_size=int(
            os.environ.get("ROLLOUT_DATABASE_POOL_MIN_SIZE", "1")
        ),
        database_pool_max_size=int(
            os.environ.get("ROLLOUT_DATABASE_POOL_MAX_SIZE", "16")
        ),
        admission=AdmissionSettings(
            reservation_ttl_sec=float(
                os.environ.get("ROLLOUT_RESERVATION_TTL_SEC", "120")
            ),
            sweep_sec=float(os.environ.get("ROLLOUT_ADMISSION_SWEEP_SEC", "1")),
            retry_after_sec=float(
                os.environ.get("ROLLOUT_ADMISSION_RETRY_AFTER_SEC", "2")
            ),
            long_poll_max_sec=float(os.environ.get("ROLLOUT_LONG_POLL_MAX_SEC", "30")),
        ),
    )


def worker_settings_from_env() -> WorkerSettings:
    return WorkerSettings(
        inventory_path=Path(os.environ["ROLLOUT_INVENTORY"]),
        worker_id=os.environ["ROLLOUT_WORKER_ID"],
        control_plane_url=_http_service_url(os.environ["ROLLOUT_CONTROL_URL"]),
        listen_host=os.environ.get("ROLLOUT_WORKER_HOST", "0.0.0.0"),
        listen_port=int(os.environ.get("ROLLOUT_WORKER_PORT", "19090")),
        poll_interval_sec=float(os.environ.get("ROLLOUT_WORKER_POLL_SEC", "1")),
        retry_max_sec=float(os.environ.get("ROLLOUT_WORKER_RETRY_MAX_SEC", "30")),
        control_request_timeout_sec=float(
            os.environ.get("ROLLOUT_WORKER_CONTROL_TIMEOUT_SEC", "15")
        ),
        policy_transport_retry_attempts=int(
            os.environ.get("ROLLOUT_POLICY_TRANSPORT_RETRY_ATTEMPTS", "20")
        ),
        policy_transport_retry_base_sec=float(
            os.environ.get("ROLLOUT_POLICY_TRANSPORT_RETRY_BASE_SEC", "0.5")
        ),
        policy_transport_retry_max_sec=float(
            os.environ.get("ROLLOUT_POLICY_TRANSPORT_RETRY_MAX_SEC", "10")
        ),
        sandbox_guard=SandboxGuardSettings(
            mode=os.environ.get("ROLLOUT_SANDBOX_GUARD_MODE", "observe"),
            sample_interval_sec=float(
                os.environ.get("ROLLOUT_SANDBOX_GUARD_SAMPLE_SEC", "2")
            ),
            oom_window_sec=float(
                os.environ.get("ROLLOUT_SANDBOX_GUARD_WINDOW_SEC", "10")
            ),
            oom_kill_threshold=int(
                os.environ.get("ROLLOUT_SANDBOX_GUARD_OOM_KILLS", "3")
            ),
            memory_full_ratio=float(
                os.environ.get("ROLLOUT_SANDBOX_GUARD_MEMORY_FULL_RATIO", "0.2")
            ),
            terminate_grace_sec=int(
                os.environ.get("ROLLOUT_SANDBOX_GUARD_TERM_SEC", "3")
            ),
            local_cleanup_timeout_sec=float(
                os.environ.get("ROLLOUT_SANDBOX_GUARD_CLEANUP_SEC", "30")
            ),
        ),
    )


def _http_service_url(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme != "http" or not parsed.netloc:
        raise ValueError(f"service URL must use http: {value}")
    return value.rstrip("/")
