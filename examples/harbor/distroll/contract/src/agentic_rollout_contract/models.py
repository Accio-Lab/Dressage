from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "agentic-rollout/v4"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExecutionState(StrEnum):
    RESERVED = "RESERVED"
    EXPIRED = "EXPIRED"
    ACCEPTED = "ACCEPTED"
    PREPARING = "PREPARING"
    WAITING_ROUTE = "WAITING_ROUTE"
    AGENT_RUNNING = "AGENT_RUNNING"
    VERIFYING = "VERIFYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


TERMINAL_EXECUTION_STATES = frozenset(
    {
        ExecutionState.SUCCEEDED,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
        ExecutionState.EXPIRED,
    }
)


class WorkerState(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    DRAINING = "DRAINING"
    LOST = "LOST"
    QUARANTINED = "QUARANTINED"


class VirtualCapacity(ContractModel):
    cpu_units: int = Field(ge=1)
    memory_mb: int = Field(ge=256)


class AllocatedCharge(ContractModel):
    cpu_units: int = Field(default=0, ge=0)
    memory_mb: int = Field(default=0, ge=0)


class SchedulingCharge(VirtualCapacity):
    def fits(self, available: VirtualCapacity | AllocatedCharge) -> bool:
        return (
            self.cpu_units <= available.cpu_units
            and self.memory_mb <= available.memory_mb
        )


class DockerTaskLimit(ContractModel):
    cpus: int = Field(ge=1)
    memory_mb: int = Field(ge=256)


class ResolvedTaskResources(ContractModel):
    charge: SchedulingCharge
    limit: DockerTaskLimit
    cpu_source: Literal["task", "default", "legacy"]
    memory_source: Literal["task", "default", "legacy"]
    task_config_digest: str = Field(min_length=1)


class ExecutionSpec(ContractModel):
    # Preserve the exact Compose ownership prefix; Harbor sanitizes dots,
    # colons and uppercase, which would otherwise make absence checks unsound.
    execution_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    task_ref: str = Field(min_length=1, max_length=500)
    model_name: str = Field(min_length=1)
    agent_name: Literal["claude-code"] = "claude-code"
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)


class ReserveRequest(ContractModel):
    client_scope: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")
    spec: ExecutionSpec
    reservation_request_id: str = Field(min_length=1, max_length=200)


class ActivateRequest(ContractModel):
    client_scope: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")
    reservation_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1, max_length=200)


class CancelRequest(ContractModel):
    client_scope: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")
    reason: str = Field(min_length=1, max_length=1000)
    quarantine: bool = False


class WorkerRegistration(ContractModel):
    contract_version: Literal["agentic-rollout/v4"] = CONTRACT_VERSION
    worker_id: str
    incarnation_id: str
    config_digest: str
    address: str
    relay_base_url: str
    labels: dict[str, str] = Field(default_factory=dict)
    virtual_capacity: VirtualCapacity
    task_resource_catalog_digest: str
    capabilities: list[str] = Field(default_factory=list)


class WorkerView(ContractModel):
    worker_id: str
    incarnation_id: str
    state: WorkerState
    virtual_capacity: VirtualCapacity
    cancel_execution_ids: list[str] = Field(default_factory=list)


class WorkerHeartbeat(ContractModel):
    worker_id: str
    incarnation_id: str
    active_execution_ids: list[str] = Field(default_factory=list)


class ExecutionClaimRequest(ContractModel):
    worker_id: str
    incarnation_id: str
    claim_request_id: str = Field(default_factory=lambda: uuid4().hex)


class WorkerDrainRequest(ContractModel):
    worker_id: str
    incarnation_id: str


class ExecutionIdentity(ContractModel):
    worker_id: str
    worker_incarnation: str
    lease_token: str = Field(repr=False)


class ExecutionTransition(ExecutionIdentity):
    state: ExecutionState
    trial_id: str | None = None
    trial_name: str | None = None
    detail: str | None = None


class ExecutionResult(ExecutionIdentity):
    state: Literal[ExecutionState.SUCCEEDED, ExecutionState.FAILED]
    trial_id: str
    trial_name: str
    rewards: dict[str, float] = Field(default_factory=dict)
    reward: float
    agent_terminal_reason: Literal["max_turns"] | None = None
    error_code: str | None = None
    error_message: str | None = None
    failure_class: str | None = None
    failure_stage: str | None = None
    retryable: bool = False


class ExecutionView(ContractModel):
    execution_id: str
    client_scope: str
    spec: ExecutionSpec | None = None
    state: ExecutionState
    request_digest: str | None = None
    reservation_request_id: str | None = None
    reservation_id: str | None = None
    reservation_expires_at: datetime | None = None
    context_id: str | None = None
    resolved_resources: ResolvedTaskResources | None = None
    resource_catalog_digest: str | None = None
    worker_id: str | None = None
    worker_incarnation: str | None = None
    claim_digest: str | None = None
    trial_id: str | None = None
    trial_name: str | None = None
    result: ExecutionResult | None = None
    result_digest: str | None = None
    failure_code: str | None = None
    resource_quarantined: bool = False
    result_acknowledged: bool = False
    cleanup_verified: bool = False
    state_version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ExecutionClaim(ExecutionIdentity):
    execution: ExecutionSpec
    client_scope: str
    context_id: str
    reservation_id: str
    resolved_resources: ResolvedTaskResources
    resource_catalog_digest: str
    execution_deadline_at: datetime
    # Fixed final settlement deadline; never renewed.
    lease_expires_at: datetime


class ResultAck(ContractModel):
    client_scope: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")
    result_digest: str


class CleanupReport(ExecutionIdentity):
    verified: Literal[True] = True


class RecoveryProof(ContractModel):
    worker_id: str
    incarnation_id: str
    worker_incarnation: str
    evidence_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    sandbox_absent: Literal[True]
    outbox_empty: Literal[True]


class CapacityWorker(ContractModel):
    worker_id: str
    state: WorkerState
    labels: dict[str, str]
    virtual_capacity: VirtualCapacity
    allocated_charge: AllocatedCharge


class CapacityView(ContractModel):
    contract_version: Literal["agentic-rollout/v4"] = CONTRACT_VERSION
    workers: list[CapacityWorker]
    reserved_executions: int = 0
    queued_executions: int = 0
    cleanup_pending_executions: int = 0


# GPU Policy owns model/version semantics. CPU only transports context identity.
class ContextPrepareRequest(ContractModel):
    policy_instance_id: str
    context_id: str
    execution_id: str
    client_scope: str
    reservation_id: str
    instance_id: str
    expected_weight_version: str = Field(min_length=1)
    model_override: str
    sampling_temperature: float = Field(default=1, ge=0)
    sampling_top_p: float = Field(default=1, gt=0, le=1)
    sampling_seed_base: int | None = None


class ContextBindRequest(ExecutionIdentity):
    execution_id: str
    reservation_id: str
    trial_id: str = Field(min_length=1)
    trial_name: str = Field(min_length=1)


class RouteControlRequest(ContractModel):
    action_id: str
    execution_id: str
    turn_id: str | None = None
    backend_session_id: str | None = None


class RouteView(ContractModel):
    context_id: str
    execution_id: str
    trial_id: str | None = None
    instance_id: str
    state: str
    policy_instance_id: str
    trajectory_retained: bool = False


class ErrorEnvelope(ContractModel):
    code: str
    message: str
    retryable: bool = False
    trace_id: str | None = None
