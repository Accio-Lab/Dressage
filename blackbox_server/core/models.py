from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from blackbox_server.config import BlackboxServerConfig, ServerConfigOverride


SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
TURN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
DEFAULT_PROXY_MAX_STEPS = 100


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ServerState(str, Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class SessionState(str, Enum):
    ACTIVE = "active"
    DESYNCED = "desynced"
    ABORTED = "aborted"


class TurnStatus(str, Enum):
    QUEUED = "queued"        # accepted, not started yet
    INFLIGHT = "inflight"    # backend call in flight
    COMMITTED = "committed"  # succeeded
    FAILED = "failed"        # deterministic failure (session not necessarily desynced)
    CANCELLED = "cancelled"  # cancellation completed
    UNKNOWN = "unknown"      # result unknown (timeout/process error); session -> DESYNCED


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class TraceEvent(BaseModel):
    turn_id: str
    seq: int
    source: str
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


class TurnUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    steps: int = 0
    tool_calls: int = 0


class AdapterResponse(BaseModel):
    outputs: list[Message] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    usage: TurnUsage = Field(default_factory=TurnUsage)
    backend_session_id: str


class BackendCapabilities(BaseModel):
    chat: bool = True
    abort: bool = True
    pause_resume: bool = False
    stream: bool = False
    multi_message_input: bool = False
    system_message: bool = False
    history_injection: bool = False


class RuntimeSystemPrompt(BaseModel):
    source_file: str
    runtime_file: str
    applies_to: str = "build"


class ProxyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sticky_header_name: str = "X-SMG-Routing-Key"
    max_steps: int | None = Field(default=DEFAULT_PROXY_MAX_STEPS, gt=0)
    default_temperature: float | None = Field(default=None, ge=0)


class BindingInfo(BaseModel):
    runtime_id: str
    blackbox_type: str
    router_raw: str
    router_base_url: str
    router_api_path: str
    bound_session_id: str
    bound_instance_id: str
    system_prompt: RuntimeSystemPrompt | None = None
    runtime_dir: str
    registered_at: datetime
    backend_options: dict[str, Any] = Field(default_factory=dict)


class BindingContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    binding: BindingInfo
    effective_config: BlackboxServerConfig


class TurnRecord(BaseModel):
    turn_id: str
    request_fingerprint: str
    status: TurnStatus
    request_messages: list[Message]
    response: AdapterResponse | None = None
    error: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class TurnContext(BaseModel):
    turn_id: str
    request_fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    deadline_seconds: float


class SessionContext(BaseModel):
    session_id: str
    state: SessionState
    blackbox_type: str
    backend_session_id: str | None = None
    router_base_url: str
    conversation_history: list[Message] = Field(default_factory=list)
    trace_events: list[TraceEvent] = Field(default_factory=list)
    turn_ledger: dict[str, TurnRecord] = Field(default_factory=dict)
    turn_count: int = 0
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegisterRequest(BaseModel):
    blackbox_type: str
    router: str
    router_api_path: str = "/v1"
    bound_session_id: str | None = None
    bound_instance_id: str | None = None
    system_prompt_file: str | None = None
    backend_options: dict[str, Any] = Field(default_factory=dict)
    server_config: ServerConfigOverride | None = None


class BackendRef(BaseModel):
    type: str
    backend_session_id: str | None = None


class RegisterResponse(BaseModel):
    request_id: str
    status: ServerState
    binding: BindingInfo
    capabilities: BackendCapabilities
    config: BlackboxServerConfig


class PauseRequest(BaseModel):
    reason: str = "weight_update"
    timeout_seconds: float | None = Field(default=None, gt=0)


class PauseResponse(BaseModel):
    request_id: str = ""
    status: str
    reason: str
    quiesced: bool
    version: str | None = None
    http_inflight_requests: int = 0
    active_sglang_generations: int = 0
    suspended_generations: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    reason: str = "weight_update"
    version: str | None = None


class ResumeResponse(BaseModel):
    request_id: str = ""
    status: str
    reason: str
    version: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class MessageRequest(BaseModel):
    turn_id: str | None = None
    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["sync", "async"] = "sync"


class MessageResponse(BaseModel):
    request_id: str
    session_id: str
    instance_id: str | None = None
    turn_id: str
    state: SessionState
    idempotent_replay: bool
    outputs: list[Message]
    backend: BackendRef
    usage: TurnUsage


class TurnSubmitResponse(BaseModel):
    """202 Accepted response returned when a turn is submitted in async mode."""

    request_id: str = ""
    session_id: str
    instance_id: str | None = None
    turn_id: str
    status: TurnStatus
    idempotent_replay: bool = False


class TurnStatusResponse(BaseModel):
    """Snapshot returned by the long-poll turn status endpoint."""

    request_id: str = ""
    session_id: str
    instance_id: str | None = None
    turn_id: str
    status: TurnStatus
    state: SessionState
    outputs: list[Message] | None = None
    usage: TurnUsage | None = None
    backend: BackendRef | None = None
    error: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class TurnCancelResponse(BaseModel):
    """Response returned by the turn cancel endpoint."""

    request_id: str = ""
    session_id: str
    instance_id: str | None = None
    turn_id: str
    status: str
    state: SessionState


class ExecuteCmdRequest(BaseModel):
    cmd: str
    timeout: float | None = Field(default=None, gt=0)


class ExecuteCmdResult(BaseModel):
    cmd: str
    stdout: str
    stderr: str
    returncode: int | None = None
    timed_out: bool
    duration_seconds: float
    started_at: datetime
    finished_at: datetime


class ExecuteCmdResponse(ExecuteCmdResult):
    request_id: str
    session_id: str
    instance_id: str | None = None


class SessionResponse(BaseModel):
    request_id: str
    session_id: str
    instance_id: str | None = None
    state: SessionState
    blackbox_type: str
    backend_session_id: str | None = None
    message_count: int
    turn_count: int
    created_at: datetime
    updated_at: datetime
    conversation_history: list[Message] | None = None
    trace_events: list[TraceEvent] | None = None
    turn_ledger: dict[str, TurnRecord] | None = None


class AbortResponse(BaseModel):
    request_id: str
    session_id: str
    instance_id: str | None = None
    action: Literal["abort"] = "abort"
    state: SessionState
    mode: Literal["best_effort", "noop"]


class SessionStats(BaseModel):
    active_count: int
    desynced_count: int
    aborted_count: int
    total_count: int
    max_sessions: int


class StatusResponse(BaseModel):
    request_id: str
    status: ServerState
    binding: BindingInfo | None = None
    sessions: SessionStats
    capabilities: BackendCapabilities | None = None
    config: BlackboxServerConfig
    implemented_backends: list[str]
    known_backends: list[str]


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    message: str
    details: dict[str, Any] | None = None
