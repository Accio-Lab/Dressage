from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from blackbox_server.adapters.base import (
    BackendAdapter,
    BackendContextOverflowError,
    BackendMaxStepsExceededError,
    BackendProcessError,
    BackendProtocolError,
    BackendTransportError,
)
from blackbox_server.adapters.factory import IMPLEMENTED_BACKENDS, KNOWN_BACKENDS, create_adapter
from blackbox_server.config import BlackboxServerConfig
from blackbox_server.core.command import (
    CommandStartError,
    execute_shell_command,
    terminate_process_group,
)
from blackbox_server.core.errors import ApiError
from blackbox_server.core.hashing import (
    binding_request_fingerprint,
    message_request_fingerprint,
    normalize_router,
    normalize_system_prompt_file,
)
from blackbox_server.core.models import (
    AbortResponse,
    BackendCapabilities,
    BackendRef,
    BindingContext,
    BindingInfo,
    ExecuteCmdRequest,
    ExecuteCmdResponse,
    PauseRequest,
    PauseResponse,
    ProxyOptions,
    Message,
    MessageRequest,
    MessageResponse,
    RegisterRequest,
    RegisterResponse,
    ResumeRequest,
    ResumeResponse,
    RuntimeSystemPrompt,
    ServerState,
    SessionContext,
    SessionResponse,
    SessionState,
    SessionStats,
    StatusResponse,
    TraceEvent,
    TurnContext,
    TurnRecord,
    TurnStatus,
    utcnow,
)
from blackbox_server.core.monitoring import BackendMonitor
from blackbox_server.runtime.paths import ensure_runtime_dir, make_runtime_id, remove_runtime_dir
from blackbox_server.store.session_store import SessionStore


LOGGER = logging.getLogger(__name__)
_POST_AGENT_COMMAND_ERROR_CODES = frozenset(
    {
        "context_overflow",
        "max_steps_exceeded",
    }
)
_TURN_MODE_KEY = "__bbs_turn_mode"
_TURN_MODE_SINGLE = "single"
_TURN_MODE_EXPLICIT = "explicit"
_DEFAULT_TURN_ID_KEY = "__bbs_default_turn_id"


class BlackboxServer:
    def __init__(self, config: BlackboxServerConfig) -> None:
        self._base_config = config
        self._effective_config = config
        self._state = ServerState.IDLE
        self._adapter: BackendAdapter | None = None
        self._binding_context: BindingContext | None = None
        self._binding_request_fingerprint: str | None = None
        self._capabilities: BackendCapabilities | None = None
        self._init_lock = asyncio.Lock()
        self._session_store = SessionStore()
        self._monitor: BackendMonitor | None = None
        self._active_cmd_processes: dict[str, asyncio.subprocess.Process] = {}
        self._active_requests = 0
        self._request_counter_lock = asyncio.Lock()
        self._no_inflight_requests = asyncio.Event()
        self._no_inflight_requests.set()
        self._shutdown_started = False
        self._pause_lock = asyncio.Lock()
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._paused = False
        self._pause_reason: str | None = None
        self._current_version: str | None = None

    @property
    def state(self) -> ServerState:
        return self._state

    @asynccontextmanager
    async def request_scope(self) -> AsyncIterator[None]:
        async with self._request_counter_lock:
            self._active_requests += 1
            self._no_inflight_requests.clear()
        try:
            yield
        finally:
            async with self._request_counter_lock:
                self._active_requests -= 1
                if self._active_requests == 0:
                    self._no_inflight_requests.set()

    async def _wait_if_paused(self) -> None:
        while True:
            async with self._pause_lock:
                if not self._paused:
                    return
                resume_event = self._resume_event
            await resume_event.wait()

    async def _wait_for_backend_call_excluding_pause(
        self,
        awaitable: Any,
        *,
        timeout: float,
    ) -> Any:
        """Wait for a backend call while excluding rollout pause time.

        During transparent preempt-and-stitch, the blackbox HTTP turn can remain
        in flight while SGLang is quiesced and weights are updated.  Counting
        that paused interval against backend_timeout would desync otherwise
        healthy blackbox sessions.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        task = asyncio.create_task(awaitable)
        try:
            while True:
                if task.done():
                    return await task
                async with self._pause_lock:
                    paused = self._paused
                    resume_event = self._resume_event
                if paused:
                    pause_started = loop.time()
                    resume_task = asyncio.create_task(resume_event.wait())
                    try:
                        done, pending = await asyncio.wait(
                            {task, resume_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for pending_task in pending:
                            pending_task.cancel()
                        if task in done:
                            return await task
                    finally:
                        if not resume_task.done():
                            resume_task.cancel()
                    deadline += loop.time() - pause_started
                    continue

                remaining = deadline - loop.time()
                if remaining <= 0:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                    raise asyncio.TimeoutError

                done, pending = await asyncio.wait({task}, timeout=remaining)
                if task in done:
                    return await task
                if pending:
                    raise asyncio.TimeoutError
        except Exception:
            if not task.done():
                task.cancel()
            raise

    async def graceful_shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._state = ServerState.SHUTTING_DOWN
        await self._terminate_all_active_cmds()
        try:
            await asyncio.wait_for(
                self._no_inflight_requests.wait(),
                timeout=self._effective_config.shutdown_timeout,
            )
        except asyncio.TimeoutError:
            LOGGER.warning("graceful shutdown timed out while waiting for in-flight requests")
        await self._stop_monitoring()
        if self._adapter is not None:
            with contextlib.suppress(Exception):
                await self._adapter.shutdown()
        if self._binding_context is not None:
            remove_runtime_dir(self._binding_context.binding.runtime_dir)

    async def register(self, request: RegisterRequest) -> tuple[RegisterResponse, bool]:
        if self._state == ServerState.SHUTTING_DOWN:
            raise ApiError(503, "service_unavailable", "Server is shutting down.")

        router_base_url, router_api_path = normalize_router(request.router, request.router_api_path)
        system_prompt_file = self._validate_system_prompt_file(request.system_prompt_file)
        adapter = create_adapter(request.blackbox_type)
        capabilities = await adapter.capabilities()
        if system_prompt_file is not None and not capabilities.system_message:
            raise ApiError(
                400,
                "request_error",
                "Current backend does not support register-time system prompt injection.",
                details={"blackbox_type": request.blackbox_type},
            )

        effective_config = (
            request.server_config.apply(self._base_config)
            if request.server_config is not None
            else self._base_config.model_copy()
        )
        proxy_options = self._parse_proxy_options(request.backend_options)
        self._validate_register_binding(request, effective_config, proxy_options)
        fingerprint = binding_request_fingerprint(request, router_base_url)

        async with self._init_lock:
            if self._state == ServerState.INITIALIZING:
                raise ApiError(409, "conflict", "Server is initializing.")
            if self._state == ServerState.SHUTTING_DOWN:
                raise ApiError(503, "service_unavailable", "Server is shutting down.")

            same_fingerprint = fingerprint == self._binding_request_fingerprint
            if same_fingerprint and self._state == ServerState.READY and self._binding_context is not None:
                return self._build_register_response(), True

            if (
                not same_fingerprint
                and self._binding_context is not None
                and await self._session_store.has_open_sessions()
            ):
                raise ApiError(
                    409,
                    "conflict",
                    "Cannot rebind while active or desynced sessions still exist.",
                )

            reset_sessions = not same_fingerprint and self._binding_context is not None
            self._state = ServerState.INITIALIZING
            self._binding_request_fingerprint = fingerprint
            new_binding_context: BindingContext | None = None
            try:
                await self._teardown_current_binding(reset_sessions=reset_sessions)
                new_binding_context = self._build_binding_context(
                    request=request,
                    router_base_url=router_base_url,
                    router_api_path=router_api_path,
                    system_prompt_file=system_prompt_file,
                    effective_config=effective_config,
                )
                self._adapter = adapter
                await self._adapter.initialize(new_binding_context)
                self._binding_context = new_binding_context
                self._capabilities = capabilities
                self._effective_config = effective_config
                await self._ensure_bound_session_initialized()
                self._state = ServerState.READY
                self._start_monitoring()
                return self._build_register_response(), False
            except ApiError:
                self._state = ServerState.ERROR
                self._adapter = None
                self._binding_context = None
                self._capabilities = None
                if new_binding_context is not None:
                    remove_runtime_dir(new_binding_context.binding.runtime_dir)
                raise
            except BackendProtocolError as exc:
                self._state = ServerState.IDLE
                self._binding_request_fingerprint = None
                self._adapter = None
                self._binding_context = None
                self._capabilities = None
                self._effective_config = self._base_config.model_copy()
                if new_binding_context is not None:
                    remove_runtime_dir(new_binding_context.binding.runtime_dir)
                raise ApiError(
                    400,
                    "request_error",
                    str(exc),
                    details={"blackbox_type": request.blackbox_type},
                ) from exc
            except Exception as exc:
                self._state = ServerState.ERROR
                self._adapter = None
                self._binding_context = None
                self._capabilities = None
                if new_binding_context is not None:
                    remove_runtime_dir(new_binding_context.binding.runtime_dir)
                raise ApiError(
                    502,
                    "backend_error",
                    f"Failed to initialize backend: {exc}",
                    details={"blackbox_type": request.blackbox_type},
                ) from exc

    async def pause_generation(self, request: PauseRequest) -> PauseResponse:
        self._ensure_state_ready_for_messages()
        assert self._adapter is not None

        async with self._pause_lock:
            already_paused = self._paused
            self._paused = True
            self._pause_reason = request.reason
            self._resume_event.clear()

        result = await self._adapter.pause(
            reason=request.reason,
            timeout_seconds=request.timeout_seconds,
        )
        if not result.get("quiesced", True):
            raise ApiError(
                503,
                "pause_timeout",
                "Rollout pause did not reach model-quiesced state before timeout.",
                details=result,
            )
        return PauseResponse(
            status=str(result.get("status") or ("already_paused" if already_paused else "paused")),
            reason=request.reason,
            quiesced=bool(result.get("quiesced", True)),
            version=result.get("version") or self._current_version,
            http_inflight_requests=int(result.get("http_inflight_requests", 0) or 0),
            active_sglang_generations=int(result.get("active_sglang_generations", 0) or 0),
            suspended_generations=int(result.get("suspended_generations", 0) or 0),
            details=dict(result),
        )

    async def resume_generation(self, request: ResumeRequest) -> ResumeResponse:
        self._ensure_state_ready_for_messages()
        assert self._adapter is not None

        result = await self._adapter.resume(
            version=request.version,
            reason=request.reason,
        )
        async with self._pause_lock:
            if request.version is not None:
                self._current_version = str(request.version)
            was_paused = self._paused
            self._paused = False
            self._pause_reason = None
            self._resume_event.set()
        return ResumeResponse(
            status=str(result.get("status") or ("resumed" if was_paused else "already_running")),
            reason=request.reason,
            version=result.get("version") or self._current_version,
            details=dict(result),
        )

    async def pause_state(self) -> dict[str, Any]:
        adapter_state: dict[str, Any] = {}
        if self._adapter is not None and hasattr(self._adapter, "pause_state"):
            with contextlib.suppress(Exception):
                adapter_state = getattr(self._adapter, "pause_state")()
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "version": self._current_version,
            "adapter": adapter_state,
        }

    async def send_message(self, session_id: str, request: MessageRequest) -> MessageResponse:
        self._ensure_state_ready_for_messages()
        await self._wait_if_paused()
        self._validate_identifier("session_id", session_id)
        if request.turn_id is not None:
            self._validate_identifier("turn_id", request.turn_id)
        capabilities = self._require_capabilities()
        self._validate_message_request(request.messages, capabilities)

        assert self._binding_context is not None
        assert self._adapter is not None

        bound_session_id = self._require_bound_session_match(session_id)
        if bound_session_id is None:
            raise ApiError(
                500,
                "internal_error",
                "Rollout binding is missing bound_session_id.",
            )
        session = await self._session_store.get(bound_session_id)
        session_lock = await self._session_store.get_lock(bound_session_id)
        if session is None or session_lock is None:
            raise ApiError(
                500,
                "internal_error",
                "Bound session was not initialized.",
                details={"session_id": bound_session_id},
            )

        async with session_lock:
            if session.state == SessionState.ABORTED:
                raise ApiError(
                    409,
                    "conflict",
                    "Session has been aborted and cannot accept new turns.",
                    details={"session_id": session_id, "state": session.state},
                )
            if session.state == SessionState.DESYNCED:
                raise ApiError(
                    409,
                    "conflict",
                    "Session is desynced and cannot accept new turns.",
                    details={"session_id": session_id, "state": session.state},
                )
            if session.turn_count >= self._effective_config.max_turns:
                raise ApiError(
                    429,
                    "too_many_requests",
                    "Session reached the max committed turn limit.",
                    details={"session_id": session_id, "max_turns": self._effective_config.max_turns},
                )

            effective_turn_id = self._resolve_turn_id(session, request.turn_id)
            request_fingerprint = message_request_fingerprint(request.messages)
            existing = session.turn_ledger.get(effective_turn_id)
            if existing is not None:
                if existing.request_fingerprint != request_fingerprint:
                    raise ApiError(
                        409,
                        "conflict",
                        "Same turn_id was already used with a different request body.",
                        details={"session_id": session_id, "turn_id": effective_turn_id},
                    )
                if existing.status == TurnStatus.COMMITTED and existing.response is not None:
                    return MessageResponse(
                        request_id="",
                        session_id=session_id,
                        instance_id=self._response_instance_id(),
                        turn_id=effective_turn_id,
                        state=session.state,
                        idempotent_replay=True,
                        outputs=existing.response.outputs,
                        backend=BackendRef(
                            type=self._binding_context.binding.blackbox_type,
                            backend_session_id=existing.response.backend_session_id,
                        ),
                        usage=existing.response.usage,
                    )
                if existing.status == TurnStatus.INFLIGHT:
                    raise ApiError(
                        409,
                        "conflict",
                        "Same turn_id is still in flight.",
                        details={"session_id": session_id, "turn_id": effective_turn_id},
                    )
                raise ApiError(
                    409,
                    "conflict",
                    "Same turn_id is in unknown state and the session is desynced.",
                    details={"session_id": session_id, "turn_id": effective_turn_id},
                )

            if not await self._adapter_health_with_retry("before_send_message"):
                await self._mark_backend_error("adapter_healthcheck_failed")
                raise ApiError(503, "service_unavailable", "Backend is unavailable.")

            now = utcnow()
            session.turn_ledger[effective_turn_id] = TurnRecord(
                turn_id=effective_turn_id,
                request_fingerprint=request_fingerprint,
                status=TurnStatus.INFLIGHT,
                request_messages=request.messages,
                created_at=now,
                updated_at=now,
            )

            turn_context = TurnContext(
                turn_id=effective_turn_id,
                request_fingerprint=request_fingerprint,
                metadata=request.metadata,
                deadline_seconds=self._effective_config.backend_timeout,
            )
            try:
                adapter_response = await self._wait_for_backend_call_excluding_pause(
                    self._adapter.send_message(session, turn_context, request.messages),
                    timeout=self._effective_config.backend_timeout,
                )
                session.backend_session_id = adapter_response.backend_session_id
                session.conversation_history.extend(request.messages)
                session.conversation_history.extend(adapter_response.outputs)
                session.trace_events.extend(adapter_response.trace_events)
                session.turn_count += 1
                session.updated_at = utcnow()
                turn_record = session.turn_ledger[effective_turn_id]
                turn_record.status = TurnStatus.COMMITTED
                turn_record.response = adapter_response
                turn_record.updated_at = utcnow()
                return MessageResponse(
                    request_id="",
                    session_id=session_id,
                    instance_id=self._response_instance_id(),
                    turn_id=effective_turn_id,
                    state=session.state,
                    idempotent_replay=False,
                    outputs=adapter_response.outputs,
                    backend=BackendRef(
                        type=self._binding_context.binding.blackbox_type,
                        backend_session_id=adapter_response.backend_session_id,
                    ),
                    usage=adapter_response.usage,
                )
            except asyncio.TimeoutError as exc:
                await self._handle_unknown_turn(
                    session=session,
                    turn_id=effective_turn_id,
                    error="backend_timeout",
                    message="Backend call exceeded backend_timeout.",
                )
                raise ApiError(
                    504,
                    "backend_timeout",
                    "Backend call exceeded backend_timeout.",
                    details={"session_id": session_id, "turn_id": effective_turn_id},
                ) from exc
            except BackendContextOverflowError as exc:
                await self._handle_unknown_turn(
                    session=session,
                    turn_id=effective_turn_id,
                    error="context_overflow",
                    message=str(exc),
                )
                raise ApiError(
                    413,
                    "context_overflow",
                    str(exc),
                    details={
                        "session_id": session_id,
                        "turn_id": effective_turn_id,
                        **exc.details(),
                    },
                ) from exc
            except BackendMaxStepsExceededError as exc:
                await self._handle_unknown_turn(
                    session=session,
                    turn_id=effective_turn_id,
                    error="max_steps_exceeded",
                    message=str(exc),
                )
                raise ApiError(
                    429,
                    "max_steps_exceeded",
                    str(exc),
                    details={
                        "session_id": session_id,
                        "turn_id": effective_turn_id,
                        **exc.details(),
                    },
                ) from exc
            except (BackendTransportError, BackendProtocolError, BackendProcessError) as exc:
                await self._handle_unknown_turn(
                    session=session,
                    turn_id=effective_turn_id,
                    error="backend_error",
                    message=str(exc),
                )
                if isinstance(exc, BackendProcessError):
                    await self._mark_backend_error("backend_process_error")
                raise ApiError(
                    502,
                    "backend_error",
                    f"Backend request failed: {exc}",
                    details={"session_id": session_id, "turn_id": effective_turn_id},
                ) from exc

    async def execute_cmd(self, session_id: str, request: ExecuteCmdRequest) -> ExecuteCmdResponse:
        self._ensure_state_ready_for_messages()
        self._validate_identifier("session_id", session_id)
        cmd, timeout = self._validate_execute_cmd_request(request)

        assert self._binding_context is not None

        bound_session_id = self._require_bound_session_match(session_id)
        if bound_session_id is None:
            raise ApiError(
                500,
                "internal_error",
                "Rollout binding is missing bound_session_id.",
            )
        session = await self._session_store.get(bound_session_id)
        session_lock = await self._session_store.get_lock(bound_session_id)
        if session is None or session_lock is None:
            raise ApiError(
                404,
                "not_found",
                "Session does not exist.",
                details={"session_id": bound_session_id},
            )

        async with session_lock:
            if session.state == SessionState.ABORTED:
                raise ApiError(
                    409,
                    "conflict",
                    "Session has been aborted and cannot execute commands.",
                    details={"session_id": session_id, "state": session.state},
                )
            if (
                session.state == SessionState.DESYNCED
                and not self._session_allows_post_agent_commands(session)
            ):
                raise ApiError(
                    409,
                    "conflict",
                    "Session is desynced and cannot execute commands.",
                    details={"session_id": session_id, "state": session.state},
                )

            try:
                result = await execute_shell_command(
                    cmd,
                    timeout=timeout,
                    on_process_start=lambda proc: self._set_active_cmd_process(session_id, proc),
                    on_process_end=lambda proc: self._clear_active_cmd_process(session_id, proc),
                )
            except CommandStartError as exc:
                raise ApiError(
                    502,
                    "backend_error",
                    f"execute_cmd failed: {exc}",
                    details={"session_id": session_id},
                ) from exc

            session.updated_at = utcnow()
            return ExecuteCmdResponse(
                request_id="",
                session_id=session_id,
                instance_id=self._response_instance_id(),
                **result.model_dump(),
            )

    async def get_session(
        self,
        session_id: str,
        *,
        include_history: bool = False,
        include_trace: bool = False,
        include_turns: bool = False,
    ) -> SessionResponse:
        self._validate_identifier("session_id", session_id)
        self._require_bound_session_match(session_id)
        session = await self._session_store.get(session_id)
        if session is None:
            raise ApiError(404, "not_found", "Session does not exist.", details={"session_id": session_id})
        return SessionResponse(
            request_id="",
            session_id=session.session_id,
            instance_id=self._response_instance_id(),
            state=session.state,
            blackbox_type=session.blackbox_type,
            backend_session_id=session.backend_session_id,
            message_count=len(session.conversation_history),
            turn_count=session.turn_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
            conversation_history=session.conversation_history if include_history else None,
            trace_events=session.trace_events if include_trace else None,
            turn_ledger=session.turn_ledger if include_turns else None,
        )

    async def abort_session(self, session_id: str) -> AbortResponse:
        self._validate_identifier("session_id", session_id)
        self._require_bound_session_match(session_id)
        session = await self._session_store.get(session_id)
        if session is None:
            raise ApiError(404, "not_found", "Session does not exist.", details={"session_id": session_id})
        session_lock = await self._session_store.get_lock(session_id)
        assert session_lock is not None
        await self._terminate_active_cmd(session_id)

        acquired = False
        try:
            try:
                await asyncio.wait_for(session_lock.acquire(), timeout=5.0)
                acquired = True
            except asyncio.TimeoutError:
                LOGGER.warning(
                    "abort_session: could not acquire session lock for %s within timeout, forcing abort",
                    session_id,
                )

            if session.state == SessionState.ABORTED:
                return AbortResponse(
                    request_id="",
                    session_id=session_id,
                    instance_id=self._response_instance_id(),
                    state=session.state,
                    mode="noop",
                )

            if self._adapter is not None:
                with contextlib.suppress(Exception):
                    await self._adapter.abort_session(session)

            session.state = SessionState.ABORTED
            session.updated_at = utcnow()

            now = utcnow()
            for turn_record in session.turn_ledger.values():
                if turn_record.status == TurnStatus.INFLIGHT:
                    turn_record.status = TurnStatus.UNKNOWN
                    turn_record.error = {"error": "aborted", "message": "Session was aborted while turn was in flight."}
                    turn_record.updated_at = now

            return AbortResponse(
                request_id="",
                session_id=session_id,
                instance_id=self._response_instance_id(),
                state=session.state,
                mode="best_effort",
            )
        finally:
            if acquired:
                session_lock.release()

    async def status(self) -> StatusResponse:
        counts = await self._session_store.counts()
        return StatusResponse(
            request_id="",
            status=self._state,
            binding=self._binding_context.binding if self._binding_context is not None else None,
            sessions=SessionStats(**counts, max_sessions=self._effective_config.max_sessions),
            capabilities=self._capabilities,
            config=self._effective_config,
            implemented_backends=IMPLEMENTED_BACKENDS,
            known_backends=KNOWN_BACKENDS,
        )

    async def _mark_backend_error(self, reason: str) -> None:
        if self._state == ServerState.ERROR:
            return
        LOGGER.error("marking backend error: %s", reason)
        self._state = ServerState.ERROR
        await self._session_store.mark_all_non_aborted_desynced()

    async def _handle_unknown_turn(
        self,
        *,
        session: SessionContext,
        turn_id: str,
        error: str,
        message: str,
    ) -> None:
        turn_record = session.turn_ledger[turn_id]
        turn_record.status = TurnStatus.UNKNOWN
        turn_record.error = {"error": error, "message": message}
        turn_record.updated_at = utcnow()
        if session.state != SessionState.ABORTED:
            session.state = SessionState.DESYNCED
        session.updated_at = utcnow()
        if self._adapter is not None:
            with contextlib.suppress(Exception):
                await self._adapter.abort_session(session)
        if self._adapter is not None and not await self._adapter_health_with_retry("unknown_turn"):
            await self._mark_backend_error(error)

    async def _adapter_health_with_retry(self, reason: str) -> bool:
        if self._adapter is None:
            return False

        attempts = max(1, self._effective_config.runtime_health_check_retries)
        delay = max(0.0, self._effective_config.runtime_health_check_retry_delay)

        for attempt in range(1, attempts + 1):
            try:
                if await self._adapter.health():
                    return True
            except Exception:
                LOGGER.warning(
                    "backend health check raised during %s on attempt %d/%d",
                    reason,
                    attempt,
                    attempts,
                    exc_info=True,
                )

            if attempt >= attempts:
                break

            LOGGER.warning(
                "backend health check failed during %s on attempt %d/%d; "
                "retrying in %.3fs",
                reason,
                attempt,
                attempts,
                delay,
            )
            if delay > 0:
                await asyncio.sleep(delay)

        return False

    @staticmethod
    def _session_allows_post_agent_commands(session: SessionContext) -> bool:
        """Allow sandbox harvesting after a known, non-retryable agent stop."""

        if session.state != SessionState.DESYNCED or not session.turn_ledger:
            return False
        latest_turn = max(
            session.turn_ledger.values(),
            key=lambda turn: (turn.updated_at, turn.created_at, turn.turn_id),
        )
        if latest_turn.status != TurnStatus.UNKNOWN:
            return False
        error = latest_turn.error
        if not isinstance(error, dict):
            return False
        return error.get("error") in _POST_AGENT_COMMAND_ERROR_CODES

    def _resolve_turn_id(self, session: SessionContext, request_turn_id: str | None) -> str:
        turn_mode = session.metadata.get(_TURN_MODE_KEY)
        default_turn_id = session.metadata.get(_DEFAULT_TURN_ID_KEY)

        if request_turn_id is None:
            if turn_mode == _TURN_MODE_EXPLICIT:
                raise ApiError(
                    400,
                    "request_error",
                    "turn_id is required after this session has used explicit turn ids.",
                    details={"session_id": session.session_id},
                )
            if turn_mode == _TURN_MODE_SINGLE and isinstance(default_turn_id, str):
                return default_turn_id
            generated_turn_id = f"turn-{uuid4().hex}"
            session.metadata[_TURN_MODE_KEY] = _TURN_MODE_SINGLE
            session.metadata[_DEFAULT_TURN_ID_KEY] = generated_turn_id
            return generated_turn_id

        if turn_mode == _TURN_MODE_SINGLE:
            if request_turn_id == default_turn_id:
                return request_turn_id
            raise ApiError(
                409,
                "conflict",
                "This session uses a generated single-turn id and cannot accept a new turn_id.",
                details={
                    "session_id": session.session_id,
                    "expected_turn_id": default_turn_id,
                    "actual_turn_id": request_turn_id,
                },
            )

        if turn_mode is None:
            session.metadata[_TURN_MODE_KEY] = _TURN_MODE_EXPLICIT
        return request_turn_id

    def _response_instance_id(self) -> str | None:
        if self._binding_context is None:
            return None
        return self._binding_context.binding.bound_instance_id

    def _start_monitoring(self) -> None:
        if self._adapter is None:
            return
        self._monitor = BackendMonitor(
            interval_seconds=self._effective_config.runtime_health_check_interval,
            check=lambda: self._adapter_health_with_retry("runtime_monitor"),
            on_failure=lambda: self._mark_backend_error("runtime_monitor_failed"),
        )
        self._monitor.start()

    async def _stop_monitoring(self) -> None:
        if self._monitor is None:
            return
        await self._monitor.stop()
        self._monitor = None

    async def _teardown_current_binding(self, *, reset_sessions: bool) -> None:
        await self._stop_monitoring()
        old_binding = self._binding_context
        if self._adapter is not None:
            with contextlib.suppress(Exception):
                await self._adapter.shutdown()
        self._adapter = None
        self._binding_context = None
        self._capabilities = None
        self._effective_config = self._base_config.model_copy()
        if reset_sessions:
            await self._session_store.reset_all()
        if old_binding is not None:
            remove_runtime_dir(old_binding.binding.runtime_dir)

    def _build_binding_context(
        self,
        *,
        request: RegisterRequest,
        router_base_url: str,
        router_api_path: str,
        system_prompt_file: str | None,
        effective_config: BlackboxServerConfig,
    ) -> BindingContext:
        runtime_id = make_runtime_id()
        runtime_dir = ensure_runtime_dir(effective_config.runtime_root, runtime_id)
        system_prompt = None
        if system_prompt_file is not None:
            if request.blackbox_type == "opencode":
                runtime_file = Path(runtime_dir) / "home" / ".config" / "opencode" / "prompts" / "build.txt"
                applies_to = "build"
            elif request.blackbox_type == "openclaw":
                runtime_file = Path(runtime_dir) / "home" / ".openclaw" / "workspace" / "AGENTS.md"
                applies_to = "openclaw"
            else:
                runtime_file = Path(runtime_dir) / "prompts" / "system.txt"
                applies_to = request.blackbox_type
            system_prompt = RuntimeSystemPrompt(
                source_file=system_prompt_file,
                runtime_file=str(runtime_file),
                applies_to=applies_to,
            )
        binding = BindingInfo(
            runtime_id=runtime_id,
            blackbox_type=request.blackbox_type,
            router_raw=request.router,
            router_base_url=router_base_url,
            router_api_path=router_api_path,
            bound_session_id=request.bound_session_id or "",
            bound_instance_id=request.bound_instance_id or "",
            system_prompt=system_prompt,
            runtime_dir=runtime_dir,
            registered_at=utcnow(),
            backend_options=request.backend_options,
        )
        return BindingContext(binding=binding, effective_config=effective_config)

    def _build_register_response(self) -> RegisterResponse:
        assert self._binding_context is not None
        assert self._capabilities is not None
        return RegisterResponse(
            request_id="",
            status=self._state,
            binding=self._binding_context.binding,
            capabilities=self._capabilities,
            config=self._effective_config,
        )

    def _ensure_state_ready_for_messages(self) -> None:
        if self._state != ServerState.READY:
            raise ApiError(
                503,
                "service_unavailable",
                f"Server is not ready. current_state={self._state}",
                details={"state": self._state},
            )

    def _require_capabilities(self) -> BackendCapabilities:
        if self._capabilities is None:
            raise ApiError(503, "service_unavailable", "Backend capabilities are unavailable.")
        return self._capabilities

    def _validate_identifier(self, field_name: str, value: str) -> None:
        pattern = (
            r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
            if field_name == "session_id"
            else r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
        )
        if not re.fullmatch(pattern, value):
            raise ApiError(
                400,
                "request_error",
                f"{field_name} has invalid format.",
                details={field_name: value},
            )

    def _validate_system_prompt_file(self, path: str | None) -> str | None:
        normalized = normalize_system_prompt_file(path)
        if normalized is None:
            return None
        if not os.path.isabs(normalized):
            raise ApiError(400, "request_error", "system_prompt_file must be an absolute path.")
        if not os.path.isfile(normalized):
            raise ApiError(400, "request_error", "system_prompt_file does not exist.")
        if not os.access(normalized, os.R_OK):
            raise ApiError(400, "request_error", "system_prompt_file is not readable.")
        return normalized

    def _inject_system_prompt(self, session: SessionContext) -> None:
        """Inject system prompt into conversation_history and trace_events when a session is first created."""
        if self._binding_context is None or self._binding_context.binding.system_prompt is None:
            return
        source_file = self._binding_context.binding.system_prompt.source_file
        try:
            content = Path(source_file).read_text(encoding="utf-8")
        except Exception:
            LOGGER.warning("failed to read system_prompt_file for history injection: %s", source_file)
            return
        if not content.strip():
            return
        session.conversation_history.insert(
            0,
            Message(role="system", content=content),
        )
        session.trace_events.append(
            TraceEvent(
                turn_id="__init__",
                seq=0,
                source="blackbox_server",
                event_type="system_prompt_injected",
                payload={
                    "source_file": source_file,
                    "content_length": len(content),
                },
                created_at=utcnow(),
            )
        )

    def _validate_message_request(
        self,
        messages: list[Message],
        capabilities: BackendCapabilities,
    ) -> None:
        if not messages:
            raise ApiError(400, "request_error", "messages must not be empty.")
        if not capabilities.multi_message_input and len(messages) != 1:
            raise ApiError(400, "request_error", "Current backend accepts exactly one input message.")
        if (
            self._binding_context is not None
            and self._binding_context.binding.blackbox_type
            in {"opencode", "openclaw", "claude_code", "codex"}
        ):
            backend = self._binding_context.binding.blackbox_type
            only_message = messages[0]
            if only_message.role != "user":
                raise ApiError(400, "request_error", f"{backend} phase 1 only accepts a user message.")
            if only_message.content is None or not only_message.content.strip():
                raise ApiError(
                    400,
                    "request_error",
                    f"{backend} phase 1 requires non-empty user content.",
                )
        for message in messages:
            if message.role == "system":
                raise ApiError(400, "request_error", "/messages does not support role=system in phase 1.")
            if not capabilities.history_injection and message.role in {"assistant", "tool"}:
                raise ApiError(
                    400,
                    "request_error",
                    "Current backend does not support assistant/tool history injection.",
                )

    def _validate_execute_cmd_request(self, request: ExecuteCmdRequest) -> tuple[str, float]:
        raw_cmd = request.cmd
        if not raw_cmd.strip():
            raise ApiError(400, "request_error", "cmd must not be empty.")
        if any(char in raw_cmd for char in ("\n", "\r", "\x00")):
            raise ApiError(400, "request_error", "cmd must be a single-line command.")
        timeout = (
            self._effective_config.execute_cmd_timeout
            if request.timeout is None
            else request.timeout
        )
        return raw_cmd.strip(), timeout

    def _set_active_cmd_process(self, session_id: str, proc: asyncio.subprocess.Process) -> None:
        self._active_cmd_processes[session_id] = proc

    def _clear_active_cmd_process(self, session_id: str, proc: asyncio.subprocess.Process) -> None:
        if self._active_cmd_processes.get(session_id) is proc:
            self._active_cmd_processes.pop(session_id, None)

    async def _terminate_active_cmd(self, session_id: str) -> None:
        await terminate_process_group(self._active_cmd_processes.get(session_id))

    async def _terminate_all_active_cmds(self) -> None:
        if not self._active_cmd_processes:
            return
        await asyncio.gather(
            *(
                terminate_process_group(proc)
                for proc in list(self._active_cmd_processes.values())
            ),
            return_exceptions=True,
        )

    def _parse_proxy_options(self, backend_options: dict[str, Any]) -> ProxyOptions:
        proxy_config = backend_options.get("proxy") if isinstance(backend_options, dict) else {}
        if proxy_config is None:
            proxy_config = {}
        if not isinstance(proxy_config, dict):
            raise ApiError(400, "request_error", "backend_options.proxy must be an object.")
        try:
            return ProxyOptions.model_validate(proxy_config)
        except Exception as exc:
            raise ApiError(400, "request_error", f"Invalid proxy config: {exc}") from exc

    def _validate_register_binding(
        self,
        request: RegisterRequest,
        effective_config: BlackboxServerConfig,
        proxy_options: ProxyOptions,
    ) -> None:
        _ = effective_config, proxy_options
        if not request.bound_session_id:
            raise ApiError(
                400,
                "request_error",
                "bound_session_id is required.",
            )
        if not request.bound_instance_id:
            raise ApiError(
                400,
                "request_error",
                "bound_instance_id is required.",
            )
        self._validate_identifier("session_id", request.bound_session_id)
        self._validate_identifier("instance_id", request.bound_instance_id)

    def _require_bound_session_match(self, session_id: str) -> str | None:
        if self._binding_context is None:
            return None
        expected = self._binding_context.binding.bound_session_id
        if session_id != expected:
            raise ApiError(
                409,
                "bound_session_mismatch",
                "This rollout binding only serves the registered session_id.",
                details={"expected_session_id": expected, "actual_session_id": session_id},
            )
        return expected

    async def _ensure_bound_session_initialized(self) -> None:
        assert self._binding_context is not None
        bound_session_id = self._binding_context.binding.bound_session_id
        session, _, created = await self._session_store.get_or_create(
            bound_session_id,
            factory=lambda: SessionContext(
                session_id=bound_session_id,
                state=SessionState.ACTIVE,
                blackbox_type=self._binding_context.binding.blackbox_type,
                router_base_url=self._binding_context.binding.router_base_url,
                created_at=utcnow(),
                updated_at=utcnow(),
                metadata={},
            ),
            max_sessions=self._effective_config.max_sessions,
        )
        if created:
            self._inject_system_prompt(session)
