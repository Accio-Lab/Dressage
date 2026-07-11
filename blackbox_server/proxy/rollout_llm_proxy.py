from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from blackbox_server.core.models import DEFAULT_PROXY_MAX_STEPS


LOGGER = logging.getLogger(__name__)
DRESSAGE_ROLLOUT_INVALIDATED_ERRORS = {
    "generation_preempted",
    "partial_rollout_staleness_exceeded",
    "trajectory_version_changed",
}


@dataclass
class _TurnScope:
    turn_id: str
    backend_session_id: str | None = None
    step_counter: int = 0
    inflight_requests: int = 0
    context_overflow_error: dict[str, Any] | None = None
    rollout_invalidated_error: dict[str, Any] | None = None
    max_steps_error: dict[str, Any] | None = None
    failed_upstream_error: dict[str, Any] | None = None
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    max_steps_exceeded: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.drained.set()


@dataclass(frozen=True)
class _TurnSnapshot:
    session_id: str
    turn_id: str | None
    backend_session_id: str | None
    step: int
    scope: _TurnScope | None = None
    max_steps_exceeded: bool = False


class RolloutLLMProxy:
    def __init__(
        self,
        *,
        upstream_origin: str,
        router_api_path: str,
        bound_session_id: str,
        bound_instance_id: str,
        sticky_header_name: str,
        max_steps: int | None = DEFAULT_PROXY_MAX_STEPS,
        default_temperature: float | None = None,
        debug_log_dir: str | Path | None = None,
    ) -> None:
        self.upstream_origin = upstream_origin.rstrip("/")
        self.router_api_path = router_api_path.rstrip("/") or "/"
        self.bound_session_id = bound_session_id
        self.bound_instance_id = bound_instance_id
        self.sticky_header_name = sticky_header_name
        self.max_steps = max_steps
        self.default_temperature = default_temperature
        self.debug_log_dir = Path(debug_log_dir) if debug_log_dir is not None else None
        self._client: httpx.AsyncClient | None = None
        self._turn_scope: _TurnScope | None = None
        self._scope_lock = asyncio.Lock()
        self._paused = False
        self._pause_reason: str | None = None
        self._current_version: str | None = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._pause_state_changed = asyncio.Event()
        self._pause_started_at: float | None = None
        self._total_paused_seconds = 0.0
        self._app = self._build_app()

    @property
    def app(self) -> FastAPI:
        return self._app

    async def open_turn(self, turn_id: str, backend_session_id: str | None = None) -> None:
        async with self._scope_lock:
            if self._turn_scope is not None:
                raise RuntimeError(f"turn scope is already active for {self._turn_scope.turn_id}")
            self._turn_scope = _TurnScope(turn_id=turn_id, backend_session_id=backend_session_id)

    async def update_turn_backend_session(self, backend_session_id: str) -> None:
        async with self._scope_lock:
            if self._turn_scope is None:
                return
            self._turn_scope.backend_session_id = backend_session_id

    async def drain_turn(self, timeout: float | None = None) -> None:
        async with self._scope_lock:
            scope = self._turn_scope
        if scope is None:
            return
        await self._wait_event_excluding_pause(scope.drained, timeout=timeout)

    @property
    def total_paused_seconds(self) -> float:
        return self._total_paused_seconds

    def pause_state(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "version": self._current_version,
            "http_inflight_requests": self._http_inflight_count(),
            "total_paused_seconds": self._total_paused_seconds,
        }

    async def pause(
        self,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self._scope_lock:
            already_paused = self._paused
            self._paused = True
            self._pause_reason = reason
            if self._pause_started_at is None:
                self._pause_started_at = loop.time()
            self._resume_event.clear()
            self._notify_pause_state_changed_locked()

        result = await self._control_post(
            "/rollout/pause",
            {
                "session_id": self.bound_session_id,
                "instance_id": self.bound_instance_id,
                "reason": reason,
                "mode": "preempt",
                "timeout_seconds": timeout_seconds,
            },
        )
        result.setdefault("status", "already_paused" if already_paused else "paused")
        result.setdefault("reason", reason)
        result["http_inflight_requests"] = self._http_inflight_count()
        return result

    async def resume(
        self,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        result = await self._control_post(
            "/rollout/resume",
            {
                "session_id": self.bound_session_id,
                "instance_id": self.bound_instance_id,
                "reason": reason,
                "version": version,
            },
        )
        loop = asyncio.get_running_loop()
        async with self._scope_lock:
            if version is not None:
                self._current_version = str(version)
            if self._pause_started_at is not None:
                self._total_paused_seconds += loop.time() - self._pause_started_at
                self._pause_started_at = None
            was_paused = self._paused
            self._paused = False
            self._pause_reason = None
            self._resume_event.set()
            self._notify_pause_state_changed_locked()
        result.setdefault("status", "resumed" if was_paused else "already_running")
        result.setdefault("reason", reason)
        result.setdefault("version", self._current_version)
        result["http_inflight_requests"] = self._http_inflight_count()
        return result

    async def clear_turn(self) -> None:
        async with self._scope_lock:
            self._turn_scope = None

    async def consume_context_overflow_error(self) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None or scope.context_overflow_error is None:
                return None
            payload = dict(scope.context_overflow_error)
            scope.context_overflow_error = None
            return payload

    async def consume_rollout_invalidated_error(self) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None or scope.rollout_invalidated_error is None:
                return None
            payload = dict(scope.rollout_invalidated_error)
            scope.rollout_invalidated_error = None
            return payload

    async def consume_failed_upstream_error(self) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None or scope.failed_upstream_error is None:
                return None
            payload = dict(scope.failed_upstream_error)
            scope.failed_upstream_error = None
            return payload

    async def wait_for_max_steps_error(
        self,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None:
                return None
            if scope.max_steps_error is not None:
                return dict(scope.max_steps_error)
            event = scope.max_steps_exceeded

        try:
            if timeout is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        async with self._scope_lock:
            if self._turn_scope is not scope or scope.max_steps_error is None:
                return None
            return dict(scope.max_steps_error)

    async def consume_max_steps_error(self) -> dict[str, Any] | None:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None or scope.max_steps_error is None:
                return None
            payload = dict(scope.max_steps_error)
            scope.max_steps_error = None
            scope.max_steps_exceeded.clear()
            return payload

    def _http_inflight_count(self) -> int:
        scope = self._turn_scope
        return 0 if scope is None else scope.inflight_requests

    def _notify_pause_state_changed_locked(self) -> None:
        event = self._pause_state_changed
        self._pause_state_changed = asyncio.Event()
        event.set()

    async def _wait_event_excluding_pause(
        self,
        event: asyncio.Event,
        *,
        timeout: float | None,
    ) -> None:
        if timeout is None:
            await event.wait()
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not event.is_set():
            async with self._scope_lock:
                paused = self._paused
                resume_event = self._resume_event
                state_changed = self._pause_state_changed
            if paused:
                pause_started = loop.time()
                event_task = asyncio.create_task(event.wait())
                resume_task = asyncio.create_task(resume_event.wait())
                try:
                    done, pending = await asyncio.wait(
                        {event_task, resume_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    if event_task in done and event_task.result():
                        return
                finally:
                    for task in (event_task, resume_task):
                        if not task.done():
                            task.cancel()
                deadline += loop.time() - pause_started
                continue

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            event_task = asyncio.create_task(event.wait())
            state_task = asyncio.create_task(state_changed.wait())
            try:
                done, pending = await asyncio.wait(
                    {event_task, state_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if event_task in done and event_task.result():
                    return
                if not done:
                    raise asyncio.TimeoutError
            finally:
                for task in (event_task, state_task):
                    if not task.done():
                        task.cancel()

    def _control_url(self, endpoint: str) -> str:
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        if self.router_api_path == "/":
            base = self.upstream_origin
        else:
            base = f"{self.upstream_origin}{self.router_api_path}".rstrip("/")
        return f"{base}{endpoint}"

    async def _control_post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "X-Session-Id": self.bound_session_id,
            "X-Instance-Id": self.bound_instance_id,
        }
        if self._current_version is not None:
            headers["X-Dressage-Expected-Version"] = str(self._current_version)
        url = self._control_url(endpoint)
        client = self._client
        if client is not None:
            response = await client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None), trust_env=False) as temp_client:
                response = await temp_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json() if response.content else {}
        return data if isinstance(data, dict) else {"data": data}

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def _lifespan(_: FastAPI):
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                limits=httpx.Limits(max_connections=100),
            )
            try:
                yield
            finally:
                if self._client is not None:
                    await self._client.aclose()
                    self._client = None

        app = FastAPI(lifespan=_lifespan)

        @app.get("/__proxy_health")
        async def _health() -> dict[str, bool]:
            return {"ok": True}

        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def _proxy(request: Request, path: str) -> Response:
            return await self._handle_proxy(request, path)

        return app

    def _is_chat_completion(self, method: str, path: str) -> bool:
        return method.upper() == "POST" and f"/{path}".rstrip("/").endswith("/chat/completions")

    def _is_anthropic_messages(self, method: str, path: str) -> bool:
        return method.upper() == "POST" and f"/{path}".rstrip("/").endswith("/messages")

    def _is_openai_responses(self, method: str, path: str) -> bool:
        return method.upper() == "POST" and f"/{path}".rstrip("/").endswith("/responses")

    async def _handle_proxy(self, request: Request, path: str) -> Response:
        is_chat = self._is_chat_completion(request.method, path)
        is_anthropic = self._is_anthropic_messages(request.method, path)
        is_responses = self._is_openai_responses(request.method, path)
        is_model_request = is_chat or is_anthropic or is_responses
        upstream_url = (
            self._openai_chat_completions_upstream_url()
            if is_anthropic or is_responses
            else self._join_upstream(path, request.url.query)
        )

        body_bytes = await request.body()
        body_json: dict[str, Any] | None = None
        is_streaming = False
        original_stream = False
        original_chat_request: dict[str, Any] | None = None
        original_anthropic_request: dict[str, Any] | None = None
        original_responses_request: dict[str, Any] | None = None
        parsed_body: Any = None

        if body_bytes:
            try:
                parsed_body = json.loads(body_bytes)
            except json.JSONDecodeError:
                pass

        if isinstance(parsed_body, dict):
            if is_chat:
                original_chat_request = dict(parsed_body)
                body_json = dict(parsed_body)
                original_stream = bool(original_chat_request.get("stream", False))
                is_streaming = original_stream
            elif is_anthropic:
                original_anthropic_request = dict(parsed_body)
                body_json = self._anthropic_messages_to_openai_chat_completion(original_anthropic_request)
                original_stream = bool(original_anthropic_request.get("stream", False))
                is_streaming = original_stream
            elif is_responses:
                original_responses_request = dict(parsed_body)
                body_json = self._openai_responses_to_chat_completion(original_responses_request)
                original_stream = bool(original_responses_request.get("stream", False))
                is_streaming = original_stream
            else:
                body_json = parsed_body
        elif is_model_request and parsed_body is not None:
            LOGGER.warning(
                "[PROXY REQUEST] Model request payload is %s, forwarding without proxy mutation",
                type(parsed_body).__name__,
            )

        LOGGER.info(
            "[PROXY REQUEST] %s /%s -> %s (is_chat=%s, is_anthropic=%s, is_responses=%s, stream=%s)",
            request.method,
            path,
            upstream_url,
            is_chat,
            is_anthropic,
            is_responses,
            original_stream,
        )
        LOGGER.info("[PROXY REQUEST] Body content: %s", self._preview_bytes(body_bytes, limit=1000))
        LOGGER.info("[PROXY REQUEST] Body size: %d bytes", len(body_bytes) if body_bytes else 0)
        LOGGER.info("[PROXY REQUEST] Path: /%s, Upstream: %s", path, upstream_url)

        snapshot = await self._enter_chat_request() if is_model_request else None
        turn_id = snapshot.turn_id if snapshot else None
        if snapshot is not None and snapshot.max_steps_exceeded:
            await self._record_max_steps_error(snapshot)
            return self._max_steps_exceeded_response(
                snapshot,
                response_format="anthropic" if is_anthropic else "openai",
            )

        if is_model_request and body_json is not None:
            tools = body_json.get("tools")
            LOGGER.info(
                "[PROXY REQUEST] Top-level request keys: %s",
                sorted(body_json.keys()),
            )
            LOGGER.info(
                "[PROXY REQUEST] model=%s, msg_count=%d, original_stream=%s, has_stream_options=%s, tool_count=%d",
                body_json.get("model"),
                len(body_json.get("messages", [])),
                original_stream,
                "stream_options" in body_json,
                len(tools) if isinstance(tools, list) else 0,
            )

            if is_anthropic:
                original_request = original_anthropic_request
                request_kind = "anthropic_messages"
            elif is_responses:
                original_request = original_responses_request
                request_kind = "openai_responses"
            else:
                original_request = original_chat_request
                request_kind = "openai_chat_completions"
            if snapshot is not None:
                self._write_model_request_tool_summary(
                    snapshot=snapshot,
                    request_kind=request_kind,
                    original_body=original_request,
                    converted_body=body_json,
                )

            if (
                body_json.get("temperature") is None
                and self.default_temperature is not None
            ):
                body_json["temperature"] = self.default_temperature

            if body_json.get("stream", False):
                stream_options = body_json.get("stream_options")
                if not isinstance(stream_options, dict):
                    stream_options = {}
                else:
                    stream_options = dict(stream_options)
                stream_options["include_usage"] = True
                body_json["stream_options"] = stream_options
            elif "stream_options" in body_json:
                removed_stream_options = body_json.pop("stream_options")
                LOGGER.info(
                    "[PROXY REQUEST] Removed stream_options because upstream stream=false: %s",
                    json.dumps(removed_stream_options, ensure_ascii=False),
                )

            LOGGER.info(
                "[PROXY REQUEST] Final upstream stream=%s, has_stream_options=%s",
                body_json.get("stream"),
                "stream_options" in body_json,
            )

            if body_json != original_request:
                body_bytes = json.dumps(body_json, ensure_ascii=False).encode("utf-8")
                LOGGER.info("[PROXY REQUEST] Final body size: %d bytes", len(body_bytes))
                LOGGER.info(
                    "[PROXY REQUEST] Final body preview: %s",
                    self._preview_bytes(body_bytes, limit=500),
                )

        headers = self._build_upstream_headers(
            request.headers,
            is_chat=is_model_request,
            is_anthropic=is_anthropic,
            turn_id=turn_id,
        )
        LOGGER.info("[PROXY REQUEST] Forwarding with headers: %s", list(headers.keys()))
        if is_model_request:
            LOGGER.info(
                "[PROXY REQUEST] Sticky header %s=%s",
                self.sticky_header_name,
                headers.get(self.sticky_header_name),
            )

        if is_anthropic and is_streaming:
            LOGGER.info("[PROXY REQUEST] Using Anthropic streaming proxy")
            return await self._stream_anthropic_messages_proxy(
                upstream_url,
                body_bytes,
                headers,
                snapshot,
            )
        if is_chat and is_streaming:
            LOGGER.info("[PROXY REQUEST] Using streaming proxy")
            return await self._stream_proxy(
                upstream_url,
                body_bytes,
                headers,
                snapshot,
            )
        if is_responses and is_streaming:
            LOGGER.info("[PROXY REQUEST] Using OpenAI Responses streaming proxy")
            return await self._stream_openai_responses_proxy(
                upstream_url,
                body_bytes,
                headers,
                snapshot,
                original_responses_request or {},
            )
        if is_anthropic:
            LOGGER.info("[PROXY REQUEST] Using Anthropic plain proxy")
            return await self._plain_anthropic_messages_proxy(
                request.method,
                upstream_url,
                body_bytes,
                headers,
                snapshot,
            )
        if is_responses:
            LOGGER.info("[PROXY REQUEST] Using OpenAI Responses plain proxy")
            return await self._plain_openai_responses_proxy(
                request.method,
                upstream_url,
                body_bytes,
                headers,
                snapshot,
                original_responses_request or {},
            )
        LOGGER.info("[PROXY REQUEST] Using plain proxy")
        return await self._plain_proxy(
            request.method,
            upstream_url,
            body_bytes,
            headers,
            snapshot,
        )

    def _join_upstream(self, path: str, query: str | None = None) -> str:
        normalized = path.lstrip("/")
        upstream_url = f"{self.upstream_origin}/{normalized}"
        if query:
            return f"{upstream_url}?{query}"
        return upstream_url

    def _openai_chat_completions_upstream_url(self) -> str:
        if self.router_api_path == "/":
            return f"{self.upstream_origin}/chat/completions"
        return f"{self.upstream_origin}{self.router_api_path}/chat/completions"

    def _build_upstream_headers(
        self,
        original_headers: Any,
        *,
        is_chat: bool,
        is_anthropic: bool = False,
        turn_id: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        reserved_headers = {"host", "content-length", "transfer-encoding"}
        if is_chat:
            reserved_headers.update(
                {
                    "accept-encoding",
                    self.sticky_header_name.lower(),
                    "x-session-id",
                    "x-instance-id",
                    "x-turn-id",
                    "x-dressage-partial-rollout",
                    "x-dressage-expected-version",
                }
            )
        if is_anthropic:
            reserved_headers.update(
                {
                    "anthropic-beta",
                    "anthropic-dangerous-direct-browser-access",
                    "anthropic-version",
                    "x-app",
                    "x-claude-code-session-id",
                }
            )
        for key, value in original_headers.items():
            key_lower = key.lower()
            if key_lower in reserved_headers:
                continue
            if is_anthropic and key_lower.startswith("x-stainless-"):
                continue
            headers[key] = value
        if is_chat:
            self._set_header(headers, self.sticky_header_name, self.bound_session_id)
            self._set_header(headers, "X-Session-Id", self.bound_session_id)
            self._set_header(headers, "X-Instance-Id", self.bound_instance_id)
            if turn_id:
                self._set_header(headers, "X-Turn-Id", turn_id)
            self._set_header(headers, "X-Dressage-Partial-Rollout", "1")
            if self._current_version is not None:
                self._set_header(headers, "X-Dressage-Expected-Version", str(self._current_version))
            self._set_header(headers, "Accept-Encoding", "identity")
        LOGGER.info("[PROXY REQUEST] upstream request headers: %s", headers)
        return headers

    @staticmethod
    def _set_header(headers: dict[str, str], name: str, value: str) -> None:
        for existing in list(headers):
            if existing.lower() == name.lower():
                del headers[existing]
        headers[name] = value

    async def _capture_snapshot(self) -> _TurnSnapshot:
        async with self._scope_lock:
            scope = self._turn_scope
            if scope is None:
                return _TurnSnapshot(
                    session_id=self.bound_session_id,
                    turn_id=None,
                    backend_session_id=None,
                    step=0,
                    scope=None,
                )
            step = scope.step_counter
            scope.step_counter += 1
            return _TurnSnapshot(
                session_id=self.bound_session_id,
                turn_id=scope.turn_id,
                backend_session_id=scope.backend_session_id,
                step=step,
                scope=scope,
            )

    async def _enter_chat_request(self) -> _TurnSnapshot:
        """Wait for rollout resume and atomically mark a chat request active."""

        while True:
            async with self._scope_lock:
                if not self._paused:
                    scope = self._turn_scope
                    if scope is None:
                        return _TurnSnapshot(
                            session_id=self.bound_session_id,
                            turn_id=None,
                            backend_session_id=None,
                            step=0,
                            scope=None,
                        )
                    if self.max_steps is not None and scope.step_counter >= self.max_steps:
                        return _TurnSnapshot(
                            session_id=self.bound_session_id,
                            turn_id=scope.turn_id,
                            backend_session_id=scope.backend_session_id,
                            step=scope.step_counter,
                            scope=scope,
                            max_steps_exceeded=True,
                        )
                    step = scope.step_counter
                    scope.step_counter += 1
                    scope.inflight_requests += 1
                    scope.drained.clear()
                    return _TurnSnapshot(
                        session_id=self.bound_session_id,
                        turn_id=scope.turn_id,
                        backend_session_id=scope.backend_session_id,
                        step=step,
                        scope=scope,
                    )
                resume_event = self._resume_event
            await resume_event.wait()

    def _max_steps_exceeded_response(
        self,
        snapshot: _TurnSnapshot,
        *,
        response_format: str = "openai",
    ) -> Response:
        details = {
            "max_steps": self.max_steps,
            "attempted_step": snapshot.step,
        }
        if response_format == "anthropic":
            payload = {
                "type": "error",
                "error": {
                    "message": "Turn exceeded max_steps.",
                    "type": "rate_limit_error",
                    "code": "max_steps_exceeded",
                    "details": details,
                },
            }
        else:
            payload = {
                "error": {
                    "message": "Turn exceeded max_steps.",
                    "type": "rate_limit_error",
                    "code": "max_steps_exceeded",
                    "details": details,
                }
            }
        return Response(
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            status_code=429,
            media_type="application/json",
        )

    async def _record_max_steps_error(self, snapshot: _TurnSnapshot) -> None:
        scope = snapshot.scope
        if scope is None:
            return
        payload = self._max_steps_error_payload(snapshot)
        async with self._scope_lock:
            scope.max_steps_error = payload
            scope.max_steps_exceeded.set()

    def _max_steps_error_payload(self, snapshot: _TurnSnapshot) -> dict[str, Any]:
        return {
            "error": "max_steps_exceeded",
            "message": "Turn exceeded max_steps.",
            "details": {
                "session_id": snapshot.session_id,
                "turn_id": snapshot.turn_id,
                "max_steps": self.max_steps,
                "attempted_step": snapshot.step,
                "backend_message": "429 Turn exceeded max_steps.",
                "raw_error_code": "rate_limit_error",
            },
        }

    async def _mark_request_started(self, scope: _TurnScope) -> None:
        async with self._scope_lock:
            scope.inflight_requests += 1
            scope.drained.clear()

    async def _mark_request_finished(self, scope: _TurnScope) -> None:
        async with self._scope_lock:
            if scope.inflight_requests > 0:
                scope.inflight_requests -= 1
            if scope.inflight_requests == 0:
                scope.drained.set()

    async def _plain_anthropic_messages_proxy(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
    ) -> Response:
        assert self._client is not None
        try:
            response = await self._send_plain_request(method, url, body, headers)
            if response.status_code >= 400:
                self._log_upstream_error(
                    url=url,
                    status_code=response.status_code,
                    response_headers=dict(response.headers),
                    response_body=response.content,
                    retried=False,
                )
                if snapshot is not None and snapshot.scope is not None:
                    await self._record_context_overflow_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    )
                    if await self._record_rollout_invalidated_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    ):
                        return self._synthetic_anthropic_message_response()
                await self._record_failed_upstream_error(
                    snapshot,
                    url=url,
                    status_code=response.status_code,
                    request_headers=headers,
                    request_body=body,
                    response_headers=dict(response.headers),
                    response_body=response.content,
                )
                if 500 <= response.status_code < 600:
                    return self._anthropic_non_retry_upstream_error_response(
                        upstream_status_code=response.status_code,
                        response_body=response.content,
                    )
                return self._anthropic_error_response_from_openai_response(response)

            try:
                payload = response.json()
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            anthropic_payload = self._openai_chat_completion_to_anthropic_message(payload)
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-length", None)
            response_headers["content-type"] = "application/json"
            return Response(
                content=json.dumps(anthropic_payload, ensure_ascii=False).encode("utf-8"),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )
        finally:
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)

    async def _plain_openai_responses_proxy(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
        original_request: dict[str, Any],
    ) -> Response:
        assert self._client is not None
        try:
            response = await self._send_plain_request(method, url, body, headers)
            if response.status_code >= 400:
                self._log_upstream_error(
                    url=url,
                    status_code=response.status_code,
                    response_headers=dict(response.headers),
                    response_body=response.content,
                    retried=False,
                )
                if snapshot is not None and snapshot.scope is not None:
                    await self._record_context_overflow_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    )
                    if await self._record_rollout_invalidated_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    ):
                        return self._synthetic_openai_response()
                await self._record_failed_upstream_error(
                    snapshot,
                    url=url,
                    status_code=response.status_code,
                    request_headers=headers,
                    request_body=body,
                    response_headers=dict(response.headers),
                    response_body=response.content,
                )
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-length", None)
            if response.status_code >= 400:
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

            try:
                payload = response.json()
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            responses_payload = self._chat_completion_to_openai_response(payload, original_request)
            response_headers["content-type"] = "application/json"
            return Response(
                content=json.dumps(responses_payload, ensure_ascii=False).encode("utf-8"),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )
        finally:
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)

    async def _plain_proxy(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
    ) -> Response:
        assert self._client is not None
        try:
            response = await self._send_plain_request(method, url, body, headers)
            if response.status_code >= 400:
                self._log_upstream_error(
                    url=url,
                    status_code=response.status_code,
                    response_headers=dict(response.headers),
                    response_body=response.content,
                    retried=False,
                )
                if snapshot is not None and snapshot.scope is not None:
                    await self._record_context_overflow_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    )
                    if await self._record_rollout_invalidated_error(
                        snapshot.scope,
                        status_code=response.status_code,
                        response_body=response.content,
                    ):
                        return self._synthetic_chat_completion_response()
                await self._record_failed_upstream_error(
                    snapshot,
                    url=url,
                    status_code=response.status_code,
                    request_headers=headers,
                    request_body=body,
                    response_headers=dict(response.headers),
                    response_body=response.content,
                )
            response_headers = dict(response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-length", None)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )
        finally:
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)

    async def _stream_proxy(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
    ) -> Response:
        assert self._client is not None
        try:
            upstream_response = await self._send_stream_request(url, body, headers)
        except Exception:
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)
            raise

        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            response_headers = dict(upstream_response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-length", None)
            self._log_upstream_error(
                url=url,
                status_code=upstream_response.status_code,
                response_headers=dict(upstream_response.headers),
                response_body=error_body,
                retried=False,
            )
            if snapshot is not None and snapshot.scope is not None:
                await self._record_context_overflow_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                )
                if await self._record_rollout_invalidated_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                ):
                    await upstream_response.aclose()
                    await self._mark_request_finished(snapshot.scope)
                    return self._synthetic_chat_completion_stream_response()
            await self._record_failed_upstream_error(
                snapshot,
                url=url,
                status_code=upstream_response.status_code,
                request_headers=headers,
                request_body=body,
                response_headers=dict(upstream_response.headers),
                response_body=error_body,
            )
            await upstream_response.aclose()
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)
            return Response(
                content=error_body,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        async def _forward():
            try:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
            finally:
                await upstream_response.aclose()
                if snapshot is not None and snapshot.scope is not None:
                    await self._mark_request_finished(snapshot.scope)

        response_headers = dict(upstream_response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("transfer-encoding", None)
        response_headers.pop("content-length", None)
        return StreamingResponse(
            _forward(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type", "text/event-stream"),
        )

    async def _stream_anthropic_messages_proxy(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
    ) -> Response:
        assert self._client is not None
        try:
            upstream_response = await self._send_stream_request(url, body, headers)
        except Exception:
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)
            raise

        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            self._log_upstream_error(
                url=url,
                status_code=upstream_response.status_code,
                response_headers=dict(upstream_response.headers),
                response_body=error_body,
                retried=False,
            )
            if snapshot is not None and snapshot.scope is not None:
                await self._record_context_overflow_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                )
                if await self._record_rollout_invalidated_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                ):
                    await upstream_response.aclose()
                    await self._mark_request_finished(snapshot.scope)
                    return self._synthetic_anthropic_message_stream_response()
            await self._record_failed_upstream_error(
                snapshot,
                url=url,
                status_code=upstream_response.status_code,
                request_headers=headers,
                request_body=body,
                response_headers=dict(upstream_response.headers),
                response_body=error_body,
            )
            await upstream_response.aclose()
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)
            if 500 <= upstream_response.status_code < 600:
                return self._anthropic_non_retry_upstream_error_response(
                    upstream_status_code=upstream_response.status_code,
                    response_body=error_body,
                )
            return self._anthropic_error_response_from_bytes(
                status_code=upstream_response.status_code,
                response_body=error_body,
            )

        async def _forward():
            try:
                async for event in self._iter_anthropic_events_from_openai_stream(upstream_response):
                    yield event
            finally:
                await upstream_response.aclose()
                if snapshot is not None and snapshot.scope is not None:
                    await self._mark_request_finished(snapshot.scope)

        return StreamingResponse(
            _forward(),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
        )

    async def _stream_openai_responses_proxy(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        snapshot: _TurnSnapshot | None,
        original_request: dict[str, Any],
    ) -> Response:
        assert self._client is not None
        try:
            upstream_response = await self._send_stream_request(url, body, headers)
        except Exception:
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)
            raise

        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            response_headers = dict(upstream_response.headers)
            response_headers.pop("content-encoding", None)
            response_headers.pop("transfer-encoding", None)
            response_headers.pop("content-length", None)
            self._log_upstream_error(
                url=url,
                status_code=upstream_response.status_code,
                response_headers=dict(upstream_response.headers),
                response_body=error_body,
                retried=False,
            )
            if snapshot is not None and snapshot.scope is not None:
                await self._record_context_overflow_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                )
                if await self._record_rollout_invalidated_error(
                    snapshot.scope,
                    status_code=upstream_response.status_code,
                    response_body=error_body,
                ):
                    await upstream_response.aclose()
                    await self._mark_request_finished(snapshot.scope)
                    return self._synthetic_openai_response_stream_response()
            await self._record_failed_upstream_error(
                snapshot,
                url=url,
                status_code=upstream_response.status_code,
                request_headers=headers,
                request_body=body,
                response_headers=dict(upstream_response.headers),
                response_body=error_body,
            )
            await upstream_response.aclose()
            if snapshot is not None and snapshot.scope is not None:
                await self._mark_request_finished(snapshot.scope)
            return Response(
                content=error_body,
                status_code=upstream_response.status_code,
                headers=response_headers,
            )

        async def _forward():
            try:
                async for event in self._iter_openai_response_events_from_chat_stream(
                    upstream_response,
                    original_request,
                ):
                    yield event
            finally:
                await upstream_response.aclose()
                if snapshot is not None and snapshot.scope is not None:
                    await self._mark_request_finished(snapshot.scope)

        return StreamingResponse(
            _forward(),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
        )

    async def _send_plain_request(
        self,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        assert self._client is not None
        request_headers = dict(headers)
        request_headers["content-length"] = str(len(body))
        LOGGER.info("[PROXY REQUEST] Setting content-length: %d", len(body))
        return await self._client.request(method=method, url=url, content=body, headers=request_headers)

    async def _send_stream_request(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        assert self._client is not None
        request_headers = dict(headers)
        request_headers["content-length"] = str(len(body))
        LOGGER.info("[PROXY REQUEST] Setting content-length: %d", len(body))
        request = self._client.build_request("POST", url, content=body, headers=request_headers)
        return await self._client.send(request, stream=True)

    def _log_upstream_error(
        self,
        *,
        url: str,
        status_code: int,
        response_headers: dict[str, str],
        response_body: bytes,
        retried: bool,
    ) -> None:
        LOGGER.warning(
            "[PROXY REQUEST] Upstream returned %d for %s "
            "(content_type=%s, content_encoding=%s, retried=%s, body=%s)",
            status_code,
            url,
            response_headers.get("content-type"),
            response_headers.get("content-encoding"),
            retried,
            self._preview_bytes(response_body, limit=2000),
        )

    async def _record_failed_upstream_error(
        self,
        snapshot: _TurnSnapshot | None,
        *,
        url: str,
        status_code: int,
        request_headers: dict[str, str],
        request_body: bytes,
        response_headers: dict[str, str],
        response_body: bytes,
    ) -> None:
        if snapshot is None or snapshot.scope is None:
            return
        payload = self._failed_upstream_error_payload(
            snapshot,
            url=url,
            status_code=status_code,
            request_headers=request_headers,
            request_body=request_body,
            response_headers=response_headers,
            response_body=response_body,
        )
        async with self._scope_lock:
            snapshot.scope.failed_upstream_error = payload

    def _failed_upstream_error_payload(
        self,
        snapshot: _TurnSnapshot,
        *,
        url: str,
        status_code: int,
        request_headers: dict[str, str],
        request_body: bytes,
        response_headers: dict[str, str],
        response_body: bytes,
    ) -> dict[str, Any]:
        body_preview = self._preview_bytes(response_body, limit=1000)
        payload: dict[str, Any] = {
            "error": "upstream_request_failed",
            "message": f"Upstream returned HTTP {status_code}"
            + (f": {body_preview}" if body_preview else ""),
            "status_code": status_code,
            "upstream_url": url,
            "turn_id": snapshot.turn_id,
            "step": snapshot.step,
        }
        if self.debug_log_dir is None:
            return payload

        safe_turn_id = self._safe_filename(snapshot.turn_id or "untagged")
        request_path = self.debug_log_dir / f"upstream_request.{safe_turn_id}.{snapshot.step}.json"
        response_path = self.debug_log_dir / f"upstream_response.{safe_turn_id}.{snapshot.step}.json"
        try:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_dump = self._upstream_request_dump(
                url=url,
                status_code=status_code,
                headers=request_headers,
                body=request_body,
            )
            response_dump = self._upstream_response_dump(
                status_code=status_code,
                headers=response_headers,
                body=response_body,
            )
            request_path.write_text(
                json.dumps(request_dump, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            response_path.write_text(
                json.dumps(response_dump, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            payload["request_path"] = str(request_path)
            payload["response_path"] = str(response_path)
        except OSError as exc:
            payload["dump_error"] = str(exc)
            LOGGER.warning("Failed to write upstream failure dump: %s", exc)
        return payload

    def _upstream_request_dump(
        self,
        *,
        url: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, Any]:
        parsed_body = self._json_body_or_none(body)
        summary: dict[str, Any] = {
            "upstream_url": url,
            "status_code": status_code,
            "headers": self._redact_headers(headers),
        }
        if isinstance(parsed_body, dict):
            tools = parsed_body.get("tools")
            messages = parsed_body.get("messages")
            summary.update(
                {
                    "top_level_keys": sorted(str(key) for key in parsed_body.keys()),
                    "message_count": len(messages) if isinstance(messages, list) else 0,
                    "tool_count": len(tools) if isinstance(tools, list) else 0,
                    "has_thinking": "thinking" in parsed_body,
                    "has_metadata": "metadata" in parsed_body,
                    "has_stream_options": "stream_options" in parsed_body,
                }
            )
            body_value: Any = parsed_body
        else:
            summary.update(
                {
                    "top_level_keys": [],
                    "message_count": 0,
                    "tool_count": 0,
                    "has_thinking": False,
                    "has_metadata": False,
                    "has_stream_options": False,
                }
            )
            body_value = self._preview_bytes(body, limit=10000)
        return {
            **summary,
            "body": body_value,
        }

    def _write_model_request_tool_summary(
        self,
        *,
        snapshot: _TurnSnapshot,
        request_kind: str,
        original_body: dict[str, Any] | None,
        converted_body: dict[str, Any],
    ) -> None:
        if self.debug_log_dir is None:
            return

        raw_tools = _tools_from_request_body(original_body)
        converted_tools = _tools_from_request_body(converted_body)
        payload = {
            "turn_id": snapshot.turn_id,
            "step": snapshot.step,
            "request_kind": request_kind,
            "raw_tool_count": len(raw_tools),
            "converted_tool_count": len(converted_tools),
            "raw_tool_names": _tool_names_from_tools(raw_tools),
            "converted_tool_names": _tool_names_from_tools(converted_tools),
            "raw_tool_types": _tool_types_from_tools(raw_tools),
            "converted_tool_types": _tool_types_from_tools(converted_tools),
        }

        safe_turn_id = self._safe_filename(snapshot.turn_id or "untagged")
        output_path = (
            self.debug_log_dir / f"tool_summary.{safe_turn_id}.{snapshot.step}.json"
        )
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.warning("Failed to write model request tool summary: %s", exc)

    def _upstream_response_dump(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
    ) -> dict[str, Any]:
        parsed_body = self._json_body_or_none(body)
        dump: dict[str, Any] = {
            "status_code": status_code,
            "headers": self._redact_headers(headers),
            "body_preview": self._preview_bytes(body, limit=10000),
        }
        if parsed_body is not None:
            dump["body"] = parsed_body
        return dump

    @staticmethod
    def _json_body_or_none(body: bytes) -> Any:
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
        redacted: dict[str, str] = {}
        for key, value in headers.items():
            lower = key.lower()
            if (
                lower in {"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}
                or "token" in lower
                or lower.endswith("-api-key")
            ):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = value
        return redacted

    @staticmethod
    def _safe_filename(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)

    async def _record_context_overflow_error(
        self,
        scope: _TurnScope,
        *,
        status_code: int,
        response_body: bytes,
    ) -> None:
        payload = self._context_overflow_payload_from_response(
            status_code=status_code,
            response_body=response_body,
        )
        if payload is None:
            return
        async with self._scope_lock:
            scope.context_overflow_error = payload

    async def _record_rollout_invalidated_error(
        self,
        scope: _TurnScope,
        *,
        status_code: int,
        response_body: bytes,
    ) -> bool:
        payload = self._rollout_invalidated_payload_from_response(
            status_code=status_code,
            response_body=response_body,
        )
        if payload is None:
            return False
        async with self._scope_lock:
            scope.rollout_invalidated_error = payload
        return True

    @staticmethod
    def _context_overflow_payload_from_response(
        *,
        status_code: int,
        response_body: bytes,
    ) -> dict[str, Any] | None:
        if status_code != 413 or not response_body:
            return None
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("error") != "context_overflow":
            return None
        return payload

    @staticmethod
    def _rollout_invalidated_payload_from_response(
        *,
        status_code: int,
        response_body: bytes,
    ) -> dict[str, Any] | None:
        if status_code != 502 or not response_body:
            return None
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        detail = payload.get("detail")
        if isinstance(detail, dict):
            candidate = detail
        else:
            candidate = payload
        if candidate.get("error") not in DRESSAGE_ROLLOUT_INVALIDATED_ERRORS:
            return None
        return {str(key): value for key, value in candidate.items()}

    @staticmethod
    def _synthetic_chat_completion_response() -> Response:
        payload = {
            "id": "chatcmpl-rollout-invalidated",
            "object": "chat.completion",
            "created": 0,
            "model": "proxy-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        return Response(
            content=json.dumps(payload).encode("utf-8"),
            status_code=200,
            media_type="application/json",
        )

    @staticmethod
    def _synthetic_chat_completion_stream_response() -> StreamingResponse:
        payload = {
            "id": "chatcmpl-rollout-invalidated",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "proxy-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        }

        async def _events():
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            _events(),
            status_code=200,
            media_type="text/event-stream",
        )

    @staticmethod
    def _synthetic_openai_response() -> Response:
        payload = {
            "id": "resp-rollout-invalidated",
            "object": "response",
            "status": "completed",
            "model": "proxy-model",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                }
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        return Response(
            content=json.dumps(payload).encode("utf-8"),
            status_code=200,
            media_type="application/json",
        )

    @staticmethod
    def _synthetic_openai_response_stream_response() -> StreamingResponse:
        async def _events():
            async for event in _openai_response_stream_events_for_text(
                response_id="resp-rollout-invalidated",
                model="proxy-model",
                text="",
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ):
                yield event

        return StreamingResponse(_events(), status_code=200, media_type="text/event-stream")

    @staticmethod
    def _synthetic_anthropic_message_response() -> Response:
        payload = {
            "id": "msg_rollout_invalidated",
            "type": "message",
            "role": "assistant",
            "model": "proxy-model",
            "content": [{"type": "text", "text": ""}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return Response(
            content=json.dumps(payload).encode("utf-8"),
            status_code=200,
            media_type="application/json",
        )

    @staticmethod
    def _synthetic_anthropic_message_stream_response() -> StreamingResponse:
        async def _events():
            yield _anthropic_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_rollout_invalidated",
                        "type": "message",
                        "role": "assistant",
                        "model": "proxy-model",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
            yield _anthropic_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                },
            )
            yield _anthropic_sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(_events(), status_code=200, media_type="text/event-stream")

    def _openai_responses_to_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        system_parts: list[str] = []
        instructions = _openai_responses_content_to_text(payload.get("instructions"))
        if instructions:
            system_parts.append(instructions)
        input_system_parts, messages = _openai_responses_input_to_chat_messages(payload.get("input"))
        system_parts.extend(input_system_parts)
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

        result: dict[str, Any] = {
            "model": payload.get("model") or "proxy-model",
            "messages": messages,
            "stream": bool(payload.get("stream", False)),
        }
        for source, target in (
            ("max_output_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("parallel_tool_calls", "parallel_tool_calls"),
        ):
            if source in payload:
                result[target] = payload[source]
        tools = _openai_responses_tools_to_chat_tools(payload.get("tools"))
        if tools:
            result["tools"] = tools
        tool_choice = _openai_responses_tool_choice_to_chat(payload.get("tool_choice"))
        if tool_choice is not None:
            result["tool_choice"] = tool_choice
        return result

    def _chat_completion_to_openai_response(
        self,
        payload: dict[str, Any],
        original_request: dict[str, Any],
    ) -> dict[str, Any]:
        choice = _first_openai_choice(payload)
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        assert isinstance(message, dict)
        output: list[dict[str, Any]] = []
        reasoning = _openai_chat_message_reasoning_to_text(message)
        text = _openai_responses_content_to_text(message.get("content"))
        tool_calls = message.get("tool_calls")
        tool_call_items = _openai_tool_calls_to_response_items(tool_calls)
        if reasoning:
            output.append(_openai_response_reasoning_item(reasoning))
        if text or not tool_call_items:
            output.append(_openai_response_message_item(text))
        output.extend(tool_call_items)
        return {
            "id": str(payload.get("id") or "resp_proxy"),
            "object": "response",
            "status": "completed",
            "model": str(payload.get("model") or original_request.get("model") or "proxy-model"),
            "output": output,
            "usage": _openai_usage_to_response_usage(payload.get("usage")),
        }

    async def _iter_openai_response_events_from_chat_stream(
        self,
        upstream_response: httpx.Response,
        original_request: dict[str, Any],
    ):
        response_id = "resp_proxy_stream"
        item_id = "msg_proxy_stream"
        model = str(original_request.get("model") or "proxy-model")
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        tool_call_deltas: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] | None = None

        yield _openai_response_sse(
            "response.created",
            {
                "type": "response.created",
                "response": _openai_response_payload(
                    response_id=response_id,
                    model=model,
                    text="",
                    usage=None,
                    status="in_progress",
                    include_output=False,
                ),
            },
        )

        async for payload_text in _iter_openai_sse_payloads(upstream_response):
            if payload_text == "[DONE]":
                break
            try:
                chunk = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("model"):
                model = str(chunk["model"])
            if isinstance(chunk.get("usage"), dict):
                usage = _openai_usage_to_response_usage(chunk.get("usage"))
            choice = _first_openai_choice(chunk)
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            assert isinstance(delta, dict)
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                reasoning_parts.append(str(reasoning))
            content = delta.get("content")
            if content:
                text = str(content)
                text_parts.append(text)
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    _accumulate_openai_stream_tool_call_delta(tool_call_deltas, tool_call)

        final_reasoning = "".join(reasoning_parts)
        final_text = "".join(text_parts)
        usage = usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        tool_call_records = _openai_stream_tool_call_records(tool_call_deltas)
        completed_output: list[dict[str, Any]] = []

        if final_reasoning:
            output_index = len(completed_output)
            reasoning_item = _openai_response_reasoning_item(
                final_reasoning,
                item_id="rs_proxy_stream",
            )
            async for event in _openai_response_stream_reasoning_item_events(
                output_index=output_index,
                item=reasoning_item,
            ):
                yield event
            completed_output.append(reasoning_item)

        if final_text or not tool_call_records:
            output_index = len(completed_output)
            async for event in _openai_response_stream_text_item_events(
                item_id=item_id,
                output_index=output_index,
                text=final_text,
            ):
                yield event
            completed_output.append(_openai_response_message_item(final_text, item_id=item_id))

        for record in tool_call_records:
            output_index = len(completed_output)
            async for event in _openai_response_stream_function_call_item_events(
                response_id=response_id,
                output_index=output_index,
                item=record["item"],
                argument_deltas=record["argument_deltas"],
            ):
                yield event
            completed_output.append(record["item"])

        yield _openai_response_sse(
            "response.completed",
            {
                "type": "response.completed",
                "response": _openai_response_payload(
                    response_id=response_id,
                    model=model,
                    text=final_text,
                    usage=usage,
                    status="completed",
                    output=completed_output,
                ),
            },
        )

    def _anthropic_messages_to_openai_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        system_text = _anthropic_content_to_text(payload.get("system"))
        if system_text:
            system_parts.append(system_text)

        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").lower() == "system":
                message_system_text = _anthropic_content_to_text(message.get("content"))
                if message_system_text:
                    system_parts.append(message_system_text)
                continue
            messages.extend(_anthropic_message_to_openai_messages(message))
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

        result: dict[str, Any] = {
            "model": payload.get("model") or "proxy-model",
            "messages": messages,
            "stream": bool(payload.get("stream", False)),
        }
        for source, target in (
            ("max_tokens", "max_tokens"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("metadata", "metadata"),
            ("thinking", "thinking"),
        ):
            if source in payload:
                result[target] = payload[source]
        if "stop_sequences" in payload:
            result["stop"] = payload["stop_sequences"]
        tools = _anthropic_tools_to_openai_tools(payload.get("tools"))
        if tools:
            result["tools"] = tools
        tool_choice = _anthropic_tool_choice_to_openai(payload.get("tool_choice"))
        if tool_choice is not None:
            result["tool_choice"] = tool_choice
        return result

    def _openai_chat_completion_to_anthropic_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        choice = _first_openai_choice(payload)
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        assert isinstance(message, dict)
        content = _openai_message_to_anthropic_content(message)
        usage = _openai_usage_to_anthropic_usage(payload.get("usage"))
        return {
            "id": str(payload.get("id") or "msg_proxy"),
            "type": "message",
            "role": "assistant",
            "model": str(payload.get("model") or "proxy-model"),
            "content": content,
            "stop_reason": _openai_finish_reason_to_anthropic(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": usage,
        }

    async def _iter_anthropic_events_from_openai_stream(self, upstream_response: httpx.Response):
        state = _AnthropicStreamState()
        yield state.message_start(model="proxy-model")
        async for payload in _iter_openai_sse_payloads(upstream_response):
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("model"):
                state.model = str(chunk["model"])
            usage = _openai_usage_to_anthropic_usage(chunk.get("usage"))
            choice = _first_openai_choice(chunk)
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            assert isinstance(delta, dict)
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                async for event in state.emit_text_like_block(
                    "thinking",
                    str(reasoning),
                    delta_type="thinking_delta",
                    delta_key="thinking",
                ):
                    yield event
            content = delta.get("content")
            if content:
                async for event in state.emit_text_like_block(
                    "text",
                    str(content),
                    delta_type="text_delta",
                    delta_key="text",
                ):
                    yield event
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    async for event in state.emit_tool_call_delta(tool_call):
                        yield event
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                async for event in state.finish(
                    stop_reason=_openai_finish_reason_to_anthropic(finish_reason),
                    usage=usage,
                ):
                    yield event
                return
        async for event in state.finish(stop_reason="end_turn", usage=None):
            yield event

    def _anthropic_error_response_from_openai_response(self, response: httpx.Response) -> Response:
        return self._anthropic_error_response_from_bytes(
            status_code=response.status_code,
            response_body=response.content,
        )

    @staticmethod
    def _anthropic_non_retry_upstream_error_response(
        *,
        upstream_status_code: int,
        response_body: bytes,
    ) -> Response:
        body_preview = RolloutLLMProxy._preview_bytes(response_body, limit=1000)
        message = f"Upstream returned HTTP {upstream_status_code}"
        if body_preview:
            message = f"{message}: {body_preview}"
        body = {
            "type": "error",
            "error": {
                "type": "upstream_error",
                "message": message,
            },
        }
        return Response(
            content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            status_code=400,
            media_type="application/json",
        )

    @staticmethod
    def _anthropic_error_response_from_bytes(*, status_code: int, response_body: bytes) -> Response:
        message = (
            RolloutLLMProxy._preview_bytes(response_body, limit=1000)
            if response_body
            else "Upstream error."
        )
        error_type = "api_error"
        try:
            payload = json.loads(response_body.decode("utf-8")) if response_body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            raw_error = payload.get("error")
            if isinstance(raw_error, dict):
                message = str(raw_error.get("message") or message)
                error_type = str(raw_error.get("type") or raw_error.get("code") or error_type)
            elif payload.get("message") is not None:
                message = str(payload.get("message"))
        body = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }
        return Response(
            content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            status_code=status_code,
            media_type="application/json",
        )

    @staticmethod
    def _preview_bytes(body: bytes, *, limit: int) -> str:
        if not body:
            return ""
        hex_limit = min(limit, 96)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            suffix = "...(truncated)" if len(body) > hex_limit else ""
            return f"<{len(body)} bytes binary; hex={body[:hex_limit].hex()}{suffix}>"
        sample = text[: min(len(text), limit)]
        control_count = sum(
            1 for char in sample if ord(char) < 32 and char not in {"\n", "\r", "\t"}
        )
        if control_count > max(8, len(sample) // 10):
            suffix = "...(truncated)" if len(body) > hex_limit else ""
            return f"<{len(body)} bytes binary; hex={body[:hex_limit].hex()}{suffix}>"
        if len(text) <= limit:
            return text
        return text[:limit] + "...(truncated)"


def _anthropic_sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _openai_response_sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _openai_responses_content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"input_text", "output_text", "text"} and item.get("text") is not None:
                parts.append(str(item["text"]))
            elif item.get("text") is not None:
                parts.append(str(item["text"]))
            elif item.get("output") is not None:
                parts.append(str(item["output"]))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return json.dumps(value, ensure_ascii=False)


def _openai_chat_message_reasoning_to_text(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning"):
        if key not in message:
            continue
        text = _openai_responses_content_to_text(message.get(key))
        if text:
            return text
    return ""


def _openai_response_reasoning_to_text(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if isinstance(summary, list):
        parts: list[str] = []
        for part in summary:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            text = part.get("text") or part.get("summary_text") or part.get("content")
            if text is not None:
                parts.append(_openai_responses_content_to_text(text))
        text = "\n".join(part for part in parts if part)
        if text:
            return text
    for key in ("content", "text", "reasoning_content", "reasoning"):
        if key in item:
            text = _openai_responses_content_to_text(item.get(key))
            if text:
                return text
    return ""


def _openai_responses_input_to_chat_messages(value: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if value is None:
        return [], []
    if isinstance(value, str):
        return [], [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return [], [{"role": "user", "content": _openai_responses_content_to_text(value)}]

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    pending_assistant_index: int | None = None
    pending_reasoning_content: str | None = None
    for item in value:
        if isinstance(item, str):
            pending_assistant_index = None
            pending_reasoning_content = None
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            pending_assistant_index = None
            pending_reasoning_content = None
            messages.append({"role": "user", "content": _openai_responses_content_to_text(item)})
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            pending_assistant_index = None
            reasoning_text = _openai_response_reasoning_to_text(item)
            if reasoning_text:
                pending_reasoning_content = (
                    f"{pending_reasoning_content}\n{reasoning_text}"
                    if pending_reasoning_content
                    else reasoning_text
                )
        elif item_type == "function_call":
            tool_call = _openai_response_function_call_to_chat_tool_call(item)
            if pending_assistant_index is not None:
                pending_message = messages[pending_assistant_index]
                tool_calls = pending_message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    tool_calls = []
                    pending_message["tool_calls"] = tool_calls
                tool_calls.append(tool_call)
            else:
                messages.append(
                    _openai_response_function_call_to_chat_message(
                        item,
                        reasoning_content=pending_reasoning_content,
                    )
                )
                pending_reasoning_content = None
                pending_assistant_index = len(messages) - 1
        elif item_type == "function_call_output":
            pending_assistant_index = None
            pending_reasoning_content = None
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or "call_proxy"),
                    "content": _openai_responses_content_to_text(item.get("output")),
                }
            )
        elif item_type == "message" or item.get("role") is not None:
            role = str(item.get("role") or "user").lower()
            content = _openai_responses_content_to_text(item.get("content"))
            if role in {"developer", "system"}:
                pending_assistant_index = None
                pending_reasoning_content = None
                if content:
                    system_parts.append(content)
                continue
            if role not in {"user", "assistant", "tool"}:
                role = "user"
            message: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant" and pending_reasoning_content:
                message["reasoning_content"] = pending_reasoning_content
                pending_reasoning_content = None
            else:
                pending_reasoning_content = None
            messages.append(message)
            if role == "assistant":
                pending_assistant_index = len(messages) - 1
            else:
                pending_assistant_index = None
        else:
            pending_assistant_index = None
            pending_reasoning_content = None
            messages.append({"role": "user", "content": json.dumps(item, ensure_ascii=False)})
    return system_parts, messages


def _openai_response_function_call_to_chat_tool_call(item: dict[str, Any]) -> dict[str, Any]:
    arguments = item.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "id": str(item.get("call_id") or item.get("id") or "call_proxy"),
        "type": "function",
        "function": {
            "name": str(item.get("name") or "tool"),
            "arguments": arguments,
        },
    }


def _openai_response_function_call_to_chat_message(
    item: dict[str, Any],
    *,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_openai_response_function_call_to_chat_tool_call(item)],
    }
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return message


def _openai_responses_tools_to_chat_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for tool in value:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        raw_function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(raw_function, dict):
            continue
        function: dict[str, Any] = {
            "name": str(raw_function.get("name") or "tool"),
            "description": str(raw_function.get("description") or ""),
            "parameters": raw_function.get("parameters") or {"type": "object", "properties": {}},
        }
        if "strict" in raw_function:
            function["strict"] = raw_function["strict"]
        tools.append({"type": "function", "function": function})
    return tools


def _tools_from_request_body(body: dict[str, Any] | None) -> list[Any]:
    if not isinstance(body, dict):
        return []
    tools = body.get("tools")
    return list(tools) if isinstance(tools, list) else []


def _tool_names_from_tools(tools: list[Any], *, prefix: str = "") -> list[str]:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        raw_function = tool.get("function")
        if isinstance(raw_function, dict):
            name = raw_function.get("name")
        else:
            name = tool.get("name")
        child_tools = tool.get("tools")
        has_child_tools = isinstance(child_tools, list)
        if name is not None and (
            isinstance(raw_function, dict)
            or tool.get("type") == "function"
            or not has_child_tools
        ):
            names.append(f"{prefix}{name}")

        if has_child_tools:
            namespace = str(tool.get("name") or tool.get("namespace") or "")
            child_prefix = f"{prefix}{namespace}." if namespace else prefix
            names.extend(_tool_names_from_tools(child_tools, prefix=child_prefix))
    return names


def _tool_types_from_tools(tools: list[Any]) -> list[str]:
    types: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            types.append(str(tool.get("type") or ""))
    return types


def _openai_responses_tool_choice_to_chat(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    if value.get("type") == "function":
        function = value.get("function") if isinstance(value.get("function"), dict) else value
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "function": {"name": str(function["name"])}}
    return None


def _openai_response_message_item(
    text: str,
    *,
    status: str = "completed",
    item_id: str = "msg_proxy",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _openai_response_reasoning_item(
    text: str,
    *,
    item_id: str = "rs_proxy",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
    }


def _openai_response_payload(
    *,
    response_id: str,
    model: str,
    text: str,
    usage: dict[str, int] | None,
    status: str,
    item_id: str = "msg_proxy",
    include_output: bool = True,
    output: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if output is not None:
        response_output = output
    elif include_output:
        response_output = [_openai_response_message_item(text, status=status, item_id=item_id)]
    else:
        response_output = []
    return {
        "id": response_id,
        "object": "response",
        "status": status,
        "model": model,
        "output": response_output,
        "usage": usage,
    }


def _openai_response_function_call_item(
    *,
    call_id: str,
    name: str,
    arguments: str,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def _openai_tool_calls_to_response_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for index, tool_call in enumerate(value):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        if not isinstance(function, dict):
            function = {}
        call_id = str(tool_call.get("id") or f"call_proxy_{index}")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        items.append(
            _openai_response_function_call_item(
                call_id=call_id,
                name=str(function.get("name") or "tool"),
                arguments=arguments,
            )
        )
    return items


def _accumulate_openai_stream_tool_call_delta(
    tool_call_deltas: dict[int, dict[str, Any]],
    tool_call: Any,
) -> None:
    if not isinstance(tool_call, dict):
        return
    raw_index = tool_call.get("index")
    try:
        index = int(raw_index)
    except (TypeError, ValueError):
        index = 0
    state = tool_call_deltas.setdefault(
        index,
        {"id": None, "type": None, "name": None, "arguments_parts": []},
    )
    if tool_call.get("id"):
        state["id"] = str(tool_call["id"])
    if tool_call.get("type"):
        state["type"] = str(tool_call["type"])
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    if not isinstance(function, dict):
        function = {}
    if function.get("name"):
        state["name"] = str(function["name"])
    if "arguments" in function and function.get("arguments") is not None:
        arguments_delta = function.get("arguments")
        if not isinstance(arguments_delta, str):
            arguments_delta = json.dumps(arguments_delta or {}, ensure_ascii=False)
        state["arguments_parts"].append(arguments_delta)


def _openai_stream_tool_call_records(
    tool_call_deltas: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in sorted(tool_call_deltas):
        state = tool_call_deltas[index]
        argument_deltas = [
            str(part)
            for part in state.get("arguments_parts", [])
            if part is not None
        ]
        if not (state.get("id") or state.get("name") or argument_deltas):
            continue
        call_id = str(state.get("id") or f"call_proxy_{index}")
        arguments = "".join(argument_deltas)
        item = _openai_response_function_call_item(
            call_id=call_id,
            name=str(state.get("name") or "tool"),
            arguments=arguments,
        )
        records.append({"item": item, "argument_deltas": argument_deltas})
    return records


def _openai_usage_to_response_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    output_tokens = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
    total_tokens = int(value.get("total_tokens", input_tokens + output_tokens) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


async def _openai_response_stream_events_for_text(
    *,
    response_id: str,
    model: str,
    text: str,
    usage: dict[str, int],
):
    item_id = "msg_proxy_stream"
    yield _openai_response_sse(
        "response.created",
        {
            "type": "response.created",
            "response": _openai_response_payload(
                response_id=response_id,
                model=model,
                text="",
                usage=None,
                status="in_progress",
                item_id=item_id,
                include_output=False,
            ),
        },
    )
    async for event in _openai_response_stream_text_item_events(
        item_id=item_id,
        output_index=0,
        text=text,
    ):
        yield event
    yield _openai_response_sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": _openai_response_payload(
                response_id=response_id,
                model=model,
                text=text,
                usage=usage,
                status="completed",
                item_id=item_id,
            ),
        },
    )


async def _openai_response_stream_reasoning_item_events(
    *,
    output_index: int,
    item: dict[str, Any],
):
    item_id = str(item["id"])
    yield _openai_response_sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "item_id": item_id,
            "output_index": output_index,
            "item": item,
        },
    )
    yield _openai_response_sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "item_id": item_id,
            "output_index": output_index,
            "item": item,
        },
    )


async def _openai_response_stream_text_item_events(
    *,
    item_id: str,
    output_index: int,
    text: str,
):
    message = _openai_response_message_item(text, item_id=item_id)
    part = {"type": "output_text", "text": text}
    yield _openai_response_sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": _openai_response_message_item("", status="in_progress", item_id=item_id),
        },
    )
    yield _openai_response_sse(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
    )
    if text:
        yield _openai_response_sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        )
    yield _openai_response_sse(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "text": text,
        },
    )
    yield _openai_response_sse(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": part,
        },
    )
    yield _openai_response_sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "item_id": item_id,
            "output_index": output_index,
            "item": message,
        },
    )


async def _openai_response_stream_function_call_item_events(
    *,
    response_id: str,
    output_index: int,
    item: dict[str, Any],
    argument_deltas: list[str],
):
    item_id = str(item["id"])
    in_progress_item = dict(item)
    in_progress_item["status"] = "in_progress"
    in_progress_item["arguments"] = ""
    deltas = list(argument_deltas)
    if not deltas and item.get("arguments"):
        deltas = [str(item["arguments"])]
    yield _openai_response_sse(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": output_index,
            "item": in_progress_item,
        },
    )
    for delta in deltas:
        yield _openai_response_sse(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": output_index,
                "delta": delta,
            },
        )
    yield _openai_response_sse(
        "response.function_call_arguments.done",
        {
            "type": "response.function_call_arguments.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "arguments": str(item.get("arguments") or ""),
            "item": item,
        },
    )
    yield _openai_response_sse(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": output_index,
            "item": item,
        },
    )


def _anthropic_content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("text") is not None:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return json.dumps(value, ensure_ascii=False)


def _anthropic_message_to_openai_messages(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = str(message.get("role") or "user")
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return [{"role": role, "content": _anthropic_content_to_text(content)}]

    if role == "assistant":
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and block.get("text") is not None:
                text_parts.append(str(block["text"]))
            elif block_type in {"thinking", "redacted_thinking"}:
                thinking = block.get("thinking") or block.get("text") or block.get("data")
                if thinking is not None:
                    reasoning_parts.append(str(thinking))
            elif block_type == "tool_use":
                tool_input = block.get("input")
                if not isinstance(tool_input, str):
                    tool_input = json.dumps(tool_input or {}, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": str(block.get("id") or "toolu_proxy"),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or "tool"),
                            "arguments": tool_input,
                        },
                    }
                )
        result: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }
        if reasoning_parts:
            result["reasoning_content"] = "\n".join(reasoning_parts)
        if tool_calls:
            result["tool_calls"] = tool_calls
        return [result]

    messages: list[dict[str, Any]] = []
    pending_text: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            pending_text.append(str(block))
            continue
        block_type = block.get("type")
        if block_type == "text" and block.get("text") is not None:
            pending_text.append(str(block["text"]))
        elif block_type == "tool_result":
            if pending_text:
                messages.append({"role": "user", "content": "\n".join(pending_text)})
                pending_text = []
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or block.get("id") or "toolu_proxy"),
                    "content": _anthropic_content_to_text(block.get("content")),
                }
            )
        else:
            pending_text.append(json.dumps(block, ensure_ascii=False))
    if pending_text or not messages:
        messages.append({"role": "user", "content": "\n".join(pending_text)})
    return messages


def _anthropic_tools_to_openai_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for tool in value:
        if not isinstance(tool, dict):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or "tool"),
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def _anthropic_tool_choice_to_openai(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "any":
            return "required"
        if value in {"auto", "none", "required"}:
            return value
        return None
    if not isinstance(value, dict):
        return None
    choice_type = value.get("type")
    if choice_type == "any":
        return "required"
    if choice_type in {"auto", "none"}:
        return choice_type
    if choice_type == "tool" and value.get("name"):
        return {"type": "function", "function": {"name": str(value["name"])}}
    return None


def _first_openai_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _openai_message_to_anthropic_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.append({"type": "thinking", "thinking": str(reasoning)})
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": str(text)})
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            raw_arguments = function.get("arguments") if isinstance(function, dict) else None
            try:
                parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except json.JSONDecodeError:
                parsed_arguments = {"arguments": raw_arguments}
            content.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or "toolu_proxy"),
                    "name": str(function.get("name") or "tool") if isinstance(function, dict) else "tool",
                    "input": parsed_arguments if isinstance(parsed_arguments, dict) else {},
                }
            )
    if not content:
        content.append({"type": "text", "text": ""})
    return content


def _openai_usage_to_anthropic_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    input_tokens = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    output_tokens = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _openai_finish_reason_to_anthropic(value: Any) -> str:
    if value in {None, ""}:
        return "end_turn"
    normalized = str(value)
    if normalized == "stop":
        return "end_turn"
    if normalized == "length":
        return "max_tokens"
    if normalized == "tool_calls":
        return "tool_use"
    return normalized


async def _iter_openai_sse_payloads(response: httpx.Response):
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            for payload in _payloads_from_sse_event(raw_event):
                yield payload
    if buffer.strip():
        for payload in _payloads_from_sse_event(buffer):
            yield payload


def _payloads_from_sse_event(raw_event: str) -> list[str]:
    payloads: list[str] = []
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        payloads.append("\n".join(data_lines))
    return payloads


class _AnthropicStreamState:
    def __init__(self) -> None:
        self.model = "proxy-model"
        self._active_block_type: str | None = None
        self._active_block_index: int | None = None
        self._next_index = 0
        self._finished = False
        self._tool_call_blocks: dict[int, int] = {}

    def message_start(self, *, model: str) -> bytes:
        self.model = model
        return _anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_proxy_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    async def emit_text_like_block(
        self,
        block_type: str,
        text: str,
        *,
        delta_type: str,
        delta_key: str,
    ):
        if self._active_block_type != block_type:
            async for event in self._stop_active_block():
                yield event
            index = self._next_index
            self._next_index += 1
            self._active_block_type = block_type
            self._active_block_index = index
            start_block = {"type": block_type}
            if block_type == "text":
                start_block["text"] = ""
            else:
                start_block["thinking"] = ""
            yield _anthropic_sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start_block},
            )
        assert self._active_block_index is not None
        yield _anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self._active_block_index,
                "delta": {"type": delta_type, delta_key: text},
            },
        )

    async def emit_tool_call_delta(self, tool_call: dict[str, Any]):
        tool_index = int(tool_call.get("index", 0) or 0)
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        if tool_index not in self._tool_call_blocks:
            async for event in self._stop_active_block():
                yield event
            index = self._next_index
            self._next_index += 1
            self._tool_call_blocks[tool_index] = index
            self._active_block_type = f"tool_use:{tool_index}"
            self._active_block_index = index
            yield _anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": str(tool_call.get("id") or f"toolu_{tool_index}"),
                        "name": str(function.get("name") or "tool"),
                        "input": {},
                    },
                },
            )
        partial_json = function.get("arguments")
        if partial_json:
            yield _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._tool_call_blocks[tool_index],
                    "delta": {"type": "input_json_delta", "partial_json": str(partial_json)},
                },
            )

    async def finish(self, *, stop_reason: str, usage: dict[str, int] | None):
        if self._finished:
            return
        async for event in self._stop_active_block():
            yield event
        self._finished = True
        yield _anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": int((usage or {}).get("output_tokens", 0) or 0)},
            },
        )
        yield _anthropic_sse("message_stop", {"type": "message_stop"})

    async def _stop_active_block(self):
        if self._active_block_index is None:
            return
        yield _anthropic_sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": self._active_block_index},
        )
        self._active_block_type = None
        self._active_block_index = None
