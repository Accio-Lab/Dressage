from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import uvicorn
from pydantic import BaseModel, ConfigDict, Field

from blackbox_server.adapters.base import (
    BackendAdapter,
    BackendCapabilities,
    BackendMaxStepsExceededError,
    BackendProcessError,
    BackendProtocolError,
    BackendTransportError,
    backend_context_overflow_from_proxy_payload,
)
from blackbox_server.core.models import (
    AdapterResponse,
    BindingContext,
    FunctionCall,
    Message,
    ProxyOptions,
    SessionContext,
    ToolCall,
    TraceEvent,
    TurnContext,
    TurnUsage,
    utcnow,
)
from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy

_OC_MSG_COUNT_KEY = "__bbs_opencode_msg_count"
_MAX_STEPS_EXCEEDED_ERROR_CODE = "max_steps_exceeded"
_MAX_STEPS_EXCEEDED_MESSAGE_FRAGMENT = "turn exceeded max_steps"
_DENY_QUESTION_PERMISSION = {
    "*": "allow",
    "question": "deny",
    "doom_loop": "deny",
}

LOGGER = logging.getLogger(__name__)


class _IncompleteHistoryError(Exception):
    pass


class _BackgroundUvicornServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        # Keep the parent process in charge of SIGINT/SIGTERM.
        yield


class OpencodeModelLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: int = Field(gt=0)
    output: int = Field(gt=0)
    input: int | None = Field(default=None, gt=0)


class OpencodeCompactionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto: bool | None = None
    prune: bool | None = None
    tail_turns: int | None = Field(default=None, ge=0)
    preserve_recent_tokens: int | None = Field(default=None, ge=0)
    reserved: int | None = Field(default=None, ge=0)


class OpencodeBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    provider_name: str
    provider_package: str
    model_id: str
    model_name: str
    model_limit: OpencodeModelLimit | None = None
    compaction: OpencodeCompactionOptions | None = None
    proxy: ProxyOptions = Field(default_factory=ProxyOptions)


class OpencodeAdapter(BackendAdapter):
    def __init__(self) -> None:
        self._binding_context: BindingContext | None = None
        self._options: OpencodeBackendOptions | None = None
        self._client: httpx.AsyncClient | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._port: int | None = None
        self._stdout_handle = None
        self._stderr_handle = None
        self._proxy: RolloutLLMProxy | None = None
        self._proxy_port: int | None = None
        self._proxy_server: uvicorn.Server | None = None
        self._proxy_task: asyncio.Task[Any] | None = None
        self._process_group_id: int | None = None

    async def initialize(self, binding_context: BindingContext) -> None:
        self._binding_context = binding_context
        options = self._parse_options(binding_context.binding.backend_options)
        self._options = options
        runtime_dir = Path(binding_context.binding.runtime_dir)
        config_dir = runtime_dir / "home" / ".config" / "opencode"
        prompts_dir = config_dir / "prompts"
        logs_dir = runtime_dir / "logs"
        run_dir = runtime_dir / "run"
        tmp_dir = runtime_dir / "tmp"
        cache_dir = runtime_dir / "home" / ".cache"
        data_dir = runtime_dir / "home" / ".local" / "share"

        for path in (config_dir, prompts_dir, logs_dir, run_dir, tmp_dir, cache_dir, data_dir):
            path.mkdir(parents=True, exist_ok=True)

        if binding_context.binding.system_prompt is not None:
            source = Path(binding_context.binding.system_prompt.source_file)
            target = Path(binding_context.binding.system_prompt.runtime_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        await self._start_proxy(binding_context, options)

        config_path = config_dir / "opencode.json"
        config_path.write_text(
            json.dumps(
                self._build_opencode_config(binding_context, options),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._port = self._find_free_port()
        self._stdout_handle = open(logs_dir / "opencode.stdout.log", "ab")
        self._stderr_handle = open(logs_dir / "opencode.stderr.log", "ab")
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(runtime_dir / "home"),
                "XDG_CONFIG_HOME": str(runtime_dir / "home" / ".config"),
                "XDG_CACHE_HOME": str(cache_dir),
                "XDG_DATA_HOME": str(data_dir),
                "TMPDIR": str(tmp_dir),
            }
        )
        binary = os.getenv("OPENCODE_BIN", "opencode")
        try:
            self._process = await asyncio.create_subprocess_exec(
                binary,
                "serve",
                "--port",
                str(self._port),
                "--hostname",
                "127.0.0.1",
                cwd=str(runtime_dir),
                env=env,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
                start_new_session=True,
            )
            self._process_group_id = self._process.pid
        except FileNotFoundError as exc:
            raise BackendProcessError(
                f"opencode binary not found. Set OPENCODE_BIN or install opencode. ({binary})"
            ) from exc

        (run_dir / "opencode.pid").write_text(str(self._process.pid), encoding="utf-8")
        (run_dir / "opencode.port").write_text(str(self._port), encoding="utf-8")
        self._client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self._port}", timeout=None)

        try:
            await self._wait_until_healthy()
        except Exception:
            await self.shutdown()
            raise

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        deadline = asyncio.get_running_loop().time() + turn_context.deadline_seconds
        if not await self.health():
            raise BackendProcessError("opencode backend is not healthy.")
        if self._proxy is not None:
            await self._proxy.open_turn(turn_context.turn_id, backend_session_id=session_context.backend_session_id)

        success = False
        try:
            backend_session_id = await self._ensure_backend_session(
                session_context,
                self._remaining_timeout(deadline, operation="ensure opencode session"),
            )
            if self._proxy is not None:
                await self._proxy.update_turn_backend_session(backend_session_id)

            message = new_messages[0]
            prev_msg_count = int(session_context.metadata.get(_OC_MSG_COUNT_KEY, 0) or 0)
            payload = {
                "parts": [{"type": "text", "text": message.content}],
                "agent": "build",
            }
            post_task = asyncio.create_task(
                self._post_message_with_connect_retry(
                    f"/session/{backend_session_id}/message",
                    json_body=payload,
                    deadline=deadline,
                    operation="send opencode message",
                )
            )
            post_result = await self._await_backend_task_or_proxy_max_steps(
                post_task,
                session_context=session_context,
                proxy=self._proxy,
            )
            _raise_if_opencode_structured_error(post_result, self._options)
            # Wait until the proxied upstream turn has fully drained so we do not
            # snapshot opencode history before the final assistant/tool messages land.
            if self._proxy is not None:
                await self._proxy.drain_turn(
                    timeout=self._remaining_timeout(deadline, operation="wait for rollout proxy drain")
                )
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()
            all_messages, outputs, trace_events, usage = await self._request_get_with_retry(
                f"/session/{backend_session_id}/message",
                deadline=deadline,
                turn_id=turn_context.turn_id,
                prev_msg_count=prev_msg_count,
            )
            session_context.metadata[_OC_MSG_COUNT_KEY] = len(all_messages)
            success = True
            return AdapterResponse(
                outputs=outputs,
                trace_events=trace_events,
                usage=usage,
                backend_session_id=backend_session_id,
            )
        finally:
            if self._proxy is not None:
                drain_timeout = None if success else 2.0
                try:
                    await self._proxy.drain_turn(timeout=drain_timeout)
                except asyncio.TimeoutError:
                    LOGGER.warning("Timed out draining rollout proxy requests for turn %s", turn_context.turn_id)
                finally:
                    await self._proxy.clear_turn()

    async def abort_session(self, session_context: SessionContext) -> bool:
        if self._client is None or session_context.backend_session_id is None:
            return False
        try:
            data = await self._request(
                "POST",
                f"/session/{session_context.backend_session_id}/abort",
                json_body={},
                timeout=5.0,
            )
        except BackendTransportError:
            return False
        return bool(data) if data is not None else False

    async def health(self) -> bool:
        if self._process is None or self._client is None:
            return False
        if self._process.returncode is not None:
            return False
        try:
            response = await self._client.get("/global/health", timeout=2.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError:
            return False
        return bool(data.get("healthy"))

    async def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            chat=True,
            abort=True,
            pause_resume=True,
            stream=False,
            multi_message_input=False,
            system_message=True,
            history_injection=False,
        )

    async def pause(
        self,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if self._proxy is None:
            return {
                "status": "not_started",
                "reason": reason,
                "quiesced": True,
                "http_inflight_requests": 0,
                "active_sglang_generations": 0,
                "suspended_generations": 0,
            }
        return await self._proxy.pause(reason=reason, timeout_seconds=timeout_seconds)

    async def resume(
        self,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        if self._proxy is None:
            return {"status": "not_started", "reason": reason, "version": version}
        return await self._proxy.resume(version=version, reason=reason)

    def pause_state(self) -> dict[str, Any]:
        if self._proxy is None:
            return {"paused": False, "http_inflight_requests": 0}
        return self._proxy.pause_state()

    async def shutdown(self) -> None:
        if self._proxy_server is not None:
            self._proxy_server.should_exit = True
        if self._proxy_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proxy_task, timeout=5.0)
            self._proxy_task = None
        self._proxy_server = None
        self._proxy = None
        self._proxy_port = None

        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.post("/instance/dispose", json={}, timeout=5.0)
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

        if self._process is not None:
            if self._process.returncode is None:
                # Try to terminate the entire process group first
                if self._process_group_id is not None:
                    try:
                        os.killpg(self._process_group_id, signal.SIGTERM)
                        await asyncio.wait_for(self._process.wait(), timeout=3.0)
                    except (ProcessLookupError, asyncio.TimeoutError):
                        pass
                
                # Fallback to individual process termination
                if self._process.returncode is None:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self._process.kill()
                        await self._process.wait()
            self._process = None
            self._process_group_id = None

        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
                setattr(self, handle_name, None)

    def _parse_options(self, backend_options: dict[str, Any]) -> OpencodeBackendOptions:
        try:
            return OpencodeBackendOptions.model_validate(backend_options)
        except Exception as exc:
            raise BackendProtocolError(f"Invalid opencode backend_options: {exc}") from exc

    def _build_opencode_config(
        self,
        binding_context: BindingContext,
        options: OpencodeBackendOptions,
    ) -> dict[str, Any]:
        provider_key = options.provider_id
        model_key = options.model_id
        base_url = (
            f"http://127.0.0.1:{self._proxy_port}{binding_context.binding.router_api_path}"
            if self._proxy_port is not None
            else binding_context.binding.router_base_url
        )
        model_config: dict[str, Any] = {"name": options.model_name, "tools": True}
        if options.model_limit is not None:
            model_config["limit"] = options.model_limit.model_dump(exclude_none=True)

        config: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                provider_key: {
                    "npm": options.provider_package,
                    "name": options.provider_name,
                    "options": {
                        "baseURL": base_url,
                        "timeout": binding_context.effective_config.router_timeout,
                    },
                    "models": {model_key: model_config},
                }
            },
            "model": f"{provider_key}/{model_key}",
            "small_model": f"{provider_key}/{model_key}",
            "permission": dict(_DENY_QUESTION_PERMISSION),
        }
        if options.compaction is not None:
            config["compaction"] = options.compaction.model_dump(exclude_none=True)

        agent_config: dict[str, Any] = {"title": {"disable": True}}
        if binding_context.binding.system_prompt is not None:
            agent_config["build"] = {"prompt": "{file:./prompts/build.txt}"}
        config["agent"] = agent_config
        return config

    async def _start_proxy(self, binding_context: BindingContext, options: OpencodeBackendOptions) -> None:
        bound_session_id = binding_context.binding.bound_session_id
        bound_instance_id = binding_context.binding.bound_instance_id
        upstream_origin = self._resolve_upstream_origin(binding_context.binding.router_base_url)
        self._proxy_port = self._find_free_port()
        LOGGER.info(
            "starting rollout proxy on port %d, upstream_origin=%s, router_api_path=%s",
            self._proxy_port,
            upstream_origin,
            binding_context.binding.router_api_path,
        )
        self._proxy = RolloutLLMProxy(
            upstream_origin=upstream_origin,
            router_api_path=binding_context.binding.router_api_path,
            bound_session_id=bound_session_id,
            bound_instance_id=bound_instance_id,
            sticky_header_name=options.proxy.sticky_header_name,
            max_steps=options.proxy.max_steps,
            default_temperature=options.proxy.default_temperature,
        )
        config = uvicorn.Config(
            self._proxy.app,
            host="127.0.0.1",
            port=self._proxy_port,
            log_level="warning",
        )
        self._proxy_server = _BackgroundUvicornServer(config)
        self._proxy_task = asyncio.create_task(self._proxy_server.serve())
        await self._wait_for_proxy()
        run_dir = Path(binding_context.binding.runtime_dir) / "run"
        (run_dir / "proxy.port").write_text(str(self._proxy_port), encoding="utf-8")
        LOGGER.info("rollout proxy started successfully on port %d", self._proxy_port)

    async def _wait_for_proxy(self, timeout: float = 5.0) -> None:
        assert self._proxy_port is not None
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://127.0.0.1:{self._proxy_port}/__proxy_health",
                        timeout=0.5,
                    )
                    if response.status_code == 200:
                        return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
        raise BackendProcessError("Timed out waiting for rollout proxy startup.")

    def _resolve_upstream_origin(self, router_base_url: str) -> str:
        raw = router_base_url
        if "://" not in raw:
            raw = f"http://{raw}"
        parsed = urlparse(raw)
        if not parsed.netloc:
            raise BackendProtocolError(f"Invalid router_base_url: {router_base_url}")
        return f"{parsed.scheme or 'http'}://{parsed.netloc}"

    async def _ensure_backend_session(self, session_context: SessionContext, timeout: float) -> str:
        if session_context.backend_session_id is not None:
            return session_context.backend_session_id
        data = await self._request("POST", "/session", json_body={}, timeout=timeout)
        backend_session_id = self._extract_session_id(data)
        session_context.backend_session_id = backend_session_id
        return backend_session_id

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None,
        timeout: float,
    ) -> Any:
        if self._client is None:
            raise BackendProcessError("opencode client has not been initialized.")
        try:
            response = await self._client.request(method, path, json=json_body, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            context_overflow_error = _proxy_context_overflow_error_from_http_status(exc)
            if context_overflow_error is not None:
                raise context_overflow_error from exc
            typed_error = _opencode_error_from_http_status(exc, self._options)
            if typed_error is not None:
                raise typed_error from exc
            raise BackendTransportError(
                f"opencode {method} {path} returned "
                f"{exc.response.status_code}: {exc.response.text[:1000]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendTransportError(str(exc)) from exc
        if not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise BackendProtocolError(f"Invalid JSON from backend: {exc}") from exc

    async def _post_message_with_connect_retry(
        self,
        path: str,
        *,
        json_body: dict[str, Any] | None,
        deadline: float,
        operation: str,
        max_retries: int = 5,
        retry_delay: float = 0.2,
    ) -> Any:
        last_exc: BackendTransportError | None = None
        for attempt in range(max_retries + 1):
            if self._process is not None and self._process.returncode is not None:
                raise BackendProcessError(
                    f"opencode exited with code {self._process.returncode}"
                )
            try:
                return await self._request(
                    "POST",
                    path,
                    json_body=json_body,
                    timeout=self._remaining_timeout(deadline, operation=operation),
                )
            except BackendTransportError as exc:
                if not isinstance(exc.__cause__, httpx.ConnectError):
                    raise
                last_exc = exc
                if attempt >= max_retries:
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= retry_delay:
                    break
                LOGGER.warning(
                    "Retrying opencode POST %s on connect error (%d/%d): %s",
                    path,
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                await asyncio.sleep(retry_delay)
        assert last_exc is not None
        raise last_exc

    async def _request_get_with_retry(
        self,
        path: str,
        *,
        deadline: float,
        turn_id: str,
        prev_msg_count: int,
        max_retries: int = 1,
        retry_delay: float = 0.2,
    ) -> tuple[list[dict[str, Any]], list[Message], list[TraceEvent], TurnUsage]:
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                data = await self._request(
                    "GET",
                    path,
                    json_body=None,
                    timeout=self._remaining_timeout(deadline, operation="fetch opencode messages"),
                )
                _raise_if_opencode_structured_error(data, self._options)
                all_messages = self._extract_messages(data)
                if len(all_messages) < prev_msg_count:
                    raise BackendProtocolError(
                        "opencode message count went backwards: "
                        f"prev={prev_msg_count}, now={len(all_messages)}"
                    )

                new_oc_messages = all_messages[prev_msg_count:]
                outputs, trace_events, usage = convert_opencode_messages(turn_id, new_oc_messages)
                if outputs:
                    return all_messages, outputs, trace_events, usage

                last_exc = _IncompleteHistoryError(
                    "opencode history fetch returned no material outputs for the current turn: "
                    f"prev={prev_msg_count}, now={len(all_messages)}"
                )
            except BackendTransportError as exc:
                last_exc = exc
            except BackendProtocolError:
                raise

            if attempt >= max_retries:
                break

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            delay = min(retry_delay, remaining)
            LOGGER.warning(
                "Retrying opencode history fetch for %s (%d/%d): %s",
                path,
                attempt + 1,
                max_retries + 1,
                last_exc,
            )
            if delay > 0:
                await asyncio.sleep(delay)

        if last_exc is None:
            raise BackendTransportError("opencode history fetch failed without an explicit error.")
        if isinstance(last_exc, _IncompleteHistoryError):
            raise BackendTransportError(str(last_exc))
        raise last_exc

    def _remaining_timeout(self, deadline: float, *, operation: str) -> float:
        pause_credit = float(getattr(self._proxy, "total_paused_seconds", 0.0)) if self._proxy is not None else 0.0
        remaining = deadline + pause_credit - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError(f"Timed out before {operation}.")
        return remaining

    async def _wait_until_healthy(self) -> None:
        assert self._binding_context is not None
        deadline = asyncio.get_running_loop().time() + self._binding_context.effective_config.health_check_timeout
        interval = self._binding_context.effective_config.health_check_interval
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise BackendProcessError(f"opencode exited early with code {self._process.returncode}")
            if await self.health():
                return
            await asyncio.sleep(interval)
        raise BackendProcessError("Timed out waiting for opencode health check.")

    def _extract_session_id(self, data: Any) -> str:
        if isinstance(data, str) and data:
            return data
        if isinstance(data, dict):
            for key in ("id", "session_id", "sessionID"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    return value
        raise BackendProtocolError("Could not extract backend_session_id from /session response.")

    def _extract_messages(self, data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("messages"), list):
                return [item for item in data["messages"] if isinstance(item, dict)]
            if isinstance(data.get("message"), dict):
                return [data["message"]]
            if "info" in data and "parts" in data:
                return [data]
        raise BackendProtocolError("Could not extract opencode messages from response.")

    def _find_free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    async def _raise_if_proxy_context_overflow(self) -> None:
        if self._proxy is None:
            return
        if not hasattr(self._proxy, "consume_context_overflow_error"):
            return
        payload = await self._proxy.consume_context_overflow_error()
        typed_error = backend_context_overflow_from_proxy_payload(payload)
        if typed_error is not None:
            raise typed_error

    async def _raise_if_proxy_rollout_invalidated(self) -> None:
        if self._proxy is None:
            return
        if not hasattr(self._proxy, "consume_rollout_invalidated_error"):
            return
        payload = await self._proxy.consume_rollout_invalidated_error()
        if payload is None:
            return
        error = payload.get("error") or "rollout_invalidated"
        message = payload.get("message") or "Dressage rollout was invalidated."
        raise BackendTransportError(f"Dressage proxy {error}: {message}")


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _structured_error_code(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict):
        for key in ("code", "type", "error"):
            raw = error.get(key)
            if raw is not None:
                return str(raw).strip().lower()
    for key in ("error_code", "code", "finish_reason"):
        raw = value.get(key)
        if raw is not None:
            return str(raw).strip().lower()
    return None


def _structured_error_message(value: dict[str, Any]) -> str | None:
    error = value.get("error")
    if isinstance(error, dict) and error.get("message") is not None:
        return str(error.get("message"))
    if value.get("message") is not None:
        return str(value.get("message"))
    return None


def _opencode_max_steps(options: OpencodeBackendOptions | None) -> int | None:
    if options is None:
        return None
    return options.proxy.max_steps


def _max_steps_error_details(
    value: dict[str, Any],
    options: OpencodeBackendOptions | None,
) -> tuple[int, int]:
    error = value.get("error")
    details = error.get("details") if isinstance(error, dict) else None
    if not isinstance(details, dict):
        details = value.get("details")
    if not isinstance(details, dict):
        details = {}

    configured_max_steps = _opencode_max_steps(options)
    max_steps = _maybe_int(details.get("max_steps")) or configured_max_steps or 0
    attempted_step = _maybe_int(details.get("attempted_step")) or max_steps
    return max_steps, attempted_step


def _max_steps_error_from_text(
    text: str,
    options: OpencodeBackendOptions | None,
) -> BackendMaxStepsExceededError | None:
    normalized = text.lower()
    if (
        _MAX_STEPS_EXCEEDED_ERROR_CODE not in normalized
        and _MAX_STEPS_EXCEEDED_MESSAGE_FRAGMENT not in normalized
    ):
        return None
    configured_max_steps = _opencode_max_steps(options) or 0
    return BackendMaxStepsExceededError(
        "Turn exceeded max_steps.",
        max_steps=configured_max_steps,
        attempted_step=configured_max_steps,
        backend_message=text[:1000],
        raw_error_code=_MAX_STEPS_EXCEEDED_ERROR_CODE,
    )


def _raise_if_opencode_structured_error(
    value: Any,
    options: OpencodeBackendOptions | None,
) -> None:
    if not isinstance(value, dict):
        return
    raw_error_code = _structured_error_code(value)
    if (
        raw_error_code == _MAX_STEPS_EXCEEDED_ERROR_CODE
        or (
            raw_error_code == "rate_limit_error"
            and (
                _MAX_STEPS_EXCEEDED_MESSAGE_FRAGMENT
                in (_structured_error_message(value) or "").lower()
            )
        )
    ):
        max_steps, attempted_step = _max_steps_error_details(value, options)
        raise BackendMaxStepsExceededError(
            "Turn exceeded max_steps.",
            max_steps=max_steps,
            attempted_step=attempted_step,
            backend_message=_structured_error_message(value),
            raw_error_code=raw_error_code,
        )


def _opencode_error_from_http_status(
    exc: httpx.HTTPStatusError,
    options: OpencodeBackendOptions | None,
) -> BackendMaxStepsExceededError | None:
    try:
        payload = exc.response.json()
    except json.JSONDecodeError:
        return _max_steps_error_from_text(exc.response.text or str(exc), options)
    if not isinstance(payload, dict):
        return _max_steps_error_from_text(exc.response.text or str(exc), options)
    try:
        _raise_if_opencode_structured_error(payload, options)
    except BackendMaxStepsExceededError as typed_error:
        return typed_error
    text_fallback = _max_steps_error_from_text(exc.response.text or str(exc), options)
    if text_fallback is not None:
        return text_fallback
    return None


def _proxy_context_overflow_error_from_http_status(
    exc: httpx.HTTPStatusError,
):
    if exc.response.status_code != 413:
        return None
    try:
        payload = exc.response.json()
    except json.JSONDecodeError:
        return None
    return backend_context_overflow_from_proxy_payload(payload)


def convert_opencode_messages(
    turn_id: str,
    oc_messages: list[dict[str, Any]],
) -> tuple[list[Message], list[TraceEvent], TurnUsage]:
    outputs: list[Message] = []
    trace_events: list[TraceEvent] = []
    usage = TurnUsage()
    seq = 0

    for oc_message in oc_messages:
        info = oc_message.get("info") or {}
        if info.get("role") == "user":
            continue

        parts = oc_message.get("parts")
        if not isinstance(parts, list):
            raise BackendProtocolError("opencode message payload is missing parts.")

        reasoning_parts: list[str] = []
        assistant_text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[Message] = []

        for part in parts:
            if not isinstance(part, dict):
                continue
            seq += 1
            part_type = part.get("type", "")

            if part_type in {"reasoning", "step-start", "step-finish", "tool"}:
                trace_events.append(
                    TraceEvent(
                        turn_id=turn_id,
                        seq=seq,
                        source="opencode",
                        event_type=str(part_type),
                        payload=part,
                        created_at=utcnow(),
                    )
                )

            if part_type == "reasoning" and part.get("text"):
                reasoning_parts.append(str(part["text"]))
            elif part_type == "text" and part.get("text"):
                assistant_text_parts.append(str(part["text"]))
            elif part_type == "tool":
                call_id = (
                    part.get("callID")
                    or part.get("callId")
                    or part.get("id")
                    or f"call_{uuid4().hex[:8]}"
                )
                tool_name = str(part.get("tool") or part.get("name") or "tool")
                state = part.get("state") or {}
                tool_input = state.get("input", {})
                tool_output = state.get("output", "")
                if not isinstance(tool_output, str):
                    tool_output = json.dumps(tool_output, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        id=str(call_id),
                        function=FunctionCall(
                            name=tool_name,
                            arguments=json.dumps(tool_input, ensure_ascii=False),
                        ),
                    )
                )
                tool_results.append(
                    Message(
                        role="tool",
                        tool_call_id=str(call_id),
                        name=tool_name,
                        content=tool_output,
                    )
                )
                usage.tool_calls += 1
            elif part_type == "step-finish":
                tokens = part.get("tokens") or part.get("usage") or {}
                usage.total_tokens += int(tokens.get("total", 0) or 0)
                usage.input_tokens += int(tokens.get("input", 0) or 0)
                usage.output_tokens += int(tokens.get("output", 0) or 0)
                usage.reasoning_tokens += int(tokens.get("reasoning", 0) or 0)
                usage.steps += 1

        reasoning_content = "\n".join(part for part in reasoning_parts if part) or None
        assistant_text = "\n".join(part for part in assistant_text_parts if part)
        assistant_content = assistant_text or None

        if tool_calls:
            outputs.append(
                Message(
                    role="assistant",
                    content=assistant_content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                )
            )
            outputs.extend(tool_results)
        elif assistant_content or reasoning_content:
            outputs.append(
                Message(
                    role="assistant",
                    content=assistant_content,
                    reasoning_content=reasoning_content,
                )
            )

    return outputs, trace_events, usage
