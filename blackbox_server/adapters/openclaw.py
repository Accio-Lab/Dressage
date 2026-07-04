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


LOGGER = logging.getLogger(__name__)

_MAX_STEPS_EXCEEDED_ERROR_CODE = "max_steps_exceeded"
_MAX_STEPS_EXCEEDED_MESSAGE_FRAGMENT = "turn exceeded max_steps"
_DENY_QUESTION_PERMISSION = {
    "*": "allow",
    "question": "deny",
    "doom_loop": "deny",
}
_DENY_QUESTION_WORKSPACE_MARKER = "<!-- blackbox-server-deny-question -->"
_DENY_QUESTION_WORKSPACE_PROMPT = f"""

{_DENY_QUESTION_WORKSPACE_MARKER}
## Blackbox Rollout Interaction Policy

Do not ask the user questions during this rollout.
Do not request confirmation or wait for external input.
If required information is missing, choose reasonable defaults and continue.
"""


class _BackgroundUvicornServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        # Keep the parent process in charge of SIGINT/SIGTERM.
        yield


class OpenClawGatewayOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: int | None = Field(default=None, ge=1, le=65535)
    auth_token: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class OpenClawRequestOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)


class OpenClawCompactionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int | None = Field(default=None, gt=0)
    reserve_tokens: int | None = Field(default=None, gt=0)
    reserve_tokens_floor: int | None = Field(default=None, ge=0)
    keep_recent_tokens: int | None = Field(default=None, ge=0)
    notify_user: bool | None = None
    max_active_transcript_bytes: int | str | None = None
    post_compaction_sections: list[str] | None = None
    model: str | None = Field(default=None, min_length=1)

    def to_openclaw_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if self.timeout_seconds is not None:
            config["timeoutSeconds"] = self.timeout_seconds
        if self.reserve_tokens is not None:
            config["reserveTokens"] = self.reserve_tokens
        if self.reserve_tokens_floor is not None:
            config["reserveTokensFloor"] = self.reserve_tokens_floor
        if self.keep_recent_tokens is not None:
            config["keepRecentTokens"] = self.keep_recent_tokens
        if self.notify_user is not None:
            config["notifyUser"] = self.notify_user
        if self.max_active_transcript_bytes is not None:
            config["maxActiveTranscriptBytes"] = self.max_active_transcript_bytes
        if self.post_compaction_sections is not None:
            config["postCompactionSections"] = self.post_compaction_sections
        if self.model is not None:
            config["model"] = self.model
        return config


class OpenClawBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = "default"

    provider_id: str = "sglang"
    model_id: str
    model_name: str | None = None
    context_window: int = Field(default=32768, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    api_key: str = "sglang-local"

    gateway: OpenClawGatewayOptions = Field(default_factory=OpenClawGatewayOptions)
    request: OpenClawRequestOptions = Field(default_factory=OpenClawRequestOptions)
    compaction: OpenClawCompactionOptions | None = None
    proxy: ProxyOptions = Field(default_factory=ProxyOptions)


class OpenClawAdapter(BackendAdapter):
    def __init__(self) -> None:
        self._binding_context: BindingContext | None = None
        self._options: OpenClawBackendOptions | None = None

        self._client: httpx.AsyncClient | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._process_group_id: int | None = None

        self._gateway_port: int | None = None
        self._gateway_token: str | None = None

        self._stdout_handle = None
        self._stderr_handle = None

        self._proxy: RolloutLLMProxy | None = None
        self._proxy_port: int | None = None
        self._proxy_server: uvicorn.Server | None = None
        self._proxy_task: asyncio.Task[Any] | None = None

        self._active_chat_task: asyncio.Task[httpx.Response] | None = None

    async def initialize(self, binding_context: BindingContext) -> None:
        self._binding_context = binding_context
        self._options = self._parse_options(binding_context.binding.backend_options)

        runtime_dir = Path(binding_context.binding.runtime_dir)
        self._prepare_runtime_dirs(runtime_dir)
        self._prepare_workspace(binding_context)

        await self._start_proxy(binding_context, self._options)

        self._gateway_port = self._options.gateway.port or self._find_free_port()
        self._gateway_token = self._options.gateway.auth_token or f"bbs-openclaw-{uuid4().hex}"

        try:
            await self._start_gateway_with_permission_fallback(binding_context, runtime_dir)
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
            raise BackendProcessError("openclaw gateway is not healthy.")

        if len(new_messages) != 1:
            raise BackendProtocolError("openclaw adapter expects exactly one input message.")
        message = new_messages[0]
        if message.role != "user":
            raise BackendProtocolError("openclaw adapter only accepts a user message.")
        if message.content is None or not message.content.strip():
            raise BackendProtocolError("openclaw adapter requires non-empty user content.")

        backend_session_id = self._ensure_backend_session_id(session_context)

        if self._proxy is not None:
            await self._proxy.open_turn(turn_context.turn_id, backend_session_id=backend_session_id)

        success = False
        try:
            chat_task = asyncio.create_task(
                self._post_chat_completions(
                    session_key=backend_session_id,
                    turn_context=turn_context,
                    user_text=message.content,
                    deadline=deadline,
                )
            )
            raw = await self._await_backend_task_or_proxy_max_steps(
                chat_task,
                session_context=session_context,
                proxy=self._proxy,
            )
            if self._proxy is not None:
                await self._proxy.drain_turn(
                    timeout=self._remaining_timeout(deadline, operation="wait for rollout proxy drain")
                )
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()

            _raise_if_openclaw_max_steps_exceeded(raw, self._options)

            outputs, trace_events, usage = convert_openclaw_chat_completion(
                turn_context.turn_id,
                raw,
            )
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
        _ = session_context
        task = self._active_chat_task
        if task is None:
            return False

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=2.0)

        if self._proxy is not None:
            with contextlib.suppress(Exception):
                await self._proxy.clear_turn()

        return True

    async def health(self) -> bool:
        if self._process is None or self._client is None:
            return False
        if self._process.returncode is not None:
            return False
        if self._gateway_token is None:
            return False
        try:
            response = await self._client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {self._gateway_token}"},
                timeout=2.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

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
        task = self._active_chat_task
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=2.0)
            self._active_chat_task = None

        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

        await self._stop_gateway_process()

        if self._proxy_server is not None:
            self._proxy_server.should_exit = True
        if self._proxy_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proxy_task, timeout=5.0)
            self._proxy_task = None
        self._proxy_server = None
        self._proxy = None
        self._proxy_port = None

        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
                setattr(self, handle_name, None)

    def _parse_options(self, backend_options: dict[str, Any]) -> OpenClawBackendOptions:
        try:
            return OpenClawBackendOptions.model_validate(backend_options)
        except Exception as exc:
            raise BackendProtocolError(f"Invalid openclaw backend_options: {exc}") from exc

    def _prepare_runtime_dirs(self, runtime_dir: Path) -> None:
        for path in (
            runtime_dir / "home" / ".openclaw",
            runtime_dir / "home" / ".openclaw" / "workspace",
            runtime_dir / "home" / ".config",
            runtime_dir / "home" / ".cache",
            runtime_dir / "home" / ".local" / "share",
            runtime_dir / "logs",
            runtime_dir / "run",
            runtime_dir / "tmp",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _prepare_workspace(self, binding_context: BindingContext) -> None:
        runtime_dir = Path(binding_context.binding.runtime_dir)
        workspace_dir = runtime_dir / "home" / ".openclaw" / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        agents_file = workspace_dir / "AGENTS.md"
        if binding_context.binding.system_prompt is not None:
            source = Path(binding_context.binding.system_prompt.source_file)
            target = Path(binding_context.binding.system_prompt.runtime_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif not agents_file.exists():
            agents_file.write_text("", encoding="utf-8")

        self._append_deny_question_workspace_prompt(agents_file)

        for name in ("SOUL.md", "TOOLS.md", "USER.md", "IDENTITY.md"):
            path = workspace_dir / name
            if not path.exists():
                path.write_text("", encoding="utf-8")

    def _append_deny_question_workspace_prompt(self, agents_file: Path) -> None:
        try:
            current = agents_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = ""
        if _DENY_QUESTION_WORKSPACE_MARKER in current:
            return
        agents_file.write_text(
            current.rstrip() + _DENY_QUESTION_WORKSPACE_PROMPT,
            encoding="utf-8",
        )

    def _build_openclaw_config(
        self,
        binding_context: BindingContext,
        options: OpenClawBackendOptions,
        *,
        permission_mode: str = "dual",
    ) -> dict[str, Any]:
        if self._proxy_port is None:
            raise BackendProcessError("rollout proxy has not been initialized.")
        if self._gateway_token is None:
            raise BackendProcessError("openclaw gateway token has not been initialized.")
        if permission_mode not in {"dual", "agents", "top", "none"}:
            raise BackendProtocolError(f"Invalid openclaw permission mode: {permission_mode}")

        runtime_dir = Path(binding_context.binding.runtime_dir)
        workspace_dir = runtime_dir / "home" / ".openclaw" / "workspace"
        model_ref = f"{options.provider_id}/{options.model_id}"
        proxy_base_url = f"http://127.0.0.1:{self._proxy_port}{binding_context.binding.router_api_path}"
        compaction_config = self._build_compaction_config(options)

        config: dict[str, Any] = {
            "gateway": {
                "mode": "local",
                "auth": {
                    "mode": "token",
                    "token": self._gateway_token,
                },
                "http": {
                    "endpoints": {
                        "chatCompletions": {"enabled": True},
                    },
                },
            },
            "agents": {
                "defaults": {
                    "workspace": str(workspace_dir),
                    "model": {
                        "primary": model_ref,
                    },
                    "compaction": compaction_config,
                    "models": {
                        model_ref: {
                            "alias": options.model_name or options.model_id,
                        },
                    },
                },
            },
            "models": {
                "mode": "merge",
                "providers": {
                    options.provider_id: {
                        "baseUrl": proxy_base_url,
                        "apiKey": options.api_key,
                        "api": "openai-completions",
                        "models": [
                            {
                                "id": options.model_id,
                                "name": options.model_name or options.model_id,
                                "reasoning": True,
                                "compat": {
                                    "thinkingFormat": "deepseek",
                                    "supportsReasoningEffort": False,
                                },
                                "input": ["text"],
                                "cost": {
                                    "input": 0,
                                    "output": 0,
                                    "cacheRead": 0,
                                    "cacheWrite": 0,
                                },
                                "maxTokens": options.max_tokens,
                            }
                        ],
                    }
                },
            },
        }
        if permission_mode in {"dual", "top"}:
            config["permission"] = dict(_DENY_QUESTION_PERMISSION)
        if permission_mode in {"dual", "agents"}:
            config["agents"]["defaults"]["permission"] = dict(_DENY_QUESTION_PERMISSION)
        return config

    def _build_compaction_config(self, options: OpenClawBackendOptions) -> dict[str, Any]:
        config: dict[str, Any] = {
            "mode": "safeguard",
            "midTurnPrecheck": {"enabled": True},
            "truncateAfterCompaction": True,
            "notifyUser": False,
        }
        if options.compaction is not None:
            config.update(options.compaction.to_openclaw_config())

        config["mode"] = "safeguard"
        config["midTurnPrecheck"] = {"enabled": True}
        config["truncateAfterCompaction"] = True
        return config

    async def _start_gateway_process(self, runtime_dir: Path) -> None:
        if self._options is None:
            raise BackendProcessError("openclaw options have not been initialized.")
        if self._gateway_port is None or self._gateway_token is None:
            raise BackendProcessError("openclaw gateway port/token have not been initialized.")

        logs_dir = runtime_dir / "logs"
        tmp_dir = runtime_dir / "tmp"
        cache_dir = runtime_dir / "home" / ".cache"
        data_dir = runtime_dir / "home" / ".local" / "share"
        self._stdout_handle = open(logs_dir / "openclaw.stdout.log", "ab")
        self._stderr_handle = open(logs_dir / "openclaw.stderr.log", "ab")

        env = os.environ.copy()
        env.update(
            {
                "HOME": str(runtime_dir / "home"),
                "XDG_CONFIG_HOME": str(runtime_dir / "home" / ".config"),
                "XDG_CACHE_HOME": str(cache_dir),
                "XDG_DATA_HOME": str(data_dir),
                "TMPDIR": str(tmp_dir),
                "SGLANG_API_KEY": self._options.api_key,
                "OPENCLAW_GATEWAY_TOKEN": self._gateway_token,
            }
        )

        binary = os.getenv("OPENCLAW_BIN", "openclaw")
        try:
            self._process = await asyncio.create_subprocess_exec(
                binary,
                "gateway",
                "run",
                "--port",
                str(self._gateway_port),
                "--bind",
                "loopback",
                "--auth",
                "token",
                "--token",
                self._gateway_token,
                *self._options.gateway.extra_args,
                cwd=str(runtime_dir),
                env=env,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
                start_new_session=True,
            )
            self._process_group_id = self._process.pid
        except FileNotFoundError as exc:
            raise BackendProcessError(
                f"openclaw binary not found. Set OPENCLAW_BIN or install openclaw. ({binary})"
            ) from exc

    async def _start_gateway_with_permission_fallback(
        self,
        binding_context: BindingContext,
        runtime_dir: Path,
    ) -> None:
        last_error: BackendProcessError | None = None
        for permission_mode in ("dual", "agents", "top", "none"):
            self._write_openclaw_config(
                binding_context,
                self._options,
                permission_mode=permission_mode,
            )
            await self._start_gateway_process(runtime_dir)
            run_dir = runtime_dir / "run"
            assert self._process is not None
            (run_dir / "openclaw.pid").write_text(str(self._process.pid), encoding="utf-8")
            (run_dir / "openclaw.port").write_text(str(self._gateway_port), encoding="utf-8")

            self._client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{self._gateway_port}",
                timeout=None,
            )

            try:
                await self._wait_until_healthy()
                if permission_mode != "dual":
                    LOGGER.warning(
                        "openclaw gateway started after disabling unsupported "
                        "permission config mode=%s; deny-question workspace prompt remains active",
                        permission_mode,
                    )
                return
            except BackendProcessError as exc:
                await self._stop_gateway_process()
                if "exited early" not in str(exc):
                    raise
                last_error = exc
                LOGGER.warning(
                    "openclaw gateway failed with permission config mode=%s; "
                    "retrying with a more conservative config: %s",
                    permission_mode,
                    exc,
                )

        if last_error is not None:
            raise BackendProcessError(
                "openclaw exited early after permission config fallback; "
                f"last error: {last_error}"
            ) from last_error
        raise BackendProcessError("openclaw gateway did not start.")

    def _write_openclaw_config(
        self,
        binding_context: BindingContext,
        options: OpenClawBackendOptions | None,
        *,
        permission_mode: str,
    ) -> None:
        if options is None:
            raise BackendProcessError("openclaw options have not been initialized.")
        config_path = Path(binding_context.binding.runtime_dir) / "home" / ".openclaw" / "openclaw.json"
        config_path.write_text(
            json.dumps(
                self._build_openclaw_config(
                    binding_context,
                    options,
                    permission_mode=permission_mode,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    async def _stop_gateway_process(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

        if self._process is not None:
            if self._process.returncode is None:
                if self._process_group_id is not None:
                    try:
                        os.killpg(self._process_group_id, signal.SIGTERM)
                        await asyncio.wait_for(self._process.wait(), timeout=3.0)
                    except (ProcessLookupError, asyncio.TimeoutError):
                        pass
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

    async def _start_proxy(self, binding_context: BindingContext, options: OpenClawBackendOptions) -> None:
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

    def _ensure_backend_session_id(self, session_context: SessionContext) -> str:
        if session_context.backend_session_id:
            return session_context.backend_session_id
        if self._binding_context is None:
            raise BackendProcessError("openclaw binding context has not been initialized.")
        binding = self._binding_context.binding
        backend_session_id = f"bbs:{binding.bound_instance_id}:{binding.bound_session_id}"
        session_context.backend_session_id = backend_session_id
        return backend_session_id

    def _agent_model_target(self, agent_id: str) -> str:
        if not agent_id or agent_id == "default":
            return "openclaw/default"
        return f"openclaw/{agent_id}"

    async def _post_chat_completions(
        self,
        *,
        session_key: str,
        turn_context: TurnContext,
        user_text: str,
        deadline: float,
    ) -> dict[str, Any]:
        if self._client is None:
            raise BackendProcessError("openclaw client has not been initialized.")
        if self._options is None:
            raise BackendProcessError("openclaw options have not been initialized.")
        if self._gateway_token is None:
            raise BackendProcessError("openclaw gateway token has not been initialized.")

        payload: dict[str, Any] = {
            "model": self._agent_model_target(self._options.agent_id),
            "user": session_key,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": user_text,
                }
            ],
        }
        if self._options.request.max_tokens is not None:
            payload["max_tokens"] = self._options.request.max_tokens
        if self._options.request.temperature is not None:
            payload["temperature"] = self._options.request.temperature
        if self._options.request.top_p is not None:
            payload["top_p"] = self._options.request.top_p

        headers = {
            "Authorization": f"Bearer {self._gateway_token}",
            "Content-Type": "application/json",
            "x-openclaw-model": f"{self._options.provider_id}/{self._options.model_id}",
            "x-openclaw-session-key": session_key,
            "x-bbs-turn-id": turn_context.turn_id,
        }

        task = asyncio.create_task(
            self._client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers,
                # This request can remain transparently suspended while the
                # Dressage proxy preempts SGLang and waits for a trainer weight
                # update.  The blackbox server's outer timeout is pause-aware,
                # so do not use httpx's fixed wall-clock timeout here.
                timeout=None,
            )
        )
        self._active_chat_task = task

        try:
            response = await task
            response.raise_for_status()
        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as exc:
            context_overflow_error = _proxy_context_overflow_error_from_http_status(exc)
            if context_overflow_error is not None:
                raise context_overflow_error from exc
            max_steps_error = _max_steps_error_from_http_status(exc, self._options)
            if max_steps_error is not None:
                raise max_steps_error from exc
            raise BackendTransportError(
                "openclaw /v1/chat/completions returned "
                f"{exc.response.status_code}: {exc.response.text[:1000]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendTransportError(str(exc)) from exc
        finally:
            if self._active_chat_task is task:
                self._active_chat_task = None

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise BackendProtocolError(f"Invalid JSON from openclaw: {exc}") from exc
        if not isinstance(data, dict):
            raise BackendProtocolError("OpenClaw chat completion response is not an object.")
        return data

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
                stderr_tail = self._openclaw_stderr_tail()
                message = f"openclaw exited early with code {self._process.returncode}"
                if stderr_tail:
                    message = f"{message}; stderr tail: {stderr_tail}"
                raise BackendProcessError(message)
            if await self.health():
                return
            await asyncio.sleep(interval)
        raise BackendProcessError("Timed out waiting for openclaw health check.")

    def _openclaw_stderr_tail(self, *, max_chars: int = 2000) -> str | None:
        if self._binding_context is None:
            return None
        logs_dir = Path(self._binding_context.binding.runtime_dir) / "logs"
        stderr_path = logs_dir / "openclaw.stderr.log"
        if not stderr_path.exists():
            return None
        try:
            data = stderr_path.read_bytes()[-max_chars:]
        except OSError:
            return None
        text = data.decode("utf-8", errors="replace").strip()
        return text or None

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


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else {}


def _first_message_content(response: dict[str, Any]) -> str | None:
    message = _first_choice(response).get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _structured_error_message(value: dict[str, Any]) -> str | None:
    error = value.get("error")
    if isinstance(error, dict) and error.get("message") is not None:
        return str(error.get("message"))
    if value.get("message") is not None:
        return str(value.get("message"))
    return _first_message_content(value)


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
    choice = _first_choice(value)
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None:
        return str(finish_reason).strip().lower()
    return None


def _proxy_max_steps(options: OpenClawBackendOptions | None) -> int | None:
    if options is None:
        return None
    return options.proxy.max_steps


def _max_steps_error_details(
    value: dict[str, Any],
    options: OpenClawBackendOptions | None,
) -> tuple[int, int]:
    error = value.get("error")
    details = error.get("details") if isinstance(error, dict) else None
    if not isinstance(details, dict):
        details = value.get("details")
    if not isinstance(details, dict):
        details = {}

    configured_max_steps = _proxy_max_steps(options)
    max_steps = _maybe_int(details.get("max_steps")) or configured_max_steps or 0
    attempted_step = _maybe_int(details.get("attempted_step")) or max_steps
    return max_steps, attempted_step


def _max_steps_error_from_text(
    text: str,
    options: OpenClawBackendOptions | None,
) -> BackendMaxStepsExceededError | None:
    normalized = text.lower()
    if (
        _MAX_STEPS_EXCEEDED_ERROR_CODE not in normalized
        and _MAX_STEPS_EXCEEDED_MESSAGE_FRAGMENT not in normalized
    ):
        return None
    configured_max_steps = _proxy_max_steps(options) or 0
    return BackendMaxStepsExceededError(
        "Turn exceeded max_steps.",
        max_steps=configured_max_steps,
        attempted_step=configured_max_steps,
        backend_message=text[:1000],
        raw_error_code=_MAX_STEPS_EXCEEDED_ERROR_CODE,
    )


def _raise_if_openclaw_max_steps_exceeded(
    response: dict[str, Any],
    options: OpenClawBackendOptions | None,
) -> None:
    raw_error_code = _structured_error_code(response)
    if (
        raw_error_code != _MAX_STEPS_EXCEEDED_ERROR_CODE
        and not (
            raw_error_code == "rate_limit_error"
            and (
                _MAX_STEPS_EXCEEDED_MESSAGE_FRAGMENT
                in (_structured_error_message(response) or "").lower()
            )
        )
    ):
        return
    max_steps, attempted_step = _max_steps_error_details(response, options)
    raise BackendMaxStepsExceededError(
        "Turn exceeded max_steps.",
        max_steps=max_steps,
        attempted_step=attempted_step,
        backend_message=_structured_error_message(response),
        raw_error_code=raw_error_code,
    )


def _proxy_context_overflow_error_from_http_status(exc: httpx.HTTPStatusError):
    if exc.response.status_code != 413:
        return None
    try:
        payload = exc.response.json()
    except json.JSONDecodeError:
        return None
    return backend_context_overflow_from_proxy_payload(payload)


def _max_steps_error_from_http_status(
    exc: httpx.HTTPStatusError,
    options: OpenClawBackendOptions | None,
) -> BackendMaxStepsExceededError | None:
    try:
        payload = exc.response.json()
    except json.JSONDecodeError:
        return _max_steps_error_from_text(exc.response.text or str(exc), options)
    if not isinstance(payload, dict):
        return _max_steps_error_from_text(exc.response.text or str(exc), options)
    raw_error_code = _structured_error_code(payload)
    if raw_error_code == _MAX_STEPS_EXCEEDED_ERROR_CODE or (
        raw_error_code == "rate_limit_error"
        and (
            _MAX_STEPS_EXCEEDED_MESSAGE_FRAGMENT
            in (_structured_error_message(payload) or "").lower()
        )
    ):
        max_steps, attempted_step = _max_steps_error_details(payload, options)
        return BackendMaxStepsExceededError(
            "Turn exceeded max_steps.",
            max_steps=max_steps,
            attempted_step=attempted_step,
            backend_message=_structured_error_message(payload),
            raw_error_code=raw_error_code,
        )
    return _max_steps_error_from_text(exc.response.text or str(exc), options)

def convert_openclaw_chat_completion(
    turn_id: str,
    response: dict[str, Any],
) -> tuple[list[Message], list[TraceEvent], TurnUsage]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BackendProtocolError("OpenClaw chat completion response has no choices.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise BackendProtocolError("OpenClaw chat completion choice is not an object.")

    raw_message = choice.get("message")
    if not isinstance(raw_message, dict):
        raise BackendProtocolError("OpenClaw chat completion choice has no message.")

    content = raw_message.get("content")
    if content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    reasoning_content = raw_message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        reasoning_content = json.dumps(reasoning_content, ensure_ascii=False)

    tool_calls: list[ToolCall] = []
    raw_tool_calls = raw_message.get("tool_calls") or []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            arguments = function.get("arguments") or "{}"
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)

            tool_calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"call_{uuid4().hex[:8]}"),
                    function=FunctionCall(
                        name=str(function.get("name") or "function"),
                        arguments=arguments,
                    ),
                )
            )

    output = Message(
        role="assistant",
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or None,
    )

    usage_raw = response.get("usage") or {}
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    completion_details = usage_raw.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {}

    usage = TurnUsage(
        total_tokens=int(usage_raw.get("total_tokens", 0) or 0),
        input_tokens=int(usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", 0)) or 0),
        output_tokens=int(usage_raw.get("completion_tokens", usage_raw.get("output_tokens", 0)) or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0),
        steps=1,
        tool_calls=len(tool_calls),
    )

    trace_events = [
        TraceEvent(
            turn_id=turn_id,
            seq=1,
            source="openclaw",
            event_type="chat_completion",
            payload=response,
            created_at=utcnow(),
        )
    ]

    return [output], trace_events, usage
