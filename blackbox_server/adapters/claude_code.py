from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
import uvicorn
from pydantic import BaseModel, ConfigDict, Field

from blackbox_server.adapters.base import (
    BackendAdapter,
    BackendCapabilities,
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
_CLAUDE_CODE_STDOUT_STREAM_LIMIT = 100 * 1024 * 1024


class _BackgroundUvicornServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        # Keep the parent process in charge of SIGINT/SIGTERM.
        yield


class ClaudeCodeModelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="proxy-model", min_length=1)
    name: str = Field(default="Dressage Proxy", min_length=1)
    supported_capabilities: list[str] = Field(
        default_factory=lambda: [
            "thinking",
            "adaptive_thinking",
            "interleaved_thinking",
        ]
    )


class ClaudeCodeGatewayOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth_token: str = "blackbox-local"


class ClaudeCodeThinkingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    interleaved: bool = True
    budget_tokens: int | None = Field(default=None, gt=0)


class ClaudeCodeCompactionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto: bool = True
    auto_compact_pct_override: int | None = Field(default=None, ge=1, le=100)


class ClaudeCodeCompatOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disable_prompt_caching: bool = True
    disable_nonessential_traffic: bool = True


class ClaudeCodeSubagentsOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


_CLAUDE_CODE_INHERITED_GATEWAY_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_TMPDIR",
)


class ClaudeCodeBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str | None = None
    model: ClaudeCodeModelOptions = Field(default_factory=ClaudeCodeModelOptions)
    max_turns: int = Field(default=20, gt=0)
    permission_mode: Literal[
        "acceptEdits",
        "auto",
        "bypassPermissions",
        "default",
        "dontAsk",
        "plan",
    ] = "default"
    setting_sources: str = "user"
    system_prompt_mode: Literal["append", "replace", "claude_md", "none"] = "append"
    gateway: ClaudeCodeGatewayOptions = Field(default_factory=ClaudeCodeGatewayOptions)
    thinking: ClaudeCodeThinkingOptions = Field(
        default_factory=ClaudeCodeThinkingOptions
    )
    compaction: ClaudeCodeCompactionOptions = Field(
        default_factory=ClaudeCodeCompactionOptions
    )
    compat: ClaudeCodeCompatOptions = Field(default_factory=ClaudeCodeCompatOptions)
    subagents: ClaudeCodeSubagentsOptions = Field(
        default_factory=ClaudeCodeSubagentsOptions
    )
    proxy: ProxyOptions = Field(default_factory=ProxyOptions)

    def resolved_executable(self) -> str:
        return self.executable or os.getenv("CLAUDE_CODE_BIN") or "claude"


@dataclass(frozen=True)
class ClaudeCodeSessionState:
    backend_session_id: str
    resume: bool


class ClaudeCodeConfigCompiler:
    def __init__(
        self,
        *,
        options: ClaudeCodeBackendOptions,
        settings_path: Path,
        system_prompt_path: Path | None = None,
    ) -> None:
        self.options = options
        self.settings_path = settings_path
        self.system_prompt_path = system_prompt_path

    def build_settings(self) -> dict[str, Any]:
        deny: list[str] = []
        if not self.options.subagents.enabled:
            deny.append("Agent")

        settings: dict[str, Any] = {
            "autoCompactEnabled": self.options.compaction.auto,
            "permissions": {
                "allow": [
                    "Bash(*)",
                    "Read(*)",
                    "Edit(*)",
                    "Write(*)",
                    "WebSearch",
                    "WebFetch",
                    "Glob",
                    "Grep",
                    "LS",
                    "NotebookRead",
                    "NotebookEdit",
                    "TodoWrite",
                    "Task"
                ],
                "deny": deny,
            },
            "env": {},
        }
        env = settings["env"]
        if self.options.compat.disable_prompt_caching:
            env["DISABLE_PROMPT_CACHING"] = "1"
        if self.options.compat.disable_nonessential_traffic:
            env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        if not self.options.thinking.enabled:
            env["CLAUDE_CODE_DISABLE_THINKING"] = "1"
            env["MAX_THINKING_TOKENS"] = "0"
        elif self.options.thinking.budget_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(self.options.thinking.budget_tokens)
        if self.options.thinking.interleaved is False:
            env["DISABLE_INTERLEAVED_THINKING"] = "1"
        if self.options.compaction.auto_compact_pct_override is not None:
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(
                self.options.compaction.auto_compact_pct_override
            )
        return settings

    def build_env(self, proxy_port: int, runtime_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        for key in _CLAUDE_CODE_INHERITED_GATEWAY_ENV_KEYS:
            env.pop(key, None)
        env.update(
            {
                "HOME": str(runtime_dir / "home"),
                "CLAUDE_CONFIG_DIR": str(runtime_dir / "home" / ".claude"),
                "CLAUDE_CODE_TMPDIR": str(runtime_dir / "tmp"),
                "TMPDIR": str(runtime_dir / "tmp"),
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{proxy_port}",
                "ANTHROPIC_AUTH_TOKEN": self.options.gateway.auth_token,
                "ANTHROPIC_CUSTOM_MODEL_OPTION": self.options.model.id,
                "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": self.options.model.name,
                "ANTHROPIC_CUSTOM_MODEL_OPTION_SUPPORTED_CAPABILITIES": ",".join(
                    self.options.model.supported_capabilities
                ),
            }
        )
        return {str(key): str(value) for key, value in env.items()}

    def build_cli_args(
        self,
        user_text: str,
        session_state: ClaudeCodeSessionState,
    ) -> list[str]:
        args = [
            self.options.resolved_executable(),
            "-p",
            user_text,
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.options.model.id,
            "--max-turns",
            str(self.options.max_turns),
            "--permission-mode",
            self.options.permission_mode,
            "--settings",
            str(self.settings_path),
            "--setting-sources",
            self.options.setting_sources,
        ]
        if session_state.resume:
            args.extend(["--resume", session_state.backend_session_id])
        else:
            args.extend(["--session-id", session_state.backend_session_id])

        if self.system_prompt_path is not None and self.options.system_prompt_mode in {
            "append",
            "replace",
        }:
            try:
                prompt_text = self.system_prompt_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise BackendProtocolError(
                    f"Could not read Claude Code system prompt: {exc}"
                ) from exc
            if self.options.system_prompt_mode == "append":
                args.extend(["--append-system-prompt", prompt_text])
            else:
                args.extend(["--system-prompt", prompt_text])
        return args


class ClaudeCodeAdapter(BackendAdapter):
    def __init__(self) -> None:
        self._binding_context: BindingContext | None = None
        self._options: ClaudeCodeBackendOptions | None = None
        self._compiler: ClaudeCodeConfigCompiler | None = None
        self._proxy: RolloutLLMProxy | None = None
        self._proxy_port: int | None = None
        self._proxy_server: uvicorn.Server | None = None
        self._proxy_task: asyncio.Task[Any] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._process_group_id: int | None = None
        self._stdout_handle = None
        self._stderr_handle = None

    async def initialize(self, binding_context: BindingContext) -> None:
        self._binding_context = binding_context
        options = self._parse_options(binding_context.binding.backend_options)
        _validate_claude_code_runtime_options(options)
        self._options = options
        runtime_dir = Path(binding_context.binding.runtime_dir)
        self._prepare_runtime_dirs(runtime_dir)
        system_prompt_path = self._prepare_system_prompt(binding_context, options)
        await self._start_proxy(binding_context, options)
        assert self._proxy_port is not None
        settings_path = runtime_dir / "home" / ".claude" / "settings.json"
        self._compiler = ClaudeCodeConfigCompiler(
            options=options,
            settings_path=settings_path,
            system_prompt_path=system_prompt_path,
        )
        settings_path.write_text(
            json.dumps(self._compiler.build_settings(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_dir = runtime_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "claude_code.settings").write_text(
            str(settings_path), encoding="utf-8"
        )
        executable = options.resolved_executable()
        if self._resolve_executable(executable) is None:
            await self.shutdown()
            raise BackendProcessError(
                f"claude code binary not found. Set CLAUDE_CODE_BIN or install Claude Code. ({executable})"
            )

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        deadline = asyncio.get_running_loop().time() + turn_context.deadline_seconds
        if not await self.health():
            raise BackendProcessError("claude_code backend is not healthy.")
        if len(new_messages) != 1:
            raise BackendProtocolError(
                "claude_code adapter expects exactly one input message."
            )
        message = new_messages[0]
        if message.role != "user":
            raise BackendProtocolError(
                "claude_code adapter only accepts a user message."
            )
        if message.content is None or not message.content.strip():
            raise BackendProtocolError(
                "claude_code adapter requires non-empty user content."
            )

        had_backend_session = session_context.backend_session_id is not None
        backend_session_id = (
            session_context.backend_session_id or self._stable_backend_session_id()
        )
        session_state = ClaudeCodeSessionState(
            backend_session_id=backend_session_id,
            resume=had_backend_session,
        )

        if self._proxy is not None:
            await self._proxy.open_turn(
                turn_context.turn_id, backend_session_id=backend_session_id
            )

        success = False
        try:
            task = asyncio.create_task(
                self._run_claude_turn(
                    user_text=message.content,
                    turn_context=turn_context,
                    session_state=session_state,
                )
            )
            try:
                (
                    outputs,
                    trace_events,
                    usage,
                ) = await self._await_backend_task_or_proxy_max_steps(
                    task,
                    session_context=session_context,
                    proxy=self._proxy,
                )
            except BackendTransportError as exc:
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()
                await self._raise_with_proxy_failed_upstream(exc)
            if self._proxy is not None:
                await self._proxy.drain_turn(
                    timeout=self._remaining_timeout(
                        deadline, operation="wait for rollout proxy drain"
                    )
                )
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()
            session_context.backend_session_id = backend_session_id
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
                    LOGGER.warning(
                        "Timed out draining rollout proxy requests for turn %s",
                        turn_context.turn_id,
                    )
                finally:
                    await self._proxy.clear_turn()

    async def abort_session(self, session_context: SessionContext) -> bool:
        _ = session_context
        terminated = await self._terminate_active_process()
        if self._proxy is not None:
            with contextlib.suppress(Exception):
                await self._proxy.clear_turn()
        return terminated

    async def health(self) -> bool:
        if self._options is None:
            return False
        return self._resolve_executable(self._options.resolved_executable()) is not None

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
        await self._terminate_active_process()
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

    def _parse_options(
        self, backend_options: dict[str, Any]
    ) -> ClaudeCodeBackendOptions:
        try:
            return ClaudeCodeBackendOptions.model_validate(backend_options)
        except Exception as exc:
            raise BackendProtocolError(
                f"Invalid claude_code backend_options: {exc}"
            ) from exc

    def _prepare_runtime_dirs(self, runtime_dir: Path) -> None:
        for path in (
            runtime_dir / "home" / ".claude",
            runtime_dir / "workspace",
            runtime_dir / "logs",
            runtime_dir / "run",
            runtime_dir / "tmp",
            runtime_dir / "prompts",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _prepare_system_prompt(
        self,
        binding_context: BindingContext,
        options: ClaudeCodeBackendOptions,
    ) -> Path | None:
        if binding_context.binding.system_prompt is None:
            return None
        source = Path(binding_context.binding.system_prompt.source_file)
        target = Path(binding_context.binding.system_prompt.runtime_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if options.system_prompt_mode == "claude_md":
            claude_md = (
                Path(binding_context.binding.runtime_dir) / "workspace" / "CLAUDE.md"
            )
            claude_md.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            return None
        if options.system_prompt_mode == "none":
            return None
        return target

    async def _run_claude_turn(
        self,
        *,
        user_text: str,
        turn_context: TurnContext,
        session_state: ClaudeCodeSessionState,
    ) -> tuple[list[Message], list[TraceEvent], TurnUsage]:
        if (
            self._binding_context is None
            or self._options is None
            or self._compiler is None
        ):
            raise BackendProcessError("claude_code adapter has not been initialized.")
        if self._proxy_port is None:
            raise BackendProcessError("rollout proxy has not been initialized.")
        runtime_dir = Path(self._binding_context.binding.runtime_dir)
        logs_dir = runtime_dir / "logs"
        workspace_dir = runtime_dir / "workspace"
        if self._stdout_handle is None:
            self._stdout_handle = open(logs_dir / "claude_code.stdout.log", "ab")
        if self._stderr_handle is None:
            self._stderr_handle = open(logs_dir / "claude_code.stderr.log", "ab")

        args = self._compiler.build_cli_args(user_text, session_state)
        env = self._compiler.build_env(self._proxy_port, runtime_dir)
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(workspace_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=self._stderr_handle,
                limit=_CLAUDE_CODE_STDOUT_STREAM_LIMIT,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise BackendProcessError(
                f"claude code binary not found. Set CLAUDE_CODE_BIN or install Claude Code. ({args[0]})"
            ) from exc
        self._process = process
        self._process_group_id = process.pid
        (runtime_dir / "run" / "claude_code.pid").write_text(
            str(process.pid), encoding="utf-8"
        )

        events: list[dict[str, Any]] = []
        assert process.stdout is not None
        try:
            async for raw_line in process.stdout:
                self._stdout_handle.write(raw_line)
                self._stdout_handle.flush()
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {"type": "stdout", "text": line}
                if isinstance(event, dict):
                    events.append(event)
            returncode = await process.wait()
        finally:
            if self._process is process:
                self._process = None
                self._process_group_id = None

        if returncode != 0:
            stderr_tail = self._stderr_tail()
            stdout_tail = self._stdout_tail()
            raise BackendTransportError(
                _build_claude_code_exit_error_message(
                    returncode,
                    stderr_tail=stderr_tail,
                    stdout_tail=stdout_tail,
                    events=events,
                )
            )
        return convert_claude_code_stream_events(turn_context.turn_id, events)

    async def _start_proxy(
        self, binding_context: BindingContext, options: ClaudeCodeBackendOptions
    ) -> None:
        bound_session_id = binding_context.binding.bound_session_id
        bound_instance_id = binding_context.binding.bound_instance_id
        upstream_origin = self._resolve_upstream_origin(
            binding_context.binding.router_base_url
        )
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
            debug_log_dir=Path(binding_context.binding.runtime_dir) / "logs",
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
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "proxy.port").write_text(str(self._proxy_port), encoding="utf-8")

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
        from urllib.parse import urlparse

        raw = router_base_url
        if "://" not in raw:
            raw = f"http://{raw}"
        parsed = urlparse(raw)
        if not parsed.netloc:
            raise BackendProtocolError(f"Invalid router_base_url: {router_base_url}")
        return f"{parsed.scheme or 'http'}://{parsed.netloc}"

    def _stable_backend_session_id(self) -> str:
        if self._binding_context is None:
            raise BackendProcessError(
                "claude_code binding context has not been initialized."
            )
        binding = self._binding_context.binding
        return str(
            uuid5(
                NAMESPACE_URL, f"{binding.bound_instance_id}:{binding.bound_session_id}"
            )
        )

    def _remaining_timeout(self, deadline: float, *, operation: str) -> float:
        pause_credit = (
            float(getattr(self._proxy, "total_paused_seconds", 0.0))
            if self._proxy is not None
            else 0.0
        )
        remaining = deadline + pause_credit - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError(f"Timed out before {operation}.")
        return remaining

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

    async def _raise_with_proxy_failed_upstream(
        self, exc: BackendTransportError
    ) -> None:
        if self._proxy is None:
            raise exc
        if not hasattr(self._proxy, "consume_failed_upstream_error"):
            raise exc
        payload = await self._proxy.consume_failed_upstream_error()
        if payload is None:
            raise exc
        parts = [str(exc)]
        message = payload.get("message")
        if message:
            parts.append(f"upstream error: {message}")
        request_path = payload.get("request_path")
        if request_path:
            parts.append(f"failed upstream payload: {request_path}")
        response_path = payload.get("response_path")
        if response_path:
            parts.append(f"failed upstream response: {response_path}")
        dump_error = payload.get("dump_error")
        if dump_error:
            parts.append(f"failed upstream dump error: {dump_error}")
        raise BackendTransportError("; ".join(parts)) from exc

    async def _terminate_active_process(self) -> bool:
        process = self._process
        if process is None:
            return False
        if process.returncode is not None:
            self._process = None
            self._process_group_id = None
            return False
        if self._process_group_id is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._process_group_id, signal.SIGTERM)
        else:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            if self._process_group_id is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._process_group_id, signal.SIGKILL)
            else:
                process.kill()
            await process.wait()
        finally:
            self._process = None
            self._process_group_id = None
        return True

    def _stderr_tail(self, *, max_chars: int = 2000) -> str | None:
        if self._binding_context is None:
            return None
        path = (
            Path(self._binding_context.binding.runtime_dir)
            / "logs"
            / "claude_code.stderr.log"
        )
        return _tail_file(path, max_chars=max_chars)

    def _stdout_tail(self, *, max_chars: int = 2000) -> str | None:
        if self._binding_context is None:
            return None
        path = (
            Path(self._binding_context.binding.runtime_dir)
            / "logs"
            / "claude_code.stdout.log"
        )
        return _tail_file(path, max_chars=max_chars)

    def _find_free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _resolve_executable(self, executable: str) -> str | None:
        if os.path.sep in executable:
            return executable if Path(executable).exists() else None
        return shutil.which(executable)


def _tail_file(path: Path, *, max_chars: int = 2000) -> str | None:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()[-max_chars:]
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace").strip()
    return text or None


def _validate_claude_code_runtime_options(options: ClaudeCodeBackendOptions) -> None:
    if options.permission_mode != "bypassPermissions":
        return
    geteuid = getattr(os, "geteuid", None)
    if not callable(geteuid):
        return
    if geteuid() != 0:
        return
    raise BackendProtocolError(
        "claude_code permission_mode=bypassPermissions cannot run as root/sudo. "
        "Use permission_mode=default or permission_mode=acceptEdits, or run BlackboxServer as a non-root user."
    )


def _claude_code_stderr_remediation(stderr_tail: str | None) -> str | None:
    if not stderr_tail:
        return None
    lower_tail = stderr_tail.lower()
    remediations: list[str] = []
    if "requires --verbose" in lower_tail and "stream-json" in lower_tail:
        remediations.append(
            "pass --verbose whenever --output-format=stream-json is used"
        )
    if (
        "dangerously-skip-permissions" in lower_tail
        or "cannot be used with root/sudo" in lower_tail
    ):
        remediations.append(
            "use permission_mode=default/acceptEdits or run BlackboxServer as a non-root user"
        )
    if (
        "anthropic_api_key" in lower_tail
        or "claude_code_oauth_token" in lower_tail
        or "authentication" in lower_tail
        or "unauthorized" in lower_tail
        or "401" in lower_tail
    ):
        remediations.append(
            "verify the Claude Code gateway token and clear inherited Anthropic/Claude auth environment"
        )
    if not remediations:
        return None
    return "; ".join(remediations)


def _build_claude_code_exit_error_message(
    returncode: int,
    *,
    stderr_tail: str | None,
    stdout_tail: str | None,
    events: list[dict[str, Any]],
) -> str:
    parts = [f"claude_code exited with code {returncode}"]
    event_error = _claude_code_stream_error_summary(events)
    if event_error:
        parts.append(f"stream-json error: {event_error}")
    if stderr_tail:
        parts.append(f"stderr tail: {stderr_tail}")
    if stdout_tail:
        parts.append(f"stdout tail: {stdout_tail}")
    remediation = _claude_code_stderr_remediation(stderr_tail)
    if remediation:
        parts.append(f"remediation: {remediation}")
    return "; ".join(parts)


def _claude_code_stream_error_summary(
    events: list[dict[str, Any]], *, max_chars: int = 1200
) -> str | None:
    summaries: list[str] = []
    for event in events[-20:]:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "")
        if event_type == "result" and event.get("is_error"):
            value = (
                event.get("result")
                or event.get("error")
                or event.get("message")
                or event
            )
            summaries.append(f"result={_compact_jsonish(value, max_chars=400)}")
            continue
        if event_type in {"error", "stderr"}:
            summaries.append(_compact_jsonish(event, max_chars=400))
            continue
        for key in ("error", "stderr"):
            value = event.get(key)
            if value:
                summaries.append(f"{key}={_compact_jsonish(value, max_chars=400)}")
        message = event.get("message")
        if isinstance(message, dict):
            for key in ("error", "stderr"):
                value = message.get(key)
                if value:
                    summaries.append(
                        f"message.{key}={_compact_jsonish(value, max_chars=400)}"
                    )
    if not summaries:
        return None
    summary = " | ".join(summaries)
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars] + "...(truncated)"


def _compact_jsonish(value: Any, *, max_chars: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def convert_claude_code_stream_events(
    turn_id: str,
    events: list[dict[str, Any]],
) -> tuple[list[Message], list[TraceEvent], TurnUsage]:
    trace_events: list[TraceEvent] = []
    outputs: list[Message] = []
    usage = TurnUsage()
    result_text: str | None = None
    seq = 0

    for event in events:
        seq += 1
        trace_events.append(
            TraceEvent(
                turn_id=turn_id,
                seq=seq,
                source="claude_code",
                event_type=str(event.get("type") or "event"),
                payload=event,
                created_at=utcnow(),
            )
        )
        event_usage = _usage_from_event(event)
        usage.total_tokens += event_usage.total_tokens
        usage.input_tokens += event_usage.input_tokens
        usage.output_tokens += event_usage.output_tokens
        usage.reasoning_tokens += event_usage.reasoning_tokens
        usage.tool_calls += event_usage.tool_calls

        if event.get("type") == "result":
            if event.get("is_error"):
                raise BackendTransportError(
                    str(
                        event.get("result")
                        or event.get("error")
                        or "Claude Code failed."
                    )
                )
            if event.get("result") is not None:
                result_text = str(event.get("result"))
            if event.get("num_turns") is not None:
                with contextlib.suppress(TypeError, ValueError):
                    usage.steps = max(usage.steps, int(event["num_turns"]))
            continue

        message = event.get("message")
        if isinstance(message, dict):
            converted = _messages_from_claude_message(message)
            outputs.extend(converted)
            if converted:
                usage.steps += 1

    if not outputs and result_text is not None:
        outputs.append(Message(role="assistant", content=result_text))
    if not outputs:
        raise BackendProtocolError(
            "claude_code stream-json output contained no assistant output."
        )
    if usage.total_tokens == 0:
        usage.total_tokens = usage.input_tokens + usage.output_tokens
    return outputs, trace_events, usage


def _messages_from_claude_message(message: dict[str, Any]) -> list[Message]:
    if message.get("role") != "assistant":
        return []
    content = message.get("content")
    if not isinstance(content, list):
        if isinstance(content, str) and content:
            return [Message(role="assistant", content=content)]
        return []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
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
                ToolCall(
                    id=str(block.get("id") or f"call_{uuid4().hex[:8]}"),
                    function=FunctionCall(
                        name=str(block.get("name") or "tool"),
                        arguments=tool_input,
                    ),
                )
            )
    return [
        Message(
            role="assistant",
            content="\n".join(text_parts) if text_parts else None,
            reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls or None,
        )
    ]


def _usage_from_event(event: dict[str, Any]) -> TurnUsage:
    raw_usage = event.get("usage")
    if not isinstance(raw_usage, dict):
        message = event.get("message")
        raw_usage = message.get("usage") if isinstance(message, dict) else {}
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    input_tokens = int(raw_usage.get("input_tokens", 0) or 0)
    output_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    cache_read = int(raw_usage.get("cache_read_input_tokens", 0) or 0)
    cache_create = int(raw_usage.get("cache_creation_input_tokens", 0) or 0)
    return TurnUsage(
        total_tokens=input_tokens + output_tokens + cache_read + cache_create,
        input_tokens=input_tokens + cache_read + cache_create,
        output_tokens=output_tokens,
    )
