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
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

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
    Message,
    ProxyOptions,
    SessionContext,
    TraceEvent,
    TurnContext,
    TurnUsage,
    utcnow,
)
from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy


LOGGER = logging.getLogger(__name__)
_CODEX_STDOUT_STREAM_LIMIT = 100 * 1024 * 1024

_CODEX_INHERITED_AUTH_ENV_KEYS = (
    "CODEX_ACCESS_TOKEN",
    "CODEX_API_KEY",
    "CODEX_HOME",
    "CODEX_SQLITE_HOME",
    "OPENAI_API_KEY",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
)


class _BackgroundUvicornServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        # Keep the parent process in charge of SIGINT/SIGTERM.
        yield


class CodexModelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="proxy-model", min_length=1)
    name: str = Field(default="Dressage Proxy", min_length=1)


class CodexBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str | None = None
    model: CodexModelOptions = Field(default_factory=CodexModelOptions)
    model_provider_id: str = Field(
        default="dressage_proxy",
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    sandbox_mode: Literal[
        "read-only",
        "workspace-write",
        "danger-full-access",
    ] = "danger-full-access"
    approval_policy: Literal["untrusted", "on-request", "never"] = "never"
    skip_git_repo_check: bool = True
    ignore_rules: bool = True
    web_search: Literal["disabled", "cached", "live"] = "disabled"
    proxy: ProxyOptions = Field(default_factory=ProxyOptions)

    def resolved_executable(self) -> str:
        return self.executable or os.getenv("CODEX_BIN") or "codex"


@dataclass(frozen=True)
class CodexSessionState:
    backend_session_id: str | None
    resume: bool


@dataclass(frozen=True)
class CodexRunResult:
    outputs: list[Message]
    trace_events: list[TraceEvent]
    usage: TurnUsage
    backend_session_id: str | None


class CodexConfigCompiler:
    def __init__(
        self,
        *,
        options: CodexBackendOptions,
        config_path: Path,
        system_prompt_path: Path | None = None,
    ) -> None:
        self.options = options
        self.config_path = config_path
        self.system_prompt_path = system_prompt_path

    def build_config(self, proxy_port: int) -> str:
        lines = [
            f"model = {_toml_string(self.options.model.id)}",
            f"model_provider = {_toml_string(self.options.model_provider_id)}",
            f"approval_policy = {_toml_string(self.options.approval_policy)}",
            f"sandbox_mode = {_toml_string(self.options.sandbox_mode)}",
            f"web_search = {_toml_string(self.options.web_search)}",
            "check_for_update_on_startup = false",
            'cli_auth_credentials_store = "file"',
            "",
        ]
        if self.system_prompt_path is not None:
            try:
                prompt_text = self.system_prompt_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise BackendProtocolError(
                    f"Could not read Codex system prompt: {exc}"
                ) from exc
            lines.append(f"developer_instructions = {_toml_string(prompt_text)}")
            lines.append("")

        lines.extend(
            [
                "[analytics]",
                "enabled = false",
                "",
                "[feedback]",
                "enabled = false",
                "",
                f"[model_providers.{self.options.model_provider_id}]",
                f"name = {_toml_string(self.options.model.name)}",
                f"base_url = {_toml_string(f'http://127.0.0.1:{proxy_port}/v1')}",
                'wire_api = "responses"',
                "",
            ]
        )
        return "\n".join(lines)

    def build_env(self, runtime_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        for key in _CODEX_INHERITED_AUTH_ENV_KEYS:
            env.pop(key, None)
        codex_home = runtime_dir / "home" / ".codex"
        sqlite_home = runtime_dir / "home" / ".codex-sqlite"
        env.update(
            {
                "HOME": str(runtime_dir / "home"),
                "CODEX_HOME": str(codex_home),
                "CODEX_SQLITE_HOME": str(sqlite_home),
                "TMPDIR": str(runtime_dir / "tmp"),
                "RUST_LOG": env.get("RUST_LOG", "error"),
            }
        )
        return {str(key): str(value) for key, value in env.items()}

    def build_cli_args(
        self,
        user_text: str,
        session_state: CodexSessionState,
    ) -> list[str]:
        args = [self.options.resolved_executable(), "exec"]
        if session_state.resume:
            if not session_state.backend_session_id:
                raise BackendProtocolError(
                    "codex resume requested without a backend_session_id."
                )
            args.extend(["resume", session_state.backend_session_id])
        args.extend(
            [
                "--json",
                "--model",
                self.options.model.id,
                "--sandbox",
                self.options.sandbox_mode,
                "--config",
                f"approval_policy={_toml_string(self.options.approval_policy)}",
                "--config",
                f'model_provider="{self.options.model_provider_id}"',
                "--config",
                f"web_search={_toml_string(self.options.web_search)}",
                "--config",
                f"log_dir={_toml_string(str(self.config_path.parent / 'log'))}",
            ]
        )
        if self.options.skip_git_repo_check:
            args.append("--skip-git-repo-check")
        if self.options.ignore_rules:
            args.append("--ignore-rules")
        args.append(user_text)
        return args


class CodexAdapter(BackendAdapter):
    def __init__(self) -> None:
        self._binding_context: BindingContext | None = None
        self._options: CodexBackendOptions | None = None
        self._compiler: CodexConfigCompiler | None = None
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
        self._options = options
        runtime_dir = Path(binding_context.binding.runtime_dir)
        self._prepare_runtime_dirs(runtime_dir)
        system_prompt_path = self._prepare_system_prompt(binding_context)
        await self._start_proxy(binding_context, options)
        assert self._proxy_port is not None
        config_path = runtime_dir / "home" / ".codex" / "config.toml"
        self._compiler = CodexConfigCompiler(
            options=options,
            config_path=config_path,
            system_prompt_path=system_prompt_path,
        )
        config_path.write_text(
            self._compiler.build_config(self._proxy_port),
            encoding="utf-8",
        )
        run_dir = runtime_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "codex.config").write_text(str(config_path), encoding="utf-8")
        executable = options.resolved_executable()
        if self._resolve_executable(executable) is None:
            await self.shutdown()
            raise BackendProcessError(
                f"codex binary not found. Set CODEX_BIN or install Codex CLI. ({executable})"
            )

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        deadline = asyncio.get_running_loop().time() + turn_context.deadline_seconds
        if not await self.health():
            raise BackendProcessError("codex backend is not healthy.")
        if len(new_messages) != 1:
            raise BackendProtocolError("codex adapter expects exactly one input message.")
        message = new_messages[0]
        if message.role != "user":
            raise BackendProtocolError("codex adapter only accepts a user message.")
        if message.content is None or not message.content.strip():
            raise BackendProtocolError("codex adapter requires non-empty user content.")

        had_backend_session = session_context.backend_session_id is not None
        target_backend_session_id = (
            session_context.backend_session_id or self._stable_backend_session_id()
        )
        session_state = CodexSessionState(
            backend_session_id=session_context.backend_session_id,
            resume=had_backend_session,
        )

        if self._proxy is not None:
            await self._proxy.open_turn(
                turn_context.turn_id,
                backend_session_id=target_backend_session_id,
            )

        success = False
        try:
            task = asyncio.create_task(
                self._run_codex_turn(
                    user_text=message.content,
                    turn_context=turn_context,
                    session_state=session_state,
                )
            )
            try:
                result = await self._await_backend_task_or_proxy_max_steps(
                    task,
                    session_context=session_context,
                    proxy=self._proxy,
                )
            except BackendTransportError as exc:
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()
                await self._raise_with_proxy_failed_upstream(exc)
            if result.backend_session_id:
                target_backend_session_id = result.backend_session_id
                if self._proxy is not None:
                    await self._proxy.update_turn_backend_session(
                        target_backend_session_id
                    )
            if self._proxy is not None:
                await self._proxy.drain_turn(
                    timeout=self._remaining_timeout(
                        deadline, operation="wait for rollout proxy drain"
                    )
                )
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()
            session_context.backend_session_id = target_backend_session_id
            success = True
            return AdapterResponse(
                outputs=result.outputs,
                trace_events=result.trace_events,
                usage=result.usage,
                backend_session_id=target_backend_session_id,
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

    def _parse_options(self, backend_options: dict[str, Any]) -> CodexBackendOptions:
        try:
            return CodexBackendOptions.model_validate(backend_options)
        except Exception as exc:
            raise BackendProtocolError(
                f"Invalid codex backend_options: {exc}"
            ) from exc

    def _prepare_runtime_dirs(self, runtime_dir: Path) -> None:
        for path in (
            runtime_dir / "home" / ".codex",
            runtime_dir / "home" / ".codex" / "log",
            runtime_dir / "home" / ".codex-sqlite",
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
    ) -> Path | None:
        if binding_context.binding.system_prompt is None:
            return None
        source = Path(binding_context.binding.system_prompt.source_file)
        target = Path(binding_context.binding.system_prompt.runtime_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    async def _run_codex_turn(
        self,
        *,
        user_text: str,
        turn_context: TurnContext,
        session_state: CodexSessionState,
    ) -> CodexRunResult:
        if (
            self._binding_context is None
            or self._options is None
            or self._compiler is None
        ):
            raise BackendProcessError("codex adapter has not been initialized.")
        if self._proxy_port is None:
            raise BackendProcessError("rollout proxy has not been initialized.")
        runtime_dir = Path(self._binding_context.binding.runtime_dir)
        logs_dir = runtime_dir / "logs"
        workspace_dir = runtime_dir / "workspace"
        if self._stdout_handle is None:
            self._stdout_handle = open(logs_dir / "codex.stdout.log", "ab")
        if self._stderr_handle is None:
            self._stderr_handle = open(logs_dir / "codex.stderr.log", "ab")

        args = self._compiler.build_cli_args(user_text, session_state)
        env = self._compiler.build_env(runtime_dir)
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(workspace_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=self._stderr_handle,
                limit=_CODEX_STDOUT_STREAM_LIMIT,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise BackendProcessError(
                f"codex binary not found. Set CODEX_BIN or install Codex CLI. ({args[0]})"
            ) from exc
        self._process = process
        self._process_group_id = process.pid
        (runtime_dir / "run" / "codex.pid").write_text(
            str(process.pid), encoding="utf-8"
        )

        events: list[dict[str, Any]] = []
        parse_error: str | None = None
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
                except json.JSONDecodeError as exc:
                    parse_error = f"invalid codex JSONL: {exc}; line={line[:1000]}"
                    break
                if not isinstance(event, dict):
                    parse_error = f"invalid codex JSONL event type: {type(event).__name__}"
                    break
                events.append(event)
            if parse_error is not None:
                await self._terminate_active_process()
                raise BackendTransportError(parse_error)
            returncode = await process.wait()
        finally:
            if self._process is process:
                self._process = None
                self._process_group_id = None

        if returncode != 0:
            stderr_tail = self._stderr_tail()
            stdout_tail = self._stdout_tail()
            raise BackendTransportError(
                _build_codex_exit_error_message(
                    returncode,
                    stderr_tail=stderr_tail,
                    stdout_tail=stdout_tail,
                    events=events,
                )
            )
        return convert_codex_jsonl_events(turn_context.turn_id, events)

    async def _start_proxy(
        self, binding_context: BindingContext, options: CodexBackendOptions
    ) -> None:
        bound_session_id = binding_context.binding.bound_session_id
        bound_instance_id = binding_context.binding.bound_instance_id
        upstream_origin = self._resolve_upstream_origin(
            binding_context.binding.router_base_url
        )
        self._proxy_port = self._find_free_port()
        LOGGER.info(
            "starting codex rollout proxy on port %d, upstream_origin=%s, router_api_path=%s",
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
        raise BackendProcessError("Timed out waiting for codex rollout proxy startup.")

    def _resolve_upstream_origin(self, router_base_url: str) -> str:
        raw = router_base_url
        if "://" not in raw:
            raw = f"http://{raw}"
        parsed = urlparse(raw)
        if not parsed.netloc:
            raise BackendProtocolError(f"Invalid router_base_url: {router_base_url}")
        return f"{parsed.scheme or 'http'}://{parsed.netloc}"

    def _stable_backend_session_id(self) -> str:
        if self._binding_context is None:
            raise BackendProcessError("codex binding context has not been initialized.")
        binding = self._binding_context.binding
        return str(
            uuid5(
                NAMESPACE_URL, f"codex:{binding.bound_instance_id}:{binding.bound_session_id}"
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
            / "codex.stderr.log"
        )
        return _tail_file(path, max_chars=max_chars)

    def _stdout_tail(self, *, max_chars: int = 2000) -> str | None:
        if self._binding_context is None:
            return None
        path = (
            Path(self._binding_context.binding.runtime_dir)
            / "logs"
            / "codex.stdout.log"
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


def _toml_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _tail_file(path: Path, *, max_chars: int = 2000) -> str | None:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()[-max_chars:]
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace").strip()
    return text or None


def _build_codex_exit_error_message(
    returncode: int,
    *,
    stderr_tail: str | None,
    stdout_tail: str | None,
    events: list[dict[str, Any]],
) -> str:
    parts = [f"codex exited with code {returncode}"]
    event_error = _codex_stream_error_summary(events)
    if event_error:
        parts.append(f"jsonl error: {event_error}")
    if stderr_tail:
        parts.append(f"stderr tail: {stderr_tail}")
    if stdout_tail:
        parts.append(f"stdout tail: {stdout_tail}")
    return "; ".join(parts)


def _codex_stream_error_summary(
    events: list[dict[str, Any]], *, max_chars: int = 1200
) -> str | None:
    summaries: list[str] = []
    for event in events[-20:]:
        event_type = str(event.get("type") or "")
        if event_type in {"turn.failed", "error"}:
            summaries.append(_compact_jsonish(event, max_chars=400))
            continue
        error = event.get("error")
        if error:
            summaries.append(f"error={_compact_jsonish(error, max_chars=400)}")
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


def convert_codex_jsonl_events(
    turn_id: str,
    events: list[dict[str, Any]],
) -> CodexRunResult:
    trace_events: list[TraceEvent] = []
    output_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = TurnUsage()
    backend_session_id: str | None = None
    tool_calls = 0
    seq = 0

    for event in events:
        seq += 1
        event_type = str(event.get("type") or "event")
        trace_events.append(
            TraceEvent(
                turn_id=turn_id,
                seq=seq,
                source="codex",
                event_type=event_type,
                payload=event,
                created_at=utcnow(),
            )
        )

        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if thread_id:
                backend_session_id = str(thread_id)
            continue

        if event_type in {"turn.failed", "error"}:
            raise BackendTransportError(_codex_event_error_message(event))

        if event_type == "turn.completed":
            event_usage = _usage_from_codex_event(event)
            usage.total_tokens = event_usage.total_tokens
            usage.input_tokens = event_usage.input_tokens
            usage.output_tokens = event_usage.output_tokens
            usage.reasoning_tokens = event_usage.reasoning_tokens
            usage.steps = max(usage.steps, 1)
            continue

        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if event_type == "item.completed" and item_type == "agent_message":
            text = item.get("text")
            if text is not None:
                output_parts.append(str(text))
        elif event_type == "item.completed" and item_type == "reasoning":
            text = (
                item.get("text")
                or item.get("summary")
                or item.get("content")
                or item.get("message")
            )
            if text is not None:
                reasoning_parts.append(str(text))
        elif event_type == "item.completed" and item_type in {
            "command_execution",
            "mcp_tool_call",
            "tool_call",
            "web_search",
        }:
            tool_calls += 1

    if not output_parts:
        raise BackendProtocolError("codex JSONL output contained no assistant output.")
    if usage.total_tokens == 0:
        usage.total_tokens = usage.input_tokens + usage.output_tokens
    usage.tool_calls = tool_calls
    return CodexRunResult(
        outputs=[
            Message(
                role="assistant",
                content="\n".join(output_parts),
                reasoning_content="\n".join(reasoning_parts)
                if reasoning_parts
                else None,
            )
        ],
        trace_events=trace_events,
        usage=usage,
        backend_session_id=backend_session_id,
    )


def _usage_from_codex_event(event: dict[str, Any]) -> TurnUsage:
    raw_usage = event.get("usage")
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    input_tokens = _int_or_zero(raw_usage.get("input_tokens"))
    output_tokens = _int_or_zero(raw_usage.get("output_tokens"))
    reasoning_tokens = _int_or_zero(raw_usage.get("reasoning_output_tokens"))
    total_tokens = _int_or_zero(raw_usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return TurnUsage(
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _int_or_zero(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _codex_event_error_message(event: dict[str, Any]) -> str:
    value = event.get("error") or event.get("message") or event
    if isinstance(value, dict):
        message = value.get("message") or value.get("error") or value.get("detail")
        if message:
            return str(message)
    return _compact_jsonish(value, max_chars=1200)
