from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import blackbox_server.adapters.codex as codex_module
from blackbox_server.adapters.base import BackendProtocolError, BackendTransportError
from blackbox_server.adapters.codex import (
    CodexAdapter,
    CodexBackendOptions,
    CodexConfigCompiler,
    CodexSessionState,
    _CODEX_STDOUT_STREAM_LIMIT,
    _build_codex_exit_error_message,
    convert_codex_jsonl_events,
)
from blackbox_server.config import BlackboxServerConfig
from blackbox_server.core.models import (
    BindingContext,
    BindingInfo,
    SessionContext,
    SessionState,
    TurnContext,
    utcnow,
)


def _make_binding_context(tmp_path: Path) -> BindingContext:
    runtime_dir = tmp_path / "runtime"
    for path in (
        runtime_dir / "home" / ".codex",
        runtime_dir / "home" / ".codex-sqlite",
        runtime_dir / "workspace",
        runtime_dir / "logs",
        runtime_dir / "run",
        runtime_dir / "tmp",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return BindingContext(
        binding=BindingInfo(
            runtime_id="bbs-test",
            blackbox_type="codex",
            router_raw="http://127.0.0.1:30000",
            router_base_url="http://127.0.0.1:30000/v1",
            router_api_path="/v1",
            bound_session_id="sess-001",
            bound_instance_id="inst-001",
            runtime_dir=str(runtime_dir),
            registered_at=utcnow(),
            backend_options={},
        ),
        effective_config=BlackboxServerConfig(router_timeout=300000),
    )


def test_codex_options_reject_unknown_fields() -> None:
    with pytest.raises(Exception, match="extra"):
        CodexBackendOptions.model_validate({"unknown": True})


def test_codex_options_default_to_full_access_noninteractive() -> None:
    options = CodexBackendOptions()

    assert options.sandbox_mode == "danger-full-access"
    assert options.approval_policy == "never"
    assert options.skip_git_repo_check is True
    assert options.ignore_rules is True
    assert options.web_search == "disabled"


def test_config_compiler_builds_config_provider_and_developer_instructions(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "system.txt"
    prompt_path.write_text("system prompt\nline 2", encoding="utf-8")
    options = CodexBackendOptions.model_validate(
        {
            "model": {"id": "proxy-model", "name": "Qwen via Dressage"},
            "model_provider_id": "dressage_proxy",
        }
    )
    compiler = CodexConfigCompiler(
        options=options,
        config_path=tmp_path / "home" / ".codex" / "config.toml",
        system_prompt_path=prompt_path,
    )

    config = compiler.build_config(4567)

    assert 'model = "proxy-model"' in config
    assert 'model_provider = "dressage_proxy"' in config
    assert 'approval_policy = "never"' in config
    assert 'sandbox_mode = "danger-full-access"' in config
    assert 'web_search = "disabled"' in config
    assert 'developer_instructions = "system prompt\\nline 2"' in config
    assert "[model_providers.dressage_proxy]" in config
    assert 'name = "Qwen via Dressage"' in config
    assert 'base_url = "http://127.0.0.1:4567/v1"' in config
    assert 'wire_api = "responses"' in config
    assert "env_key" not in config


def test_config_compiler_missing_prompt_file_raises_protocol_error(
    tmp_path: Path,
) -> None:
    compiler = CodexConfigCompiler(
        options=CodexBackendOptions(),
        config_path=tmp_path / "config.toml",
        system_prompt_path=tmp_path / "missing.txt",
    )

    with pytest.raises(BackendProtocolError):
        compiler.build_config(4567)


def test_config_compiler_sanitizes_inherited_codex_and_openai_auth_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "CODEX_ACCESS_TOKEN",
        "CODEX_API_KEY",
        "CODEX_HOME",
        "CODEX_SQLITE_HOME",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
    ):
        monkeypatch.setenv(key, f"parent-{key}")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    compiler = CodexConfigCompiler(
        options=CodexBackendOptions(),
        config_path=tmp_path / "home" / ".codex" / "config.toml",
    )

    env = compiler.build_env(tmp_path)

    assert "CODEX_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["HOME"] == str(tmp_path / "home")
    assert env["CODEX_HOME"] == str(tmp_path / "home" / ".codex")
    assert env["CODEX_SQLITE_HOME"] == str(tmp_path / "home" / ".codex-sqlite")
    assert env["TMPDIR"] == str(tmp_path / "tmp")
    assert env["PATH"] == "/usr/local/bin:/usr/bin"


def test_config_compiler_builds_cli_args_for_first_turn_and_resume(
    tmp_path: Path,
) -> None:
    options = CodexBackendOptions.model_validate(
        {
            "executable": "/bin/codex",
            "model": {"id": "proxy-model", "name": "Dressage Proxy"},
        }
    )
    compiler = CodexConfigCompiler(
        options=options,
        config_path=tmp_path / "home" / ".codex" / "config.toml",
    )

    first = compiler.build_cli_args(
        "hello",
        CodexSessionState(backend_session_id=None, resume=False),
    )
    resumed = compiler.build_cli_args(
        "hello again",
        CodexSessionState(backend_session_id="thread-001", resume=True),
    )

    assert first[:2] == ["/bin/codex", "exec"]
    assert "resume" not in first
    assert "--json" in first
    assert first[first.index("--model") + 1] == "proxy-model"
    assert first[first.index("--sandbox") + 1] == "danger-full-access"
    assert "--ask-for-approval" not in first
    approval_config_index = first.index('approval_policy="never"')
    assert first[approval_config_index - 1] == "--config"
    assert "--skip-git-repo-check" in first
    assert "--ignore-rules" in first
    assert first[-1] == "hello"
    assert resumed[:4] == ["/bin/codex", "exec", "resume", "thread-001"]
    assert "--json" in resumed
    assert resumed[-1] == "hello again"


def test_config_compiler_resume_requires_backend_session_id(tmp_path: Path) -> None:
    compiler = CodexConfigCompiler(
        options=CodexBackendOptions(),
        config_path=tmp_path / "config.toml",
    )

    with pytest.raises(BackendProtocolError, match="backend_session_id"):
        compiler.build_cli_args(
            "hello",
            CodexSessionState(backend_session_id=None, resume=True),
        )


def test_convert_codex_jsonl_events_extracts_output_trace_usage_and_thread() -> None:
    events = [
        {"type": "thread.started", "thread_id": "thread-001"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"id": "item-1", "type": "reasoning", "text": "think"},
        },
        {
            "type": "item.completed",
            "item": {"id": "item-2", "type": "command_execution", "command": "ls"},
        },
        {
            "type": "item.completed",
            "item": {"id": "item-3", "type": "agent_message", "text": "done"},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "reasoning_output_tokens": 2,
            },
        },
    ]

    result = convert_codex_jsonl_events("turn-1", events)

    assert result.backend_session_id == "thread-001"
    assert result.outputs[0].role == "assistant"
    assert result.outputs[0].content == "done"
    assert result.outputs[0].reasoning_content == "think"
    assert [event.source for event in result.trace_events] == ["codex"] * len(events)
    assert result.usage.total_tokens == 10
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 3
    assert result.usage.reasoning_tokens == 2
    assert result.usage.steps == 1
    assert result.usage.tool_calls == 1


def test_convert_codex_jsonl_events_joins_multiple_agent_messages() -> None:
    result = convert_codex_jsonl_events(
        "turn-1",
        [
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "first"},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "second"},
            },
        ],
    )

    assert result.outputs[0].content == "first\nsecond"


def test_convert_codex_jsonl_events_raises_on_failed_turn() -> None:
    with pytest.raises(BackendTransportError, match="boom"):
        convert_codex_jsonl_events(
            "turn-1",
            [{"type": "turn.failed", "error": {"message": "boom"}}],
        )


def test_convert_codex_jsonl_events_raises_on_missing_agent_message() -> None:
    with pytest.raises(BackendProtocolError, match="no assistant output"):
        convert_codex_jsonl_events("turn-1", [{"type": "turn.completed"}])


def test_codex_exit_message_includes_event_error_and_tails() -> None:
    message = _build_codex_exit_error_message(
        1,
        stderr_tail="stderr detail",
        stdout_tail='{"type":"turn.failed","error":{"message":"bad"}}',
        events=[{"type": "turn.failed", "error": {"message": "bad"}}],
    )

    assert "codex exited with code 1" in message
    assert "jsonl error:" in message
    assert "bad" in message
    assert "stderr tail: stderr detail" in message
    assert "stdout tail:" in message


def test_run_codex_turn_uses_large_stdout_stream_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeStdout:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines

        def __aiter__(self) -> "FakeStdout":
            return self

        async def __anext__(self) -> bytes:
            if not self._lines:
                raise StopAsyncIteration
            return self._lines.pop(0)

    class FakeProcess:
        pid = 4321
        returncode = 0

        def __init__(self, lines: list[bytes]) -> None:
            self.stdout = FakeStdout(lines)

        async def wait(self) -> int:
            return self.returncode

    captured_args: tuple[object, ...] = ()
    captured_kwargs: dict[str, object] = {}
    stdout_lines = [
        (json.dumps({"type": "thread.started", "thread_id": "thread-001"}) + "\n").encode(
            "utf-8"
        ),
        (
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "done"},
                }
            )
            + "\n"
        ).encode("utf-8"),
        (json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}) + "\n").encode(
            "utf-8"
        ),
    ]

    async def fake_create_subprocess_exec(
        *args: object, **kwargs: object
    ) -> FakeProcess:
        nonlocal captured_args
        captured_args = args
        captured_kwargs.update(kwargs)
        return FakeProcess(stdout_lines.copy())

    async def run_test() -> None:
        binding_context = _make_binding_context(tmp_path)
        runtime_dir = Path(binding_context.binding.runtime_dir)
        options = CodexBackendOptions.model_validate({"executable": "/bin/codex"})
        adapter = CodexAdapter()
        adapter._binding_context = binding_context
        adapter._options = options
        adapter._compiler = CodexConfigCompiler(
            options=options,
            config_path=runtime_dir / "home" / ".codex" / "config.toml",
        )
        adapter._proxy_port = 4567

        monkeypatch.setattr(
            codex_module.asyncio,
            "create_subprocess_exec",
            fake_create_subprocess_exec,
        )
        try:
            result = await adapter._run_codex_turn(
                user_text="hello",
                turn_context=TurnContext(
                    turn_id="turn-1",
                    request_fingerprint="fp-turn-1",
                    deadline_seconds=30.0,
                ),
                session_state=CodexSessionState(
                    backend_session_id=None,
                    resume=False,
                ),
            )
        finally:
            await adapter.shutdown()

        assert captured_args[:2] == ("/bin/codex", "exec")
        assert captured_kwargs["limit"] == _CODEX_STDOUT_STREAM_LIMIT
        assert captured_kwargs["cwd"] == str(runtime_dir / "workspace")
        env = captured_kwargs["env"]
        assert isinstance(env, dict)
        assert env["CODEX_HOME"] == str(runtime_dir / "home" / ".codex")
        assert result.backend_session_id == "thread-001"
        assert result.outputs[0].content == "done"
        assert result.usage.total_tokens == 3

    asyncio.run(run_test())


def test_run_codex_turn_rejects_invalid_jsonl_and_terminates_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeStdout:
        def __init__(self) -> None:
            self._lines = [b"not-json\n"]

        def __aiter__(self) -> "FakeStdout":
            return self

        async def __anext__(self) -> bytes:
            if not self._lines:
                raise StopAsyncIteration
            return self._lines.pop(0)

    class FakeProcess:
        pid = 4321

        def __init__(self) -> None:
            self.stdout = FakeStdout()
            self.returncode: int | None = None
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    fake_process = FakeProcess()
    killed_process_groups: list[tuple[int, int]] = []

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        _ = args, kwargs
        return fake_process

    def fake_killpg(process_group_id: int, sig: int) -> None:
        killed_process_groups.append((process_group_id, sig))

    async def run_test() -> None:
        binding_context = _make_binding_context(tmp_path)
        runtime_dir = Path(binding_context.binding.runtime_dir)
        options = CodexBackendOptions.model_validate({"executable": "/bin/codex"})
        adapter = CodexAdapter()
        adapter._binding_context = binding_context
        adapter._options = options
        adapter._compiler = CodexConfigCompiler(
            options=options,
            config_path=runtime_dir / "home" / ".codex" / "config.toml",
        )
        adapter._proxy_port = 4567

        monkeypatch.setattr(
            codex_module.asyncio,
            "create_subprocess_exec",
            fake_create_subprocess_exec,
        )
        monkeypatch.setattr(codex_module.os, "killpg", fake_killpg)
        try:
            with pytest.raises(BackendTransportError, match="invalid codex JSONL"):
                await adapter._run_codex_turn(
                    user_text="hello",
                    turn_context=TurnContext(
                        turn_id="turn-1",
                        request_fingerprint="fp-turn-1",
                        deadline_seconds=30.0,
                    ),
                    session_state=CodexSessionState(
                        backend_session_id=None,
                        resume=False,
                    ),
                )
        finally:
            await adapter.shutdown()

        assert killed_process_groups

    asyncio.run(run_test())


def test_abort_session_terminates_process_and_clears_proxy_turn() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    class FakeProxy:
        def __init__(self) -> None:
            self.cleared = False

        async def clear_turn(self) -> None:
            self.cleared = True

    async def run_test() -> None:
        process = FakeProcess()
        proxy = FakeProxy()
        adapter = CodexAdapter()
        adapter._process = process  # type: ignore[assignment]
        adapter._process_group_id = None
        adapter._proxy = proxy  # type: ignore[assignment]
        session = SessionContext(
            session_id="sess-001",
            state=SessionState.ACTIVE,
            blackbox_type="codex",
            router_base_url="http://127.0.0.1:30000/v1",
            created_at=utcnow(),
            updated_at=utcnow(),
        )

        terminated = await adapter.abort_session(session)

        assert terminated is True
        assert process.terminated is True
        assert proxy.cleared is True

    asyncio.run(run_test())
