from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

import blackbox_server.adapters.claude_code as claude_code_module
from blackbox_server.adapters.base import BackendProtocolError, BackendTransportError
from blackbox_server.adapters.claude_code import (
    ClaudeCodeAdapter,
    ClaudeCodeBackendOptions,
    ClaudeCodeConfigCompiler,
    ClaudeCodeSessionState,
    _CLAUDE_CODE_STDOUT_STREAM_LIMIT,
    _build_claude_code_exit_error_message,
    _claude_code_stderr_remediation,
    _claude_code_stream_error_summary,
    _validate_claude_code_runtime_options,
    convert_claude_code_stream_events,
)
from blackbox_server.config import BlackboxServerConfig
from blackbox_server.core.models import BindingContext, BindingInfo, TurnContext, utcnow

EXPECTED_CLAUDE_CODE_PERMISSION_ALLOW = [
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
    "Task",
]


def _make_binding_context(tmp_path: Path) -> BindingContext:
    runtime_dir = tmp_path / "runtime"
    for path in (
        runtime_dir / "home" / ".claude",
        runtime_dir / "workspace",
        runtime_dir / "logs",
        runtime_dir / "run",
        runtime_dir / "tmp",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return BindingContext(
        binding=BindingInfo(
            runtime_id="bbs-test",
            blackbox_type="claude_code",
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


def test_claude_code_options_reject_unknown_fields() -> None:
    with pytest.raises(Exception, match="extra"):
        ClaudeCodeBackendOptions.model_validate({"unknown": True})


def test_claude_code_options_default_to_root_safe_permission_mode() -> None:
    assert ClaudeCodeBackendOptions().permission_mode == "default"


def test_config_compiler_defaults_keep_thinking_and_disable_prompt_caching(tmp_path: Path) -> None:
    options = ClaudeCodeBackendOptions()
    compiler = ClaudeCodeConfigCompiler(
        options=options,
        settings_path=tmp_path / "settings.json",
    )

    settings = compiler.build_settings()

    assert settings["autoCompactEnabled"] is True
    assert settings["permissions"]["allow"] == EXPECTED_CLAUDE_CODE_PERMISSION_ALLOW
    assert settings["permissions"]["deny"] == ["Agent"]
    assert settings["env"]["DISABLE_PROMPT_CACHING"] == "1"
    assert settings["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert "CLAUDE_CODE_DISABLE_THINKING" not in settings["env"]
    assert "MAX_THINKING_TOKENS" not in settings["env"]
    assert "DISABLE_INTERLEAVED_THINKING" not in settings["env"]
    assert "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS" not in settings["env"]


def test_config_compiler_can_disable_subagents(tmp_path: Path) -> None:
    options = ClaudeCodeBackendOptions.model_validate(
        {"subagents": {"enabled": False}}
    )
    compiler = ClaudeCodeConfigCompiler(
        options=options,
        settings_path=tmp_path / "settings.json",
    )

    settings = compiler.build_settings()

    assert settings["permissions"]["allow"] == EXPECTED_CLAUDE_CODE_PERMISSION_ALLOW
    assert settings["permissions"]["deny"] == ["Agent"]


def test_config_compiler_can_enable_subagents(tmp_path: Path) -> None:
    options = ClaudeCodeBackendOptions.model_validate(
        {"subagents": {"enabled": True}}
    )
    compiler = ClaudeCodeConfigCompiler(
        options=options,
        settings_path=tmp_path / "settings.json",
    )

    settings = compiler.build_settings()

    assert settings["permissions"]["allow"] == EXPECTED_CLAUDE_CODE_PERMISSION_ALLOW
    assert settings["permissions"]["deny"] == []


def test_config_compiler_can_disable_thinking_and_interleaving(tmp_path: Path) -> None:
    options = ClaudeCodeBackendOptions.model_validate(
        {
            "thinking": {
                "enabled": False,
                "interleaved": False,
                "budget_tokens": None,
            }
        }
    )
    compiler = ClaudeCodeConfigCompiler(
        options=options,
        settings_path=tmp_path / "settings.json",
    )

    env = compiler.build_settings()["env"]

    assert env["CLAUDE_CODE_DISABLE_THINKING"] == "1"
    assert env["MAX_THINKING_TOKENS"] == "0"
    assert env["DISABLE_INTERLEAVED_THINKING"] == "1"


def test_config_compiler_builds_secret_env_and_model_capabilities(tmp_path: Path) -> None:
    options = ClaudeCodeBackendOptions.model_validate(
        {
            "model": {
                "id": "proxy-model",
                "name": "Qwen via Dressage",
                "supported_capabilities": ["thinking", "interleaved_thinking"],
            },
            "gateway": {"auth_token": "token-test"},
        }
    )
    compiler = ClaudeCodeConfigCompiler(
        options=options,
        settings_path=tmp_path / "settings.json",
    )

    env = compiler.build_env(4567, tmp_path)

    assert env["HOME"] == str(tmp_path / "home")
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "home" / ".claude")
    assert env["CLAUDE_CODE_TMPDIR"] == str(tmp_path / "tmp")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4567"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "token-test"
    assert env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "proxy-model"
    assert env["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"] == "Qwen via Dressage"
    assert env["ANTHROPIC_CUSTOM_MODEL_OPTION_SUPPORTED_CAPABILITIES"] == (
        "thinking,interleaved_thinking"
    )


def test_config_compiler_sanitizes_inherited_claude_auth_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = {
        "ANTHROPIC_API_KEY": "parent-api-key",
        "ANTHROPIC_AUTH_TOKEN": "parent-auth-token",
        "ANTHROPIC_BASE_URL": "https://parent.example",
        "ANTHROPIC_CUSTOM_HEADERS": "X-Parent: yes",
        "CLAUDE_CODE_OAUTH_TOKEN": "parent-oauth",
        "CLAUDE_CONFIG_DIR": "/tmp/parent-claude",
        "CLAUDE_CODE_TMPDIR": "/tmp/parent-tmp",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    options = ClaudeCodeBackendOptions.model_validate(
        {"gateway": {"auth_token": "compiled-token"}}
    )
    compiler = ClaudeCodeConfigCompiler(
        options=options,
        settings_path=tmp_path / "settings.json",
    )

    env = compiler.build_env(4567, tmp_path)

    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_CUSTOM_HEADERS" not in env
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert env["ANTHROPIC_AUTH_TOKEN"] == "compiled-token"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4567"
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "home" / ".claude")
    assert env["CLAUDE_CODE_TMPDIR"] == str(tmp_path / "tmp")
    assert env["PATH"] == "/usr/local/bin:/usr/bin"


def test_config_compiler_builds_cli_args_for_session_and_resume(tmp_path: Path) -> None:
    prompt_path = tmp_path / "system.txt"
    prompt_path.write_text("system prompt", encoding="utf-8")
    options = ClaudeCodeBackendOptions.model_validate(
        {
            "executable": "/bin/claude",
            "model": {"id": "proxy-model", "name": "Dressage Proxy"},
            "max_turns": 3,
            "permission_mode": "bypassPermissions",
        }
    )
    compiler = ClaudeCodeConfigCompiler(
        options=options,
        settings_path=tmp_path / "settings.json",
        system_prompt_path=prompt_path,
    )

    first = compiler.build_cli_args(
        "hello",
        ClaudeCodeSessionState(backend_session_id="00000000-0000-0000-0000-000000000001", resume=False),
    )
    resumed = compiler.build_cli_args(
        "hello again",
        ClaudeCodeSessionState(backend_session_id="00000000-0000-0000-0000-000000000001", resume=True),
    )

    assert first[:3] == ["/bin/claude", "-p", "hello"]
    assert "--output-format" in first
    assert "stream-json" in first
    assert "--verbose" in first
    assert first[first.index("--model") + 1] == "proxy-model"
    assert first[first.index("--max-turns") + 1] == "3"
    assert first[first.index("--settings") + 1] == str(tmp_path / "settings.json")
    assert "--session-id" in first
    assert "--append-system-prompt" in first
    assert first[first.index("--append-system-prompt") + 1] == "system prompt"
    assert "--resume" in resumed
    assert "--session-id" not in resumed


def test_run_claude_turn_uses_large_stdout_stream_limit(
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

        def __init__(self, lines: list[bytes]) -> None:
            self.stdout = FakeStdout(lines)

        async def wait(self) -> int:
            return 0

    captured_kwargs: dict[str, object] = {}
    assistant_event = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
        },
    }
    result_event = {"type": "result", "is_error": False, "num_turns": 1}
    stdout_lines = [
        (json.dumps(assistant_event) + "\n").encode("utf-8"),
        (json.dumps(result_event) + "\n").encode("utf-8"),
    ]

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        _ = args
        captured_kwargs.update(kwargs)
        return FakeProcess(stdout_lines.copy())

    async def run_test() -> None:
        binding_context = _make_binding_context(tmp_path)
        runtime_dir = Path(binding_context.binding.runtime_dir)
        options = ClaudeCodeBackendOptions.model_validate({"executable": "/bin/claude"})
        adapter = ClaudeCodeAdapter()
        adapter._binding_context = binding_context
        adapter._options = options
        adapter._compiler = ClaudeCodeConfigCompiler(
            options=options,
            settings_path=runtime_dir / "home" / ".claude" / "settings.json",
        )
        adapter._proxy_port = 4567

        monkeypatch.setattr(
            claude_code_module.asyncio,
            "create_subprocess_exec",
            fake_create_subprocess_exec,
        )
        try:
            outputs, _trace_events, usage = await adapter._run_claude_turn(
                user_text="hello",
                turn_context=TurnContext(
                    turn_id="turn-1",
                    request_fingerprint="fp-turn-1",
                    deadline_seconds=30.0,
                ),
                session_state=ClaudeCodeSessionState(
                    backend_session_id="00000000-0000-0000-0000-000000000001",
                    resume=False,
                ),
            )
        finally:
            await adapter.shutdown()

        assert captured_kwargs["limit"] == _CLAUDE_CODE_STDOUT_STREAM_LIMIT
        assert outputs[0].content == "done"
        assert usage.steps == 1

    asyncio.run(run_test())


def test_stdout_stream_limit_reads_jsonl_line_over_asyncio_default() -> None:
    async def run_test() -> None:
        code = """
import json
import sys

event = {
    "type": "assistant",
    "message": {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "x" * (70 * 1024)},
            {"type": "text", "text": "done"},
        ],
    },
}
sys.stdout.write(json.dumps(event) + "\\n")
sys.stdout.flush()
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_CLAUDE_CODE_STDOUT_STREAM_LIMIT,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        raw_line = await process.stdout.readline()
        stderr = await process.stderr.read()
        returncode = await process.wait()

        assert returncode == 0, stderr.decode("utf-8", errors="replace")
        assert len(raw_line) > 65536
        event = json.loads(raw_line.decode("utf-8"))
        assert event["message"]["content"][0]["type"] == "thinking"
        assert event["message"]["content"][1]["text"] == "done"

    asyncio.run(run_test())


def test_config_compiler_missing_prompt_file_raises_protocol_error(tmp_path: Path) -> None:
    compiler = ClaudeCodeConfigCompiler(
        options=ClaudeCodeBackendOptions(),
        settings_path=tmp_path / "settings.json",
        system_prompt_path=tmp_path / "missing.txt",
    )

    with pytest.raises(BackendProtocolError):
        compiler.build_cli_args(
            "hello",
            ClaudeCodeSessionState(backend_session_id="00000000-0000-0000-0000-000000000001", resume=False),
        )


def test_root_rejects_bypass_permissions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("blackbox_server.adapters.claude_code.os.geteuid", lambda: 0)
    options = ClaudeCodeBackendOptions.model_validate({"permission_mode": "bypassPermissions"})

    with pytest.raises(BackendProtocolError, match="cannot run as root"):
        _validate_claude_code_runtime_options(options)


def test_root_allows_default_permission_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("blackbox_server.adapters.claude_code.os.geteuid", lambda: 0)

    _validate_claude_code_runtime_options(ClaudeCodeBackendOptions())


def test_known_claude_code_stderr_remediation() -> None:
    assert "pass --verbose" in (
        _claude_code_stderr_remediation(
            "Error: When using --print, --output-format=stream-json requires --verbose"
        )
        or ""
    )
    assert "non-root" in (
        _claude_code_stderr_remediation(
            "--dangerously-skip-permissions cannot be used with root/sudo privileges"
        )
        or ""
    )


def test_claude_code_exit_message_includes_stdout_tail() -> None:
    message = _build_claude_code_exit_error_message(
        1,
        stderr_tail=None,
        stdout_tail='{"type":"system","message":"startup"}\nplain failure',
        events=[],
    )

    assert "claude_code exited with code 1" in message
    assert "stdout tail:" in message
    assert "plain failure" in message


def test_claude_code_exit_message_includes_stream_json_result_error() -> None:
    message = _build_claude_code_exit_error_message(
        1,
        stderr_tail="stderr detail",
        stdout_tail=None,
        events=[{"type": "result", "is_error": True, "result": "upstream exploded"}],
    )

    assert "stream-json error:" in message
    assert "upstream exploded" in message
    assert "stderr tail: stderr detail" in message


def test_claude_code_stream_error_summary_extracts_nested_error_fields() -> None:
    summary = _claude_code_stream_error_summary(
        [
            {"type": "assistant", "message": {"error": {"type": "api_error", "message": "bad gateway"}}},
            {"type": "event", "stderr": "transport closed"},
        ]
    )

    assert summary is not None
    assert "bad gateway" in summary
    assert "transport closed" in summary


def test_claude_code_adapter_appends_failed_upstream_payload_path() -> None:
    class FakeProxy:
        async def consume_failed_upstream_error(self) -> dict[str, object]:
            return {
                "message": "Upstream returned HTTP 500: Internal Server Error",
                "request_path": "/tmp/runtime/logs/upstream_request.turn-001.0.json",
                "response_path": "/tmp/runtime/logs/upstream_response.turn-001.0.json",
            }

    async def run_test() -> None:
        adapter = ClaudeCodeAdapter()
        adapter._proxy = FakeProxy()  # type: ignore[assignment]
        with pytest.raises(BackendTransportError) as exc_info:
            await adapter._raise_with_proxy_failed_upstream(BackendTransportError("claude_code exited with code 1"))
        message = str(exc_info.value)
        assert "claude_code exited with code 1" in message
        assert "Internal Server Error" in message
        assert "failed upstream payload: /tmp/runtime/logs/upstream_request.turn-001.0.json" in message
        assert "failed upstream response: /tmp/runtime/logs/upstream_response.turn-001.0.json" in message

    asyncio.run(run_test())


def test_convert_claude_code_stream_events_extracts_reasoning_text_tool_and_usage() -> None:
    events = [
        {"type": "system", "subtype": "init", "session_id": "sess"},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "think"},
                    {"type": "text", "text": "visible"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"cmd": "ls"}},
                ],
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        },
        {"type": "result", "is_error": False, "num_turns": 2},
    ]

    outputs, trace_events, usage = convert_claude_code_stream_events("turn-1", events)

    assert outputs[0].role == "assistant"
    assert outputs[0].reasoning_content == "think"
    assert outputs[0].content == "visible"
    assert outputs[0].tool_calls[0].id == "toolu_1"
    assert json.loads(outputs[0].tool_calls[0].function.arguments) == {"cmd": "ls"}
    assert [event.source for event in trace_events] == ["claude_code", "claude_code", "claude_code"]
    assert usage.input_tokens == 2
    assert usage.output_tokens == 3
    assert usage.steps == 2


def test_convert_claude_code_stream_events_raises_on_result_error() -> None:
    with pytest.raises(BackendTransportError, match="boom"):
        convert_claude_code_stream_events(
            "turn-1",
            [{"type": "result", "is_error": True, "result": "boom"}],
        )
