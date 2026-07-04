from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

from blackbox_server.adapters.base import (
    BackendAdapter,
    BackendContextOverflowError,
    BackendMaxStepsExceededError,
)
from blackbox_server.adapters.openclaw import OpenClawAdapter
from blackbox_server.adapters.opencode import OpencodeAdapter
from blackbox_server.app import create_app
from blackbox_server.config import BlackboxServerConfig, ServerConfigOverride
from blackbox_server.core.hashing import binding_request_fingerprint
from blackbox_server.core.models import (
    AdapterResponse,
    BackendCapabilities,
    ExecuteCmdResult,
    Message,
    RegisterRequest,
    SessionContext,
    TraceEvent,
    TurnContext,
    TurnUsage,
    utcnow,
)


class FakeAdapter(BackendAdapter):
    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self, binding_context) -> None:
        self.initialized = True

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        session_context.backend_session_id = session_context.backend_session_id or "oc-session-1"
        content = new_messages[0].content or ""
        return AdapterResponse(
            outputs=[
                Message(
                    role="assistant",
                    content=f"echo: {content}",
                    reasoning_content=f"thought: {content}",
                )
            ],
            trace_events=[
                TraceEvent(
                    turn_id=turn_context.turn_id,
                    seq=1,
                    source="fake",
                    event_type="reasoning",
                    payload={"text": "fake trace"},
                    created_at=utcnow(),
                )
            ],
            usage=TurnUsage(total_tokens=10, input_tokens=4, output_tokens=6, steps=1),
            backend_session_id=session_context.backend_session_id,
        )

    async def abort_session(self, session_context: SessionContext) -> bool:
        return True

    async def health(self) -> bool:
        return True

    async def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            chat=True,
            abort=True,
            stream=False,
            multi_message_input=False,
            system_message=True,
            history_injection=False,
        )

    async def shutdown(self) -> None:
        self.shutdown_called = True


class ContextOverflowAdapter(FakeAdapter):
    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        raise BackendContextOverflowError(
            "openclaw context overflow: input_tokens exceeds configured context_window",
            context_window=32768,
            input_tokens=40000,
            max_tokens=8192,
            raw_error_code="context_length_exceeded",
        )


class MaxStepsExceededAdapter(FakeAdapter):
    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        raise BackendMaxStepsExceededError(
            "Turn exceeded max_steps.",
            max_steps=1,
            attempted_step=1,
        )


class SlowAdapter(FakeAdapter):
    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        await asyncio.sleep(0.2)
        return await super().send_message(session_context, turn_context, new_messages)


class FlakyHealthAdapter(FakeAdapter):
    def __init__(self, failures_before_success: int) -> None:
        super().__init__()
        self.failures_before_success = failures_before_success
        self.health_calls = 0

    async def health(self) -> bool:
        self.health_calls += 1
        return self.health_calls > self.failures_before_success


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "build.txt"
    path.write_text("system prompt", encoding="utf-8")
    return path


def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter: BackendAdapter) -> TestClient:
    monkeypatch.setattr("blackbox_server.core.server.create_adapter", lambda _: adapter)
    config = BlackboxServerConfig(runtime_root=str(tmp_path / "runtime"), backend_timeout=0.05)
    return TestClient(create_app(config))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def test_http_requests_are_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    caplog.set_level(logging.INFO, logger="blackbox_server.http")

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert any(
        record.name == "blackbox_server.http"
        and "request completed" in record.message
        and "method=GET" in record.message
        and "path=/health" in record.message
        and "status_code=200" in record.message
        for record in caplog.records
    )


def register_payload(
    prompt_file: Path,
    *,
    bound_session_id: str = "sess-001",
    bound_instance_id: str = "inst-001",
    sticky_header_name: str | None = None,
) -> dict:
    payload = {
        "blackbox_type": "opencode",
        "router": "127.0.0.1:30000",
        "bound_session_id": bound_session_id,
        "bound_instance_id": bound_instance_id,
        "system_prompt_file": str(prompt_file),
        "backend_options": {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
        },
    }
    if sticky_header_name is not None:
        payload["backend_options"]["proxy"] = {"sticky_header_name": sticky_header_name}
    return payload


def openclaw_register_payload(
    prompt_file: Path,
    *,
    bound_session_id: str = "sess-001",
    bound_instance_id: str = "inst-001",
) -> dict:
    return {
        "blackbox_type": "openclaw",
        "router": "127.0.0.1:30000",
        "router_api_path": "/v1",
        "bound_session_id": bound_session_id,
        "bound_instance_id": bound_instance_id,
        "system_prompt_file": str(prompt_file),
        "backend_options": {
            "agent_id": "default",
            "provider_id": "sglang",
            "model_id": "Qwen/Qwen2.5-32B-Instruct",
            "model_name": "Qwen2.5 32B via SGLang Router",
            "context_window": 32768,
            "max_tokens": 8192,
            "api_key": "sglang-local",
            "request": {"max_tokens": 4096},
            "proxy": {},
        },
    }


def claude_code_register_payload(
    prompt_file: Path,
    *,
    bound_session_id: str = "sess-001",
    bound_instance_id: str = "inst-001",
) -> dict:
    return {
        "blackbox_type": "claude_code",
        "router": "127.0.0.1:30000",
        "router_api_path": "/v1",
        "bound_session_id": bound_session_id,
        "bound_instance_id": bound_instance_id,
        "system_prompt_file": str(prompt_file),
        "backend_options": {
            "model": {
                "id": "proxy-model",
                "name": "Dressage Proxy",
                "supported_capabilities": [
                    "thinking",
                    "adaptive_thinking",
                    "interleaved_thinking",
                ],
            },
            "proxy": {},
        },
    }


def test_http_request_and_response_bodies_are_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
    caplog: pytest.LogCaptureFixture,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    caplog.set_level(logging.INFO, logger="blackbox_server.http")

    with client:
        response = client.post("/v1/rollout/register", json=register_payload(prompt_file))

    assert response.status_code == 200
    log_output = "\n".join(
        record.message for record in caplog.records if record.name == "blackbox_server.http"
    )
    assert "request_body=" in log_output
    assert '"blackbox_type":"opencode"' in log_output
    assert "response_body=" in log_output
    assert '"status":"ready"' in log_output


def test_register_runtime_root_override_creates_runtime_under_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_file: Path
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    runtime_root = tmp_path / "register-runtime"

    with client:
        register_response = client.post(
            "/v1/rollout/register",
            json={
                **register_payload(prompt_file),
                "server_config": {
                    "runtime_root": str(runtime_root),
                    "backend_timeout": 0.05,
                },
            },
        )

        assert register_response.status_code == 200
        register_data = register_response.json()
        runtime_dir = Path(register_data["binding"]["runtime_dir"])
        assert register_data["config"]["runtime_root"] == str(runtime_root)
        assert runtime_dir.parent == runtime_root
        assert runtime_dir.is_dir()


def test_server_config_override_runtime_root_applies_and_affects_fingerprint():
    runtime_root = "/workspace_sandbox/blackbox_server_runtime"
    override = ServerConfigOverride(runtime_root=runtime_root)

    assert override.apply(BlackboxServerConfig()).runtime_root == runtime_root
    assert override.explicit_values()["runtime_root"] == runtime_root

    base_request = RegisterRequest(
        blackbox_type="opencode",
        router="127.0.0.1:30000",
        bound_session_id="sess-001",
        bound_instance_id="inst-001",
    )
    override_request = base_request.model_copy(update={"server_config": override})

    assert binding_request_fingerprint(base_request, "http://127.0.0.1:30000") != (
        binding_request_fingerprint(override_request, "http://127.0.0.1:30000")
    )


def test_status_lists_blackbox_backends_as_implemented(tmp_path: Path):
    client = TestClient(create_app(BlackboxServerConfig(runtime_root=str(tmp_path / "runtime"))))

    with client:
        response = client.get("/v1/status")

    assert response.status_code == 200
    data = response.json()
    assert "openclaw" in data["implemented_backends"]
    assert "openclaw" in data["known_backends"]
    assert "claude_code" in data["implemented_backends"]
    assert "claude_code" in data["known_backends"]


def test_claude_code_register_uses_prompt_runtime_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())

    with client:
        response = client.post("/v1/rollout/register", json=claude_code_register_payload(prompt_file))

    assert response.status_code == 200
    data = response.json()
    assert data["binding"]["blackbox_type"] == "claude_code"
    system_prompt = data["binding"]["system_prompt"]
    assert system_prompt["runtime_file"].endswith("/prompts/system.txt")
    assert system_prompt["applies_to"] == "claude_code"


def test_claude_code_message_validation_rejects_non_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())

    with client:
        register_response = client.post("/v1/rollout/register", json=claude_code_register_payload(prompt_file))
        assert register_response.status_code == 200
        response = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-bad", "messages": [{"role": "assistant", "content": "bad"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "request_error"
    assert "claude_code phase 1 only accepts a user message" in response.json()["message"]


def test_openclaw_register_uses_agents_workspace_prompt_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())

    with client:
        response = client.post("/v1/rollout/register", json=openclaw_register_payload(prompt_file))

    assert response.status_code == 200
    data = response.json()
    assert data["binding"]["blackbox_type"] == "openclaw"
    system_prompt = data["binding"]["system_prompt"]
    assert system_prompt["runtime_file"].endswith("/home/.openclaw/workspace/AGENTS.md")
    assert system_prompt["applies_to"] == "openclaw"


def test_openclaw_message_validation_rejects_non_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())

    with client:
        register_response = client.post("/v1/rollout/register", json=openclaw_register_payload(prompt_file))
        assert register_response.status_code == 200
        response = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-bad", "messages": [{"role": "assistant", "content": "bad"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "request_error"
    assert "openclaw phase 1 only accepts a user message" in response.json()["message"]


def test_openclaw_message_validation_rejects_empty_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())

    with client:
        register_response = client.post("/v1/rollout/register", json=openclaw_register_payload(prompt_file))
        assert register_response.status_code == 200
        response = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-empty", "messages": [{"role": "user", "content": "  "}]},
        )

    assert response.status_code == 400
    assert "openclaw phase 1 requires non-empty user content" in response.json()["message"]


def test_register_rejects_invalid_openclaw_backend_options_as_request_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, OpenClawAdapter())
    payload = openclaw_register_payload(prompt_file)
    payload["backend_options"]["endpoint"] = "responses"

    with client:
        response = client.post("/v1/rollout/register", json=payload)
        status_response = client.get("/v1/status")

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "request_error"
    assert "Invalid openclaw backend_options" in data["message"]
    assert "endpoint" in data["message"]
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "idle"


def test_register_send_message_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_file: Path):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200
        register_data = register_response.json()
        assert register_data["status"] == "ready"
        assert register_data["binding"]["blackbox_type"] == "opencode"
        assert register_data["binding"]["runtime_id"].startswith("bbs-")
        assert "instance_id" not in register_data["binding"]
        assert register_data["binding"]["bound_session_id"] == "sess-001"
        assert register_data["binding"]["bound_instance_id"] == "inst-001"
        assert register_data["capabilities"]["multi_message_input"] is False

        message_payload = {
            "turn_id": "turn-1",
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"source": "test"},
        }
        message_response = client.post("/v1/sessions/sess-001/messages", json=message_payload)
        assert message_response.status_code == 200
        message_data = message_response.json()
        assert message_data["session_id"] == "sess-001"
        assert message_data["instance_id"] == "inst-001"
        assert message_data["idempotent_replay"] is False
        assert message_data["outputs"][0]["content"] == "echo: hello"
        assert message_data["outputs"][0]["reasoning_content"] == "thought: hello"
        assert message_data["backend"]["backend_session_id"] == "oc-session-1"

        replay_response = client.post("/v1/sessions/sess-001/messages", json=message_payload)
        assert replay_response.status_code == 200
        replay_data = replay_response.json()
        assert replay_data["instance_id"] == "inst-001"
        assert replay_data["idempotent_replay"] is True
        assert replay_data["outputs"][0]["content"] == "echo: hello"
        assert replay_data["outputs"][0]["reasoning_content"] == "thought: hello"

        session_response = client.get(
            "/v1/sessions/sess-001",
            params={"include_history": "true", "include_trace": "true", "include_turns": "true"},
        )
        assert session_response.status_code == 200
        session_data = session_response.json()
        assert session_data["instance_id"] == "inst-001"
        assert session_data["state"] == "active"
        assert session_data["turn_count"] == 1
        assert len(session_data["conversation_history"]) == 3
        assert session_data["conversation_history"][2]["content"] == "echo: hello"
        assert session_data["conversation_history"][2]["reasoning_content"] == "thought: hello"
        ledger_output = session_data["turn_ledger"]["turn-1"]["response"]["outputs"][0]
        assert ledger_output["reasoning_content"] == "thought: hello"
        assert session_data["turn_ledger"]["turn-1"]["status"] == "committed"

        abort_response = client.post("/v1/sessions/sess-001/abort")
        assert abort_response.status_code == 200
        abort_data = abort_response.json()
        assert abort_data["state"] == "aborted"
        assert abort_data["instance_id"] == "inst-001"



def test_context_overflow_returns_413_without_marking_server_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, ContextOverflowAdapter())

    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-overflow",
                "messages": [{"role": "user", "content": "too long"}],
            },
        )
        status_response = client.get("/v1/status")
        session_response = client.get(
            "/v1/sessions/sess-001",
            params={"include_turns": "true"},
        )
        execute_response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "printf harvested"},
        )
        next_message_response = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-after-overflow",
                "messages": [{"role": "user", "content": "continue"}],
            },
        )

    assert response.status_code == 413
    body = response.json()
    assert body["error"] == "context_overflow"
    assert body["details"]["input_tokens"] == 40000
    assert body["details"]["context_window"] == 32768
    assert status_response.json()["status"] == "ready"
    assert session_response.json()["state"] == "desynced"
    turn = session_response.json()["turn_ledger"]["turn-overflow"]
    assert turn["status"] == "unknown"
    assert turn["error"]["error"] == "context_overflow"
    assert execute_response.status_code == 200
    assert execute_response.json()["stdout"] == "harvested"
    assert next_message_response.status_code == 409
    assert "desynced" in next_message_response.json()["message"]


def test_max_steps_exceeded_returns_429_without_marking_server_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, MaxStepsExceededAdapter())

    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-max-steps",
                "messages": [{"role": "user", "content": "keep going"}],
            },
        )
        status_response = client.get("/v1/status")
        session_response = client.get(
            "/v1/sessions/sess-001",
            params={"include_turns": "true"},
        )
        execute_response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "printf harvested"},
        )

    assert response.status_code == 429
    body = response.json()
    assert body["error"] == "max_steps_exceeded"
    assert body["message"] == "Turn exceeded max_steps."
    assert body["details"]["max_steps"] == 1
    assert body["details"]["attempted_step"] == 1
    assert body["details"]["session_id"] == "sess-001"
    assert body["details"]["turn_id"] == "turn-max-steps"
    assert status_response.json()["status"] == "ready"
    assert session_response.json()["state"] == "desynced"
    turn = session_response.json()["turn_ledger"]["turn-max-steps"]
    assert turn["status"] == "unknown"
    assert turn["error"]["error"] == "max_steps_exceeded"
    assert execute_response.status_code == 200
    assert execute_response.json()["stdout"] == "harvested"


def test_execute_cmd_success_and_default_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "  printf ok  "},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess-001"
    assert data["instance_id"] == "inst-001"
    assert data["cmd"] == "printf ok"
    assert data["stdout"] == "ok"
    assert data["returncode"] == 0
    assert data["timed_out"] is False


def test_execute_cmd_nonzero_exit_returns_command_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "printf err >&2 && exit 7"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["stdout"] == ""
    assert data["stderr"] == "err"
    assert data["returncode"] == 7
    assert data["timed_out"] is False


def test_execute_cmd_timeout_returns_command_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={
                "cmd": f"'{sys.executable}' -c 'import time; time.sleep(30)'",
                "timeout": 0.05,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["timed_out"] is True
    assert data["returncode"] is not None


def test_execute_cmd_explicit_timeout_overrides_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    calls: list[tuple[str, float]] = []

    async def fake_execute_shell_command(cmd: str, *, timeout: float, **kwargs) -> ExecuteCmdResult:
        del kwargs
        calls.append((cmd, timeout))
        now = utcnow()
        return ExecuteCmdResult(
            cmd=cmd,
            stdout="ok",
            stderr="",
            returncode=0,
            timed_out=False,
            duration_seconds=0.001,
            started_at=now,
            finished_at=now,
        )

    monkeypatch.setattr("blackbox_server.core.server.execute_shell_command", fake_execute_shell_command)
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post(
            "/v1/rollout/register",
            json={
                **register_payload(prompt_file),
                "server_config": {"execute_cmd_timeout": 42.0},
            },
        )
        assert register_response.status_code == 200

        default_response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "echo default"},
        )
        explicit_response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "echo explicit", "timeout": 3.5},
        )

    assert default_response.status_code == 200
    assert explicit_response.status_code == 200
    assert calls == [
        ("echo default", 42.0),
        ("echo explicit", 3.5),
    ]


@pytest.mark.parametrize("cmd", ["", "   ", "echo one\necho two", "echo one\recho two", "echo \x00"])
def test_execute_cmd_rejects_invalid_cmd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
    cmd: str,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post("/v1/sessions/sess-001/execute_cmd", json={"cmd": cmd})

    assert response.status_code == 400
    assert response.json()["error"] == "request_error"


def test_execute_cmd_rejects_invalid_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "echo nope", "timeout": 0},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "request_error"


def test_execute_cmd_rejects_aborted_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200
        abort_response = client.post("/v1/sessions/sess-001/abort")
        assert abort_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "echo nope"},
        )

    assert response.status_code == 409
    assert "cannot execute commands" in response.json()["message"]


def test_execute_cmd_rejects_desynced_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, SlowAdapter())
    with client:
        register_response = client.post(
            "/v1/rollout/register",
            json={
                **register_payload(prompt_file),
                "server_config": {"backend_timeout": 0.01},
            },
        )
        assert register_response.status_code == 200
        message_response = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-timeout", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert message_response.status_code == 504

        response = client.post(
            "/v1/sessions/sess-001/execute_cmd",
            json={"cmd": "echo nope"},
        )

    assert response.status_code == 409
    assert "desynced" in response.json()["message"]


def test_execute_cmd_enforces_bound_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        response = client.post(
            "/v1/sessions/sess-999/execute_cmd",
            json={"cmd": "echo nope"},
        )

    assert response.status_code == 409
    assert response.json()["error"] == "bound_session_mismatch"


def test_timeout_marks_session_desynced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_file: Path):
    client = make_client(tmp_path, monkeypatch, SlowAdapter())
    with client:
        register_response = client.post(
            "/v1/rollout/register",
            json={
                **register_payload(prompt_file),
                "server_config": {"backend_timeout": 0.01},
            },
        )
        assert register_response.status_code == 200

        message_response = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-timeout",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert message_response.status_code == 504

        session_response = client.get(
            "/v1/sessions/sess-001",
            params={"include_turns": "true"},
        )
        assert session_response.status_code == 200
        session_data = session_response.json()
        assert session_data["state"] == "desynced"
        assert session_data["turn_ledger"]["turn-timeout"]["status"] == "unknown"

        conflict_response = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-next",
                "messages": [{"role": "user", "content": "retry"}],
            },
        )
        assert conflict_response.status_code == 409


def test_message_fingerprint_includes_reasoning_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        payload = {
            "turn_id": "turn-reasoning-fp",
            "messages": [{"role": "user", "content": "hello", "reasoning_content": "first"}],
        }
        first = client.post("/v1/sessions/sess-001/messages", json=payload)
        assert first.status_code == 200
        assert first.json()["idempotent_replay"] is False

        replay = client.post("/v1/sessions/sess-001/messages", json=payload)
        assert replay.status_code == 200
        assert replay.json()["idempotent_replay"] is True

        changed_reasoning = {
            "turn_id": "turn-reasoning-fp",
            "messages": [{"role": "user", "content": "hello", "reasoning_content": "second"}],
        }
        conflict = client.post("/v1/sessions/sess-001/messages", json=changed_reasoning)
        assert conflict.status_code == 409


def test_rebind_rejected_when_active_session_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        first_register = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert first_register.status_code == 200
        message_response = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-1", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert message_response.status_code == 200

        second_register = client.post(
            "/v1/rollout/register",
            json={**register_payload(prompt_file), "router": "127.0.0.1:30001"},
        )
        assert second_register.status_code == 409


def test_proxy_bound_session_is_precreated_and_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post(
            "/v1/rollout/register",
            json=register_payload(
                prompt_file,
                bound_session_id="sess-001",
                bound_instance_id="inst-001",
            ),
        )
        assert register_response.status_code == 200
        register_data = register_response.json()
        assert register_data["binding"]["bound_session_id"] == "sess-001"
        assert register_data["binding"]["bound_instance_id"] == "inst-001"

        session_response = client.get("/v1/sessions/sess-001", params={"include_history": "true"})
        assert session_response.status_code == 200
        session_data = session_response.json()
        assert session_data["session_id"] == "sess-001"
        assert session_data["instance_id"] == "inst-001"
        assert session_data["state"] == "active"
        assert session_data["message_count"] == 1

        wrong_session_response = client.post(
            "/v1/sessions/sess-999/messages",
            json={"turn_id": "turn-1", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert wrong_session_response.status_code == 409
        assert wrong_session_response.json()["error"] == "bound_session_mismatch"

        ok_message = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-1", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert ok_message.status_code == 200
        assert ok_message.json()["instance_id"] == "inst-001"

        status_response = client.get("/v1/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["sessions"]["total_count"] == 1
        assert status_data["sessions"]["active_count"] == 1


def test_register_rejects_invalid_bound_session_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        payload_without_session = register_payload(prompt_file)
        payload_without_session.pop("bound_session_id")
        missing_bound_session = client.post(
            "/v1/rollout/register",
            json=payload_without_session,
        )
        assert missing_bound_session.status_code == 400

        payload_without_instance = register_payload(prompt_file)
        payload_without_instance.pop("bound_instance_id")
        missing_bound_instance = client.post(
            "/v1/rollout/register",
            json=payload_without_instance,
        )
        assert missing_bound_instance.status_code == 400

        removed_proxy_field = client.post(
            "/v1/rollout/register",
            json={
                **register_payload(prompt_file),
                "backend_options": {
                    **register_payload(prompt_file)["backend_options"],
                    "proxy": {"enabled": False},
                },
            },
        )
        assert removed_proxy_field.status_code == 400


def test_register_rejects_invalid_opencode_backend_options_as_request_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, OpencodeAdapter())
    payload = register_payload(prompt_file)
    payload["backend_options"]["model_limit"] = {"context": 200000}

    with client:
        response = client.post("/v1/rollout/register", json=payload)
        status_response = client.get("/v1/status")

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "request_error"
    assert "Invalid opencode backend_options" in data["message"]
    assert "model_limit" in data["message"]
    assert "output" in data["message"]
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "idle"


def test_get_session_no_longer_returns_logprob_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        session_response = client.get("/v1/sessions/sess-001", params={"include_logprobs": "true"})
        assert session_response.status_code == 200
        assert "logprob_records" not in session_response.json()


def test_omitted_turn_id_uses_single_turn_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        first = client.post(
            "/v1/sessions/sess-001/messages",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert first.status_code == 200
        first_data = first.json()
        generated_turn_id = first_data["turn_id"]
        assert generated_turn_id.startswith("turn-")
        assert first_data["idempotent_replay"] is False

        replay = client.post(
            "/v1/sessions/sess-001/messages",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert replay.status_code == 200
        replay_data = replay.json()
        assert replay_data["turn_id"] == generated_turn_id
        assert replay_data["idempotent_replay"] is True

        different_body = client.post(
            "/v1/sessions/sess-001/messages",
            json={"messages": [{"role": "user", "content": "different"}]},
        )
        assert different_body.status_code == 409

        new_turn = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-2", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert new_turn.status_code == 409

        session = client.get("/v1/sessions/sess-001", params={"include_turns": "true"})
        assert session.status_code == 200
        assert list(session.json()["turn_ledger"].keys()) == [generated_turn_id]


def test_explicit_turn_id_mode_requires_explicit_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        register_response = client.post("/v1/rollout/register", json=register_payload(prompt_file))
        assert register_response.status_code == 200

        first = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-1", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert first.status_code == 200

        second = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-2", "messages": [{"role": "user", "content": "next"}]},
        )
        assert second.status_code == 200
        assert second.json()["turn_id"] == "turn-2"

        omitted = client.post(
            "/v1/sessions/sess-001/messages",
            json={"messages": [{"role": "user", "content": "missing"}]},
        )
        assert omitted.status_code == 400


def test_main_process_exits_cleanly_on_sigint(tmp_path: Path):
    port = _find_free_port()
    runtime_root = tmp_path / "runtime"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "BBS_HOST": "127.0.0.1",
            "BBS_PORT": str(port),
            "BBS_RUNTIME_ROOT": str(runtime_root),
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "blackbox_server.main"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        startup_deadline = time.time() + 10
        while time.time() < startup_deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.fail("blackbox_server.main did not become healthy in time")

        assert proc.poll() is None
        os.kill(proc.pid, signal.SIGINT)

        exit_deadline = time.time() + 5
        while time.time() < exit_deadline and proc.poll() is None:
            time.sleep(0.1)

        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()


def test_transient_health_failure_is_retried_before_returning_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt_file: Path,
):
    adapter = FlakyHealthAdapter(failures_before_success=2)
    client = make_client(tmp_path, monkeypatch, adapter)

    with client:
        register_response = client.post(
            "/v1/rollout/register",
            json={
                **register_payload(prompt_file),
                "server_config": {
                    "backend_timeout": 0.05,
                    "runtime_health_check_interval": 999.0,
                    "runtime_health_check_retries": 3,
                    "runtime_health_check_retry_delay": 0.0,
                },
            },
        )
        assert register_response.status_code == 200

        message_response = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-transient-health",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert message_response.status_code == 200
        assert message_response.json()["outputs"][0]["content"] == "echo: hello"
        assert adapter.health_calls == 3
