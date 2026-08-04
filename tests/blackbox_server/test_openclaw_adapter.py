from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

import blackbox_server.adapters.openclaw as openclaw_module
from blackbox_server.adapters.base import (
    BackendContextOverflowError,
    BackendMaxStepsExceededError,
    BackendProtocolError,
    BackendTransportError,
)
from blackbox_server.adapters.openclaw import (
    OpenClawAdapter,
    OpenClawBackendOptions,
    _max_steps_error_from_http_status,
    _proxy_context_overflow_error_from_http_status,
    _raise_if_openclaw_max_steps_exceeded,
    convert_openclaw_chat_completion,
)
from blackbox_server.config import BlackboxServerConfig
from blackbox_server.core.models import (
    BindingContext,
    BindingInfo,
    Message,
    ProxyOptions,
    RuntimeSystemPrompt,
    SessionContext,
    SessionState,
    TurnContext,
    utcnow,
)


class FakeProxy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.max_steps_payload: dict[str, Any] | None = None
        self.rollout_invalidated_payload: dict[str, Any] | None = None
        self.max_steps_event = asyncio.Event()

    async def open_turn(self, turn_id: str, backend_session_id: str | None = None) -> None:
        self.calls.append(("open", (turn_id, backend_session_id)))

    async def drain_turn(self, timeout: float | None = None) -> None:
        self.calls.append(("drain", timeout))

    async def consume_context_overflow_error(self) -> dict[str, Any] | None:
        self.calls.append(("consume_context_overflow_error", None))
        return None

    async def consume_rollout_invalidated_error(self) -> dict[str, Any] | None:
        self.calls.append(("consume_rollout_invalidated_error", None))
        if self.rollout_invalidated_payload is None:
            return None
        payload = dict(self.rollout_invalidated_payload)
        self.rollout_invalidated_payload = None
        return payload

    async def wait_for_max_steps_error(self, timeout: float | None = None) -> dict[str, Any] | None:
        if self.max_steps_payload is not None:
            return dict(self.max_steps_payload)
        try:
            if timeout is None:
                await self.max_steps_event.wait()
            else:
                await asyncio.wait_for(self.max_steps_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if self.max_steps_payload is None:
            return None
        return dict(self.max_steps_payload)

    async def consume_max_steps_error(self) -> dict[str, Any] | None:
        if self.max_steps_payload is None:
            return None
        payload = dict(self.max_steps_payload)
        self.max_steps_payload = None
        self.max_steps_event.clear()
        return payload

    def trigger_max_steps_error(self, payload: dict[str, Any]) -> None:
        self.max_steps_payload = payload
        self.max_steps_event.set()

    def trigger_rollout_invalidated_error(self, payload: dict[str, Any]) -> None:
        self.rollout_invalidated_payload = payload

    async def clear_turn(self) -> None:
        self.calls.append(("clear", None))


class FailedUpstreamProxy(FakeProxy):
    async def consume_failed_upstream_error(self) -> dict[str, Any]:
        self.calls.append(("consume_failed_upstream_error", None))
        return {
            "message": "Upstream returned HTTP 503: unavailable",
            "response_path": "/tmp/upstream-response.json",
        }


def _make_binding_context(
    tmp_path: Path,
    *,
    system_prompt: RuntimeSystemPrompt | None = None,
    backend_options: dict[str, Any] | None = None,
) -> BindingContext:
    return BindingContext(
        binding=BindingInfo(
            runtime_id="bbs-test",
            blackbox_type="openclaw",
            router_raw="http://127.0.0.1:30000",
            router_base_url="http://127.0.0.1:30000/v1",
            router_api_path="/v1",
            bound_session_id="sess-001",
            bound_instance_id="inst-001",
            system_prompt=system_prompt,
            runtime_dir=str(tmp_path / "runtime"),
            registered_at=utcnow(),
            backend_options=backend_options or _minimal_backend_options(),
        ),
        effective_config=BlackboxServerConfig(router_timeout=300000),
    )


def _make_session_context(**overrides: Any) -> SessionContext:
    defaults: dict[str, Any] = {
        "session_id": "sess-001",
        "state": SessionState.ACTIVE,
        "blackbox_type": "openclaw",
        "backend_session_id": None,
        "router_base_url": "http://127.0.0.1:30000/v1",
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "metadata": {},
    }
    defaults.update(overrides)
    return SessionContext(**defaults)


def _make_turn_context(turn_id: str = "turn-1", deadline: float = 30.0) -> TurnContext:
    return TurnContext(
        turn_id=turn_id,
        request_fingerprint=f"fp-{turn_id}",
        deadline_seconds=deadline,
    )


def _minimal_backend_options() -> dict[str, Any]:
    return {
        "agent_id": "default",
        "provider_id": "sglang",
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "model_name": "Qwen2.5 32B via SGLang Router",
        "context_window": 32768,
        "max_tokens": 8192,
        "api_key": "sglang-local",
        "proxy": {"sticky_header_name": "X-SMG-Routing-Key"},
    }


def _minimal_options() -> OpenClawBackendOptions:
    return OpenClawBackendOptions.model_validate(_minimal_backend_options())


def test_parse_openclaw_options_rejects_responses_endpoint_field() -> None:
    adapter = OpenClawAdapter()
    options = {**_minimal_backend_options(), "endpoint": "responses"}

    with pytest.raises(BackendProtocolError, match="endpoint"):
        adapter._parse_options(options)


def test_parse_openclaw_options_rejects_unknown_compaction_field() -> None:
    adapter = OpenClawAdapter()
    options = {
        **_minimal_backend_options(),
        "compaction": {
            "mode": "default",
        },
    }

    with pytest.raises(BackendProtocolError, match="mode"):
        adapter._parse_options(options)


def test_build_openclaw_config_uses_proxy_chat_completions_only(tmp_path: Path) -> None:
    adapter = OpenClawAdapter()
    adapter._proxy_port = 4567
    adapter._gateway_token = "token-test"
    binding_context = _make_binding_context(tmp_path)

    config = adapter._build_openclaw_config(binding_context, _minimal_options())
    expected_permission = {"*": "allow", "question": "deny", "doom_loop": "deny"}

    assert config["permission"] == expected_permission
    assert config["gateway"]["mode"] == "local"
    assert config["gateway"]["auth"] == {"mode": "token", "token": "token-test"}
    assert config["gateway"]["http"]["endpoints"] == {
        "chatCompletions": {"enabled": True}
    }
    assert "responses" not in config["gateway"]["http"]["endpoints"]

    provider = config["models"]["providers"]["sglang"]
    assert provider["baseUrl"] == "http://127.0.0.1:4567/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "sglang-local"
    assert provider["models"][0]["id"] == "Qwen/Qwen2.5-32B-Instruct"
    assert "contextWindow" not in provider["models"][0]

    assert config["agents"]["defaults"]["model"]["primary"] == "sglang/Qwen/Qwen2.5-32B-Instruct"
    assert config["agents"]["defaults"]["workspace"].endswith(".openclaw/workspace")
    assert config["agents"]["defaults"]["permission"] == expected_permission
    assert config["agents"]["defaults"]["compaction"] == {
        "mode": "safeguard",
        "midTurnPrecheck": {"enabled": True},
        "truncateAfterCompaction": True,
        "notifyUser": False,
    }
    assert "model" not in config["agents"]["defaults"]["compaction"]


def test_build_openclaw_config_can_disable_permission_for_legacy_openclaw(
    tmp_path: Path,
) -> None:
    adapter = OpenClawAdapter()
    adapter._proxy_port = 4567
    adapter._gateway_token = "token-test"
    binding_context = _make_binding_context(tmp_path)

    config = adapter._build_openclaw_config(
        binding_context,
        _minimal_options(),
        permission_mode="none",
    )

    assert "permission" not in config
    assert "permission" not in config["agents"]["defaults"]


def test_start_proxy_passes_limits_and_default_temperature_to_rollout_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeRolloutLLMProxy:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

            async def app(scope, receive, send):
                return None

            self.app = app

    class FakeBackgroundUvicornServer:
        def __init__(self, config: object) -> None:
            self.config = config

        async def serve(self) -> None:
            return None

    async def no_wait_for_proxy() -> None:
        return None

    adapter = OpenClawAdapter()
    binding_context = _make_binding_context(tmp_path)
    (Path(binding_context.binding.runtime_dir) / "run").mkdir(parents=True)
    options = OpenClawBackendOptions.model_validate(
        {
            **_minimal_backend_options(),
            "proxy": {
                "sticky_header_name": "X-SMG-Routing-Key",
                "max_steps": 7,
                "default_temperature": 0.25,
            },
        }
    )

    monkeypatch.setattr(openclaw_module, "RolloutLLMProxy", FakeRolloutLLMProxy)
    monkeypatch.setattr(
        openclaw_module,
        "_BackgroundUvicornServer",
        FakeBackgroundUvicornServer,
    )
    monkeypatch.setattr(adapter, "_find_free_port", lambda: 4567)
    monkeypatch.setattr(adapter, "_wait_for_proxy", no_wait_for_proxy)

    asyncio.run(adapter._start_proxy(binding_context, options))

    assert captured["upstream_origin"] == "http://127.0.0.1:30000"
    assert captured["router_api_path"] == "/v1"
    assert captured["bound_session_id"] == "sess-001"
    assert captured["bound_instance_id"] == "inst-001"
    assert captured["sticky_header_name"] == "X-SMG-Routing-Key"
    assert captured["max_steps"] == 7
    assert captured["default_temperature"] == 0.25


def test_build_openclaw_config_maps_compaction_options(tmp_path: Path) -> None:
    adapter = OpenClawAdapter()
    adapter._proxy_port = 4567
    adapter._gateway_token = "token-test"
    backend_options = {
        **_minimal_backend_options(),
        "compaction": {
            "timeout_seconds": 900,
            "reserve_tokens": 8192,
            "reserve_tokens_floor": 8192,
            "keep_recent_tokens": 16000,
            "notify_user": True,
            "max_active_transcript_bytes": "20mb",
            "post_compaction_sections": ["Session Startup", "Red Lines"],
            "model": "sglang/Qwen/Qwen2.5-32B-Instruct",
        },
    }
    binding_context = _make_binding_context(tmp_path, backend_options=backend_options)

    config = adapter._build_openclaw_config(
        binding_context,
        OpenClawBackendOptions.model_validate(backend_options),
    )

    assert config["agents"]["defaults"]["compaction"] == {
        "mode": "safeguard",
        "midTurnPrecheck": {"enabled": True},
        "truncateAfterCompaction": True,
        "notifyUser": True,
        "timeoutSeconds": 900,
        "reserveTokens": 8192,
        "reserveTokensFloor": 8192,
        "keepRecentTokens": 16000,
        "maxActiveTranscriptBytes": "20mb",
        "postCompactionSections": ["Session Startup", "Red Lines"],
        "model": "sglang/Qwen/Qwen2.5-32B-Instruct",
    }


def test_prepare_workspace_copies_system_prompt_to_agents_md(tmp_path: Path) -> None:
    prompt = tmp_path / "system_prompt.txt"
    prompt.write_text("You are a coding agent.", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    system_prompt = RuntimeSystemPrompt(
        source_file=str(prompt),
        runtime_file=str(runtime_dir / "home" / ".openclaw" / "workspace" / "AGENTS.md"),
        applies_to="openclaw",
    )
    binding_context = _make_binding_context(tmp_path, system_prompt=system_prompt)
    adapter = OpenClawAdapter()

    adapter._prepare_runtime_dirs(runtime_dir)
    adapter._prepare_workspace(binding_context)

    workspace = runtime_dir / "home" / ".openclaw" / "workspace"
    agents_text = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "You are a coding agent." in agents_text
    assert "Do not ask the user questions during this rollout." in agents_text
    assert "blackbox-server-deny-question" in agents_text
    for name in ("SOUL.md", "TOOLS.md", "USER.md", "IDENTITY.md"):
        assert (workspace / name).exists()



def test_openclaw_usage_over_context_window_is_not_adapter_overflow() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "backend reported an over-budget turn",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 40000,
            "completion_tokens": 8,
            "total_tokens": 40008,
        },
    }

    outputs, _, usage = convert_openclaw_chat_completion("turn-1", raw)

    assert outputs[0].content == "backend reported an over-budget turn"
    assert usage.input_tokens == 40000


def test_openclaw_structured_context_code_is_not_adapter_overflow() -> None:
    raw = {
        "error": {"code": "context_length_exceeded", "message": "too large"},
        "usage": {"prompt_tokens": 1000},
    }

    _raise_if_openclaw_max_steps_exceeded(raw, _minimal_options())


def test_openclaw_context_overflow_detected_from_proxy_http_status() -> None:
    request = httpx.Request("POST", "http://openclaw.test/v1/chat/completions")
    response = httpx.Response(
        413,
        json={
            "error": "context_overflow",
            "message": "Dressage proxy context window overflow.",
            "details": {
                "phase": "input_output",
                "context_window": 8,
                "input_tokens": 6,
                "output_tokens": 3,
                "total_tokens": 9,
                "max_tokens": 4,
                "last_proxy_step_recorded": True,
            },
        },
        request=request,
    )
    exc = httpx.HTTPStatusError("context overflow", request=request, response=response)

    typed = _proxy_context_overflow_error_from_http_status(exc)

    assert isinstance(typed, BackendContextOverflowError)
    assert typed.context_window == 8
    assert typed.input_tokens == 6
    assert typed.max_tokens == 4
    assert typed.raw_error_code == "context_overflow"
    assert typed.details()["phase"] == "input_output"


def test_openclaw_max_steps_detected_from_structured_code() -> None:
    options = OpenClawBackendOptions.model_validate(
        {
            **_minimal_backend_options(),
            "proxy": {"sticky_header_name": "X-SMG-Routing-Key", "max_steps": 2},
        }
    )
    raw = {
        "error": {
            "code": "max_steps_exceeded",
            "message": "Turn exceeded max_steps.",
            "details": {"max_steps": 2, "attempted_step": 3},
        },
    }

    with pytest.raises(BackendMaxStepsExceededError) as exc_info:
        _raise_if_openclaw_max_steps_exceeded(raw, options)

    assert exc_info.value.max_steps == 2
    assert exc_info.value.attempted_step == 3
    assert exc_info.value.backend_message == "Turn exceeded max_steps."
    assert exc_info.value.raw_error_code == "max_steps_exceeded"


def test_openclaw_max_steps_detected_from_http_status() -> None:
    options = OpenClawBackendOptions.model_validate(
        {
            **_minimal_backend_options(),
            "proxy": {"sticky_header_name": "X-SMG-Routing-Key", "max_steps": 2},
        }
    )
    request = httpx.Request("POST", "http://openclaw.test/v1/chat/completions")
    response = httpx.Response(
        429,
        json={
            "error": {
                "code": "max_steps_exceeded",
                "message": "Turn exceeded max_steps.",
                "details": {"max_steps": 2, "attempted_step": 3},
            }
        },
        request=request,
    )
    exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

    typed = _max_steps_error_from_http_status(exc, options)

    assert typed is not None
    assert typed.max_steps == 2
    assert typed.attempted_step == 3
    assert typed.backend_message == "Turn exceeded max_steps."
    assert typed.raw_error_code == "max_steps_exceeded"


def test_openclaw_max_steps_detected_from_gateway_rate_limit_payload() -> None:
    options = OpenClawBackendOptions.model_validate(
        {
            **_minimal_backend_options(),
            "proxy": {"sticky_header_name": "X-SMG-Routing-Key", "max_steps": 1},
        }
    )
    request = httpx.Request("POST", "http://openclaw.test/v1/chat/completions")
    response = httpx.Response(
        429,
        json={
            "error": {
                "message": "429 Turn exceeded max_steps.",
                "type": "rate_limit_error",
            }
        },
        request=request,
    )
    exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

    typed = _max_steps_error_from_http_status(exc, options)

    assert typed is not None
    assert typed.max_steps == 1
    assert typed.attempted_step == 1
    assert typed.backend_message == "429 Turn exceeded max_steps."
    assert typed.raw_error_code == "rate_limit_error"


def test_openclaw_max_steps_detected_from_wrapped_http_status_text() -> None:
    options = OpenClawBackendOptions.model_validate(
        {
            **_minimal_backend_options(),
            "proxy": {"sticky_header_name": "X-SMG-Routing-Key", "max_steps": 1},
        }
    )
    request = httpx.Request("POST", "http://openclaw.test/v1/chat/completions")
    response = httpx.Response(
        500,
        text=(
            "OpenClaw provider failed after upstream 429 "
            "max_steps_exceeded: Turn exceeded max_steps."
        ),
        request=request,
    )
    exc = httpx.HTTPStatusError("provider failed", request=request, response=response)

    typed = _max_steps_error_from_http_status(exc, options)

    assert typed is not None
    assert typed.max_steps == 1
    assert typed.attempted_step == 1
    assert typed.raw_error_code == "max_steps_exceeded"


def test_convert_openclaw_chat_completion_text_usage_and_tool_calls() -> None:
    raw = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "thinking",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": {"cmd": "ls"},
                            },
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }

    outputs, trace_events, usage = convert_openclaw_chat_completion("turn-1", raw)

    assert len(outputs) == 1
    assert outputs[0].role == "assistant"
    assert outputs[0].content == "ok"
    assert outputs[0].reasoning_content == "thinking"
    assert outputs[0].tool_calls[0].function.name == "bash"
    assert outputs[0].tool_calls[0].function.arguments == json.dumps({"cmd": "ls"}, ensure_ascii=False)
    assert trace_events[0].source == "openclaw"
    assert trace_events[0].event_type == "chat_completion"
    assert usage.input_tokens == 3
    assert usage.output_tokens == 4
    assert usage.total_tokens == 7
    assert usage.reasoning_tokens == 2
    assert usage.tool_calls == 1
    assert usage.steps == 1


def test_convert_openclaw_chat_completion_rejects_missing_choices() -> None:
    with pytest.raises(BackendProtocolError, match="no choices"):
        convert_openclaw_chat_completion("turn-1", {"usage": {}})


def test_send_message_posts_chat_completion_payload_and_headers(tmp_path: Path) -> None:
    adapter = OpenClawAdapter()
    adapter._binding_context = _make_binding_context(tmp_path)
    adapter._options = _minimal_options()
    adapter._gateway_token = "token-test"
    adapter._proxy = FakeProxy()
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = dict(request.headers)
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello back",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://openclaw.test",
        )
        session = _make_session_context()
        with patch.object(adapter, "health", return_value=True):
            response = await adapter.send_message(
                session,
                _make_turn_context("turn-1"),
                [Message(role="user", content="hello")],
            )
        await adapter._client.aclose()

        assert response.backend_session_id == "bbs:inst-001:sess-001"
        assert session.backend_session_id == "bbs:inst-001:sess-001"
        assert response.outputs[0].content == "hello back"

    asyncio.run(run_test())

    assert seen["path"] == "/v1/chat/completions"
    payload = seen["payload"]
    assert payload["model"] == "openclaw/default"
    assert payload["user"] == "bbs:inst-001:sess-001"
    assert payload["stream"] is False
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert "input" not in payload
    assert "instructions" not in payload

    headers = seen["headers"]
    assert headers["authorization"] == "Bearer token-test"
    assert headers["x-openclaw-model"] == "sglang/Qwen/Qwen2.5-32B-Instruct"
    assert headers["x-openclaw-session-key"] == "bbs:inst-001:sess-001"
    assert headers["x-bbs-turn-id"] == "turn-1"

    assert adapter._proxy.calls[0] == ("open", ("turn-1", "bbs:inst-001:sess-001"))
    assert adapter._proxy.calls[1][0] == "drain"
    assert isinstance(adapter._proxy.calls[1][1], float)
    assert adapter._proxy.calls[2] == ("consume_context_overflow_error", None)
    assert adapter._proxy.calls[3] == ("consume_rollout_invalidated_error", None)
    assert adapter._proxy.calls[4] == ("clear", None)


def test_send_message_raises_proxy_rollout_invalidated_error(tmp_path: Path) -> None:
    adapter = OpenClawAdapter()
    adapter._binding_context = _make_binding_context(tmp_path)
    adapter._options = _minimal_options()
    adapter._gateway_token = "token-test"
    adapter._proxy = FakeProxy()

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        adapter._proxy.trigger_rollout_invalidated_error(
            {
                "error": "partial_rollout_staleness_exceeded",
                "message": "Partial rollout model version span exceeded limit.",
            }
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [{"message": {"role": "assistant", "content": ""}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://openclaw.test",
        )
        session = _make_session_context()
        with patch.object(adapter, "health", return_value=True):
            with pytest.raises(
                BackendTransportError,
                match="partial_rollout_staleness_exceeded",
            ):
                await adapter.send_message(
                    session,
                    _make_turn_context("turn-1"),
                    [Message(role="user", content="hello")],
                )
        await adapter._client.aclose()

    asyncio.run(run_test())

    assert ("consume_rollout_invalidated_error", None) in adapter._proxy.calls
    assert ("clear", None) in adapter._proxy.calls


def test_send_message_preserves_proxy_failure_when_backend_raises(
    tmp_path: Path,
) -> None:
    adapter = OpenClawAdapter()
    adapter._binding_context = _make_binding_context(tmp_path)
    adapter._options = _minimal_options()
    adapter._proxy = FailedUpstreamProxy()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(
                adapter,
                "_post_chat_completions",
                side_effect=BackendProtocolError("invalid streamed response"),
            ),
        ):
            with pytest.raises(BackendTransportError) as exc_info:
                await adapter.send_message(
                    _make_session_context(),
                    _make_turn_context("turn-1"),
                    [Message(role="user", content="hello")],
                )

        message = str(exc_info.value)
        assert "invalid streamed response" in message
        assert "Upstream returned HTTP 503" in message
        assert "failed upstream response: /tmp/upstream-response.json" in message

    asyncio.run(run_test())

    calls = adapter._proxy.calls
    assert calls[0] == ("open", ("turn-1", "bbs:inst-001:sess-001"))
    assert calls[1] == ("drain", 2.0)
    assert calls[2] == ("consume_context_overflow_error", None)
    assert calls[3] == ("consume_rollout_invalidated_error", None)
    assert calls[4] == ("consume_failed_upstream_error", None)
    assert calls[5] == ("clear", None)


def test_send_message_cancels_openclaw_chat_on_proxy_max_steps(tmp_path: Path) -> None:
    adapter = OpenClawAdapter()
    adapter._binding_context = _make_binding_context(tmp_path)
    adapter._options = _minimal_options()
    adapter._gateway_token = "token-test"
    adapter._proxy = FakeProxy()
    post_started = asyncio.Event()
    post_cancelled = asyncio.Event()
    payload = {
        "error": "max_steps_exceeded",
        "message": "Turn exceeded max_steps.",
        "details": {
            "session_id": "sess-001",
            "turn_id": "turn-1",
            "max_steps": 2,
            "attempted_step": 2,
            "backend_message": "429 Turn exceeded max_steps.",
            "raw_error_code": "rate_limit_error",
        },
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        post_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            post_cancelled.set()
            raise

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://openclaw.test",
        )
        session = _make_session_context()
        with patch.object(adapter, "health", return_value=True):
            send_task = asyncio.create_task(
                adapter.send_message(
                    session,
                    _make_turn_context("turn-1"),
                    [Message(role="user", content="hello")],
                )
            )
            await asyncio.wait_for(post_started.wait(), timeout=1.0)
            assert adapter._active_chat_task is not None
            assert isinstance(adapter._proxy, FakeProxy)
            adapter._proxy.trigger_max_steps_error(payload)
            with pytest.raises(BackendMaxStepsExceededError) as exc_info:
                await asyncio.wait_for(send_task, timeout=1.0)
        await adapter._client.aclose()

        assert exc_info.value.max_steps == 2
        assert exc_info.value.attempted_step == 2
        assert exc_info.value.backend_message == "429 Turn exceeded max_steps."
        assert exc_info.value.raw_error_code == "rate_limit_error"
        assert post_cancelled.is_set()
        assert adapter._active_chat_task is None
        assert ("clear", None) in adapter._proxy.calls

    asyncio.run(run_test())


def test_abort_session_cancels_active_chat_task() -> None:
    adapter = OpenClawAdapter()
    adapter._proxy = FakeProxy()

    async def never() -> httpx.Response:
        await asyncio.sleep(3600)
        return httpx.Response(200)

    async def run_test() -> None:
        task = asyncio.create_task(never())
        adapter._active_chat_task = task
        session = _make_session_context()

        assert await adapter.abort_session(session) is True
        assert task.cancelled()
        assert adapter._proxy.calls == [("clear", None)]

        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run_test())


def test_abort_session_noop_without_active_task() -> None:
    adapter = OpenClawAdapter()

    async def run_test() -> None:
        assert await adapter.abort_session(_make_session_context()) is False

    asyncio.run(run_test())
