from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import uvicorn

import blackbox_server.adapters.opencode as opencode_module
from blackbox_server.adapters.base import (
    BackendContextOverflowError,
    BackendMaxStepsExceededError,
    BackendProcessError,
    BackendProtocolError,
    BackendTransportError,
)
from blackbox_server.adapters.opencode import (
    _OC_MSG_COUNT_KEY,
    _BackgroundUvicornServer,
    OpencodeAdapter,
    OpencodeBackendOptions,
    convert_opencode_messages,
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


def test_convert_opencode_messages_separates_reasoning_and_tools():
    oc_messages = [
        {
            "info": {"role": "assistant"},
            "parts": [
                {"type": "reasoning", "text": "need a command"},
                {"type": "text", "text": "I will inspect the logs."},
                {
                    "type": "tool",
                    "callID": "call_1",
                    "tool": "bash",
                    "state": {
                        "input": {"command": "cat /tmp/log.txt"},
                        "output": "log output",
                    },
                },
                {
                    "type": "step-finish",
                    "tokens": {"total": 5, "input": 2, "output": 3, "reasoning": 1},
                },
            ],
        }
    ]

    outputs, trace_events, usage = convert_opencode_messages("turn-1", oc_messages)

    assert outputs[0].role == "assistant"
    assert outputs[0].content == "I will inspect the logs."
    assert outputs[0].reasoning_content == "need a command"
    assert "<think>" not in outputs[0].content
    assert "</think>" not in outputs[0].content
    assert outputs[0].tool_calls[0].function.name == "bash"
    assert outputs[1].role == "tool"
    assert outputs[1].content == "log output"
    assert len(trace_events) == 3
    assert trace_events[0].payload == {"type": "reasoning", "text": "need a command"}
    assert usage.total_tokens == 5
    assert usage.reasoning_tokens == 1
    assert usage.tool_calls == 1


def test_convert_opencode_messages_joins_multiple_reasoning_parts():
    outputs, trace_events, usage = convert_opencode_messages(
        "turn-1",
        [
            {
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "reasoning", "text": "first"},
                    {"type": "reasoning", "text": "second"},
                    {"type": "text", "text": "visible"},
                    {"type": "step-finish", "tokens": {"total": 2, "input": 1, "output": 1}},
                ],
            }
        ],
    )

    assert len(outputs) == 1
    assert outputs[0].content == "visible"
    assert outputs[0].reasoning_content == "first\nsecond"
    assert [event.event_type for event in trace_events] == [
        "reasoning",
        "reasoning",
        "step-finish",
    ]
    assert usage.steps == 1


def test_convert_opencode_messages_keeps_reasoning_only_output():
    outputs, trace_events, usage = convert_opencode_messages(
        "turn-1",
        [
            {
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "reasoning", "text": "only thought"},
                    {"type": "step-finish", "tokens": {"total": 1, "reasoning": 1}},
                ],
            }
        ],
    )

    assert len(outputs) == 1
    assert outputs[0].role == "assistant"
    assert outputs[0].content is None
    assert outputs[0].reasoning_content == "only thought"
    assert [event.event_type for event in trace_events] == ["reasoning", "step-finish"]
    assert usage.reasoning_tokens == 1


def test_convert_opencode_messages_ignores_step_only_output():
    outputs, trace_events, usage = convert_opencode_messages(
        "turn-1",
        [_oc_assistant_msg_step_only()],
    )

    assert outputs == []
    assert [event.event_type for event in trace_events] == ["step-start", "step-finish"]
    assert usage.steps == 1


def test_proxy_options_defaults_to_reasonable_max_steps_with_explicit_opt_out():
    assert ProxyOptions().max_steps == 100
    assert ProxyOptions(max_steps=None).max_steps is None


def _make_session_context(**overrides: Any) -> SessionContext:
    defaults: dict[str, Any] = {
        "session_id": "test-session",
        "state": SessionState.ACTIVE,
        "blackbox_type": "opencode",
        "backend_session_id": "oc-session-1",
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


def _oc_user_msg(text: str) -> dict[str, Any]:
    return {
        "info": {"role": "user"},
        "parts": [{"type": "text", "text": text}],
    }


def _oc_assistant_msg_with_tool() -> dict[str, Any]:
    return {
        "info": {"role": "assistant"},
        "parts": [
            {"type": "reasoning", "text": "thinking..."},
            {"type": "text", "text": "Let me check."},
            {
                "type": "tool",
                "callID": "call_abc",
                "tool": "bash",
                "state": {"input": {"command": "ls"}, "output": "file.txt"},
            },
            {"type": "step-finish", "tokens": {"total": 10, "input": 4, "output": 6}},
            {"type": "text", "text": "Done."},
        ],
    }


def _oc_assistant_msg_simple(text: str = "Hello!") -> dict[str, Any]:
    return {
        "info": {"role": "assistant"},
        "parts": [
            {"type": "text", "text": text},
            {"type": "step-finish", "tokens": {"total": 3, "input": 1, "output": 2}},
        ],
    }


def _oc_assistant_msg_step_only() -> dict[str, Any]:
    return {
        "info": {"role": "assistant"},
        "parts": [
            {"type": "step-start", "id": "step_1"},
            {"type": "step-finish", "tokens": {"total": 0, "input": 0, "output": 0}},
        ],
    }


class FakeProxy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.max_steps_payload: dict[str, Any] | None = None
        self.max_steps_event = asyncio.Event()

    async def open_turn(self, turn_id: str, backend_session_id: str | None = None) -> None:
        self.calls.append(("open", (turn_id, backend_session_id)))

    async def update_turn_backend_session(self, backend_session_id: str) -> None:
        self.calls.append(("update", backend_session_id))

    async def drain_turn(self, timeout: float | None = None) -> None:
        self.calls.append(("drain", timeout))

    async def consume_context_overflow_error(self) -> dict[str, Any] | None:
        return None

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

    async def clear_turn(self) -> None:
        self.calls.append(("clear", None))


class FailedUpstreamProxy(FakeProxy):
    async def consume_context_overflow_error(self) -> None:
        self.calls.append(("consume_context_overflow_error", None))
        return None

    async def consume_rollout_invalidated_error(self) -> None:
        self.calls.append(("consume_rollout_invalidated_error", None))
        return None

    async def consume_failed_upstream_error(self) -> dict[str, Any]:
        self.calls.append(("consume_failed_upstream_error", None))
        return {
            "message": "Upstream returned HTTP 503: unavailable",
            "request_path": "/tmp/upstream-request.json",
        }


def _make_binding_context(
    *,
    system_prompt: RuntimeSystemPrompt | None = None,
    runtime_dir: str = "/tmp/runtime",
    **backend_options: Any,
) -> BindingContext:
    return BindingContext(
        binding=BindingInfo(
            runtime_id="bbs-test",
            blackbox_type="opencode",
            router_raw="http://127.0.0.1:30000",
            router_base_url="http://127.0.0.1:30000/v1",
            router_api_path="/v1",
            bound_session_id="sess-001",
            bound_instance_id="inst-001",
            system_prompt=system_prompt,
            runtime_dir=runtime_dir,
            registered_at=utcnow(),
            backend_options=backend_options,
        ),
        effective_config=BlackboxServerConfig(router_timeout=300000),
    )


def test_send_message_uses_get_for_full_trace():
    adapter = OpencodeAdapter()

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        assert timeout > 0
        if method == "POST":
            return {"message": "ignored"}
        if method == "GET":
            return [_oc_user_msg("hello"), _oc_assistant_msg_with_tool()]
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()
    turn = _make_turn_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            resp = await adapter.send_message(session, turn, [Message(role="user", content="hello")])

        assert resp.outputs[0].role == "assistant"
        assert resp.outputs[0].content == "Let me check.\nDone."
        assert resp.outputs[0].reasoning_content == "thinking..."
        assert resp.outputs[0].tool_calls[0].function.name == "bash"
        assert resp.outputs[1].role == "tool"
        assert resp.outputs[1].content == "file.txt"
        assert any(event.event_type == "reasoning" for event in resp.trace_events)
        assert any(event.event_type == "tool" for event in resp.trace_events)
        assert resp.usage.tool_calls == 1
        assert session.metadata[_OC_MSG_COUNT_KEY] == 2

    asyncio.run(run_test())


def test_send_message_drains_and_clears_proxy_scope():
    adapter = OpencodeAdapter()
    adapter._proxy = FakeProxy()

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        if method == "POST":
            return {}
        return [_oc_user_msg("hello"), _oc_assistant_msg_simple("Hi!")]

    session = _make_session_context(backend_session_id=None)

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context("turn-1"),
                [Message(role="user", content="hello")],
            )

        assert resp.backend_session_id == "oc-session-1"
        assert adapter._proxy.calls[0] == ("open", ("turn-1", None))
        assert adapter._proxy.calls[1] == ("update", "oc-session-1")
        assert adapter._proxy.calls[2][0] == "drain"
        assert isinstance(adapter._proxy.calls[2][1], float)
        assert adapter._proxy.calls[3] == ("clear", None)

    asyncio.run(run_test())


def test_send_message_best_effort_drains_proxy_on_failure():
    adapter = OpencodeAdapter()
    adapter._proxy = FakeProxy()
    session = _make_session_context(backend_session_id=None)

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_ensure_backend_session", side_effect=BackendTransportError("boom")),
        ):
            with pytest.raises(BackendTransportError, match="boom"):
                await adapter.send_message(
                    session,
                    _make_turn_context("turn-1"),
                    [Message(role="user", content="hello")],
                )

        assert adapter._proxy.calls == [
            ("open", ("turn-1", None)),
            ("drain", 2.0),
            ("clear", None),
        ]

    asyncio.run(run_test())


def test_send_message_preserves_proxy_failure_when_backend_raises():
    adapter = OpencodeAdapter()
    adapter._proxy = FailedUpstreamProxy()
    session = _make_session_context(backend_session_id=None)

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(
                adapter,
                "_ensure_backend_session",
                return_value="oc-session-1",
            ),
            patch.object(
                adapter,
                "_post_message_with_connect_retry",
                side_effect=BackendProtocolError("invalid streamed response"),
            ),
        ):
            with pytest.raises(BackendTransportError) as exc_info:
                await adapter.send_message(
                    session,
                    _make_turn_context("turn-1"),
                    [Message(role="user", content="hello")],
                )

        message = str(exc_info.value)
        assert "invalid streamed response" in message
        assert "Upstream returned HTTP 503" in message
        assert "failed upstream payload: /tmp/upstream-request.json" in message

    asyncio.run(run_test())

    assert adapter._proxy.calls == [
        ("open", ("turn-1", None)),
        ("update", "oc-session-1"),
        ("drain", 2.0),
        ("consume_context_overflow_error", None),
        ("consume_rollout_invalidated_error", None),
        ("consume_failed_upstream_error", None),
        ("clear", None),
    ]


def test_send_message_aborts_pending_opencode_post_on_proxy_max_steps():
    adapter = OpencodeAdapter()
    adapter._proxy = FakeProxy()
    session = _make_session_context(backend_session_id=None)
    turn = _make_turn_context("turn-1")
    post_started = asyncio.Event()
    post_cancelled = asyncio.Event()
    abort_called = asyncio.Event()
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

    async def never_post(*args: Any, **kwargs: Any) -> Any:
        post_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            post_cancelled.set()
            raise

    async def mock_abort(session_context: SessionContext) -> bool:
        assert session_context is session
        abort_called.set()
        return True

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
            patch.object(adapter, "_post_message_with_connect_retry", side_effect=never_post),
            patch.object(adapter, "abort_session", side_effect=mock_abort),
        ):
            send_task = asyncio.create_task(
                adapter.send_message(
                    session,
                    turn,
                    [Message(role="user", content="hello")],
                )
            )
            await asyncio.wait_for(post_started.wait(), timeout=1.0)
            assert isinstance(adapter._proxy, FakeProxy)
            adapter._proxy.trigger_max_steps_error(payload)
            with pytest.raises(BackendMaxStepsExceededError) as exc_info:
                await asyncio.wait_for(send_task, timeout=1.0)

        assert exc_info.value.max_steps == 2
        assert exc_info.value.attempted_step == 2
        assert exc_info.value.backend_message == "429 Turn exceeded max_steps."
        assert exc_info.value.raw_error_code == "rate_limit_error"
        assert abort_called.is_set()
        assert post_cancelled.is_set()
        assert ("clear", None) in adapter._proxy.calls

    asyncio.run(run_test())


def test_build_opencode_config_uses_proxy_base_url():
    adapter = OpencodeAdapter()
    adapter._proxy_port = 4567
    binding_context = _make_binding_context(
        provider_id="sglang",
        provider_name="Remote SGLang",
        provider_package="@ai-sdk/openai-compatible",
        model_id="qwen35-a3b",
        model_name="Qwen 3.5 35B A3B",
        proxy={},
    )
    options = OpencodeBackendOptions(
        provider_id="sglang",
        provider_name="Remote SGLang",
        provider_package="@ai-sdk/openai-compatible",
        model_id="qwen35-a3b",
        model_name="Qwen 3.5 35B A3B",
        proxy=ProxyOptions(),
    )

    config = adapter._build_opencode_config(binding_context, options)

    provider_options = config["provider"]["sglang"]["options"]
    assert provider_options["baseURL"] == "http://127.0.0.1:4567/v1"
    assert "limit" not in config["provider"]["sglang"]["models"]["qwen35-a3b"]
    assert "compaction" not in config
    assert config["permission"] == {"*": "allow", "question": "deny", "doom_loop": "deny"}
    assert config["agent"] == {"title": {"disable": True}}


def test_start_proxy_passes_default_temperature_to_rollout_proxy(
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

    runtime_dir = tmp_path / "runtime"
    (runtime_dir / "run").mkdir(parents=True)
    adapter = OpencodeAdapter()
    binding_context = _make_binding_context(
        provider_id="sglang",
        provider_name="Remote SGLang",
        provider_package="@ai-sdk/openai-compatible",
        model_id="qwen35-a3b",
        model_name="Qwen 3.5 35B A3B",
        proxy={"default_temperature": 0.25},
        runtime_dir=str(runtime_dir),
    )
    options = OpencodeBackendOptions(
        provider_id="sglang",
        provider_name="Remote SGLang",
        provider_package="@ai-sdk/openai-compatible",
        model_id="qwen35-a3b",
        model_name="Qwen 3.5 35B A3B",
        proxy=ProxyOptions(default_temperature=0.25),
    )

    monkeypatch.setattr(opencode_module, "RolloutLLMProxy", FakeRolloutLLMProxy)
    monkeypatch.setattr(
        opencode_module,
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
    assert captured["default_temperature"] == 0.25


def test_build_opencode_config_includes_model_limit_and_compaction():
    adapter = OpencodeAdapter()
    adapter._proxy_port = 4567
    backend_options = {
        "provider_id": "sglang",
        "provider_name": "Remote SGLang",
        "provider_package": "@ai-sdk/openai-compatible",
        "model_id": "qwen35-a3b",
        "model_name": "Qwen 3.5 35B A3B",
        "model_limit": {"context": 200000, "input": 180000, "output": 65536},
        "compaction": {
            "auto": True,
            "prune": True,
            "tail_turns": 2,
            "preserve_recent_tokens": 12000,
            "reserved": 20000,
        },
        "proxy": {},
    }
    binding_context = _make_binding_context(**backend_options)
    options = adapter._parse_options(backend_options)

    config = adapter._build_opencode_config(binding_context, options)

    model = config["provider"]["sglang"]["models"]["qwen35-a3b"]
    assert model["limit"] == {"context": 200000, "output": 65536, "input": 180000}
    assert config["compaction"] == {
        "auto": True,
        "prune": True,
        "tail_turns": 2,
        "preserve_recent_tokens": 12000,
        "reserved": 20000,
    }


def test_build_opencode_config_preserves_system_prompt_and_title_agent_config():
    adapter = OpencodeAdapter()
    adapter._proxy_port = 4567
    binding_context = _make_binding_context(
        system_prompt=RuntimeSystemPrompt(
            source_file="/tmp/source-build.txt",
            runtime_file="/tmp/runtime/home/.config/opencode/prompts/build.txt",
        ),
        provider_id="sglang",
        provider_name="Remote SGLang",
        provider_package="@ai-sdk/openai-compatible",
        model_id="qwen35-a3b",
        model_name="Qwen 3.5 35B A3B",
        proxy={},
    )
    options = OpencodeBackendOptions(
        provider_id="sglang",
        provider_name="Remote SGLang",
        provider_package="@ai-sdk/openai-compatible",
        model_id="qwen35-a3b",
        model_name="Qwen 3.5 35B A3B",
        proxy=ProxyOptions(),
    )

    config = adapter._build_opencode_config(binding_context, options)

    assert config["agent"]["build"] == {"prompt": "{file:./prompts/build.txt}"}
    assert config["agent"]["title"] == {"disable": True}


def test_parse_opencode_options_rejects_unknown_backend_option():
    adapter = OpencodeAdapter()

    with pytest.raises(BackendProtocolError, match="unexpected"):
        adapter._parse_options(
            {
                "provider_id": "sglang",
                "provider_name": "Remote SGLang",
                "provider_package": "@ai-sdk/openai-compatible",
                "model_id": "qwen35-a3b",
                "model_name": "Qwen 3.5 35B A3B",
                "unexpected": True,
            }
        )


def test_background_uvicorn_server_does_not_install_signal_handlers(monkeypatch: pytest.MonkeyPatch):
    async def app(scope, receive, send):
        return None

    signal_calls: list[tuple[object, object]] = []

    def _record_signal(sig, handler):
        signal_calls.append((sig, handler))
        return signal.SIG_DFL

    monkeypatch.setattr("blackbox_server.adapters.opencode.signal.signal", _record_signal)

    server = _BackgroundUvicornServer(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    with server.capture_signals():
        pass

    assert signal_calls == []


def test_send_message_multi_turn_skips_previous_messages():
    adapter = OpencodeAdapter()
    turn1_messages = [_oc_user_msg("hello"), _oc_assistant_msg_simple("Hi!")]
    turn2_messages = turn1_messages + [_oc_user_msg("bye"), _oc_assistant_msg_simple("Goodbye!")]
    get_calls = {"count": 0}

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        assert timeout > 0
        if method == "POST":
            return {}
        if method == "GET":
            get_calls["count"] += 1
            return turn1_messages if get_calls["count"] == 1 else turn2_messages
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            resp1 = await adapter.send_message(
                session,
                _make_turn_context("turn-1"),
                [Message(role="user", content="hello")],
            )
            resp2 = await adapter.send_message(
                session,
                _make_turn_context("turn-2"),
                [Message(role="user", content="bye")],
            )

        assert len(resp1.outputs) == 1
        assert resp1.outputs[0].content == "Hi!"
        assert len(resp2.outputs) == 1
        assert resp2.outputs[0].content == "Goodbye!"
        assert session.metadata[_OC_MSG_COUNT_KEY] == 4

    asyncio.run(run_test())


def test_send_message_without_tools_keeps_plain_text_output():
    adapter = OpencodeAdapter()

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        assert timeout > 0
        if method == "POST":
            return {}
        if method == "GET":
            return [_oc_user_msg("hello"), _oc_assistant_msg_simple("Hello there!")]
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context(),
                [Message(role="user", content="hello")],
            )

        assert len(resp.outputs) == 1
        assert resp.outputs[0].role == "assistant"
        assert resp.outputs[0].content == "Hello there!"
        assert resp.outputs[0].tool_calls is None
        assert resp.usage.total_tokens == 3

    asyncio.run(run_test())


def test_send_message_detects_message_count_regression():
    adapter = OpencodeAdapter()

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        if method == "POST":
            return {}
        if method == "GET":
            return [_oc_user_msg("hello")]
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context(metadata={_OC_MSG_COUNT_KEY: 5})

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            with pytest.raises(BackendProtocolError, match="went backwards"):
                await adapter.send_message(
                    session,
                    _make_turn_context(),
                    [Message(role="user", content="hello")],
                )

        assert session.metadata[_OC_MSG_COUNT_KEY] == 5

    asyncio.run(run_test())


def test_send_message_retries_get_after_transport_error():
    adapter = OpencodeAdapter()
    attempts = {"get": 0}

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        assert timeout > 0
        if method == "POST":
            return {}
        if method == "GET":
            attempts["get"] += 1
            if attempts["get"] == 1:
                raise BackendTransportError("connection reset")
            return [_oc_user_msg("hi"), _oc_assistant_msg_simple("Hey!")]
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=AsyncMock()),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context(),
                [Message(role="user", content="hi")],
            )

        assert attempts["get"] == 2
        assert len(resp.outputs) == 1
        assert resp.outputs[0].content == "Hey!"

    asyncio.run(run_test())


def test_send_message_retries_when_get_only_returns_new_user_message():
    adapter = OpencodeAdapter()
    attempts = {"get": 0}
    first_turn = [_oc_user_msg("hello"), _oc_assistant_msg_simple("Hi!")]
    second_turn_incomplete = first_turn + [_oc_user_msg("bye")]
    second_turn_complete = second_turn_incomplete + [_oc_assistant_msg_simple("Goodbye!")]

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        assert timeout > 0
        if method == "POST":
            return {}
        if method == "GET":
            attempts["get"] += 1
            return second_turn_incomplete if attempts["get"] == 1 else second_turn_complete
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context(metadata={_OC_MSG_COUNT_KEY: len(first_turn)})

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=AsyncMock()),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context("turn-2"),
                [Message(role="user", content="bye")],
            )

        assert attempts["get"] == 2
        assert len(resp.outputs) == 1
        assert resp.outputs[0].content == "Goodbye!"
        assert session.metadata[_OC_MSG_COUNT_KEY] == len(second_turn_complete)

    asyncio.run(run_test())


def test_send_message_retries_when_get_only_returns_placeholder_assistant_message():
    adapter = OpencodeAdapter()
    attempts = {"get": 0}
    incomplete_turn = [_oc_user_msg("hello"), _oc_assistant_msg_step_only()]
    complete_turn = incomplete_turn + [_oc_assistant_msg_simple("Done!")]

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        assert timeout > 0
        if method == "POST":
            return {}
        if method == "GET":
            attempts["get"] += 1
            return incomplete_turn if attempts["get"] == 1 else complete_turn
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=AsyncMock()),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context("turn-1"),
                [Message(role="user", content="hello")],
            )

        assert attempts["get"] == 2
        assert len(resp.outputs) == 1
        assert resp.outputs[0].content == "Done!"
        assert session.metadata[_OC_MSG_COUNT_KEY] == len(complete_turn)

    asyncio.run(run_test())


def test_send_message_fails_when_get_never_returns_assistant_message():
    adapter = OpencodeAdapter()
    attempts = {"get": 0}
    first_turn = [_oc_user_msg("hello"), _oc_assistant_msg_simple("Hi!")]
    second_turn_incomplete = first_turn + [_oc_user_msg("bye")]

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        assert timeout > 0
        if method == "POST":
            return {}
        if method == "GET":
            attempts["get"] += 1
            return second_turn_incomplete
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context(metadata={_OC_MSG_COUNT_KEY: len(first_turn)})

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=AsyncMock()),
        ):
            with pytest.raises(BackendTransportError, match="no material outputs"):
                await adapter.send_message(
                    session,
                    _make_turn_context("turn-2"),
                    [Message(role="user", content="bye")],
                )

        assert attempts["get"] == 2
        assert session.metadata[_OC_MSG_COUNT_KEY] == len(first_turn)

    asyncio.run(run_test())


def test_send_message_ignores_context_overflow_from_structured_post_payload():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "model_limit": {"context": 32768, "input": 24576, "output": 8192},
        }
    )

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        if method == "POST":
            return {
                "error": {
                    "code": "context_length_exceeded",
                    "message": "prompt is too large",
                },
                "usage": {"prompt_tokens": 40000},
            }
        if method == "GET":
            return [_oc_user_msg("hello"), _oc_assistant_msg_simple("still handled")]
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context("turn-1"),
                [Message(role="user", content="hello")],
            )

        assert resp.outputs[0].content == "still handled"

    asyncio.run(run_test())


def test_send_message_ignores_context_overflow_from_structured_get_payload():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "model_limit": {"context": 32768, "input": 24576, "output": 8192},
        }
    )

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        if method == "POST":
            return {}
        if method == "GET":
            return {
                "error": {
                    "code": "prompt_too_long",
                    "message": "history exceeded context window",
                },
                "usage": {"input_tokens": 41000},
            }
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            with pytest.raises(BackendProtocolError, match="Could not extract opencode messages"):
                await adapter.send_message(
                    session,
                    _make_turn_context("turn-1"),
                    [Message(role="user", content="hello")],
                )

    asyncio.run(run_test())


def test_request_preserves_context_overflow_http_status_as_transport_error():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "model_limit": {"context": 32768, "input": 24576, "output": 8192},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            413,
            json={
                "error": {
                    "code": "context_length_exceeded",
                    "message": "request exceeded context",
                },
                "usage": {"prompt_tokens": 50000},
            },
        )

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://opencode.test",
        )
        with pytest.raises(BackendTransportError, match="returned 413"):
            await adapter._request(
                "GET",
                "/session/oc-session-1/message",
                json_body=None,
                timeout=1.0,
            )
        await adapter._client.aclose()

    asyncio.run(run_test())


def test_request_detects_proxy_context_overflow_http_status_payload():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
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
        )

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://opencode.test",
        )
        with pytest.raises(BackendContextOverflowError) as exc_info:
            await adapter._request(
                "GET",
                "/session/oc-session-1/message",
                json_body=None,
                timeout=1.0,
            )
        await adapter._client.aclose()

        assert exc_info.value.context_window == 8
        assert exc_info.value.input_tokens == 6
        assert exc_info.value.max_tokens == 4
        assert exc_info.value.raw_error_code == "context_overflow"
        assert exc_info.value.details()["phase"] == "input_output"

    asyncio.run(run_test())


def test_request_detects_max_steps_exceeded_from_http_status_payload():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "proxy": {"max_steps": 3},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "max_steps_exceeded",
                    "message": "Turn exceeded max_steps.",
                    "details": {"max_steps": 3, "attempted_step": 4},
                }
            },
        )

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://opencode.test",
        )
        with pytest.raises(BackendMaxStepsExceededError) as exc_info:
            await adapter._request(
                "POST",
                "/session/oc-session-1/message",
                json_body={},
                timeout=1.0,
            )
        await adapter._client.aclose()

        assert exc_info.value.max_steps == 3
        assert exc_info.value.attempted_step == 4
        assert exc_info.value.backend_message == "Turn exceeded max_steps."
        assert exc_info.value.raw_error_code == "max_steps_exceeded"

    asyncio.run(run_test())


def test_request_detects_max_steps_exceeded_from_gateway_rate_limit_payload():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "proxy": {"max_steps": 1},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "429 Turn exceeded max_steps.",
                    "type": "rate_limit_error",
                }
            },
        )

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://opencode.test",
        )
        with pytest.raises(BackendMaxStepsExceededError) as exc_info:
            await adapter._request(
                "POST",
                "/session/oc-session-1/message",
                json_body={},
                timeout=1.0,
            )
        await adapter._client.aclose()

        assert exc_info.value.max_steps == 1
        assert exc_info.value.attempted_step == 1
        assert exc_info.value.backend_message == "429 Turn exceeded max_steps."
        assert exc_info.value.raw_error_code == "rate_limit_error"

    asyncio.run(run_test())


def test_request_detects_max_steps_exceeded_from_wrapped_http_status_text():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "proxy": {"max_steps": 1},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            text=(
                "Provider request failed: upstream returned 429 "
                "max_steps_exceeded: Turn exceeded max_steps."
            ),
        )

    async def run_test() -> None:
        adapter._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://opencode.test",
        )
        with pytest.raises(BackendMaxStepsExceededError) as exc_info:
            await adapter._request(
                "POST",
                "/session/oc-session-1/message",
                json_body={},
                timeout=1.0,
            )
        await adapter._client.aclose()

        assert exc_info.value.max_steps == 1
        assert exc_info.value.attempted_step == 1
        assert exc_info.value.raw_error_code == "max_steps_exceeded"

    asyncio.run(run_test())


def test_send_message_allows_step_token_usage_over_model_limit():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "model_limit": {"context": 12, "input": 8, "output": 4},
        }
    )

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        if method == "POST":
            return {}
        if method == "GET":
            return [
                _oc_user_msg("hello"),
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {"type": "text", "text": "too big"},
                        {"type": "step-finish", "tokens": {"input": 13, "output": 1, "total": 14}},
                    ],
                },
            ]
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context("turn-1"),
                [Message(role="user", content="hello")],
            )

        assert resp.outputs[0].content == "too big"
        assert resp.usage.input_tokens == 13

    asyncio.run(run_test())


def test_send_message_allows_multi_step_usage_under_model_limit():
    adapter = OpencodeAdapter()
    adapter._options = adapter._parse_options(
        {
            "provider_id": "sglang",
            "provider_name": "Remote SGLang",
            "provider_package": "@ai-sdk/openai-compatible",
            "model_id": "qwen35-a3b",
            "model_name": "Qwen 3.5 35B A3B",
            "model_limit": {"context": 20, "input": 16, "output": 4},
        }
    )

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert path == "/session/oc-session-1/message"
        if method == "POST":
            return {}
        if method == "GET":
            return [
                _oc_user_msg("hello"),
                {
                    "info": {"role": "assistant"},
                    "parts": [
                        {"type": "text", "text": "first"},
                        {"type": "step-finish", "tokens": {"input": 8, "output": 1, "total": 9}},
                        {"type": "text", "text": "second"},
                        {"type": "step-finish", "tokens": {"input": 9, "output": 1, "total": 10}},
                    ],
                },
            ]
        raise AssertionError(f"unexpected method: {method}")

    session = _make_session_context()

    async def run_test() -> None:
        with (
            patch.object(adapter, "health", return_value=True),
            patch.object(adapter, "_request", side_effect=mock_request),
            patch.object(adapter, "_ensure_backend_session", return_value="oc-session-1"),
        ):
            resp = await adapter.send_message(
                session,
                _make_turn_context("turn-1"),
                [Message(role="user", content="hello")],
            )

        assert resp.outputs[0].content == "first\nsecond"
        assert resp.usage.steps == 2
        assert resp.usage.input_tokens == 17

    asyncio.run(run_test())


def test_request_get_with_retry_fails_immediately_when_deadline_is_exhausted():
    adapter = OpencodeAdapter()
    request_mock = AsyncMock(side_effect=AssertionError("request should not run"))

    async def run_test() -> None:
        loop = asyncio.get_running_loop()
        with patch.object(adapter, "_request", request_mock):
            with pytest.raises(asyncio.TimeoutError, match="fetch opencode messages"):
                await adapter._request_get_with_retry(
                    "/session/oc-session-1/message",
                    deadline=loop.time() - 0.01,
                    turn_id="turn-1",
                    prev_msg_count=0,
                )

        request_mock.assert_not_awaited()

    asyncio.run(run_test())


def _make_transport_error_from(cause: httpx.HTTPError) -> BackendTransportError:
    err = BackendTransportError(str(cause))
    err.__cause__ = cause
    return err


def test_post_message_with_connect_retry_retries_then_succeeds():
    adapter = OpencodeAdapter()
    attempts = {"count": 0}

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        assert method == "POST"
        assert timeout > 0
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise _make_transport_error_from(
                httpx.ConnectError("All connection attempts failed")
            )
        return {}

    async def run_test() -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30.0
        with (
            patch.object(adapter, "_request", side_effect=mock_request),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=AsyncMock()),
        ):
            result = await adapter._post_message_with_connect_retry(
                "/session/sid/message",
                json_body={"x": 1},
                deadline=deadline,
                operation="send opencode message",
            )

        assert result == {}
        assert attempts["count"] == 2

    asyncio.run(run_test())


def test_post_message_with_connect_retry_does_not_retry_on_non_connect_transport_error():
    adapter = OpencodeAdapter()
    attempts = {"count": 0}

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        attempts["count"] += 1
        raise _make_transport_error_from(httpx.ReadTimeout("read timeout"))

    async def run_test() -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30.0
        with (
            patch.object(adapter, "_request", side_effect=mock_request),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=AsyncMock()),
        ):
            with pytest.raises(BackendTransportError, match="read timeout"):
                await adapter._post_message_with_connect_retry(
                    "/session/sid/message",
                    json_body={"x": 1},
                    deadline=deadline,
                    operation="send opencode message",
                )

        assert attempts["count"] == 1

    asyncio.run(run_test())


def test_post_message_with_connect_retry_aborts_when_process_already_exited():
    adapter = OpencodeAdapter()
    adapter._process = SimpleNamespace(returncode=137)
    request_mock = AsyncMock(side_effect=AssertionError("request should not run"))

    async def run_test() -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30.0
        with patch.object(adapter, "_request", request_mock):
            with pytest.raises(BackendProcessError, match="exited with code 137"):
                await adapter._post_message_with_connect_retry(
                    "/session/sid/message",
                    json_body={"x": 1},
                    deadline=deadline,
                    operation="send opencode message",
                )

        request_mock.assert_not_awaited()

    asyncio.run(run_test())


def test_post_message_with_connect_retry_exhausts_max_retries():
    adapter = OpencodeAdapter()
    attempts = {"count": 0}

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        attempts["count"] += 1
        raise _make_transport_error_from(
            httpx.ConnectError("All connection attempts failed")
        )

    async def run_test() -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30.0
        with (
            patch.object(adapter, "_request", side_effect=mock_request),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=AsyncMock()),
        ):
            with pytest.raises(BackendTransportError, match="All connection attempts failed"):
                await adapter._post_message_with_connect_retry(
                    "/session/sid/message",
                    json_body={"x": 1},
                    deadline=deadline,
                    operation="send opencode message",
                    max_retries=2,
                )

        assert attempts["count"] == 3

    asyncio.run(run_test())


def test_post_message_with_connect_retry_stops_when_remaining_budget_below_delay():
    adapter = OpencodeAdapter()
    attempts = {"count": 0}

    async def mock_request(method: str, path: str, *, json_body: dict[str, Any] | None, timeout: float) -> Any:
        attempts["count"] += 1
        raise _make_transport_error_from(
            httpx.ConnectError("All connection attempts failed")
        )

    sleep_mock = AsyncMock()

    async def run_test() -> None:
        loop = asyncio.get_running_loop()
        # Deadline is well below retry_delay (0.2s default), so after the first
        # failure the budget check should trip and we re-raise without sleeping.
        deadline = loop.time() + 0.05
        with (
            patch.object(adapter, "_request", side_effect=mock_request),
            patch("blackbox_server.adapters.opencode.asyncio.sleep", new=sleep_mock),
        ):
            with pytest.raises(BackendTransportError, match="All connection attempts failed"):
                await adapter._post_message_with_connect_retry(
                    "/session/sid/message",
                    json_body={"x": 1},
                    deadline=deadline,
                    operation="send opencode message",
                )

        assert attempts["count"] == 1
        sleep_mock.assert_not_awaited()

    asyncio.run(run_test())
