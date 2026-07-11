from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy


class MockAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


def _make_proxy(
    sticky_header_name: str = "X-SMG-Routing-Key",
    *,
    max_steps: int | None = 100,
    default_temperature: float | None = None,
    debug_log_dir: Any = None,
) -> RolloutLLMProxy:
    return RolloutLLMProxy(
        upstream_origin="http://127.0.0.1:30000",
        router_api_path="/v1",
        bound_session_id="sess-001",
        bound_instance_id="inst-001",
        sticky_header_name=sticky_header_name,
        max_steps=max_steps,
        default_temperature=default_temperature,
        debug_log_dir=debug_log_dir,
    )


def _sse_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def _sse_events_from_body(body: bytes) -> list[tuple[str | None, dict[str, Any]]]:
    events: list[tuple[str | None, dict[str, Any]]] = []
    for raw_event in body.decode("utf-8").strip().split("\n\n"):
        event_name: str | None = None
        data_lines: list[str] = []
        for line in raw_event.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def test_rollout_proxy_defaults_to_registered_router_for_anthropic_messages():
    proxy = RolloutLLMProxy(
        upstream_origin="http://127.0.0.1:8800",
        router_api_path="/v1",
        bound_session_id="sess-001",
        bound_instance_id="inst-001",
        sticky_header_name="X-SMG-Routing-Key",
    )
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={"model": "proxy-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert response.json()["content"] == [{"type": "text", "text": "OK"}]

    asyncio.run(run_test())

    assert captured["url"] == "http://127.0.0.1:8800/v1/chat/completions"
    assert "x-backend-host" not in captured["headers"]


def test_rollout_proxy_forces_identity_encoding_for_model_requests():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages",
                headers={"accept-encoding": "gzip, deflate, br, zstd"},
                json={"model": "proxy-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200

    asyncio.run(run_test())

    assert captured["headers"]["accept-encoding"] == "identity"


def test_rollout_proxy_returns_non_retry_anthropic_error_for_unclassified_stream_5xx():
    proxy = _make_proxy()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            502,
            headers={"content-type": "application/json", "content-encoding": "br"},
            content=b"\x8b\xffcompressed-error-body",
        )

    async def run_test() -> httpx.Response:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "proxy-model",
                    "max_tokens": 1,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()
        return response

    response = asyncio.run(run_test())
    payload = response.json()

    assert response.status_code == 400
    assert payload["error"]["type"] == "upstream_error"
    assert "HTTP 502" in payload["error"]["message"]
    assert "binary; hex=" in payload["error"]["message"]
    assert "\ufffd" not in payload["error"]["message"]


def test_rollout_proxy_dumps_failed_anthropic_upstream_payload(tmp_path):
    proxy = _make_proxy(debug_log_dir=tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            500,
            headers={"content-type": "text/plain"},
            content=b"Internal Server Error",
        )

    async def run_test() -> tuple[httpx.Response, dict[str, Any] | None]:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages",
                headers={"authorization": "Bearer secret-token"},
                json={
                    "model": "proxy-model",
                    "max_tokens": 128,
                    "stream": True,
                    "metadata": {"source": "test"},
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [
                        {
                            "name": "Bash",
                            "description": "run commands",
                            "input_schema": {"type": "object"},
                        }
                    ],
                },
            )
        await proxy.drain_turn(timeout=1.0)
        payload = await proxy.consume_failed_upstream_error()
        await proxy.clear_turn()
        await proxy._client.aclose()
        return response, payload

    response, payload = asyncio.run(run_test())

    assert response.status_code == 400
    assert payload is not None
    assert payload["message"] == "Upstream returned HTTP 500: Internal Server Error"
    request_path = payload["request_path"]
    response_path = payload["response_path"]
    request_dump = json.loads(Path(request_path).read_text(encoding="utf-8"))
    response_dump = json.loads(Path(response_path).read_text(encoding="utf-8"))

    auth_key = next(key for key in request_dump["headers"] if key.lower() == "authorization")
    assert request_dump["headers"][auth_key] == "<redacted>"
    assert request_dump["status_code"] == 500
    assert request_dump["message_count"] == 1
    assert request_dump["tool_count"] == 1
    assert request_dump["has_thinking"] is True
    assert request_dump["has_metadata"] is True
    assert request_dump["has_stream_options"] is True
    assert request_dump["body"]["stream_options"] == {"include_usage": True}
    assert request_dump["body"]["tools"][0]["function"]["name"] == "Bash"
    assert response_dump["status_code"] == 500
    assert response_dump["body_preview"] == "Internal Server Error"


def test_rollout_proxy_does_not_dump_successful_payloads(tmp_path):
    proxy = _make_proxy(debug_log_dir=tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={"model": "proxy-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()
        assert response.status_code == 200

    asyncio.run(run_test())

    assert list(tmp_path.glob("upstream_request.*.json")) == []
    assert list(tmp_path.glob("upstream_response.*.json")) == []


def test_rollout_proxy_uses_bound_ids_for_headers_and_untagged_snapshot():
    proxy = _make_proxy()

    async def run_test() -> None:
        snapshot = await proxy._capture_snapshot()
        headers = proxy._build_upstream_headers(
            {
                "authorization": "Bearer x",
                "x-session-id": "bad-session",
                "X-Instance-Id": "bad-instance",
                "X-Turn-ID": "bad-turn",
            },
            is_chat=True,
        )
        assert headers["authorization"] == "Bearer x"
        assert headers["X-SMG-Routing-Key"] == "sess-001"
        assert headers["X-Session-Id"] == "sess-001"
        assert headers["X-Instance-Id"] == "inst-001"
        assert "X-Turn-Id" not in headers
        assert snapshot.session_id == "sess-001"
        assert snapshot.turn_id is None
        assert snapshot.backend_session_id is None

    asyncio.run(run_test())


def test_rollout_proxy_scopes_increment_steps_and_inject_turn_header():
    proxy = _make_proxy()

    async def run_test() -> None:
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        first = await proxy._capture_snapshot()
        second = await proxy._capture_snapshot()
        headers = proxy._build_upstream_headers({}, is_chat=True, turn_id=first.turn_id)
        await proxy.clear_turn()
        assert first.turn_id == "turn-001"
        assert first.step == 0
        assert second.step == 1
        assert second.backend_session_id == "oc-session-1"
        assert headers["X-Turn-Id"] == "turn-001"

    asyncio.run(run_test())


def test_rollout_proxy_matches_any_chat_completion_prefix():
    proxy = _make_proxy()

    assert proxy._is_chat_completion("POST", "v1/chat/completions") is True
    assert proxy._is_chat_completion("POST", "custom-prefix/chat/completions") is True
    assert proxy._is_chat_completion("GET", "v1/chat/completions") is False
    assert proxy._is_chat_completion("POST", "v1/responses") is False
    assert proxy._is_openai_responses("POST", "v1/responses") is True
    assert proxy._is_openai_responses("POST", "custom-prefix/responses") is True
    assert proxy._is_openai_responses("GET", "v1/responses") is False
    assert proxy._is_openai_responses("POST", "v1/chat/completions") is False


def test_rollout_proxy_bridges_openai_responses_to_chat_completions():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "proxy-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "visible",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "Bash",
                                        "arguments": "{\"cmd\":\"ls\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/responses",
                json={
                    "model": "proxy-model",
                    "instructions": "system prompt",
                    "input": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": "developer prompt"}],
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hello"}],
                        },
                        {
                            "type": "message",
                            "role": "system",
                            "content": [{"type": "input_text", "text": "late system"}],
                        },
                        {
                            "type": "message",
                            "role": "critic",
                            "content": [{"type": "input_text", "text": "unknown role text"}],
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "Bash",
                            "arguments": "{\"cmd\":\"pwd\"}",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "repo",
                        },
                    ],
                    "max_output_tokens": 128,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "parallel_tool_calls": False,
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "name": "Bash",
                            "description": "run shell commands",
                            "parameters": {"type": "object", "properties": {}},
                        },
                        {"type": "web_search_preview"},
                    ],
                    "tool_choice": {"type": "function", "name": "Bash"},
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "chatcmpl-1"
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["model"] == "proxy-model"
        assert body["usage"] == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"] == [{"type": "output_text", "text": "visible"}]
        assert body["output"][1] == {
            "type": "function_call",
            "id": "fc_call_2",
            "call_id": "call_2",
            "name": "Bash",
            "arguments": "{\"cmd\":\"ls\"}",
            "status": "completed",
        }

    asyncio.run(run_test())

    assert str(captured["url"]).endswith("/v1/chat/completions")
    payload = captured["payload"]
    assert payload == {
        "model": "proxy-model",
        "messages": [
            {"role": "system", "content": "system prompt\n\ndeveloper prompt\n\nlate system"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "unknown role text"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{\"cmd\":\"pwd\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "repo"},
        ],
        "stream": False,
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
        "parallel_tool_calls": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "run shell commands",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}},
    }
    roles = [message["role"] for message in payload["messages"]]
    assert roles.count("system") == 1
    assert "developer" not in roles
    assert set(roles) <= {"system", "user", "assistant", "tool"}
    headers = captured["headers"]
    assert headers["x-smg-routing-key"] == "sess-001"
    assert headers["x-session-id"] == "sess-001"
    assert headers["x-instance-id"] == "inst-001"
    assert headers["x-turn-id"] == "turn-001"


def test_rollout_proxy_coalesces_responses_message_and_function_call_history():
    proxy = _make_proxy()

    payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I'll call a tool."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{\"prompt\":\"solve\"}",
                },
            ],
        }
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": "I'll call a tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"prompt\":\"solve\"}",
                    },
                }
            ],
        }
    ]


def test_rollout_proxy_round_trips_responses_reasoning_message_and_function_call():
    proxy = _make_proxy()

    response_payload = proxy._chat_completion_to_openai_response(
        {
            "id": "chatcmpl-1",
            "model": "proxy-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "I should delegate.",
                        "content": "I'll spawn a helper.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "spawn_agent",
                                    "arguments": "{\"prompt\":\"solve\"}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"model": "proxy-model"},
    )

    assert [item["type"] for item in response_payload["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert response_payload["output"][0]["summary"] == [
        {"type": "summary_text", "text": "I should delegate."}
    ]

    replay_payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                *response_payload["output"],
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "agent-1",
                },
            ],
        }
    )

    assert replay_payload["messages"] == [
        {
            "role": "assistant",
            "content": "I'll spawn a helper.",
            "reasoning_content": "I should delegate.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"prompt\":\"solve\"}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "agent-1"},
    ]


def test_rollout_proxy_attaches_reasoning_to_standalone_function_call_group():
    proxy = _make_proxy()

    payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Need a helper."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{\"prompt\":\"solve\"}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "wait_agent",
                    "arguments": "{\"agent_id\":\"agent-1\"}",
                },
            ],
        }
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Need a helper.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"prompt\":\"solve\"}",
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "wait_agent",
                        "arguments": "{\"agent_id\":\"agent-1\"}",
                    },
                },
            ],
        }
    ]


def test_rollout_proxy_clears_pending_reasoning_on_boundaries():
    proxy = _make_proxy()

    payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Do not leak."}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_0",
                    "output": "result",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{}",
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Attach me."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "break"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "No reasoning."}],
                },
            ],
        }
    )

    assert payload["messages"] == [
        {"role": "tool", "tool_call_id": "call_0", "content": "result"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "spawn_agent", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "break"},
        {"role": "assistant", "content": "No reasoning."},
    ]


def test_rollout_proxy_coalesces_multiple_responses_function_calls_in_order():
    proxy = _make_proxy()

    payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Two calls."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{\"prompt\":\"solve\"}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "wait_agent",
                    "arguments": "{\"agent_id\":\"agent-1\"}",
                },
            ],
        }
    )

    assistant = payload["messages"][0]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Two calls."
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == [
        "spawn_agent",
        "wait_agent",
    ]


def test_rollout_proxy_coalesces_standalone_responses_function_calls():
    proxy = _make_proxy()

    payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{\"prompt\":\"solve\"}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "wait_agent",
                    "arguments": "{\"agent_id\":\"agent-1\"}",
                },
            ],
        }
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"prompt\":\"solve\"}",
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "wait_agent",
                        "arguments": "{\"agent_id\":\"agent-1\"}",
                    },
                },
            ],
        }
    ]


def test_rollout_proxy_does_not_coalesce_function_calls_across_boundaries():
    proxy = _make_proxy()

    payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_0",
                    "output": "result",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{\"prompt\":\"solve\"}",
                },
            ],
        }
    )

    assert payload["messages"] == [
        {"role": "assistant", "content": "Done."},
        {"role": "tool", "tool_call_id": "call_0", "content": "result"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"prompt\":\"solve\"}",
                    },
                }
            ],
        },
    ]


def test_rollout_proxy_clears_pending_assistant_on_user_system_and_unknown_items():
    proxy = _make_proxy()

    payload = proxy._openai_responses_to_chat_completion(
        {
            "model": "proxy-model",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "First."}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer note"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_after_developer",
                    "name": "spawn_agent",
                    "arguments": "{}",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Second."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "user break"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_after_user",
                    "name": "wait_agent",
                    "arguments": "{}",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Third."}],
                },
                {"type": "unknown", "value": "break"},
                {
                    "type": "function_call",
                    "call_id": "call_after_unknown",
                    "name": "exec_command",
                    "arguments": "{}",
                },
            ],
        }
    )

    assert payload["messages"] == [
        {"role": "system", "content": "developer note"},
        {"role": "assistant", "content": "First."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_after_developer",
                    "type": "function",
                    "function": {"name": "spawn_agent", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": "Second."},
        {"role": "user", "content": "user break"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_after_user",
                    "type": "function",
                    "function": {"name": "wait_agent", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": "Third."},
        {"role": "user", "content": "{\"type\": \"unknown\", \"value\": \"break\"}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_after_unknown",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": "{}"},
                }
            ],
        },
    ]


def test_rollout_proxy_bridges_coalesced_responses_history_to_chat_completions():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "proxy-model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/responses",
                json={
                    "model": "proxy-model",
                    "input": [
                        {"type": "message", "role": "user", "content": "hi"},
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "I'll spawn."}],
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "spawn_agent",
                            "arguments": "{\"prompt\":\"solve\"}",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "agent-1",
                        },
                    ],
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200

    asyncio.run(run_test())

    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "I'll spawn.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"prompt\":\"solve\"}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "agent-1"},
    ]


def test_rollout_proxy_keeps_text_when_chat_tool_calls_are_empty():
    proxy = _make_proxy()

    payload = proxy._chat_completion_to_openai_response(
        {
            "id": "chatcmpl-1",
            "model": "proxy-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "plain text",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        {"model": "proxy-model"},
    )

    assert [item["type"] for item in payload["output"]] == ["message"]
    assert payload["output"][0]["content"] == [{"type": "output_text", "text": "plain text"}]


def test_rollout_proxy_omits_empty_text_when_chat_completion_only_calls_tool():
    proxy = _make_proxy()

    payload = proxy._chat_completion_to_openai_response(
        {
            "id": "chatcmpl-1",
            "model": "proxy-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "spawn_agent",
                                    "arguments": "{\"prompt\":\"solve\"}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"model": "proxy-model"},
    )

    assert [item["type"] for item in payload["output"]] == ["function_call"]
    assert payload["output"][0]["name"] == "spawn_agent"


def test_rollout_proxy_writes_openai_responses_tool_summary(tmp_path: Path):
    proxy = _make_proxy(debug_log_dir=tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "proxy-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/responses",
                json={
                    "model": "proxy-model",
                    "input": "hi",
                    "stream": False,
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "description": "spawn an agent",
                            "parameters": {"type": "object", "properties": {}},
                        },
                        {
                            "type": "namespace",
                            "name": "collaboration",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "wait_agent",
                                    "description": "wait for an agent",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {},
                                    },
                                }
                            ],
                        },
                    ],
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200

    asyncio.run(run_test())

    summary_path = tmp_path / "tool_summary.turn-001.0.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["request_kind"] == "openai_responses"
    assert summary["raw_tool_count"] == 2
    assert summary["converted_tool_count"] == 1
    assert summary["raw_tool_names"] == ["spawn_agent", "collaboration.wait_agent"]
    assert summary["converted_tool_names"] == ["spawn_agent"]
    assert summary["raw_tool_types"] == ["function", "namespace"]
    assert summary["converted_tool_types"] == ["function"]


def test_rollout_proxy_streams_chat_chunks_as_openai_response_events():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "model": "proxy-model",
                            "choices": [{"delta": {"content": "hel"}}],
                        }
                    ),
                    _sse_event({"id": "chatcmpl-stream", "choices": [{"delta": {"content": "lo"}}]}),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                        }
                    ),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> bytes:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "proxy-model", "input": "hello", "stream": True},
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        return body

    body = asyncio.run(run_test())
    events = _sse_events_from_body(body)
    assert [name for name, _ in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[3][1]["delta"] == "hello"
    completed = events[-1][1]["response"]
    assert completed["status"] == "completed"
    assert completed["output"][0]["content"] == [{"type": "output_text", "text": "hello"}]
    assert completed["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert str(captured["url"]).endswith("/v1/chat/completions")
    assert captured["payload"] == {
        "model": "proxy-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_rollout_proxy_streams_chat_tool_calls_as_openai_response_events():
    proxy = _make_proxy()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "model": "proxy-model",
                            "choices": [{"delta": {"content": "I'll use "}}],
                        }
                    ),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [{"delta": {"content": "a helper."}}],
                        }
                    ),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_spawn",
                                                "type": "function",
                                                "function": {
                                                    "name": "spawn_agent",
                                                    "arguments": "{\"prompt\":",
                                                },
                                            }
                                        ]
                                    }
                                }
                            ],
                        }
                    ),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"arguments": "\"solve\"}"},
                                            }
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                        }
                    ),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> bytes:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "proxy-model", "input": "spawn", "stream": True},
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        return body

    body = asyncio.run(run_test())
    events = _sse_events_from_body(body)
    event_names = [name for name, _ in events]
    assert event_names == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[3][1]["delta"] == "I'll use a helper."
    function_added = events[7][1]
    assert function_added["output_index"] == 1
    assert function_added["item"]["type"] == "function_call"
    assert function_added["item"]["status"] == "in_progress"
    assert function_added["item"]["arguments"] == ""
    assert events[8][1]["delta"] == "{\"prompt\":"
    assert events[9][1]["delta"] == "\"solve\"}"
    assert events[10][1]["arguments"] == "{\"prompt\":\"solve\"}"
    completed = events[-1][1]["response"]
    assert completed["status"] == "completed"
    assert [item["type"] for item in completed["output"]] == ["message", "function_call"]
    assert completed["output"][0]["content"] == [
        {"type": "output_text", "text": "I'll use a helper."}
    ]
    function_call = completed["output"][1]
    assert function_call["name"] == "spawn_agent"
    assert function_call["arguments"] == "{\"prompt\":\"solve\"}"
    assert completed["usage"] == {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}


def test_rollout_proxy_streams_chat_reasoning_with_message_and_tool_call_output():
    proxy = _make_proxy()

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "model": "proxy-model",
                            "choices": [{"delta": {"reasoning_content": "think-"}}],
                        }
                    ),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [{"delta": {"reasoning": "then-act"}}],
                        }
                    ),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [{"delta": {"content": "I'll use a helper."}}],
                        }
                    ),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_spawn",
                                                "type": "function",
                                                "function": {
                                                    "name": "spawn_agent",
                                                    "arguments": "{\"prompt\":",
                                                },
                                            }
                                        ]
                                    }
                                }
                            ],
                        }
                    ),
                    _sse_event(
                        {
                            "id": "chatcmpl-stream",
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"arguments": "\"solve\"}"},
                                            }
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                        }
                    ),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> bytes:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "proxy-model", "input": "spawn", "stream": True},
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        return body

    body = asyncio.run(run_test())
    events = _sse_events_from_body(body)
    completed = events[-1][1]["response"]
    assert [item["type"] for item in completed["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert completed["output"][0]["summary"] == [
        {"type": "summary_text", "text": "think-then-act"}
    ]
    assert completed["output"][1]["content"] == [
        {"type": "output_text", "text": "I'll use a helper."}
    ]
    assert completed["output"][2]["name"] == "spawn_agent"
    assert completed["output"][2]["arguments"] == "{\"prompt\":\"solve\"}"
    assert events[1][1]["item"]["type"] == "reasoning"
    assert events[1][1]["output_index"] == 0
    assert events[3][1]["item"]["type"] == "message"
    assert events[3][1]["output_index"] == 1
    function_added = next(
        payload
        for name, payload in events
        if name == "response.output_item.added"
        and payload["item"]["type"] == "function_call"
    )
    assert function_added["output_index"] == 2


def test_rollout_proxy_rejects_openai_responses_after_max_steps():
    proxy = _make_proxy(max_steps=1)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/responses",
                json={"model": "proxy-model", "input": "hi", "stream": False},
            )
            second = await client.post(
                "/v1/responses",
                json={"model": "proxy-model", "input": "again", "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "max_steps_exceeded"
        assert request_count == 1
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_records_failed_upstream_for_openai_responses():
    proxy = _make_proxy()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, headers={"content-type": "text/plain"}, content=b"boom")

    async def run_test() -> tuple[httpx.Response, dict[str, Any] | None]:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="codex-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/responses",
                json={"model": "proxy-model", "input": "hi", "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        payload = await proxy.consume_failed_upstream_error()
        await proxy.clear_turn()
        await proxy._client.aclose()
        return response, payload

    response, payload = asyncio.run(run_test())

    assert response.status_code == 500
    assert response.text == "boom"
    assert payload is not None
    assert payload["status_code"] == 500
    assert payload["message"] == "Upstream returned HTTP 500: boom"
    assert str(payload["upstream_url"]).endswith("/v1/chat/completions")


def test_rollout_proxy_preserves_stream_options_without_logprob_injection():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event(
                        {"id": "resp-1", "choices": [{"delta": {"reasoning_content": "thinking"}}]}
                    ),
                    _sse_event({"id": "resp-1", "choices": [{"delta": {"content": "ok"}}]}),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [],
                    "stream": True,
                    "stream_options": {"include_usage": False, "extra": "keep-me"},
                },
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert b'"reasoning_content": "thinking"' in body
        assert b'"content": "ok"' in body
        assert body.endswith(b"data: [DONE]\n\n")
        assert captured["payload"] == {
            "model": "gpt-test",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True, "extra": "keep-me"},
        }
        headers = captured["headers"]
        assert headers["x-smg-routing-key"] == "sess-001"
        assert headers["x-session-id"] == "sess-001"
        assert headers["x-instance-id"] == "inst-001"
        assert headers["x-turn-id"] == "turn-001"

    asyncio.run(run_test())


def test_rollout_proxy_does_not_inject_logprobs_for_non_stream_requests():
    proxy = _make_proxy()
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "thinking",
                        }
                    }
                ],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"] == {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "thinking",
        }
        assert requests == [{"model": "gpt-test", "messages": [], "stream": False}]

    asyncio.run(run_test())


def test_rollout_proxy_injects_default_temperature_when_missing():
    proxy = _make_proxy(default_temperature=0.7)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert requests == [
            {
                "model": "gpt-test",
                "messages": [],
                "stream": False,
                "temperature": 0.7,
            }
        ]

    asyncio.run(run_test())


def test_rollout_proxy_preserves_explicit_zero_temperature():
    proxy = _make_proxy(default_temperature=0.7)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [],
                    "stream": False,
                    "temperature": 0.0,
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert requests == [
            {
                "model": "gpt-test",
                "messages": [],
                "stream": False,
                "temperature": 0.0,
            }
        ]

    asyncio.run(run_test())


def test_rollout_proxy_injects_default_temperature_for_stream_requests():
    proxy = _make_proxy(default_temperature=0.7)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event({"id": "resp-1", "choices": [{"delta": {"content": "ok"}}]}),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [],
                    "stream": True,
                    "stream_options": {"include_usage": False},
                },
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert b'"content": "ok"' in body
        assert captured["payload"] == {
            "model": "gpt-test",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.7,
        }

    asyncio.run(run_test())


def test_rollout_proxy_allows_chat_steps_under_default_limit():
    proxy = _make_proxy()
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            second = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 200
        assert request_count == 2

    asyncio.run(run_test())


def test_rollout_proxy_allows_unlimited_chat_steps_when_max_steps_disabled():
    proxy = _make_proxy(max_steps=None)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            for _ in range(3):
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-test", "messages": [], "stream": False},
                )
                assert response.status_code == 200
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert request_count == 3

    asyncio.run(run_test())


def test_rollout_proxy_rejects_chat_completion_after_max_steps_without_inflight_leak():
    proxy = _make_proxy(max_steps=1)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            second = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json() == {
            "error": {
                "message": "Turn exceeded max_steps.",
                "type": "rate_limit_error",
                "code": "max_steps_exceeded",
                "details": {
                    "max_steps": 1,
                    "attempted_step": 1,
                },
            }
        }
        assert request_count == 1
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_resets_max_steps_for_next_turn():
    proxy = _make_proxy(max_steps=1)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
            first = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            await proxy.drain_turn(timeout=1.0)
            await proxy.clear_turn()

            await proxy.open_turn("turn-002", backend_session_id="oc-session-1")
            second = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            await proxy.drain_turn(timeout=1.0)
            await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 200
        assert request_count == 2

    asyncio.run(run_test())


def test_rollout_proxy_bridges_anthropic_messages_to_openai_chat_completion():
    proxy = _make_proxy(default_temperature=0.7)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "proxy-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "thinking",
                            "content": "visible",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Bash",
                                        "arguments": "{\"cmd\":\"ls\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages?beta=true",
                json={
                    "model": "proxy-model",
                    "max_tokens": 128,
                    "stream": False,
                    "system": "system prompt",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}],
                        }
                    ],
                    "tools": [
                        {
                            "name": "Bash",
                            "description": "run commands",
                            "input_schema": {"type": "object"},
                        }
                    ],
                    "tool_choice": {"type": "tool", "name": "Bash"},
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"
        assert body["stop_reason"] == "tool_use"
        assert body["usage"] == {"input_tokens": 2, "output_tokens": 3}
        assert body["content"] == [
            {"type": "thinking", "thinking": "thinking"},
            {"type": "text", "text": "visible"},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "Bash",
                "input": {"cmd": "ls"},
            },
        ]

        assert str(captured["url"]).endswith("/v1/chat/completions")
        payload = captured["payload"]
        assert payload["model"] == "proxy-model"
        assert payload["max_tokens"] == 128
        assert payload["temperature"] == 0.7
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert payload["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
        ]
        assert payload["tools"][0]["function"]["name"] == "Bash"
        assert payload["tool_choice"] == {"type": "function", "function": {"name": "Bash"}}
        headers = captured["headers"]
        assert headers["x-smg-routing-key"] == "sess-001"
        assert headers["x-session-id"] == "sess-001"
        assert headers["x-instance-id"] == "inst-001"
        assert headers["x-turn-id"] == "turn-001"

    asyncio.run(run_test())


def test_rollout_proxy_hoists_mid_conversation_anthropic_system_messages():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": "proxy-model",
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages",
                headers={
                    "anthropic-beta": "mid-conversation-system-2026-04-07",
                    "anthropic-version": "2023-06-01",
                    "x-app": "cli",
                    "x-claude-code-session-id": "claude-session",
                    "x-stainless-lang": "js",
                    "authorization": "Bearer blackbox-local",
                },
                json={
                    "model": "proxy-model",
                    "max_tokens": 128,
                    "system": "top system",
                    "messages": [
                        {"role": "user", "content": "first user"},
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "mid system"}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "assistant text"}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "tool output"},
                                {"type": "text", "text": "second user"},
                            ],
                        },
                    ],
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200

    asyncio.run(run_test())

    payload = captured["payload"]
    assert payload["messages"] == [
        {"role": "system", "content": "top system\n\nmid system"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "assistant text"},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "tool output"},
        {"role": "user", "content": "second user"},
    ]
    assert [message["role"] for message in payload["messages"]].count("system") == 1

    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["authorization"] == "Bearer blackbox-local"
    assert headers["x-smg-routing-key"] == "sess-001"
    assert "anthropic-beta" not in headers
    assert "anthropic-version" not in headers
    assert "x-app" not in headers
    assert "x-claude-code-session-id" not in headers
    assert "x-stainless-lang" not in headers


def test_rollout_proxy_streams_openai_reasoning_as_anthropic_thinking_delta():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event(
                        {
                            "id": "chunk-1",
                            "model": "proxy-model",
                            "choices": [{"delta": {"reasoning_content": "think-1"}}],
                        }
                    ),
                    _sse_event({"id": "chunk-2", "choices": [{"delta": {"content": "visible"}}]}),
                    _sse_event({"id": "chunk-3", "choices": [{"delta": {"reasoning_content": "think-2"}}]}),
                    _sse_event(
                        {
                            "id": "chunk-4",
                            "choices": [{"delta": {}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
                        }
                    ),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/messages",
                json={
                    "model": "proxy-model",
                    "max_tokens": 128,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert b"event: message_start" in body
        assert b'"type": "thinking_delta", "thinking": "think-1"' in body
        assert b'"type": "text_delta", "text": "visible"' in body
        assert b'"type": "thinking_delta", "thinking": "think-2"' in body
        assert b"event: message_stop" in body
        assert captured["payload"]["stream"] is True
        assert captured["payload"]["stream_options"] == {"include_usage": True}

    asyncio.run(run_test())


def test_rollout_proxy_rejects_anthropic_messages_after_max_steps():
    proxy = _make_proxy(max_steps=1)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/messages",
                json={"model": "proxy-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            )
            second = await client.post(
                "/v1/messages",
                json={"model": "proxy-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            )
        await proxy.drain_turn(timeout=1.0)
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["type"] == "error"
        assert second.json()["error"]["code"] == "max_steps_exceeded"
        assert request_count == 1
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_records_rollout_invalidated_for_anthropic_messages():
    proxy = _make_proxy()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={"detail": {"error": "generation_preempted", "message": "stale"}},
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="claude-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={"model": "proxy-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
            )
        await proxy.drain_turn(timeout=1.0)
        payload = await proxy.consume_rollout_invalidated_error()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert response.json()["content"] == [{"type": "text", "text": ""}]
        assert payload == {"error": "generation_preempted", "message": "stale"}

    asyncio.run(run_test())


def test_rollout_proxy_recognizes_partial_staleness_invalidated_response():
    proxy = _make_proxy()
    payload = {
        "detail": {
            "error": "partial_rollout_staleness_exceeded",
            "message": "Partial rollout model version span exceeded limit.",
            "session_id": "sess-001",
            "versions": ["v1", "v2", "v3"],
            "version_span": 3,
            "version_switches": 2,
            "max_preempts": 1,
            "max_version_span": 2,
        }
    }

    recorded = proxy._rollout_invalidated_payload_from_response(
        status_code=502,
        response_body=json.dumps(payload).encode("utf-8"),
    )

    assert recorded == payload["detail"]
