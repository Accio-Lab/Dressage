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
