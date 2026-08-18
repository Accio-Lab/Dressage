"""Tests for blackbox tool-call side-effect hooks."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from dressage.proxy.server import create_app
from dressage.proxy.session_manager import SessionManager
from dressage.proxy.tool_call_hooks import (
    ToolCallContext,
    ToolCallHook,
    ToolCallHookChain,
    ToolCallHookError,
    build_tool_call_hook_chain,
    build_tool_call_idempotency_key,
    load_tool_call_hooks,
    register_tool_call_hook,
    registered_tool_call_hooks,
    reset_tool_call_hook_registry,
)
from dressage.proxy.trajectory_store import TrajectoryStore


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class RecordingHook(ToolCallHook):
    name = "recording"
    order = 100

    def __init__(self) -> None:
        self.before_events: list[ToolCallContext] = []
        self.after_events: list[ToolCallContext] = []

    async def before_tool_call(self, ctx: ToolCallContext) -> None:
        self.before_events.append(ctx)

    async def after_tool_call(self, ctx: ToolCallContext) -> None:
        self.after_events.append(ctx)


class FailingHook(ToolCallHook):
    name = "failing"

    def __init__(self) -> None:
        self.calls = 0

    async def before_tool_call(self, ctx: ToolCallContext) -> None:
        self.calls += 1
        raise RuntimeError("boom")


class SlowHook(ToolCallHook):
    name = "slow"

    def __init__(self) -> None:
        self.calls = 0

    async def before_tool_call(self, ctx: ToolCallContext) -> None:
        self.calls += 1
        await asyncio.sleep(10)


class SelectiveHook(RecordingHook):
    name = "selective"
    order = 50

    def applies_to(self, ctx: ToolCallContext) -> bool:
        return ctx.session_id == "sess-hooks"


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        tools=None,
        chat_template=None,
        return_assistant_tokens_mask=False,
        **_,
    ):
        rendered = ""
        masks: list[int] = []
        for message in messages:
            role = message.get("role", "unknown")
            chunk = f"<{role}>"
            rendered += chunk
            masks.extend([0] * len(chunk))
            content = message.get("content")
            if content is not None:
                text = str(content)
                rendered += text
                if role == "assistant":
                    masks.extend([1] * len(text))
                else:
                    masks.extend([0] * len(text))
        if add_generation_prompt:
            rendered += "<assistant>"
            masks.extend([0] * len("<assistant>"))
        if not tokenize:
            return rendered
        payload = {"input_ids": [ord(ch) for ch in rendered]}
        if return_assistant_tokens_mask:
            payload["assistant_masks"] = list(masks)
        return payload if return_dict else payload["input_ids"]


class _FakeResponse:
    def __init__(self, text: str):
        self.output_ids = [ord(ch) for ch in text]
        self.output_token_logprobs = [-0.1 for _ in text]
        self.output_token_texts = list(text)
        self.text = text
        self.meta_info: dict[str, Any] = {}
        self.finish_reason = "stop"
        self.input_token_logprobs_raw = []
        self.all_token_ids = list(self.output_ids)
        self.all_logprobs = list(self.output_token_logprobs)
        self.output_versions: list[str] = []
        self.weight_version = "v1"
        self.rollout_epoch = 1
        self.all_logprobs_invalid = False
        self.input_token_texts: list[str] = []
        self.routed_experts_chunks: list[Any] = []


class FakeSGLangClient:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    async def generate(self, input_ids, sampling_params, **kwargs):
        text = self._responses.pop(0)
        return _FakeResponse(text)

    async def parse_function_call(self, text, tools=None, **kwargs):
        if "<tool_call>" in text:
            payload = json.loads(text.split("<tool_call>")[1].split("</tool_call>")[0])
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call000001",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": payload["name"],
                            "arguments": json.dumps(payload["arguments"]),
                        },
                    }
                ],
            }
        return {"content": text, "tool_calls": None}

    async def separate_reasoning(self, text, **kwargs):
        return {"reasoning_content": None, "content": text}

    async def close(self) -> None:
        return None


def make_hook_client(
    *responses: str,
    chain: ToolCallHookChain | None,
) -> tuple[TestClient, SessionManager]:
    session_manager = SessionManager()
    app = create_app(
        sglang_router_url="http://router.test",
        tokenizer=FakeTokenizer(),
        session_manager=session_manager,
        trajectory_store=TrajectoryStore(min_group_size=1, group_timeout=0.0),
        sglang_client=FakeSGLangClient(list(responses)),
        model_mask_type="qwen3_5",
        model_tool_call_type="hermes",
        tool_call_parse_backend="local",
        token_build_mode="snapshot",
        tool_call_hook_chain=chain,
    )
    return TestClient(app), session_manager


# ---------------------------------------------------------------------------
# Chain semantics
# ---------------------------------------------------------------------------


def _ctx(session_id: str = "sess-1", **overrides: Any) -> ToolCallContext:
    defaults: dict[str, Any] = {
        "instance_id": "inst-1",
        "session_id": session_id,
        "turn_id": "turn-1",
        "step_index": 0,
        "sandbox_id": "sbx-1",
    }
    defaults.update(overrides)
    return ToolCallContext(**defaults)


def test_chain_runs_before_ascending_and_after_descending():
    order_log: list[str] = []

    class Ordered(ToolCallHook):
        def __init__(self, name: str, order: int) -> None:
            self.name = name
            self.order = order

        async def before_tool_call(self, ctx) -> None:
            order_log.append(f"before:{self.name}")

        async def after_tool_call(self, ctx) -> None:
            order_log.append(f"after:{self.name}")

    chain = ToolCallHookChain([Ordered("c", 300), Ordered("a", 100), Ordered("b", 200)])
    asyncio.run(chain.run_before(_ctx()))
    asyncio.run(chain.run_after(_ctx()))

    assert order_log == [
        "before:a",
        "before:b",
        "before:c",
        "after:c",
        "after:b",
        "after:a",
    ]


def test_optional_hook_failure_is_recorded_and_chain_continues():
    hook = FailingHook()
    recorder = RecordingHook()
    chain = ToolCallHookChain([hook, recorder])
    metadata: dict[str, Any] = {}
    ctx = _ctx(stage_metadata=metadata)

    asyncio.run(chain.run_before(ctx))

    assert hook.calls == 1
    assert len(recorder.before_events) == 1
    failures = metadata["tool_call_hook_failures"]
    assert failures[0]["hook"] == "failing"
    assert failures[0]["required"] is False
    assert "boom" in failures[0]["error"]["message"]


def test_required_hook_failure_raises():
    class RequiredFailing(FailingHook):
        name = "required-failing"
        required = True

    chain = ToolCallHookChain([RequiredFailing()])
    with pytest.raises(ToolCallHookError):
        asyncio.run(chain.run_before(_ctx()))


def test_before_timeout_applies_to_before_stage():
    class RequiredSlow(SlowHook):
        name = "required-slow"
        required = True

    chain = ToolCallHookChain([RequiredSlow()], before_timeout=0.05)
    with pytest.raises(ToolCallHookError):
        asyncio.run(chain.run_before(_ctx()))
    assert chain.before_timeout == 0.05


def test_idempotency_key_suppresses_duplicate_dispatches():
    recorder = RecordingHook()
    chain = ToolCallHookChain([recorder])
    key = build_tool_call_idempotency_key(
        session_id="s", turn_id="t", step_index=1, stage="before"
    )
    ctx = _ctx(idempotency_key=key)

    asyncio.run(chain.run_before(ctx))
    asyncio.run(chain.run_before(ctx))

    assert len(recorder.before_events) == 1


def test_applies_to_filters_hooks():
    selective = SelectiveHook()
    chain = ToolCallHookChain([selective])
    asyncio.run(chain.run_before(_ctx(session_id="other-session")))
    assert selective.before_events == []
    asyncio.run(chain.run_before(_ctx(session_id="sess-hooks")))
    assert len(selective.before_events) == 1


def test_idempotency_keys_expire_after_ttl():
    recorder = RecordingHook()
    chain = ToolCallHookChain([recorder], key_ttl_seconds=0.0)
    key = build_tool_call_idempotency_key(
        session_id="s", turn_id="t", step_index=1, stage="before"
    )
    ctx = _ctx(idempotency_key=key)

    asyncio.run(chain.run_before(ctx))
    asyncio.run(chain.run_before(ctx))

    # ttl=0 means every claim is fresh: the second dispatch is NOT a
    # duplicate and both run.
    assert len(recorder.before_events) == 2


def test_purge_session_releases_keys_for_rebuilt_sessions():
    recorder = RecordingHook()
    chain = ToolCallHookChain([recorder])
    key = build_tool_call_idempotency_key(
        session_id="sess-rebuild", turn_id="t", step_index=0, stage="before"
    )
    ctx = _ctx(session_id="sess-rebuild", idempotency_key=key)

    asyncio.run(chain.run_before(ctx))
    asyncio.run(chain.run_before(ctx))
    assert len(recorder.before_events) == 1

    assert chain.purge_session("sess-rebuild") == 1
    assert chain.purge_session("sess-rebuild") == 0  # idempotent

    # After discard the same key dispatches again (rebuilt session).
    asyncio.run(chain.run_before(ctx))
    assert len(recorder.before_events) == 2


def test_discard_endpoint_purges_hook_keys():
    hook = RecordingHook()
    client, _ = make_hook_client(
        TOOL_CALL_TEXT,
        chain=ToolCallHookChain([hook]),
    )
    headers = {
        "X-Session-Id": "sess-purge",
        "X-Instance-Id": "inst-1",
        "X-Dressage-Sandbox-Id": "sbx-1",
    }
    first = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "find x"}],
        },
    )
    assert first.status_code == 200
    assert len(hook.before_events) == 1

    discarded = client.post("/session/discard", json={"session_id": "sess-purge"})
    assert discarded.status_code == 200
    assert discarded.json()["tool_call_hook_keys_purged"] == 1


# ---------------------------------------------------------------------------
# Registry / loader
# ---------------------------------------------------------------------------


def test_registry_decorator_and_duplicate_rejection():
    reset_tool_call_hook_registry()
    try:

        @register_tool_call_hook
        class Decorated(RecordingHook):
            name = "decorated"

        assert "decorated" in registered_tool_call_hooks()
        with pytest.raises(ValueError):

            @register_tool_call_hook
            class Conflicting(RecordingHook):
                name = "decorated"

        hooks = load_tool_call_hooks(["decorated"])
        assert len(hooks) == 1
        assert isinstance(hooks[0], Decorated)
        with pytest.raises(ValueError):
            load_tool_call_hooks(["missing-hook"])
    finally:
        reset_tool_call_hook_registry()


def test_load_hooks_from_file_path(tmp_path):
    hook_file = tmp_path / "my_snapshot_hook.py"
    hook_file.write_text(
        "from dressage.proxy.tool_call_hooks import ToolCallHook, "
        "register_tool_call_hook\n"
        "\n"
        "\n"
        "@register_tool_call_hook\n"
        "class FileHook(ToolCallHook):\n"
        "    name = 'file-hook'\n",
        encoding="utf-8",
    )
    reset_tool_call_hook_registry()
    try:
        hooks = load_tool_call_hooks([f"{hook_file}:FileHook"])
        assert hooks[0].name == "file-hook"
        # Bare path works when the file registers exactly one hook.
        hooks = load_tool_call_hooks([str(hook_file)])
        assert hooks[0].name == "file-hook"
    finally:
        reset_tool_call_hook_registry()


def test_build_chain_returns_none_without_hooks():
    assert build_tool_call_hook_chain(None) is None
    assert build_tool_call_hook_chain([]) is None
    assert build_tool_call_hook_chain([RecordingHook()]) is not None


# ---------------------------------------------------------------------------
# Proxy pipeline integration
# ---------------------------------------------------------------------------


TOOL_CALL_TEXT = '<tool_call>{"name": "search", "arguments": {"q": "x"}}</tool_call>'


def test_hooks_dispatch_on_tool_boundaries():
    hook = RecordingHook()
    client, session_manager = make_hook_client(
        TOOL_CALL_TEXT,
        "done",
        chain=ToolCallHookChain([hook]),
    )

    first = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-hooks",
            "X-Instance-Id": "inst-1",
            "X-Dressage-Sandbox-Id": "sbx-e2b-1",
        },
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "find x"}],
        },
    )
    assert first.status_code == 200
    assert first.json()["choices"][0]["finish_reason"] == "tool_calls"

    # before_tool_call fired once for the parsed tool_calls of step 0.
    assert len(hook.before_events) == 1
    before_ctx = hook.before_events[0]
    assert before_ctx.session_id == "sess-hooks"
    assert before_ctx.sandbox_id == "sbx-e2b-1"
    assert before_ctx.tool_calls[0]["function"]["name"] == "search"
    assert hook.after_events == []

    # Second request carries the tool result -> after_tool_call fires.
    session = session_manager.get_session("sess-hooks")
    tool_call_id = session.full_messages[-1]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-hooks",
            "X-Instance-Id": "inst-1",
            "X-Dressage-Sandbox-Id": "sbx-e2b-1",
        },
        json={
            "model": "fake-model",
            "messages": session.full_messages
            + [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "result",
                }
            ],
        },
    )
    assert second.status_code == 200
    assert len(hook.after_events) == 1
    assert hook.after_events[0].session_id == "sess-hooks"
    # Step 1 also produced plain text (no tool_calls) -> no extra before.
    assert len(hook.before_events) == 1


def test_hook_metadata_lands_in_finalized_trajectory():
    class MetadataHook(RecordingHook):
        name = "metadata-hook"

        async def before_tool_call(self, ctx: ToolCallContext) -> None:
            ctx.stage_metadata["before_snapshot"] = {"sandbox_id": ctx.sandbox_id}
            await super().before_tool_call(ctx)

    hook = MetadataHook()
    client, session_manager = make_hook_client(
        TOOL_CALL_TEXT,
        "done",
        chain=ToolCallHookChain([hook]),
    )
    headers = {
        "X-Session-Id": "sess-meta",
        "X-Instance-Id": "inst-1",
        "X-Dressage-Sandbox-Id": "sbx-meta",
    }
    first = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "find x"}],
        },
    )
    assert first.status_code == 200
    session = session_manager.get_session("sess-meta")
    tool_call_id = session.full_messages[-1]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": session.full_messages
            + [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "result",
                }
            ],
        },
    )
    assert second.status_code == 200

    finalized = client.post(
        "/session/finalize",
        json={"session_id": "sess-meta", "instance_id": "inst-1"},
    )
    assert finalized.status_code == 200
    read = client.post("/trajectory/read", json={"trajectory_id": "sess-meta"})
    items = read.json()["data"]
    assert items, "expected finalized trajectory segments"
    step0 = next(
        item
        for item in items
        if item.get("extra_info", {}).get("segment_view") == "timeline"
        and "tool_call_hooks" in item.get("extra_info", {})
    )
    hooks_meta = step0["extra_info"]["tool_call_hooks"]
    assert hooks_meta["before_snapshot"] == {"sandbox_id": "sbx-meta"}


def test_no_chain_is_full_noop():
    client, _ = make_hook_client("hello", chain=None)
    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-noop"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200


def test_required_hook_failure_returns_502():
    class RequiredFailing(FailingHook):
        name = "required-failing"
        required = True

    client, _ = make_hook_client(
        TOOL_CALL_TEXT,
        chain=ToolCallHookChain([RequiredFailing()]),
    )
    response = client.post(
        "/v1/chat/completions",
        headers={
            "X-Session-Id": "sess-fail",
            "X-Dressage-Sandbox-Id": "sbx-1",
        },
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "find x"}],
        },
    )
    assert response.status_code == 502
    assert response.json()["detail"]["error"] == "tool_call_hook_failed"
    assert response.json()["detail"]["stage"] == "before"


def test_missing_sandbox_id_skips_dispatch():
    hook = RecordingHook()
    client, _ = make_hook_client(
        TOOL_CALL_TEXT,
        chain=ToolCallHookChain([hook]),
    )
    response = client.post(
        "/v1/chat/completions",
        headers={"X-Session-Id": "sess-nosbx"},
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "find x"}],
        },
    )
    assert response.status_code == 200
    assert hook.before_events == []
    assert hook.after_events == []


# ---------------------------------------------------------------------------
# RolloutLLMProxy header forwarding
# ---------------------------------------------------------------------------


def test_rollout_proxy_forwards_bound_sandbox_id_header():
    from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy

    proxy = RolloutLLMProxy(
        upstream_origin="http://127.0.0.1:8800",
        router_api_path="/v1",
        bound_session_id="sess-001",
        bound_instance_id="inst-001",
        bound_sandbox_id="sbx-e2b-42",
        sticky_header_name="X-SMG-Routing-Key",
    )
    captured: dict[str, Any] = {}

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
        await proxy.open_turn("turn-001")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "proxy-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()
        assert response.status_code == 200

    asyncio.run(run_test())

    assert captured["headers"]["x-dressage-sandbox-id"] == "sbx-e2b-42"
    assert captured["headers"]["x-session-id"] == "sess-001"


def test_rollout_proxy_omits_header_without_sandbox_id():
    from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy

    proxy = RolloutLLMProxy(
        upstream_origin="http://127.0.0.1:8800",
        router_api_path="/v1",
        bound_session_id="sess-001",
        bound_instance_id="inst-001",
        sticky_header_name="X-SMG-Routing-Key",
    )
    captured: dict[str, Any] = {}

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
        await proxy.open_turn("turn-001")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "proxy-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

    asyncio.run(run_test())

    assert "x-dressage-sandbox-id" not in captured["headers"]


# ---------------------------------------------------------------------------
# Register binding plumbing
# ---------------------------------------------------------------------------


def test_register_request_accepts_bound_sandbox_id():
    from blackbox_server.core.models import RegisterRequest

    request = RegisterRequest(
        blackbox_type="opencode",
        router="http://127.0.0.1:8800",
        bound_session_id="sess-1",
        bound_instance_id="inst-1",
        bound_sandbox_id="sbx-1",
    )
    assert request.bound_sandbox_id == "sbx-1"

    default_request = RegisterRequest(
        blackbox_type="opencode",
        router="http://127.0.0.1:8800",
        bound_session_id="sess-1",
        bound_instance_id="inst-1",
    )
    assert default_request.bound_sandbox_id is None


def test_blackbox_client_register_payload_carries_sandbox_id():
    from dressage.paddock.blackbox.client import BlackboxServerClient
    from dressage.sandbox.types import SandboxEndpoint

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"ok": True})

    async def run_test() -> None:
        client = BlackboxServerClient(
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        await client.register_agent(
            SandboxEndpoint(url="http://sandbox.test"),
            trajectory_id="traj-1",
            instance_id="inst-1",
            session_id="sess-1",
            router_url="http://proxy.test",
            blackbox_type="opencode",
            backend_options={},
            server_config={},
            sandbox_id="sbx-native-1",
        )
        await client._client.aclose()

    asyncio.run(run_test())

    assert captured["json"]["bound_sandbox_id"] == "sbx-native-1"


def test_binding_fingerprint_distinguishes_sandbox_id():
    from blackbox_server.core.hashing import binding_request_fingerprint
    from blackbox_server.core.models import RegisterRequest

    base = {
        "blackbox_type": "opencode",
        "router": "http://127.0.0.1:8800",
        "bound_session_id": "sess-1",
        "bound_instance_id": "inst-1",
    }
    without_sandbox = RegisterRequest(**base)
    with_sandbox = RegisterRequest(**base, bound_sandbox_id="sbx-1")

    assert binding_request_fingerprint(
        without_sandbox, "http://127.0.0.1:8800"
    ) != binding_request_fingerprint(with_sandbox, "http://127.0.0.1:8800")
