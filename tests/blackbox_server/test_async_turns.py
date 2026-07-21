from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from blackbox_server.adapters.base import BackendAdapter
from blackbox_server.app import create_app
from blackbox_server.config import BlackboxServerConfig
from blackbox_server.core.models import (
    AdapterResponse,
    BackendCapabilities,
    Message,
    RegisterRequest,
    SessionContext,
    TraceEvent,
    TurnContext,
    TurnRecord,
    TurnStatus,
    TurnUsage,
    utcnow,
)
from blackbox_server.core.server import BlackboxServer
from dressage.paddock.blackbox.client import BlackboxServerClient
from dressage.sandbox.types import SandboxEndpoint


class FakeAdapter(BackendAdapter):
    def __init__(self) -> None:
        self.calls = 0
        self.aborts = 0

    async def initialize(self, binding_context) -> None:  # noqa: D401
        return None

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        self.calls += 1
        session_context.backend_session_id = session_context.backend_session_id or "oc-session-1"
        content = new_messages[0].content or ""
        return AdapterResponse(
            outputs=[Message(role="assistant", content=f"echo: {content}")],
            trace_events=[
                TraceEvent(
                    turn_id=turn_context.turn_id,
                    seq=1,
                    source="fake",
                    event_type="reasoning",
                    payload={"text": "fake"},
                    created_at=utcnow(),
                )
            ],
            usage=TurnUsage(total_tokens=10, input_tokens=4, output_tokens=6, steps=1),
            backend_session_id=session_context.backend_session_id,
        )

    async def abort_session(self, session_context: SessionContext) -> bool:
        self.aborts += 1
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
        return None


class SlowAdapter(FakeAdapter):
    def __init__(self, delay: float = 0.2) -> None:
        super().__init__()
        self.delay = delay

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        self.calls += 1
        await asyncio.sleep(self.delay)
        session_context.backend_session_id = session_context.backend_session_id or "oc-session-1"
        content = new_messages[0].content or ""
        return AdapterResponse(
            outputs=[Message(role="assistant", content=f"echo: {content}")],
            trace_events=[],
            usage=TurnUsage(total_tokens=1),
            backend_session_id=session_context.backend_session_id,
        )


def make_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter: BackendAdapter,
    *,
    backend_timeout: float = 5.0,
) -> TestClient:
    monkeypatch.setattr("blackbox_server.core.server.create_adapter", lambda _: adapter)
    config = BlackboxServerConfig(
        runtime_root=str(tmp_path / "runtime"), backend_timeout=backend_timeout
    )
    return TestClient(create_app(config))


def register_payload() -> dict:
    return {
        "blackbox_type": "opencode",
        "router": "127.0.0.1:30000",
        "bound_session_id": "sess-001",
        "bound_instance_id": "inst-001",
        "backend_options": {"proxy": {}},
    }


def _register(client: TestClient) -> None:
    resp = client.post("/v1/rollout/register", json=register_payload())
    assert resp.status_code == 200


def test_async_submit_poll_to_committed_matches_sync_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = FakeAdapter()
    client = make_client(tmp_path, monkeypatch, adapter)
    with client:
        _register(client)
        submit = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-1",
                "mode": "async",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert submit.status_code == 202
        body = submit.json()
        assert body["status"] == "queued"
        assert body["turn_id"] == "turn-1"
        assert body["idempotent_replay"] is False

        poll = client.get("/v1/sessions/sess-001/turns/turn-1", params={"wait": 5})
        assert poll.status_code == 200
        data = poll.json()
        assert data["status"] == "committed"
        assert data["state"] == "active"
        assert data["outputs"][0]["content"] == "echo: hello"
        assert data["backend"]["backend_session_id"] == "oc-session-1"

        # Same output as the synchronous mode.
        sync = client.post(
            "/v1/sessions/sess-001/messages",
            json={"turn_id": "turn-2", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert sync.status_code == 200
        assert sync.json()["outputs"][0]["content"] == "echo: hello"


def test_async_submit_requires_turn_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        _register(client)
        resp = client.post(
            "/v1/sessions/sess-001/messages",
            json={"mode": "async", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 400
    assert resp.json()["error"] == "request_error"
    assert "turn_id is required" in resp.json()["message"]


def test_async_duplicate_submit_reuses_same_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = SlowAdapter(delay=0.2)
    client = make_client(tmp_path, monkeypatch, adapter)
    with client:
        _register(client)
        payload = {
            "turn_id": "turn-dup",
            "mode": "async",
            "messages": [{"role": "user", "content": "hello"}],
        }
        first = client.post("/v1/sessions/sess-001/messages", json=payload)
        assert first.status_code == 202
        second = client.post("/v1/sessions/sess-001/messages", json=payload)
        assert second.status_code == 202
        assert second.json()["idempotent_replay"] is True
        assert second.json()["turn_id"] == "turn-dup"

        poll = client.get("/v1/sessions/sess-001/turns/turn-dup", params={"wait": 5})
        assert poll.status_code == 200
        assert poll.json()["status"] == "committed"
        # The turn executed exactly once despite the duplicate submission.
        assert adapter.calls == 1


def test_async_same_turn_id_different_body_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        _register(client)
        first = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-x",
                "mode": "async",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert first.status_code == 202
        poll = client.get("/v1/sessions/sess-001/turns/turn-x", params={"wait": 5})
        assert poll.json()["status"] == "committed"

        conflict = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-x",
                "mode": "async",
                "messages": [{"role": "user", "content": "different"}],
            },
        )
    assert conflict.status_code == 409


def test_async_distinct_turn_while_active_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = SlowAdapter(delay=0.3)
    client = make_client(tmp_path, monkeypatch, adapter)
    with client:
        _register(client)
        first = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-active",
                "mode": "async",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert first.status_code == 202
        conflict = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-second",
                "mode": "async",
                "messages": [{"role": "user", "content": "again"}],
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["details"]["active_turn_id"] == "turn-active"
        # Let the active turn finish to drain the background task cleanly.
        client.get("/v1/sessions/sess-001/turns/turn-active", params={"wait": 5})


def test_execute_cmd_conflicts_with_active_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = SlowAdapter(delay=0.3)
    client = make_client(tmp_path, monkeypatch, adapter)
    with client:
        _register(client)
        submit = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-cmd",
                "mode": "async",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert submit.status_code == 202
        conflict = client.post(
            "/v1/sessions/sess-001/execute_cmd", json={"cmd": "printf hi"}
        )
        assert conflict.status_code == 409
        assert conflict.json()["details"]["active_turn_id"] == "turn-cmd"

        poll = client.get("/v1/sessions/sess-001/turns/turn-cmd", params={"wait": 5})
        assert poll.json()["status"] == "committed"

        ok = client.post("/v1/sessions/sess-001/execute_cmd", json={"cmd": "printf hi"})
        assert ok.status_code == 200
        assert ok.json()["stdout"] == "hi"


def test_long_poll_returns_current_state_then_early_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = SlowAdapter(delay=0.5)
    client = make_client(tmp_path, monkeypatch, adapter)
    with client:
        _register(client)
        submit = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-poll",
                "mode": "async",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert submit.status_code == 202

        short = client.get("/v1/sessions/sess-001/turns/turn-poll", params={"wait": 0.05})
        assert short.status_code == 200
        assert short.json()["status"] in {"queued", "inflight"}

        done = client.get("/v1/sessions/sess-001/turns/turn-poll", params={"wait": 5})
        assert done.status_code == 200
        assert done.json()["status"] == "committed"


def test_get_unknown_turn_returns_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client = make_client(tmp_path, monkeypatch, FakeAdapter())
    with client:
        _register(client)
        resp = client.get("/v1/sessions/sess-001/turns/turn-missing")
    assert resp.status_code == 404


def test_cancel_inflight_turn_then_terminal_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = SlowAdapter(delay=1.0)
    client = make_client(tmp_path, monkeypatch, adapter)
    with client:
        _register(client)
        submit = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-cancel",
                "mode": "async",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert submit.status_code == 202

        cancel = client.post("/v1/sessions/sess-001/turns/turn-cancel/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancel_requested"

        poll = client.get("/v1/sessions/sess-001/turns/turn-cancel", params={"wait": 5})
        assert poll.json()["status"] == "cancelled"

        # Cancelling a terminal turn is idempotent.
        again = client.post("/v1/sessions/sess-001/turns/turn-cancel/cancel")
        assert again.status_code == 200
        assert again.json()["status"] == "cancelled"


def test_backend_timeout_marks_turn_unknown_and_session_desynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    adapter = SlowAdapter(delay=0.3)
    client = make_client(tmp_path, monkeypatch, adapter, backend_timeout=0.01)
    with client:
        _register(client)
        submit = client.post(
            "/v1/sessions/sess-001/messages",
            json={
                "turn_id": "turn-timeout",
                "mode": "async",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert submit.status_code == 202

        poll = client.get("/v1/sessions/sess-001/turns/turn-timeout", params={"wait": 5})
        assert poll.status_code == 200
        data = poll.json()
        assert data["status"] == "unknown"
        assert data["error"]["error"] == "backend_timeout"
        assert data["error"]["http_status"] == 504

        session = client.get("/v1/sessions/sess-001", params={"include_turns": "true"})
        assert session.json()["state"] == "desynced"


def test_cancel_queued_turn_is_synchronously_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    asyncio.run(_run_cancel_queued_turn(tmp_path, monkeypatch))


async def _run_cancel_queued_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr("blackbox_server.core.server.create_adapter", lambda _: adapter)
    config = BlackboxServerConfig(
        runtime_root=str(tmp_path / "runtime"),
        backend_timeout=5.0,
        runtime_health_check_interval=999.0,
    )
    server = BlackboxServer(config)
    await server.register(
        RegisterRequest(
            blackbox_type="opencode",
            router="127.0.0.1:30000",
            bound_session_id="sess-001",
            bound_instance_id="inst-001",
            backend_options={"proxy": {}},
        )
    )
    try:
        session = await server._session_store.get("sess-001")
        now = utcnow()
        session.turn_ledger["turn-q"] = TurnRecord(
            turn_id="turn-q",
            request_fingerprint="fp",
            status=TurnStatus.QUEUED,
            request_messages=[Message(role="user", content="hi")],
            created_at=now,
            updated_at=now,
        )
        server._turn_events[("sess-001", "turn-q")] = asyncio.Event()

        response = await server.cancel_turn("sess-001", "turn-q")
        assert response.status == "cancelled"
        assert session.turn_ledger["turn-q"].status == TurnStatus.CANCELLED
        assert server._turn_events[("sess-001", "turn-q")].is_set()
    finally:
        await server.graceful_shutdown()


def test_client_call_agent_reuses_turn_id_across_retry(monkeypatch: pytest.MonkeyPatch):
    asyncio.run(_run_client_call_agent_reuses_turn_id(monkeypatch))


async def _run_client_call_agent_reuses_turn_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_AGENT_REQUEST_INITIAL_DELAY_SEC", "0")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_AGENT_REQUEST_MAX_DELAY_SEC", "0")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_AGENT_POLL_WAIT_SEC", "0")

    submit_turn_ids: list[str] = []
    counters = {"post": 0, "get": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages"):
            body = json.loads(request.content.decode())
            assert body["mode"] == "async"
            submit_turn_ids.append(body["turn_id"])
            counters["post"] += 1
            if counters["post"] == 1:
                raise httpx.ConnectError("transient connect failure")
            return httpx.Response(
                202,
                json={
                    "session_id": "sess-1",
                    "turn_id": body["turn_id"],
                    "status": "queued",
                    "idempotent_replay": counters["post"] > 1,
                },
            )
        if "/turns/" in request.url.path:
            counters["get"] += 1
            turn_id = request.url.path.rsplit("/", 1)[-1]
            if counters["get"] == 1:
                return httpx.Response(
                    200,
                    json={
                        "session_id": "sess-1",
                        "turn_id": turn_id,
                        "status": "inflight",
                        "state": "active",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "session_id": "sess-1",
                    "turn_id": turn_id,
                    "status": "committed",
                    "state": "active",
                    "outputs": [{"role": "assistant", "content": "done"}],
                    "backend": {"type": "opencode", "backend_session_id": "oc-1"},
                    "usage": {"total_tokens": 3},
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BlackboxServerClient(client=http_client)
    endpoint = SandboxEndpoint(url="http://sandbox.test", headers={})
    payload = await client.call_agent(
        endpoint,
        trajectory_id="sess-1",
        session_id="sess-1",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert payload["outputs"][0]["content"] == "done"
    assert counters["post"] == 2  # submit retried once
    assert counters["get"] == 2  # polled inflight then committed
    assert len(set(submit_turn_ids)) == 1  # same turn_id reused across retry
    await http_client.aclose()


def test_client_call_agent_sync_mode_via_env(monkeypatch: pytest.MonkeyPatch):
    asyncio.run(_run_client_call_agent_sync_mode(monkeypatch))


async def _run_client_call_agent_sync_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRESSAGE_BLACKBOX_AGENT_CALL_MODE", "sync")

    counters = {"post": 0, "get": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages"):
            body = json.loads(request.content.decode())
            assert body["mode"] == "sync"
            assert body["turn_id"].startswith("turn-")
            counters["post"] += 1
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "session_id": "sess-1",
                    "turn_id": body["turn_id"],
                    "state": "active",
                    "idempotent_replay": False,
                    "outputs": [{"role": "assistant", "content": "done"}],
                    "backend": {"type": "opencode", "backend_session_id": "oc-1"},
                    "usage": {"total_tokens": 3},
                },
            )
        if "/turns/" in request.url.path:
            counters["get"] += 1
            raise AssertionError("sync mode must not poll the turns endpoint")
        raise AssertionError(f"unexpected path {request.url.path}")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BlackboxServerClient(client=http_client)
    endpoint = SandboxEndpoint(url="http://sandbox.test", headers={})
    payload = await client.call_agent(
        endpoint,
        trajectory_id="sess-1",
        session_id="sess-1",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert payload["outputs"][0]["content"] == "done"
    assert counters["post"] == 1
    assert counters["get"] == 0  # no polling in sync mode
    await http_client.aclose()
