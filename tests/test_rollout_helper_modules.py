from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from dressage.paddock import lifecycle
from dressage.paddock.blackbox.execute_hooks import (
    execute_blackbox_cmds_for_stage,
    parse_blackbox_execute_cmds,
)
from dressage.paddock.blackbox.failures import (
    expected_abort_from_call_agent_exception,
    failure_from_call_agent_exception,
    failure_from_payload_state,
    record_blackbox_abort_for_retry,
    record_agent_early_stop_metadata,
    record_agent_failure_metadata,
)
from dressage.rollout.artifacts.writer import RolloutArtifactWriter
from dressage.rollout.generate import runtime as generate_runtime
from dressage.rollout.generate import whitebox_agent


class RuntimeDummyPaddock:
    pass


class SampleLike:
    prompt = "hello"
    label = "label"
    session_id = "bbs-sess"
    group_index = 7
    index = 3
    status = "pending"
    response = ""
    tokens = []
    response_length = 0
    loss_mask = None
    rollout_log_probs = None
    reward = None

    def __init__(self) -> None:
        self.metadata = {}


class ExecutePaddock:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    async def execute_cmd(self, state, **kwargs):
        self.calls.append(("execute_cmd", state, kwargs))
        if self.fail:
            request = httpx.Request("POST", "http://sandbox/execute_cmd")
            response = httpx.Response(
                503,
                json={"error": "not_ready"},
                request=request,
            )
            raise httpx.HTTPStatusError("not ready", request=request, response=response)
        return {
            "stdout": "ok",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
        }


class SlowTerminatePaddock:
    def __init__(self) -> None:
        self.calls = []

    async def terminate(self, session_id, env_args=None):
        self.calls.append(("terminate", session_id, env_args))
        await asyncio.sleep(0.01)
        return {"deleted": True}


def test_rollout_artifact_writer_writes_session_samples_and_error(
    monkeypatch,
    tmp_path,
):
    asyncio.run(
        _run_rollout_artifact_writer_writes_session_samples_and_error(
            monkeypatch,
            tmp_path,
        )
    )


async def _run_rollout_artifact_writer_writes_session_samples_and_error(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("DRESSAGE_LOG_WRITE_MODE", "await")
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR", str(tmp_path / "payload"))
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_ERROR_LOG_DIR", str(tmp_path / "errors"))

    writer = RolloutArtifactWriter()
    session_path = await writer.write_session_payload(
        {"success": True, "data": []},
        session_id="bbs-sess",
        instance_id="inst",
    )
    assert session_path == tmp_path / "payload" / "inst" / "bbs-sess" / "session.json"
    assert session_path.exists()

    segment = {
        "uid": "seg-0",
        "segment_index": 0,
        "tokens": [1, 2],
        "full_loss_mask": [0, 1],
        "full_logprobs": [0.0, -0.1],
        "messages": [{"role": "assistant", "content": "done"}],
    }
    await writer.write_segment_samples(
        SampleLike(),
        args=SimpleNamespace(max_tokens_per_gpu=8, context_parallel_size=1),
        segments=[segment],
        base_metadata={"source": "test"},
        session_id="bbs-sess",
        instance_id="inst",
        agent_response="fallback",
    )
    assert (tmp_path / "payload" / "inst" / "bbs-sess" / "samples" / "0.json").exists()

    error_path = await writer.write_error(
        RuntimeError("boom"),
        sample=SampleLike(),
        metadata={"source": "test"},
        session_id="bbs-sess",
        instance_id="inst",
        blackbox_type="opencode",
        env_args={"sandbox_image": "img"},
        state={"ok": False},
        agent_response="",
    )
    assert error_path == tmp_path / "errors" / "inst" / "bbs-sess" / "error.json"
    assert error_path.exists()


def test_lifecycle_terminate_timeout_can_be_drained(monkeypatch):
    asyncio.run(_run_lifecycle_terminate_timeout_can_be_drained(monkeypatch))


async def _run_lifecycle_terminate_timeout_can_be_drained(monkeypatch):
    paddock = SlowTerminatePaddock()
    monkeypatch.setenv("DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC", "0.001")

    await lifecycle.terminate_paddock_best_effort(
        paddock,
        session_id="bbs-sess",
        env_args={"sandbox_image": "img"},
    )
    await lifecycle.drain_terminate_tasks()

    assert paddock.calls == [
        ("terminate", "bbs-sess", {"sandbox_image": "img"}),
    ]


def test_generate_runtime_paddock_env_args_only_uses_supported_keys():
    env_args = generate_runtime.paddock_env_args_from_metadata(
        {
            "sandbox_timeout_sec": 12,
            "sandbox_image": "img",
            "sandbox_cmd": ["serve"],
            "sandbox_extra_params": {"env_key": "env"},
            "image": "legacy",
            "cmd": "legacy",
            "e2b_template": "legacy",
        },
        extra_env_args={"blackbox_type": "openclaw"},
    )

    assert env_args == {
        "sandbox_timeout_sec": 12,
        "sandbox_image": "img",
        "sandbox_cmd": ["serve"],
        "sandbox_extra_params": {"env_key": "env"},
        "blackbox_type": "openclaw",
    }


def test_generate_runtime_maybe_await_handles_values_and_coroutines():
    async def run():
        async def coro():
            return "async"

        assert await generate_runtime.maybe_await("sync") == "sync"
        assert await generate_runtime.maybe_await(coro()) == "async"

    asyncio.run(run())


def test_generate_runtime_get_proxy_client_uses_proxy_url(monkeypatch):
    class FakeProxyClient:
        def __init__(self, url):
            self.url = url

    previous = generate_runtime._PROXY_CLIENT
    monkeypatch.setenv("DRESSAGE_PROXY_URL", "http://proxy.test:8800")
    monkeypatch.setattr(generate_runtime, "ProxyClient", FakeProxyClient)
    generate_runtime._PROXY_CLIENT = None
    try:
        first = generate_runtime.get_proxy_client()
        second = generate_runtime.get_proxy_client()
    finally:
        generate_runtime._PROXY_CLIENT = previous

    assert first is second
    assert first.url == "http://proxy.test:8800"


def test_generate_runtime_get_paddock_from_env_mode_rules(monkeypatch):
    import dressage.paddock.factory as paddock_factory

    previous = generate_runtime._PADDOCK
    monkeypatch.delenv("DRESSAGE_PADDOCK_CLASS", raising=False)
    monkeypatch.setenv("DRESSAGE_PADDOCK_MODE", "whitebox")
    monkeypatch.setattr(
        paddock_factory,
        "create_paddock_from_env",
        lambda: RuntimeDummyPaddock(),
    )
    generate_runtime._PADDOCK = None
    try:
        with pytest.raises(ValueError, match="does not support whitebox"):
            generate_runtime.get_paddock_from_env(allow_whitebox_mode=False)

        paddock = generate_runtime.get_paddock_from_env(allow_whitebox_mode=True)
    finally:
        generate_runtime._PADDOCK = previous

    assert isinstance(paddock, RuntimeDummyPaddock)


def test_generate_runtime_caches_paddocks_by_explicit_mode(monkeypatch):
    import dressage.paddock.factory as paddock_factory

    previous = generate_runtime._PADDOCK
    previous_by_mode = generate_runtime._PADDOCK_BY_MODE
    created = []

    def create_paddock_from_env(*, mode=None):
        paddock = RuntimeDummyPaddock()
        created.append((mode, paddock))
        return paddock

    monkeypatch.delenv("DRESSAGE_PADDOCK_CLASS", raising=False)
    monkeypatch.setattr(
        paddock_factory,
        "create_paddock_from_env",
        create_paddock_from_env,
    )
    generate_runtime._PADDOCK = None
    generate_runtime._PADDOCK_BY_MODE = {}
    try:
        blackbox = generate_runtime.get_paddock_from_env(
            allow_whitebox_mode=False,
            mode="blackbox",
        )
        assert (
            generate_runtime.get_paddock_from_env(
                allow_whitebox_mode=False,
                mode="blackbox",
            )
            is blackbox
        )

        whitebox = generate_runtime.get_paddock_from_env(
            allow_whitebox_mode=True,
            mode="whitebox",
        )
        assert (
            generate_runtime.get_paddock_from_env(
                allow_whitebox_mode=True,
                mode="whitebox",
            )
            is whitebox
        )
    finally:
        generate_runtime._PADDOCK = previous
        generate_runtime._PADDOCK_BY_MODE = previous_by_mode

    assert blackbox is not whitebox
    assert [(mode, type(paddock)) for mode, paddock in created] == [
        ("blackbox", RuntimeDummyPaddock),
        ("whitebox", RuntimeDummyPaddock),
    ]


def test_generate_runtime_get_paddock_from_env_class_override(monkeypatch):
    previous = generate_runtime._PADDOCK
    monkeypatch.setenv(
        "DRESSAGE_PADDOCK_CLASS",
        f"{__name__}.RuntimeDummyPaddock",
    )
    monkeypatch.setenv("DRESSAGE_PADDOCK_MODE", "whitebox")
    generate_runtime._PADDOCK = None
    try:
        paddock = generate_runtime.get_paddock_from_env(allow_whitebox_mode=False)
    finally:
        generate_runtime._PADDOCK = previous

    assert isinstance(paddock, RuntimeDummyPaddock)


def test_whitebox_agent_session_id_reuses_sample_id_and_prefixes(monkeypatch):
    sample = SimpleNamespace(session_id="sample-session")

    assert whitebox_agent._agent_session_id(sample, "wb") == "wb-sample-session"


def test_whitebox_agent_session_id_uses_existing_prefix():
    sample = SimpleNamespace(session_id="wb-existing")

    assert whitebox_agent._agent_session_id(sample, "wb") == "wb-existing"


def test_whitebox_agent_session_id_generates_uuid_when_missing(monkeypatch):
    monkeypatch.setattr(
        whitebox_agent.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="abc123"),
    )

    sample = SimpleNamespace(session_id=None)

    assert whitebox_agent._agent_session_id(sample, "wb") == "wb-abc123"


def test_whitebox_stamp_runtime_metadata_writes_sample_session_id():
    sample = SimpleNamespace(session_id="old", metadata={})

    whitebox_agent._stamp_runtime_metadata(sample, "wb-new", "inst-1")

    assert sample.session_id == "wb-new"
    assert sample.metadata["session_id"] == "wb-new"
    assert sample.metadata["instance_id"] == "inst-1"


def test_blackbox_failures_parse_semantic_http_and_payload_state():
    request = httpx.Request("POST", "http://sandbox/messages")
    response = httpx.Response(
        413,
        json={
            "error": "context_overflow",
            "message": "too long",
            "details": {"state": "desynced", "tokens": 123},
        },
        request=request,
    )
    exc = httpx.HTTPStatusError("too long", request=request, response=response)

    failure = failure_from_call_agent_exception(exc)
    assert failure is not None
    assert failure.kind == "context_overflow"
    metadata = {}
    record_agent_failure_metadata(metadata, failure)
    record_agent_early_stop_metadata(metadata, failure)
    assert metadata["blackbox_agent_error_kind"] == "context_overflow"
    assert metadata["blackbox_agent_early_stop"] is True

    payload_failure = failure_from_payload_state(
        {"state": "failed"},
        agent_response="agent text",
    )
    assert payload_failure is not None
    assert payload_failure.kind == "agent_failed_state"


def test_blackbox_failures_parse_expected_generation_preempted_abort():
    exc = _http_status_error(
        502,
        {
            "error": "backend_error",
            "message": (
                "Backend request failed: Dressage proxy generation_preempted: "
                "SGLang generation was interrupted while partial rollout resume is disabled"
            ),
        },
    )

    assert expected_abort_from_call_agent_exception(exc) == "generation_preempted"


def test_blackbox_failures_ignore_non_expected_abort_http_errors():
    assert (
        expected_abort_from_call_agent_exception(
            _http_status_error(
                502,
                {
                    "error": "backend_error",
                    "message": "Backend request failed: ordinary backend failure",
                },
            )
        )
        is None
    )
    assert (
        expected_abort_from_call_agent_exception(
            _http_status_error(
                503,
                {
                    "error": "backend_error",
                    "message": "Dressage proxy generation_preempted",
                },
            )
        )
        is None
    )
    assert expected_abort_from_call_agent_exception(_http_status_error(502)) is None
    assert expected_abort_from_call_agent_exception(RuntimeError("boom")) is None


def test_blackbox_failures_record_abort_for_retry_metadata():
    metadata = {"dressage_retry_count": 2}
    exc = RuntimeError("boom")

    record_blackbox_abort_for_retry(metadata, "bbs-sess", exc)

    assert metadata["blackbox_error"] == "boom"
    assert metadata["blackbox_failure_history"] == [
        {
            "session_id": "bbs-sess",
            "error_type": "RuntimeError",
            "error": "boom",
            "retry_count": 2,
        }
    ]


def _http_status_error(
    status_code: int,
    json_body: dict | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://sandbox/messages")
    if json_body is None:
        response = httpx.Response(status_code, content=b"not-json", request=request)
    else:
        response = httpx.Response(status_code, json=json_body, request=request)
    return httpx.HTTPStatusError("request failed", request=request, response=response)

def test_blackbox_failures_record_abort_includes_http_response_body():
    metadata = {"dressage_retry_count": 1}
    request = httpx.Request("POST", "http://proxy/v1/chat/completions")
    response = httpx.Response(
        502,
        json={
            "detail": {
                "error": "partial_rollout_staleness_exceeded",
                "version_span": 3,
            }
        },
        request=request,
    )
    exc = httpx.HTTPStatusError("bad gateway", request=request, response=response)

    record_blackbox_abort_for_retry(metadata, "bbs-sess", exc)

    assert "partial_rollout_staleness_exceeded" in metadata["blackbox_error"]
    assert metadata["blackbox_http_status_code"] == 502
    assert "partial_rollout_staleness_exceeded" in metadata["blackbox_http_response_body"]
    assert metadata["blackbox_http_response_json"]["detail"]["version_span"] == 3
    history = metadata["blackbox_failure_history"][0]
    assert "partial_rollout_staleness_exceeded" in history["error"]
    assert "partial_rollout_staleness_exceeded" in history["http_response_body"]


def test_execute_hooks_run_stage_and_record_optional_http_failure():
    asyncio.run(_run_execute_hooks_run_stage_and_record_optional_http_failure())


async def _run_execute_hooks_run_stage_and_record_optional_http_failure():
    schedule = parse_blackbox_execute_cmds(
        {
            "before_agent": [
                {
                    "name": "ok",
                    "cmd": "echo ok",
                    "required": True,
                }
            ],
            "after_agent": [
                {
                    "name": "optional",
                    "cmd": "echo optional",
                    "required": False,
                }
            ],
        }
    )

    metadata = {}
    paddock = ExecutePaddock()
    await execute_blackbox_cmds_for_stage(
        paddock,
        {"state": "ok"},
        metadata,
        schedule=schedule,
        session_id="bbs-sess",
        stage="before_agent",
    )
    assert metadata["execute_cmds"][0]["cmd_result"]["returncode"] == 0

    optional_metadata = {}
    failing_paddock = ExecutePaddock(fail=True)
    await execute_blackbox_cmds_for_stage(
        failing_paddock,
        {"state": "ok"},
        optional_metadata,
        schedule=schedule,
        session_id="bbs-sess",
        stage="after_agent",
    )
    assert optional_metadata["execute_cmds"][0]["cmd_error"]["summary"] == "not ready"
    assert optional_metadata["execute_cmds"][0]["http"]["response"]["status_code"] == 503
