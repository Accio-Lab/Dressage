from __future__ import annotations

import asyncio
import json
import logging
import sys
import types

import httpx
import pytest
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules["transformers"] = transformers

from dressage.paddock import lifecycle as paddock_lifecycle
from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_OPENCODE_COMPACTION,
    dynamic_backend_defaults_for,
)
from dressage.rollout.generate import blackbox_dispatch
from dressage.rollout.generate import runtime as generate_runtime
from dressage.rollout.prewarm import store as prewarm_store
from dressage.rollout.prewarm.store import PrewarmHandle
from dressage.paddock.blackbox.execute_hooks import (
    parse_blackbox_execute_cmds,
)
from dressage.rollout.artifacts import samples as trajectory_sample
from dressage.rollout.artifacts.writer import DEFAULT_WRITER, RolloutArtifactWriter


@dataclass
class SampleLike:
    prompt: str = "hello"
    label: str | None = None
    group_index: int | None = 7
    index: int | None = 3
    session_id: str | None = "sess-7"
    metadata: dict = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    reward: float | None = None

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING


def _encode_routed_experts(values, *, num_layers=1, topk=1):
    import base64

    import numpy as np

    array = np.asarray(values, dtype=np.int32).reshape(-1, num_layers, topk)
    return base64.b64encode(array.tobytes()).decode("ascii")


def _last_segment_sample(result, *, expected_count: int | None = None):
    assert isinstance(result, list)
    if expected_count is not None:
        assert len(result) == expected_count
    assert result
    return result[-1]


class FakePaddock:
    def __init__(self):
        self.calls = []

    async def init(self, session_id, env_type=None, env_args=None):
        self.calls.append(("init", session_id, env_type, env_args))
        return {"sandbox_url": "http://sandbox.test"}

    async def register_agent(self, state, **kwargs):
        self.calls.append(("register_agent", state, kwargs))
        return {"ok": True}

    async def call_agent(self, state, **kwargs):
        self.calls.append(("call_agent", state, kwargs))
        return {"response": "agent done"}

    async def execute_cmd(self, state, **kwargs):
        self.calls.append(("execute_cmd", state, kwargs))
        return {
            "cmd": kwargs["cmd"],
            "stdout": f"ran {kwargs['cmd']}",
            "stderr": "",
            "returncode": 0,
            "timed_out": False,
            "duration_seconds": 0.1,
            "started_at": "2026-05-29T00:00:00Z",
            "finished_at": "2026-05-29T00:00:01Z",
            "request_id": f"req-{len(self.calls)}",
            "session_id": kwargs["session_id"],
            "instance_id": "7",
        }

    async def terminate(self, session_id, env_args=None):
        self.calls.append(("terminate", session_id, env_args))
        return {"deleted": True}


class NoSlotPaddock(FakePaddock):
    async def init(self, session_id, env_type=None, env_args=None):
        self.calls.append(("init", session_id, env_type, env_args))
        raise TimeoutError("no blackbox slot available")


class FailingTerminatePaddock(FakePaddock):
    async def terminate(self, session_id, env_args=None):
        self.calls.append(("terminate", session_id, env_args))
        raise RuntimeError("destroy returned 504\nFor more information")


class SlowTerminatePaddock(FakePaddock):
    async def terminate(self, session_id, env_args=None):
        self.calls.append(("terminate", session_id, env_args))
        await asyncio.sleep(1)
        return {"deleted": True}


class FailingRegisterPaddock(FakePaddock):
    async def register_agent(self, state, **kwargs):
        self.calls.append(("register_agent", state, kwargs))
        raise RuntimeError("duplicate session")




class FailingHttpCallPaddock(FakePaddock):
    async def call_agent(self, state, **kwargs):
        self.calls.append(("call_agent", state, kwargs))
        request = httpx.Request(
            "POST",
            "http://sandbox.test/v1/sessions/bbs-sess-7/messages",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        response = httpx.Response(
            502,
            json={"error": "bad_gateway", "message": "backend failed"},
            request=request,
        )
        raise httpx.HTTPStatusError(
            "Server error '502 Bad Gateway' for url 'http://sandbox.test/v1/sessions/bbs-sess-7/messages'",
            request=request,
            response=response,
        )


class GenerationPreemptedCallPaddock(FakePaddock):
    async def call_agent(self, state, **kwargs):
        self.calls.append(("call_agent", state, kwargs))
        request = httpx.Request(
            "POST",
            "http://sandbox.test/v1/sessions/bbs-sess-7/messages",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        response = httpx.Response(
            502,
            json={
                "request_id": "req-preempted",
                "error": "backend_error",
                "message": (
                    "Backend request failed: Dressage proxy generation_preempted: "
                    "SGLang generation was interrupted while partial rollout resume is disabled"
                ),
                "details": {
                    "session_id": "bbs-sess-7",
                    "turn_id": "turn-preempted",
                },
            },
            request=request,
        )
        raise httpx.HTTPStatusError(
            "Server error '502 Bad Gateway' for url "
            "'http://sandbox.test/v1/sessions/bbs-sess-7/messages'",
            request=request,
            response=response,
        )


class FailingOnceRegisterPaddock(FakePaddock):
    def __init__(self):
        super().__init__()
        self.register_attempts = 0

    async def register_agent(self, state, **kwargs):
        self.calls.append(("register_agent", state, kwargs))
        self.register_attempts += 1
        if self.register_attempts == 1:
            raise RuntimeError("duplicate session")
        return {"ok": True}


class FailingExecuteCmdPaddock(FakePaddock):
    async def execute_cmd(self, state, **kwargs):
        self.calls.append(("execute_cmd", state, kwargs))
        return {
            "cmd": kwargs["cmd"],
            "stdout": "bad",
            "stderr": "failed",
            "returncode": 2,
            "timed_out": False,
            "duration_seconds": 0.1,
            "started_at": "2026-05-29T00:00:00Z",
            "finished_at": "2026-05-29T00:00:01Z",
            "request_id": "req-failed",
            "session_id": kwargs["session_id"],
            "instance_id": "7",
        }


class ContextOverflowPaddock(FakePaddock):
    async def call_agent(self, state, **kwargs):
        self.calls.append(("call_agent", state, kwargs))
        request = httpx.Request(
            "POST",
            "http://sandbox.test/v1/sessions/bbs-sess-7/messages",
            json={"messages": kwargs["messages"]},
        )
        response = httpx.Response(
            413,
            json={
                "error": "context_overflow",
                "message": (
                    "openclaw context overflow: input_tokens exceeds "
                    "configured context_window"
                ),
                "details": {
                    "input_tokens": 40000,
                    "context_window": 32768,
                    "max_tokens": 8192,
                    "raw_error_code": "context_length_exceeded",
                },
            },
            request=request,
        )
        raise httpx.HTTPStatusError(
            "Client error '413 Payload Too Large'",
            request=request,
            response=response,
        )


class MaxStepsExceededPaddock(FakePaddock):
    async def call_agent(self, state, **kwargs):
        self.calls.append(("call_agent", state, kwargs))
        request = httpx.Request(
            "POST",
            "http://sandbox.test/v1/sessions/bbs-sess-7/messages",
            json={"messages": kwargs["messages"]},
        )
        response = httpx.Response(
            429,
            json={
                "error": "max_steps_exceeded",
                "message": "Turn exceeded max_steps.",
                "details": {
                    "max_steps": 1,
                    "attempted_step": 1,
                    "raw_error_code": "max_steps_exceeded",
                },
            },
            request=request,
        )
        raise httpx.HTTPStatusError(
            "Client error '429 Too Many Requests'",
            request=request,
            response=response,
        )


class BackendTimeoutPaddock(FakePaddock):
    async def call_agent(self, state, **kwargs):
        self.calls.append(("call_agent", state, kwargs))
        request = httpx.Request(
            "POST",
            "http://sandbox.test/v1/sessions/bbs-sess-7/messages",
            json={"messages": kwargs["messages"]},
        )
        response = httpx.Response(
            504,
            json={
                "error": "backend_timeout",
                "message": "Backend call exceeded backend_timeout.",
                "details": {
                    "session_id": "bbs-sess-7",
                    "turn_id": "turn-timeout",
                },
            },
            request=request,
        )
        raise httpx.HTTPStatusError(
            "Server error '504 Gateway Timeout'",
            request=request,
            response=response,
        )


class OptionalHttpExecuteCmdPaddock(FakePaddock):
    async def execute_cmd(self, state, **kwargs):
        self.calls.append(("execute_cmd", state, kwargs))
        request = httpx.Request(
            "POST",
            "http://sandbox.test/v1/sessions/bbs-sess-7/execute_cmd",
            json={"cmd": kwargs["cmd"], "timeout": kwargs["timeout"]},
        )
        response = httpx.Response(
            503,
            json={
                "error": "service_unavailable",
                "message": "Server is not ready. current_state=ServerState.ERROR",
            },
            request=request,
        )
        raise httpx.HTTPStatusError(
            "Server error '503 Service Unavailable'",
            request=request,
            response=response,
        )


def test_generate_runtime_default_paddock_uses_mode_provider_factory(monkeypatch):
    class DummyPaddock:
        pass

    previous = generate_runtime._PADDOCK

    import dressage.paddock.factory as paddock_factory

    monkeypatch.delenv("DRESSAGE_PADDOCK_CLASS", raising=False)
    monkeypatch.setattr(
        paddock_factory,
        "create_paddock_from_env",
        lambda: DummyPaddock(),
    )
    generate_runtime._PADDOCK = None
    try:
        paddock = generate_runtime.get_paddock_from_env(allow_whitebox_mode=False)
    finally:
        generate_runtime._PADDOCK = previous

    assert isinstance(paddock, DummyPaddock)


def test_generate_runtime_rejects_whitebox_mode_for_blackbox(monkeypatch):
    previous = generate_runtime._PADDOCK
    monkeypatch.delenv("DRESSAGE_PADDOCK_CLASS", raising=False)
    monkeypatch.setenv("DRESSAGE_PADDOCK_MODE", "whitebox")
    generate_runtime._PADDOCK = None
    try:
        with pytest.raises(ValueError, match="does not support whitebox"):
            generate_runtime.get_paddock_from_env(allow_whitebox_mode=False)
    finally:
        generate_runtime._PADDOCK = previous


def test_blackbox_dispatch_passes_max_steps_env_in_register_payload(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_passes_max_steps_env_in_register_payload(monkeypatch)
    )


async def _run_blackbox_dispatch_passes_max_steps_env_in_register_payload(monkeypatch):
    paddock = FakePaddock()
    monkeypatch.setenv("DRESSAGE_BLACKBOX_MAX_STEPS", "23")
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    result = await blackbox_dispatch.generate(_rollout_args(), SampleLike(), {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert isinstance(result, list)
    register_call = next(call for call in paddock.calls if call[0] == "register_agent")
    assert register_call[2]["backend_options"]["proxy"]["max_steps"] == 23


class FakeProxy:
    def __init__(self):
        self.calls = []

    async def finalize_session(self, session_id, *, instance_id=None, label=None):
        self.calls.append(("finalize", session_id, instance_id, label))
        return {"success": True}

    async def read_trajectory(self, **kwargs):
        self.calls.append(("read", kwargs))
        return {
            "success": True,
            "data": [
                {
                    "uid": "seg-0",
                    "segment_index": 0,
                    "timestamp": 1.0,
                    "tokens": [10, 11],
                    "full_loss_mask": [0, 1],
                    "full_logprobs": [0.0, -0.1],
                    "full_versions": ["v0", "v1"],
                    "messages": [{"role": "assistant", "content": "old"}],
                    "extra_info": {"segment": 0},
                },
                {
                    "uid": "seg-1",
                    "segment_index": 1,
                    "timestamp": 2.0,
                    "tokens": [20, 21, 22],
                    "full_loss_mask": [0, 1, 1],
                    "full_logprobs": [0.0, -0.2, -0.3],
                    "full_versions": ["v0", "v2", "v2"],
                    "messages": [{"role": "assistant", "content": "final"}],
                    "extra_info": {"segment": 1},
                },
            ],
        }


def _rollout_args(
    *,
    max_tokens_per_gpu: int = 8,
    context_parallel_size: int = 2,
    rollout_max_response_len: int = 4,
    rollout_temperature: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_tokens_per_gpu=max_tokens_per_gpu,
        context_parallel_size=context_parallel_size,
        rollout_max_response_len=rollout_max_response_len,
        rollout_temperature=rollout_temperature,
    )


def _blackbox_execute_cmds(
    *,
    before_agent: list[dict] | None = None,
    after_agent: list[dict] | None = None,
) -> dict:
    payload = {}
    if before_agent is not None:
        payload["before_agent"] = before_agent
    if after_agent is not None:
        payload["after_agent"] = after_agent
    return payload


def test_ensure_blackbox_session_id_reuses_prefixes_and_writes_generated(monkeypatch):
    prefixed = SampleLike(session_id="bbs-existing")
    assert blackbox_dispatch.ensure_blackbox_session_id(prefixed) == "bbs-existing"
    assert prefixed.session_id == "bbs-existing"

    unprefixed = SampleLike(session_id="existing")
    assert blackbox_dispatch.ensure_blackbox_session_id(unprefixed) == "bbs-existing"
    assert unprefixed.session_id == "bbs-existing"

    monkeypatch.setattr(prewarm_store.uuid, "uuid4", lambda: "generated")
    missing = SampleLike(session_id=None)
    assert blackbox_dispatch.ensure_blackbox_session_id(missing) == "bbs-generated"
    assert missing.session_id == "bbs-generated"


def test_chat_messages_from_prompt_copies_list_and_wraps_text():
    prompt_messages = [{"role": "user", "content": "hi"}]

    messages = blackbox_dispatch._chat_messages_from_prompt(prompt_messages)
    messages[0]["content"] = "changed"

    assert prompt_messages == [{"role": "user", "content": "hi"}]
    assert blackbox_dispatch._chat_messages_from_prompt("hello") == [
        {"role": "user", "content": "hello"},
    ]


def test_parse_blackbox_execute_cmds_accepts_stage_dict():
    schedule = parse_blackbox_execute_cmds(
        _blackbox_execute_cmds(
            before_agent=[
                {
                    "name": "before",
                    "cmd": " echo before ",
                    "timeout": 5,
                    "required": True,
                }
            ],
            after_agent=[
                {
                    "name": "after",
                    "cmd": "echo after",
                    "required": False,
                }
            ],
        )
    )

    assert [command.name for command in schedule["before_agent"]] == ["before"]
    assert schedule["before_agent"][0].cmd == "echo before"
    assert schedule["before_agent"][0].timeout == 5.0
    assert schedule["before_agent"][0].required is True
    assert [command.name for command in schedule["after_agent"]] == ["after"]
    assert schedule["after_agent"][0].timeout is None


def test_parse_blackbox_execute_cmds_rejects_legacy_list_shape():
    legacy_stage_key = "when"
    legacy_stage = "before_agent"
    with pytest.raises(ValueError, match="dict"):
        parse_blackbox_execute_cmds(
            [
                {
                    legacy_stage_key: legacy_stage,
                    "name": "legacy",
                    "cmd": "echo legacy",
                    "required": True,
                }
            ]
        )


def test_parse_blackbox_execute_cmds_rejects_unknown_stage():
    with pytest.raises(ValueError, match="during_agent"):
        parse_blackbox_execute_cmds({"during_agent": []})


def test_parse_blackbox_execute_cmds_rejects_non_list_stage():
    with pytest.raises(ValueError, match="before_agent"):
        parse_blackbox_execute_cmds({"before_agent": {"name": "not-a-list"}})


@pytest.mark.parametrize(
    ("command", "match"),
    [
        ({"cmd": "echo missing-name", "required": True}, "name"),
        ({"name": "missing_cmd", "required": True}, "cmd"),
        ({"name": "missing_required", "cmd": "echo nope"}, "required"),
    ],
)
def test_parse_blackbox_execute_cmds_rejects_missing_required_keys(command, match):
    with pytest.raises(ValueError, match=match):
        parse_blackbox_execute_cmds({"before_agent": [command]})


def test_parse_blackbox_execute_cmds_rejects_unknown_command_key():
    with pytest.raises(ValueError, match="extra"):
        parse_blackbox_execute_cmds(
            {
                "before_agent": [
                    {
                        "name": "bad",
                        "cmd": "echo bad",
                        "required": True,
                        "extra": "nope",
                    }
                ]
            }
        )


def test_parse_blackbox_execute_cmds_requires_explicit_bool_required():
    with pytest.raises(ValueError, match="required"):
        parse_blackbox_execute_cmds(
            {
                "before_agent": [
                    {
                        "name": "bad",
                        "cmd": "echo bad",
                        "required": "true",
                    }
                ]
            }
        )


def test_extract_routed_experts_combines_partial_last_step_chunks():
    args = SimpleNamespace(num_layers=1, moe_router_topk=1)
    segment = {
        "routed_experts_chunks": [
            {
                "data": _encode_routed_experts([10, 11, 12, 13]),
                "prefix_token_count": 3,
                "output_token_count": 2,
                "is_first_chunk": True,
            },
            {
                "data": _encode_routed_experts([20, 21, 22, 23, 24, 25]),
                "prefix_token_count": 5,
                "output_token_count": 2,
                "is_first_chunk": False,
            },
        ],
    }

    routed = trajectory_sample.extract_routed_experts(
        segment,
        args,
        expected_token_count=7,
    )

    assert routed.shape == (6, 1, 1)
    assert routed.reshape(-1).tolist() == [10, 11, 12, 13, 24, 25]


def test_extract_routed_experts_combines_partial_tito_parts():
    args = SimpleNamespace(num_layers=1, moe_router_topk=1)
    segment = {
        "routed_experts_parts": [
            {
                "prefix_token_count": 0,
                "concat_token_count": 4,
                "is_first_step": True,
                "chunks": [
                    {
                        "data": _encode_routed_experts([1, 2]),
                        "prefix_token_count": 2,
                        "output_token_count": 1,
                        "is_first_chunk": True,
                    },
                    {
                        "data": _encode_routed_experts([90, 91, 3]),
                        "prefix_token_count": 3,
                        "output_token_count": 1,
                        "is_first_chunk": False,
                    },
                ],
            },
            {
                "prefix_token_count": 4,
                "concat_token_count": 3,
                "is_first_step": False,
                "chunks": [
                    {
                        "data": _encode_routed_experts([10, 11, 12, 13, 14, 15]),
                        "prefix_token_count": 6,
                        "output_token_count": 1,
                        "is_first_chunk": True,
                    },
                ],
            },
        ],
    }

    routed = trajectory_sample.extract_routed_experts(
        segment,
        args,
        expected_token_count=7,
    )

    assert routed.shape == (6, 1, 1)
    assert routed.reshape(-1).tolist() == [1, 2, 3, 13, 14, 15]


def _dynamic_backend_options(
    *,
    context: int = 16,
    output: int = 4,
    input_tokens: int = 12,
    reserved: int = 3,
    default_temperature: float = 1.0,
) -> dict:
    return {
        "provider_id": "sglang",
        "provider_name": "Dressage Proxy",
        "provider_package": "@ai-sdk/openai-compatible",
        "model_id": "proxy-model",
        "model_name": "Dressage Proxy",
        "proxy": {"default_temperature": default_temperature},
        "model_limit": {
            "context": context,
            "output": output,
            "input": input_tokens,
        },
        "compaction": {
            **DEFAULT_OPENCODE_COMPACTION,
            "reserved": reserved,
        },
    }


def test_blackbox_dispatch_writes_last_segment_to_sample(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_writes_last_segment_to_sample(monkeypatch))


async def _run_blackbox_dispatch_writes_last_segment_to_sample(monkeypatch):
    paddock = FakePaddock()
    proxy = FakeProxy()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)

    args = _rollout_args(
        max_tokens_per_gpu=4,
        context_parallel_size=1,
        rollout_max_response_len=2,
    )
    sample = SampleLike(metadata={"reward_fn": "constant"})
    result = await blackbox_dispatch.generate(args, sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result, expected_count=2)
    assert segment_sample.tokens == [20, 21, 22]
    assert segment_sample.response_length == 2
    assert segment_sample.loss_mask == [1, 1]
    assert segment_sample.rollout_log_probs == [-0.2, -0.3]
    assert segment_sample.response == "final"
    assert segment_sample.status == SampleLike.Status.COMPLETED
    assert segment_sample.metadata["instance_id"] == "7"
    assert segment_sample.metadata["session_id"] == "bbs-sess-7"
    assert segment_sample.metadata["segment_count"] == 2
    assert segment_sample.metadata["selected_segment_index"] == 1
    assert segment_sample.metadata["all_segment_uids"] == ["seg-0", "seg-1"]
    assert segment_sample.metadata["execute_cmds"] == []
    assert "truncated" not in segment_sample.metadata
    assert proxy.calls[-1] == (
        "read",
        {"trajectory_id": "bbs-sess-7", "instance_id": "7", "drain": True},
    )
    assert not [call for call in paddock.calls if call[0] == "execute_cmd"]
    assert paddock.calls[-1][0] == "terminate"


def test_blackbox_dispatch_claims_prewarm_handle(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_claims_prewarm_handle(monkeypatch))


async def _run_blackbox_dispatch_claims_prewarm_handle(monkeypatch):
    default_paddock = FakePaddock()
    prewarmed_paddock = FakePaddock()
    proxy = FakeProxy()
    state = {"sandbox_url": "http://prewarmed.test"}
    handle = PrewarmHandle(
        session_id="bbs-sess-7",
        group_id=21,
        paddock=prewarmed_paddock,
        state=state,
        env_args={"from": "prewarm"},
    )

    async def claim(session_id):
        assert session_id == "bbs-sess-7"
        return handle

    monkeypatch.setattr(generate_runtime, "_PADDOCK", default_paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)
    monkeypatch.setattr(blackbox_dispatch, "claim_prewarm", claim)

    result = await blackbox_dispatch.generate(_rollout_args(), SampleLike(), {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert _last_segment_sample(result).status == SampleLike.Status.COMPLETED
    assert default_paddock.calls == []
    assert not [item for item in prewarmed_paddock.calls if item[0] == "init"]
    assert prewarmed_paddock.calls[0][0:2] == ("register_agent", state)
    assert prewarmed_paddock.calls[-1] == (
        "terminate",
        "bbs-sess-7",
        {"from": "prewarm"},
    )


def test_blackbox_dispatch_prefixes_generated_session_id(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_prefixes_generated_session_id(monkeypatch))


async def _run_blackbox_dispatch_prefixes_generated_session_id(monkeypatch):
    fixed_uuid = "12345678-1234-5678-1234-567812345678"
    expected_session_id = f"bbs-{fixed_uuid}"
    paddock = FakePaddock()
    proxy = FakeProxy()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)
    monkeypatch.setattr(prewarm_store.uuid, "uuid4", lambda: fixed_uuid)
    monkeypatch.setenv("DRESSAGE_PROXY_URL", "http://127.0.0.1:8800")

    sample = SampleLike(session_id=None)
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    _last_segment_sample(result, expected_count=2)
    assert sample.session_id == expected_session_id
    assert sample.metadata["session_id"] == expected_session_id
    assert paddock.calls[0] == ("init", expected_session_id, None, {})
    assert paddock.calls[1] == (
        "register_agent",
        {"sandbox_url": "http://sandbox.test"},
            {
                "instance_id": "7",
                "session_id": expected_session_id,
                "router_url": "http://127.0.0.1:8800",
                "blackbox_type": "opencode",
                "backend_options": _dynamic_backend_options(),
            },
    )
    assert paddock.calls[2] == (
        "call_agent",
        {"sandbox_url": "http://sandbox.test"},
        {
            "session_id": expected_session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {
                "source": "dressage",
                "execute_cmds": sample.metadata["execute_cmds"],
                "session_id": expected_session_id,
                "instance_id": "7",
            },
        },
    )
    assert paddock.calls[-1] == ("terminate", expected_session_id, {})
    assert sample.metadata["execute_cmds"] == []
    assert not [call for call in paddock.calls if call[0] == "execute_cmd"]
    assert proxy.calls == [
        ("finalize", expected_session_id, "7", None),
        (
            "read",
            {"trajectory_id": expected_session_id, "instance_id": "7", "drain": True},
        ),
    ]


def test_blackbox_dispatch_records_execute_cmds_in_actual_order(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_records_execute_cmds_in_actual_order(monkeypatch))


async def _run_blackbox_dispatch_records_execute_cmds_in_actual_order(monkeypatch):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={
            "blackbox_execute_cmds": _blackbox_execute_cmds(
                before_agent=[
                    {
                        "name": "before_one",
                        "cmd": " echo before-one ",
                        "timeout": None,
                        "required": True,
                    },
                    {
                        "name": "before_two",
                        "cmd": "echo before-two",
                        "timeout": 1,
                        "required": True,
                    },
                ],
                after_agent=[
                    {
                        "name": "after_one",
                        "cmd": "echo after-one",
                        "timeout": 5,
                        "required": True,
                    },
                ],
            )
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result, expected_count=2)
    assert segment_sample.status == SampleLike.Status.COMPLETED
    assert "execute_cmd_results" not in segment_sample.metadata
    assert [item["name"] for item in segment_sample.metadata["execute_cmds"]] == [
        "before_one",
        "before_two",
        "after_one",
    ]
    assert [item["cmd"] for item in segment_sample.metadata["execute_cmds"]] == [
        "echo before-one",
        "echo before-two",
        "echo after-one",
    ]
    assert [item["stage"] for item in segment_sample.metadata["execute_cmds"]] == [
        "before_agent",
        "before_agent",
        "after_agent",
    ]
    assert [call[0] for call in paddock.calls] == [
        "init",
        "register_agent",
        "execute_cmd",
        "execute_cmd",
        "call_agent",
        "execute_cmd",
        "terminate",
    ]


def test_blackbox_dispatch_required_execute_cmd_failure_aborts(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_required_execute_cmd_failure_aborts(monkeypatch))


async def _run_blackbox_dispatch_required_execute_cmd_failure_aborts(monkeypatch):
    paddock = FailingExecuteCmdPaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={
            "blackbox_execute_cmds": _blackbox_execute_cmds(
                before_agent=[
                    {
                        "name": "required_failure",
                        "cmd": "exit 2",
                        "timeout": 5,
                        "required": True,
                    },
                ],
            )
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result is sample
    assert result.status == SampleLike.Status.ABORTED
    assert sample.metadata["execute_cmds"][0]["name"] == "required_failure"
    assert sample.metadata["execute_cmds"][0]["cmd_result"]["returncode"] == 2
    assert "required execute_cmd failed" in sample.metadata["blackbox_error"]
    assert not [call for call in paddock.calls if call[0] == "call_agent"]


def test_blackbox_dispatch_optional_execute_cmd_failure_continues(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_optional_execute_cmd_failure_continues(monkeypatch))


async def _run_blackbox_dispatch_optional_execute_cmd_failure_continues(monkeypatch):
    paddock = FailingExecuteCmdPaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={
            "blackbox_execute_cmds": _blackbox_execute_cmds(
                before_agent=[
                    {
                        "name": "optional_failure",
                        "cmd": "exit 2",
                        "timeout": 5,
                        "required": False,
                    },
                ],
            )
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result, expected_count=2)
    assert segment_sample.status == SampleLike.Status.COMPLETED
    assert segment_sample.metadata["execute_cmds"][0]["required"] is False
    assert segment_sample.metadata["execute_cmds"][0]["cmd_result"]["returncode"] == 2
    assert any(call[0] == "call_agent" for call in paddock.calls)


def test_blackbox_dispatch_context_overflow_is_reported_before_after_agent_cmd(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_context_overflow_is_reported_before_after_agent_cmd(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_context_overflow_is_reported_before_after_agent_cmd(monkeypatch):
    paddock = ContextOverflowPaddock()
    proxy = FakeProxy()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)

    sample = SampleLike(
        metadata={
            "blackbox_execute_cmds": _blackbox_execute_cmds(
                before_agent=[
                    {
                        "name": "env_check",
                        "cmd": "python -V && ls -la",
                        "timeout": 30,
                        "required": False,
                    },
                ],
                after_agent=[
                    {
                        "name": "inspect_files",
                        "cmd": "find . -maxdepth 2 -type f",
                        "timeout": 30,
                        "required": False,
                    },
                ],
            )
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert isinstance(result, list)
    assert len(result) == 2
    assert "blackbox_error" not in sample.metadata
    for segment_sample in result:
        assert segment_sample.status == SampleLike.Status.COMPLETED
        assert segment_sample.metadata["blackbox_agent_early_stop"] is True
        assert (
            segment_sample.metadata["blackbox_agent_early_stop_kind"]
            == "context_overflow"
        )
        assert segment_sample.metadata["blackbox_agent_error_kind"] == "context_overflow"
        assert segment_sample.metadata["blackbox_agent_http_status_code"] == 413
        assert segment_sample.metadata["blackbox_agent_error_details"] == {
            "input_tokens": 40000,
            "context_window": 32768,
            "max_tokens": 8192,
            "raw_error_code": "context_length_exceeded",
        }
        assert [item["name"] for item in segment_sample.metadata["execute_cmds"]] == [
            "env_check",
            "inspect_files",
        ]
    assert [call[0] for call in paddock.calls] == [
        "init",
        "register_agent",
        "execute_cmd",
        "call_agent",
        "execute_cmd",
        "terminate",
    ]
    assert proxy.calls == [
        ("finalize", "bbs-sess-7", "7", None),
        (
            "read",
            {
                "trajectory_id": "bbs-sess-7",
                "instance_id": "7",
                "drain": True,
            },
        ),
    ]


def test_blackbox_dispatch_max_steps_exceeded_is_reported(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_max_steps_exceeded_is_reported(monkeypatch))


async def _run_blackbox_dispatch_max_steps_exceeded_is_reported(monkeypatch):
    paddock = MaxStepsExceededPaddock()
    proxy = FakeProxy()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)

    sample = SampleLike(
        metadata={
            "blackbox_execute_cmds": _blackbox_execute_cmds(
                before_agent=[
                    {
                        "name": "env_check",
                        "cmd": "python -V && ls -la",
                        "timeout": 30,
                        "required": False,
                    },
                ],
                after_agent=[
                    {
                        "name": "inspect_files",
                        "cmd": "find . -maxdepth 2 -type f",
                        "timeout": 30,
                        "required": False,
                    },
                ],
            )
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert isinstance(result, list)
    assert len(result) == 2
    assert "blackbox_error" not in sample.metadata
    for segment_sample in result:
        assert segment_sample.status == SampleLike.Status.COMPLETED
        assert segment_sample.metadata["blackbox_agent_early_stop"] is True
        assert (
            segment_sample.metadata["blackbox_agent_early_stop_kind"]
            == "max_steps_exceeded"
        )
        assert segment_sample.metadata["blackbox_agent_error_kind"] == "max_steps_exceeded"
        assert segment_sample.metadata["blackbox_agent_http_status_code"] == 429
        assert segment_sample.metadata["blackbox_agent_error_details"] == {
            "max_steps": 1,
            "attempted_step": 1,
            "raw_error_code": "max_steps_exceeded",
        }
        assert [item["name"] for item in segment_sample.metadata["execute_cmds"]] == [
            "env_check",
            "inspect_files",
        ]
    assert [call[0] for call in paddock.calls] == [
        "init",
        "register_agent",
        "execute_cmd",
        "call_agent",
        "execute_cmd",
        "terminate",
    ]
    assert proxy.calls == [
        ("finalize", "bbs-sess-7", "7", None),
        (
            "read",
            {
                "trajectory_id": "bbs-sess-7",
                "instance_id": "7",
                "drain": True,
            },
        ),
    ]


def test_blackbox_dispatch_backend_timeout_still_aborts(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_backend_timeout_still_aborts(monkeypatch))


async def _run_blackbox_dispatch_backend_timeout_still_aborts(monkeypatch):
    paddock = BackendTimeoutPaddock()
    proxy = FakeProxy()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result is sample
    assert result.status == SampleLike.Status.ABORTED
    assert sample.metadata["blackbox_agent_error_kind"] == "backend_timeout"
    assert sample.metadata["blackbox_agent_http_status_code"] == 504
    assert sample.metadata["blackbox_agent_error_details"] == {
        "session_id": "bbs-sess-7",
        "turn_id": "turn-timeout",
    }
    assert "blackbox_agent_early_stop" not in sample.metadata
    assert "blackbox agent backend timeout" in sample.metadata["blackbox_error"]
    assert [call[0] for call in paddock.calls] == [
        "init",
        "register_agent",
        "call_agent",
        "terminate",
    ]
    assert proxy.calls == []


def test_blackbox_dispatch_optional_execute_cmd_http_failure_continues(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_optional_execute_cmd_http_failure_continues(monkeypatch)
    )


async def _run_blackbox_dispatch_optional_execute_cmd_http_failure_continues(monkeypatch):
    paddock = OptionalHttpExecuteCmdPaddock()
    proxy = FakeProxy()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)

    sample = SampleLike(
        metadata={
            "blackbox_execute_cmds": _blackbox_execute_cmds(
                before_agent=[
                    {
                        "name": "optional_http_failure",
                        "cmd": "echo optional",
                        "timeout": 5,
                        "required": False,
                    },
                ],
            )
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result, expected_count=2)
    assert segment_sample.status == SampleLike.Status.COMPLETED
    record = segment_sample.metadata["execute_cmds"][0]
    assert record["name"] == "optional_http_failure"
    assert record["required"] is False
    assert record["cmd_error"]["type"].endswith("HTTPStatusError")
    assert record["http"]["response"]["status_code"] == 503
    assert any(call[0] == "call_agent" for call in paddock.calls)
    assert proxy.calls[0][0] == "finalize"


def test_blackbox_dispatch_execute_cmd_http_409_aborts(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_execute_cmd_http_409_aborts(monkeypatch))


async def _run_blackbox_dispatch_execute_cmd_http_409_aborts(monkeypatch):
    class ConflictExecuteCmdPaddock(FakePaddock):
        async def execute_cmd(self, state, **kwargs):
            self.calls.append(("execute_cmd", state, kwargs))
            request = httpx.Request(
                "POST",
                "http://sandbox.test/v1/sessions/bbs-sess-7/execute_cmd",
                json={"cmd": kwargs["cmd"], "timeout": kwargs["timeout"]},
            )
            response = httpx.Response(
                409,
                json={"error": "session_aborted"},
                request=request,
            )
            raise httpx.HTTPStatusError(
                "Client error '409 Conflict'",
                request=request,
                response=response,
            )

    paddock = ConflictExecuteCmdPaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={
            "blackbox_execute_cmds": _blackbox_execute_cmds(
                before_agent=[
                    {
                        "name": "conflict",
                        "cmd": "echo conflict",
                        "timeout": 5,
                        "required": True,
                    },
                ],
            )
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result.status == SampleLike.Status.ABORTED
    assert sample.metadata["execute_cmds"][0]["name"] == "conflict"
    assert sample.metadata["execute_cmds"][0]["http"]["response"]["status_code"] == 409
    assert "409 Conflict" in sample.metadata["blackbox_error"]


def test_blackbox_dispatch_abort_clears_session_for_retry(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_abort_clears_session_for_retry(monkeypatch))


async def _run_blackbox_dispatch_abort_clears_session_for_retry(monkeypatch):
    paddock = FailingRegisterPaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result is sample
    assert result.status == SampleLike.Status.ABORTED
    assert sample.session_id is None
    assert "session_id" not in sample.metadata
    assert sample.metadata["last_failed_session_id"] == "bbs-sess-7"
    assert sample.metadata["blackbox_error"] == "duplicate session"
    assert sample.metadata["blackbox_failure_history"] == [
        {
            "session_id": "bbs-sess-7",
            "error_type": "RuntimeError",
            "error": "duplicate session",
            "retry_count": 0,
        }
    ]
    assert paddock.calls[-1] == ("terminate", "bbs-sess-7", {})


def test_blackbox_dispatch_retry_uses_new_session_id_after_abort(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_retry_uses_new_session_id_after_abort(monkeypatch)
    )


async def _run_blackbox_dispatch_retry_uses_new_session_id_after_abort(monkeypatch):
    retry_uuid = "12345678-1234-5678-1234-567812345678"
    retry_session_id = f"bbs-{retry_uuid}"
    paddock = FailingOnceRegisterPaddock()
    proxy = FakeProxy()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", proxy)
    monkeypatch.setattr(prewarm_store.uuid, "uuid4", lambda: retry_uuid)

    sample = SampleLike()
    first = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert first.status == SampleLike.Status.ABORTED
    assert sample.session_id is None
    assert sample.metadata["last_failed_session_id"] == "bbs-sess-7"

    second = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(second, expected_count=2)
    assert segment_sample.status == SampleLike.Status.COMPLETED
    assert sample.session_id == retry_session_id
    assert segment_sample.metadata["session_id"] == retry_session_id
    assert "blackbox_error" not in segment_sample.metadata
    assert segment_sample.metadata["last_failed_session_id"] == "bbs-sess-7"
    assert (
        segment_sample.metadata["blackbox_failure_history"][0]["session_id"]
        == "bbs-sess-7"
    )
    assert [
        call[1] for call in paddock.calls if call[0] == "init"
    ] == ["bbs-sess-7", retry_session_id]
    register_calls = [call for call in paddock.calls if call[0] == "register_agent"]
    assert register_calls[1][2]["session_id"] == retry_session_id
    assert any(
        call[0] == "call_agent" and call[2]["session_id"] == retry_session_id
        for call in paddock.calls
    )
    assert proxy.calls == [
        ("finalize", retry_session_id, "7", None),
        (
            "read",
            {"trajectory_id": retry_session_id, "instance_id": "7", "drain": True},
        ),
    ]


def test_blackbox_dispatch_merges_dynamic_opencode_backend_options(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_merges_dynamic_opencode_backend_options(monkeypatch))


async def _run_blackbox_dispatch_merges_dynamic_opencode_backend_options(monkeypatch):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    backend_options = {
        "provider_id": "custom-provider",
        "model_limit": {"output": 2},
        "compaction": {"tail_turns": 4},
        "unknown": {"kept": True},
    }
    sample = SampleLike(metadata={"backend_options": backend_options})
    result = await blackbox_dispatch.generate(
        _rollout_args(
            max_tokens_per_gpu=6,
            context_parallel_size=2,
            rollout_max_response_len=4,
        ),
        sample,
        {},
    )
    await paddock_lifecycle.drain_terminate_tasks()

    assert _last_segment_sample(result).status == SampleLike.Status.COMPLETED
    assert backend_options == {
        "provider_id": "custom-provider",
        "model_limit": {"output": 2},
        "compaction": {"tail_turns": 4},
        "unknown": {"kept": True},
    }
    assert paddock.calls[1][2]["backend_options"] == {
        "provider_id": "custom-provider",
        "provider_name": "Dressage Proxy",
        "provider_package": "@ai-sdk/openai-compatible",
        "model_id": "proxy-model",
        "model_name": "Dressage Proxy",
        "proxy": {"default_temperature": 1.0},
        "model_limit": {"context": 12, "output": 2, "input": 8},
        "compaction": {
            "auto": True,
            "prune": True,
            "tail_turns": 4,
            "reserved": 2,
        },
        "unknown": {"kept": True},
    }


def test_blackbox_dispatch_caps_dynamic_opencode_reserved():
    options = dynamic_backend_defaults_for(
        "opencode",
        _rollout_args(
            max_tokens_per_gpu=100_000,
            context_parallel_size=1,
            rollout_max_response_len=1_000,
        ),
    )

    assert options["model_limit"] == {
        "context": 100_000,
        "output": 1_000,
        "input": 99_000,
    }
    assert options["compaction"]["reserved"] == 8192


def test_blackbox_dispatch_uses_explicit_opencode_compact_threshold(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_COMPACT_THRESHOLD", "10")

    options = dynamic_backend_defaults_for(
        "opencode",
        _rollout_args(
            max_tokens_per_gpu=6,
            context_parallel_size=2,
            rollout_max_response_len=4,
        ),
    )

    assert options["model_limit"] == {
        "context": 12,
        "output": 4,
        "input": 12,
    }
    assert options["compaction"]["reserved"] == 2


def test_blackbox_dispatch_preserves_explicit_reserved_override(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_preserves_explicit_reserved_override(monkeypatch))


async def _run_blackbox_dispatch_preserves_explicit_reserved_override(monkeypatch):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={"backend_options": {"compaction": {"reserved": 1234}}}
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert _last_segment_sample(result).status == SampleLike.Status.COMPLETED
    assert paddock.calls[1][2]["backend_options"]["compaction"]["reserved"] == 1234


def test_blackbox_dispatch_preserves_explicit_none_backend_option_sections(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_preserves_explicit_none_backend_option_sections(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_preserves_explicit_none_backend_option_sections(
    monkeypatch,
):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={"backend_options": {"model_limit": None, "compaction": None}}
    )
    result = await blackbox_dispatch.generate(None, sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert _last_segment_sample(result).status == SampleLike.Status.COMPLETED
    assert paddock.calls[1][2]["backend_options"] == {
        "provider_id": "sglang",
        "provider_name": "Dressage Proxy",
        "provider_package": "@ai-sdk/openai-compatible",
        "model_id": "proxy-model",
        "model_name": "Dressage Proxy",
        "model_limit": None,
        "compaction": None,
    }


def test_blackbox_dispatch_aborts_invalid_dynamic_opencode_limit(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_aborts_invalid_dynamic_opencode_limit(monkeypatch))


async def _run_blackbox_dispatch_aborts_invalid_dynamic_opencode_limit(monkeypatch):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike()
    result = await blackbox_dispatch.generate(
        _rollout_args(
            max_tokens_per_gpu=2,
            context_parallel_size=2,
            rollout_max_response_len=4,
        ),
        sample,
        {},
    )

    assert result is sample
    assert result.status == SampleLike.Status.ABORTED
    assert paddock.calls == []
    assert "--max-tokens-per-gpu" in result.metadata["blackbox_error"]
    assert "--rollout-max-response-len" in result.metadata["blackbox_error"]


def test_blackbox_dispatch_does_not_inject_dynamic_options_for_non_opencode(
    monkeypatch,
):
    asyncio.run(
        _run_blackbox_dispatch_does_not_inject_dynamic_options_for_non_opencode(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_does_not_inject_dynamic_options_for_non_opencode(
    monkeypatch,
):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={
            "blackbox_type": "custom",
            "backend_options": {"unknown": "kept"},
        }
    )
    result = await blackbox_dispatch.generate(None, sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert _last_segment_sample(result).status == SampleLike.Status.COMPLETED
    assert paddock.calls[1][2]["blackbox_type"] == "custom"
    assert paddock.calls[1][2]["backend_options"] == {"unknown": "kept"}


def test_blackbox_dispatch_merges_dynamic_openclaw_backend_options(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_merges_dynamic_openclaw_backend_options(monkeypatch))


async def _run_blackbox_dispatch_merges_dynamic_openclaw_backend_options(monkeypatch):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(metadata={"blackbox_type": "openclaw"})
    result = await blackbox_dispatch.generate(
        _rollout_args(
            max_tokens_per_gpu=6,
            context_parallel_size=2,
            rollout_max_response_len=4,
        ),
        sample,
        {},
    )
    await paddock_lifecycle.drain_terminate_tasks()

    assert _last_segment_sample(result).status == SampleLike.Status.COMPLETED
    assert paddock.calls[0] == (
        "init",
        "bbs-sess-7",
        None,
        {"blackbox_type": "openclaw"},
    )
    backend_options = paddock.calls[1][2]["backend_options"]
    assert backend_options == {
        "agent_id": "default",
        "provider_id": "sglang",
        "model_id": "proxy-model",
        "model_name": "Dressage Proxy",
        "api_key": "sglang-local",
        "proxy": {"default_temperature": 1.0},
        "context_window": 12,
        "max_tokens": 4,
        "request": {"max_tokens": 4},
        "compaction": {
            "reserve_tokens": 2,
            "reserve_tokens_floor": 2,
        },
    }
    assert "model_limit" not in backend_options
    assert "provider_package" not in backend_options


def test_blackbox_dispatch_passes_non_dict_opencode_backend_options_through(
    monkeypatch,
):
    asyncio.run(
        _run_blackbox_dispatch_passes_non_dict_opencode_backend_options_through(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_passes_non_dict_opencode_backend_options_through(
    monkeypatch,
):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(metadata={"backend_options": "invalid-options"})
    result = await blackbox_dispatch.generate(None, sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert _last_segment_sample(result).status == SampleLike.Status.COMPLETED
    assert paddock.calls[1][2]["backend_options"] == "invalid-options"


def test_blackbox_dispatch_maps_multi_turn_suffix_to_sample(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_maps_multi_turn_suffix_to_sample(monkeypatch))


async def _run_blackbox_dispatch_maps_multi_turn_suffix_to_sample(monkeypatch):
    class MultiTurnProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            self.calls.append(("read", kwargs))
            return {
                "success": True,
                "data": [
                    {
                        "uid": "seg-multi",
                        "segment_index": 0,
                        "timestamp": 1.0,
                        "tokens": [10, 11, 20, 30, 21],
                        "full_loss_mask": [0, 0, 1, 0, 1],
                        "full_logprobs": [0.0, 0.0, -0.1, 0.0, -0.2],
                        "full_versions": ["v0", "v0", "v1", "v0", "v2"],
                        "messages": [{"role": "assistant", "content": "multi"}],
                        "extra_info": {"segment": "multi"},
                    },
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", MultiTurnProxy())

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result)
    assert segment_sample.tokens == [10, 11, 20, 30, 21]
    assert segment_sample.response_length == 3
    assert segment_sample.loss_mask == [1, 0, 1]
    assert segment_sample.rollout_log_probs == [-0.1, 0.0, -0.2]
    assert segment_sample.metadata["full_versions"] == ["v0", "v0", "v1", "v0", "v2"]
    assert segment_sample.metadata["dressage_start_token_version"] == "v1"
    assert segment_sample.metadata["dressage_end_token_version"] == "v2"
    assert "response_versions" not in segment_sample.metadata
    assert "response_version_spans" not in segment_sample.metadata
    assert segment_sample.metadata["dressage_partial_rollout"] is True
    assert segment_sample.status == SampleLike.Status.COMPLETED


def test_blackbox_dispatch_truncates_segment_to_token_cap(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_truncates_segment_to_token_cap(monkeypatch))


async def _run_blackbox_dispatch_truncates_segment_to_token_cap(monkeypatch):
    class LongSegmentProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            self.calls.append(("read", kwargs))
            return {
                "success": True,
                "data": [
                    {
                        "uid": "seg-long",
                        "segment_index": 0,
                        "timestamp": 1.0,
                        "tokens": [30, 31, 32, 33, 34],
                        "full_loss_mask": [0, 0, 1, 0, 1],
                        "full_logprobs": [0.0, 0.0, -0.2, 0.0, -0.4],
                        "full_versions": ["v0", "v0", "v1", "v0", "v2"],
                        "messages": [{"role": "assistant", "content": "long final"}],
                        "extra_info": {"segment": "long"},
                    },
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", LongSegmentProxy())

    args = _rollout_args(
        max_tokens_per_gpu=2,
        context_parallel_size=2,
        rollout_max_response_len=2,
    )
    sample = SampleLike()
    result = await blackbox_dispatch.generate(args, sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result)
    assert segment_sample.tokens == [30, 31, 32, 33]
    assert segment_sample.response_length == 2
    assert segment_sample.loss_mask == [1, 0]
    assert segment_sample.rollout_log_probs == [-0.2, 0.0]
    assert segment_sample.metadata["full_versions"] == ["v0", "v0", "v1", "v0"]
    assert segment_sample.metadata["dressage_start_token_version"] == "v1"
    assert segment_sample.metadata["dressage_end_token_version"] == "v1"
    assert "response_versions" not in segment_sample.metadata
    assert "response_version_spans" not in segment_sample.metadata
    assert "dressage_partial_rollout" not in segment_sample.metadata
    assert segment_sample.response == "long final"
    assert segment_sample.status == SampleLike.Status.COMPLETED
    assert segment_sample.metadata["truncated"] is True


def test_blackbox_dispatch_clears_batch_level_partial_flag_when_output_version_does_not_change(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_clears_batch_level_partial_flag_when_output_version_does_not_change(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_clears_batch_level_partial_flag_when_output_version_does_not_change(monkeypatch):
    class SingleVersionProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            self.calls.append(("read", kwargs))
            return {
                "success": True,
                "data": [
                    {
                        "uid": "seg-single-version",
                        "segment_index": 0,
                        "timestamp": 1.0,
                        "tokens": [10, 11, 12, 13],
                        "full_loss_mask": [0, 1, 0, 1],
                        "full_logprobs": [0.0, -0.1, 0.0, -0.2],
                        "full_versions": ["-1", "v1", "-1", "v1"],
                        "messages": [{"role": "assistant", "content": "single"}],
                    }
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", SingleVersionProxy())

    sample = SampleLike(
        metadata={
            "dressage_start_rollout_id": 3,
            "dressage_async_group_id": 9,
            "dressage_partial_rollout": True,
            "dressage_start_token_version": "stale-start",
            "dressage_end_token_version": "stale-end",
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result)
    assert segment_sample.loss_mask == [1, 0, 1]
    assert segment_sample.metadata["dressage_start_rollout_id"] == 3
    assert segment_sample.metadata["dressage_start_token_version"] == "v1"
    assert segment_sample.metadata["dressage_end_token_version"] == "v1"
    assert "dressage_async_group_id" not in segment_sample.metadata
    assert "dressage_partial_rollout" not in segment_sample.metadata
    assert "response_versions" not in segment_sample.metadata
    assert "response_version_spans" not in segment_sample.metadata
    assert segment_sample.status == SampleLike.Status.COMPLETED


def test_blackbox_dispatch_skips_version_metadata_when_full_versions_missing(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_skips_version_metadata_when_full_versions_missing(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_skips_version_metadata_when_full_versions_missing(monkeypatch):
    class NoVersionProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            self.calls.append(("read", kwargs))
            return {
                "success": True,
                "data": [
                    {
                        "uid": "seg-no-version",
                        "segment_index": 0,
                        "timestamp": 1.0,
                        "tokens": [10, 11, 12, 13],
                        "full_loss_mask": [0, 1, 0, 1],
                        "full_logprobs": [0.0, -0.1, 0.0, -0.2],
                        "messages": [{"role": "assistant", "content": "no-version"}],
                    }
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", NoVersionProxy())

    sample = SampleLike(
        metadata={
            "dressage_partial_rollout": True,
            "full_versions": ["stale"],
            "version_spans": [{"version": "stale"}],
            "dressage_start_token_version": "stale-start",
            "dressage_end_token_version": "stale-end",
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result)
    assert segment_sample.response == "no-version"
    assert segment_sample.loss_mask == [1, 0, 1]
    assert "dressage_partial_rollout" not in segment_sample.metadata
    assert "full_versions" not in segment_sample.metadata
    assert "version_spans" not in segment_sample.metadata
    assert "dressage_start_token_version" not in segment_sample.metadata
    assert "dressage_end_token_version" not in segment_sample.metadata
    assert segment_sample.status == SampleLike.Status.COMPLETED


def test_blackbox_dispatch_marks_partial_only_for_multiple_trainable_output_versions(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_marks_partial_only_for_multiple_trainable_output_versions(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_marks_partial_only_for_multiple_trainable_output_versions(monkeypatch):
    class MultiVersionProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            self.calls.append(("read", kwargs))
            return {
                "success": True,
                "data": [
                    {
                        "uid": "seg-token-level-partial",
                        "segment_index": 0,
                        "timestamp": 1.0,
                        "tokens": [10, 11, 12, 13, 14],
                        "full_loss_mask": [0, 1, 1, 0, 1],
                        "full_logprobs": [0.0, -0.1, -0.2, 0.0, -0.3],
                        "full_versions": ["-1", "v0", "v1", "-1", "v1"],
                        "messages": [{"role": "assistant", "content": "multi-version"}],
                    }
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", MultiVersionProxy())

    sample = SampleLike(metadata={"dressage_start_rollout_id": 4})
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result)
    assert segment_sample.metadata["dressage_start_rollout_id"] == 4
    assert segment_sample.metadata["dressage_start_token_version"] == "v0"
    assert segment_sample.metadata["dressage_end_token_version"] == "v1"
    assert segment_sample.metadata["dressage_partial_rollout"] is True
    assert "dressage_async_group_id" not in segment_sample.metadata
    assert "response_versions" not in segment_sample.metadata
    assert "response_version_spans" not in segment_sample.metadata
    assert segment_sample.status == SampleLike.Status.COMPLETED


def test_blackbox_dispatch_masks_nonlast_version_tokens_when_proxy_flagged(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_masks_nonlast_version_tokens_when_proxy_flagged(monkeypatch)
    )


async def _run_blackbox_dispatch_masks_nonlast_version_tokens_when_proxy_flagged(monkeypatch):
    class MultiVersionProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            self.calls.append(("read", kwargs))
            return {
                "success": True,
                "data": [
                    {
                        "uid": "seg-token-level-partial",
                        "segment_index": 0,
                        "timestamp": 1.0,
                        "tokens": [10, 11, 12, 13, 14],
                        "full_loss_mask": [0, 1, 1, 0, 1],
                        "full_logprobs": [0.0, -0.1, -0.2, 0.0, -0.3],
                        "full_versions": ["-1", "v0", "v1", "-1", "v1"],
                        "messages": [{"role": "assistant", "content": "multi-version"}],
                        "extra_info": {"mask_nonlast_version_tokens": True},
                    }
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", MultiVersionProxy())

    sample = SampleLike(metadata={"dressage_start_rollout_id": 4})
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result)
    assert segment_sample.response_length == 4
    assert segment_sample.loss_mask == [0, 1, 0, 1]
    assert segment_sample.rollout_log_probs == [-0.1, -0.2, 0.0, -0.3]
    assert segment_sample.metadata["dressage_start_rollout_id"] == 4
    assert segment_sample.metadata["dressage_start_token_version"] == "v0"
    assert segment_sample.metadata["dressage_end_token_version"] == "v1"
    assert segment_sample.metadata["dressage_partial_rollout"] is True
    assert segment_sample.status == SampleLike.Status.COMPLETED


def test_blackbox_dispatch_ignores_context_sentinel_versions_for_partial_flag(monkeypatch):
    asyncio.run(
        _run_blackbox_dispatch_ignores_context_sentinel_versions_for_partial_flag(
            monkeypatch
        )
    )


async def _run_blackbox_dispatch_ignores_context_sentinel_versions_for_partial_flag(monkeypatch):
    class ContextSentinelProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            self.calls.append(("read", kwargs))
            return {
                "success": True,
                "data": [
                    {
                        "uid": "seg-context-sentinel",
                        "segment_index": 0,
                        "timestamp": 1.0,
                        "tokens": [10, 11, 12, 13, 14],
                        "full_loss_mask": [0, 1, 0, 0, 1],
                        "full_logprobs": [0.0, -0.1, 0.0, 0.0, -0.2],
                        "full_versions": ["-1", "v1", "-1", "-1", "v1"],
                        "messages": [{"role": "assistant", "content": "sentinel"}],
                    }
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", ContextSentinelProxy())

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result)
    assert segment_sample.metadata["dressage_start_token_version"] == "v1"
    assert segment_sample.metadata["dressage_end_token_version"] == "v1"
    assert "dressage_partial_rollout" not in segment_sample.metadata
    assert "response_versions" not in segment_sample.metadata
    assert "response_version_spans" not in segment_sample.metadata
    assert segment_sample.status == SampleLike.Status.COMPLETED


def test_blackbox_dispatch_passes_e2b_metadata(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_passes_e2b_metadata(monkeypatch))


async def _run_blackbox_dispatch_passes_e2b_metadata(monkeypatch):
    paddock = FakePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())

    sample = SampleLike(
        metadata={
            "sandbox_extra_params": {
                "e2b_envs": {"A": "1"},
                "e2b_metadata": {"trace": "abc"},
            },
            "sandbox_timeout_sec": 120,
        }
    )
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    _last_segment_sample(result)
    assert paddock.calls[0] == (
        "init",
        "bbs-sess-7",
        None,
        {
            "sandbox_extra_params": {
                "e2b_envs": {"A": "1"},
                "e2b_metadata": {"trace": "abc"},
            },
            "sandbox_timeout_sec": 120,
        },
    )


def test_blackbox_dispatch_aborts_invalid_alignment(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_aborts_invalid_alignment(monkeypatch))


async def _run_blackbox_dispatch_aborts_invalid_alignment(monkeypatch):
    class BadProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            return {
                "success": True,
                "data": [
                    {
                        "tokens": [1, 2],
                        "messages": [],
                    }
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", BadProxy())

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result.status == SampleLike.Status.ABORTED
    assert "blackbox_error" in result.metadata
    assert "full_loss_mask" in result.metadata["blackbox_error"]


def test_blackbox_dispatch_aborts_invalid_new_field_alignment(monkeypatch):
    asyncio.run(_run_blackbox_dispatch_aborts_invalid_new_field_alignment(monkeypatch))


async def _run_blackbox_dispatch_aborts_invalid_new_field_alignment(monkeypatch):
    class BadProxy(FakeProxy):
        async def read_trajectory(self, **kwargs):
            return {
                "success": True,
                "data": [
                    {
                        "tokens": [1, 2],
                        "full_loss_mask": [0],
                        "full_logprobs": [0.0, -0.1],
                        "full_versions": ["v0", "v1"],
                        "messages": [],
                    }
                ],
            }

    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", BadProxy())

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result.status == SampleLike.Status.ABORTED
    assert "blackbox_error" in result.metadata
    assert "full_loss_mask length" in result.metadata["blackbox_error"]


def test_blackbox_dispatch_terminate_failure_is_warning_only(monkeypatch, caplog):
    asyncio.run(
        _run_blackbox_dispatch_terminate_failure_is_warning_only(monkeypatch, caplog)
    )


async def _run_blackbox_dispatch_terminate_failure_is_warning_only(monkeypatch, caplog):
    paddock = FailingTerminatePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    caplog.set_level(logging.WARNING, logger=paddock_lifecycle.__name__)

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})

    segment_sample = _last_segment_sample(result, expected_count=2)
    assert segment_sample.tokens == [20, 21, 22]
    assert segment_sample.response == "final"
    assert segment_sample.status == SampleLike.Status.COMPLETED
    assert "blackbox_error" not in segment_sample.metadata

    await paddock_lifecycle.drain_terminate_tasks()

    warning_records = [
        record
        for record in caplog.records
        if "failed to terminate sandbox" in record.getMessage()
    ]
    assert len(warning_records) == 1
    assert "session_id=bbs-sess-7" in warning_records[0].getMessage()
    assert "destroy returned 504" in warning_records[0].getMessage()
    assert "\n" not in warning_records[0].getMessage()
    assert warning_records[0].exc_info is None


def test_blackbox_dispatch_terminate_timeout_default_is_30s(monkeypatch):
    monkeypatch.delenv("DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC", raising=False)

    assert paddock_lifecycle.terminate_timeout_sec() == 30.0


def test_blackbox_dispatch_terminate_timeout_is_warning_only(monkeypatch, caplog):
    asyncio.run(
        _run_blackbox_dispatch_terminate_timeout_is_warning_only(monkeypatch, caplog)
    )


async def _run_blackbox_dispatch_terminate_timeout_is_warning_only(monkeypatch, caplog):
    paddock = SlowTerminatePaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    monkeypatch.setenv("DRESSAGE_PADDOCK_TERMINATE_TIMEOUT_SEC", "0.001")
    caplog.set_level(logging.WARNING, logger=paddock_lifecycle.__name__)

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})

    segment_sample = _last_segment_sample(result, expected_count=2)
    assert segment_sample.tokens == [20, 21, 22]
    assert segment_sample.response == "final"
    assert segment_sample.status == SampleLike.Status.COMPLETED
    assert "blackbox_error" not in segment_sample.metadata

    await paddock_lifecycle.drain_terminate_tasks()

    warning_records = [
        record
        for record in caplog.records
        if "timed out waiting for sandbox release RPC" in record.getMessage()
    ]
    assert len(warning_records) == 1
    assert "session_id=bbs-sess-7" in warning_records[0].getMessage()
    assert warning_records[0].exc_info is None


def test_blackbox_dispatch_logs_trajectory_payload(monkeypatch, tmp_path):
    asyncio.run(_run_blackbox_dispatch_logs_trajectory_payload(monkeypatch, tmp_path))


async def _run_blackbox_dispatch_logs_trajectory_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DRESSAGE_LOG_WRITE_MODE", "await")

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    log_path = tmp_path / "7" / "bbs-sess-7" / "session.json"
    assert log_path.exists()
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["success"] is True
    assert [segment["uid"] for segment in payload["data"]] == ["seg-0", "seg-1"]
    assert "trajectory_payload_log_path" not in _last_segment_sample(result).metadata


def test_blackbox_dispatch_logs_all_segment_samples(monkeypatch, tmp_path):
    asyncio.run(_run_blackbox_dispatch_logs_all_segment_samples(monkeypatch, tmp_path))


async def _run_blackbox_dispatch_logs_all_segment_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DRESSAGE_LOG_WRITE_MODE", "await")

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    segment_sample = _last_segment_sample(result, expected_count=2)
    assert segment_sample.tokens == [20, 21, 22]
    assert segment_sample.response == "final"

    samples_dir = tmp_path / "7" / "bbs-sess-7" / "samples"
    first_payload = json.loads((samples_dir / "0.json").read_text(encoding="utf-8"))
    assert first_payload["session_id"] == "bbs-sess-7"
    assert first_payload["trajectory_id"] == "bbs-sess-7"
    assert first_payload["instance_id"] == "7"
    assert first_payload["segment_index"] == 0
    assert first_payload["segment_uid"] == "seg-0"
    assert first_payload["tokens"] == [10, 11]
    assert first_payload["response"] == "old"
    assert first_payload["response_length"] == 1
    assert first_payload["loss_mask"] == [1]
    assert first_payload["rollout_log_probs"] == [-0.1]
    assert first_payload["status"] == "completed"
    assert first_payload["metadata"]["selected_segment_index"] == 0

    second_payload = json.loads((samples_dir / "1.json").read_text(encoding="utf-8"))
    assert second_payload["session_id"] == "bbs-sess-7"
    assert second_payload["trajectory_id"] == "bbs-sess-7"
    assert second_payload["instance_id"] == "7"
    assert second_payload["segment_index"] == 1
    assert second_payload["segment_uid"] == "seg-1"
    assert second_payload["tokens"] == [20, 21, 22]
    assert second_payload["response"] == "final"
    assert second_payload["response_length"] == 2
    assert second_payload["loss_mask"] == [1, 1]
    assert second_payload["rollout_log_probs"] == [-0.2, -0.3]
    assert second_payload["status"] == "completed"
    assert second_payload["metadata"]["selected_segment_index"] == 1


def test_blackbox_dispatch_logs_in_background_mode(monkeypatch, tmp_path):
    asyncio.run(_run_blackbox_dispatch_logs_in_background_mode(monkeypatch, tmp_path))


async def _run_blackbox_dispatch_logs_in_background_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_runtime, "_PADDOCK", FakePaddock())
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("DRESSAGE_LOG_WRITE_MODE", raising=False)

    await blackbox_dispatch.generate(_rollout_args(), SampleLike(), {})
    await paddock_lifecycle.drain_terminate_tasks()
    await DEFAULT_WRITER.drain()

    assert (tmp_path / "7" / "bbs-sess-7" / "session.json").exists()
    assert (tmp_path / "7" / "bbs-sess-7" / "samples" / "0.json").exists()
    assert (tmp_path / "7" / "bbs-sess-7" / "samples" / "1.json").exists()


def test_blackbox_dispatch_log_write_falls_back_after_executor_shutdown(
    monkeypatch,
    tmp_path,
):
    asyncio.run(
        _run_blackbox_dispatch_log_write_falls_back_after_executor_shutdown(
            monkeypatch,
            tmp_path,
        )
    )


async def _run_blackbox_dispatch_log_write_falls_back_after_executor_shutdown(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("DRESSAGE_LOG_WRITE_MODE", raising=False)
    loop = asyncio.get_running_loop()

    def fail_run_in_executor(executor, func, *args):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(loop, "run_in_executor", fail_run_in_executor)

    log_path = tmp_path / "session.json"
    writer = RolloutArtifactWriter()
    result = await writer.write_json(log_path, {"ok": True})

    assert result == log_path
    assert json.loads(log_path.read_text(encoding="utf-8")) == {"ok": True}
    assert not writer._write_tasks



def test_blackbox_dispatch_logs_trajectory_error(monkeypatch, tmp_path, caplog):
    asyncio.run(_run_blackbox_dispatch_logs_trajectory_error(monkeypatch, tmp_path, caplog))


def test_blackbox_dispatch_suppresses_generation_preempted_error_log(
    monkeypatch,
    tmp_path,
    caplog,
):
    asyncio.run(
        _run_blackbox_dispatch_suppresses_generation_preempted_error_log(
            monkeypatch,
            tmp_path,
            caplog,
        )
    )


async def _run_blackbox_dispatch_suppresses_generation_preempted_error_log(
    monkeypatch,
    tmp_path,
    caplog,
):
    paddock = GenerationPreemptedCallPaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DRESSAGE_LOG_WRITE_MODE", "await")
    caplog.set_level(logging.WARNING, logger=blackbox_dispatch.__name__)

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result is sample
    assert sample.status == SampleLike.Status.ABORTED
    assert sample.metadata["blackbox_expected_abort"] == "generation_preempted"
    assert "blackbox_error" not in sample.metadata
    assert "blackbox_error_log_path" not in sample.metadata
    assert "blackbox_failure_history" not in sample.metadata

    error_path = tmp_path / "7" / "bbs-sess-7" / "error.json"
    assert not error_path.exists()

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == blackbox_dispatch.__name__
    ]
    assert not any("blackbox rollout failed" in message for message in warning_messages)


async def _run_blackbox_dispatch_logs_trajectory_error(monkeypatch, tmp_path, caplog):
    paddock = FailingHttpCallPaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DRESSAGE_LOG_WRITE_MODE", "await")
    caplog.set_level(logging.WARNING, logger=blackbox_dispatch.__name__)

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result is sample
    assert sample.status == SampleLike.Status.ABORTED
    assert "blackbox_error" in sample.metadata
    assert "blackbox_error_log_path" in sample.metadata

    error_path = tmp_path / "7" / "bbs-sess-7" / "error.json"
    assert error_path.exists()
    payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["session_id"] == "bbs-sess-7"
    assert payload["http"]["response"]["status_code"] == 502
    assert "backend failed" in payload["http"]["response"]["body"]
    assert "HTTPStatusError" in payload["error"]["type"]
    assert "Traceback" in payload["error"]["traceback"]

    warning_messages = [record.getMessage() for record in caplog.records]
    assert any("blackbox rollout failed for session_id=bbs-sess-7" in message for message in warning_messages)


def test_blackbox_dispatch_logs_no_slot_timeout_as_warning(monkeypatch, tmp_path, caplog):
    asyncio.run(
        _run_blackbox_dispatch_logs_no_slot_timeout_as_warning(
            monkeypatch,
            tmp_path,
            caplog,
        )
    )


async def _run_blackbox_dispatch_logs_no_slot_timeout_as_warning(
    monkeypatch,
    tmp_path,
    caplog,
):
    paddock = NoSlotPaddock()
    monkeypatch.setattr(generate_runtime, "_PADDOCK", paddock)
    monkeypatch.setattr(generate_runtime, "_PROXY_CLIENT", FakeProxy())
    monkeypatch.setenv("DRESSAGE_TRAJECTORY_ERROR_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DRESSAGE_LOG_WRITE_MODE", "await")
    caplog.set_level(logging.WARNING, logger=blackbox_dispatch.__name__)

    sample = SampleLike()
    result = await blackbox_dispatch.generate(_rollout_args(), sample, {})
    await paddock_lifecycle.drain_terminate_tasks()

    assert result is sample
    assert sample.status == SampleLike.Status.ABORTED
    assert sample.metadata["blackbox_error"] == "no blackbox slot available"
    assert "blackbox_error_log_path" in sample.metadata

    error_path = tmp_path / "7" / "bbs-sess-7" / "error.json"
    assert error_path.exists()
    payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert payload["error"]["type"] == "builtins.TimeoutError"
    assert payload["error"]["message"] == "no blackbox slot available"
    assert "Traceback" in payload["error"]["traceback"]
    assert payload["http"] is None

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == blackbox_dispatch.__name__
    ]
    assert warning_messages == [
        "blackbox rollout failed for session_id=bbs-sess-7: no blackbox slot available"
    ]
