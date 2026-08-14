from __future__ import annotations

import asyncio
import logging
import sys
import types

import httpx

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - local minimal test environment
    fake_transformers = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        pass

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    sys.modules["transformers"] = fake_transformers

from dressage.proxy.generation_controller import GenerationController
from dressage.proxy.sglang_client import SGLangResponse, SGLangRouterClient


def test_sglang_router_client_wait_until_ready_uses_workers_health():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        assert request.url.path == "/workers"
        attempts += 1
        return httpx.Response(
            200,
            json={
                "workers": [
                    {
                        "url": "http://127.0.0.1:30000",
                        "is_healthy": attempts >= 2,
                        "connection_mode": "Http",
                    }
                ]
            },
        )

    async def run_test() -> dict:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ) as client:
            router = SGLangRouterClient("http://router.test", client=client)
            return await router.wait_until_ready(timeout_seconds=1.0, interval_seconds=0.01)

    result = asyncio.run(run_test())

    assert result["ready"] is True
    assert result["attempts"] == 2
    assert result["worker_count"] == 1
    assert result["healthy_worker_count"] == 1


def test_generation_controller_resume_keeps_paused_when_backend_not_ready():
    class NotReadyClient:
        async def wait_until_ready(self):
            return {"ready": False, "error": "router returned no healthy HTTP workers"}

    async def run_test() -> dict:
        controller = GenerationController(NotReadyClient())  # type: ignore[arg-type]
        await controller.pause(reason="unit", timeout_seconds=0.01)
        result = await controller.resume(version="v1", reason="unit")
        assert controller.state()["paused"] is True
        assert controller.current_version is None
        return result

    result = asyncio.run(run_test())

    assert result["status"] == "backend_not_ready"
    assert result["paused"] is True
    assert result["readiness"]["ready"] is False


def test_generation_controller_resume_releases_after_backend_ready():
    class ReadyClient:
        async def wait_until_ready(self):
            return {"ready": True, "healthy_worker_count": 1}

    async def run_test() -> dict:
        controller = GenerationController(ReadyClient())  # type: ignore[arg-type]
        await controller.pause(reason="unit", timeout_seconds=0.01)
        result = await controller.resume(version="v1", reason="unit")
        assert controller.state()["paused"] is False
        assert controller.current_version == "v1"
        return result

    result = asyncio.run(run_test())

    assert result["status"] == "resumed"
    assert result["readiness"] == {"ready": True, "healthy_worker_count": 1}


def test_generation_controller_pause_logs_aborted_rids(caplog):
    class BlockingAbortClient:
        def __init__(self):
            self.generate_started = asyncio.Event()
            self.abort_called = asyncio.Event()
            self.request_id = None
            self.abort_calls = []

        async def generate(
            self,
            input_ids,
            sampling_params,
            *,
            routing_key=None,
            request_id=None,
            return_logprob=True,
        ):
            del sampling_params, return_logprob
            self.request_id = request_id
            self.generate_started.set()
            await self.abort_called.wait()
            return SGLangResponse(
                input_token_ids=list(input_ids),
                input_token_logprobs_raw=[-0.1] * len(input_ids),
                input_token_texts=[""] * len(input_ids),
                output_ids=[42],
                output_token_logprobs=[-0.2],
                output_token_texts=["*"],
                output_versions=["v0"],
                all_token_ids=list(input_ids) + [42],
                all_logprobs=[-0.1] * len(input_ids) + [-0.2],
                text="*",
                meta_info={"finish_reason": {"type": "abort"}, "weight_version": "v0"},
                finish_reason="abort",
            )

        async def abort_request(self, request_id, *, routing_key=None):
            self.abort_calls.append(
                {"request_id": request_id, "routing_key": routing_key}
            )
            self.abort_called.set()
            return {
                "success": True,
                "request_id": request_id,
                "rid": request_id,
                "targets": [
                    {
                        "target": "http://worker.test/abort_request",
                        "status_code": 200,
                    }
                ],
                "errors": [],
            }

    async def run_test():
        client = BlockingAbortClient()
        controller = GenerationController(
            client,
            partial_rollout=True,
        )  # type: ignore[arg-type]
        generate_task = asyncio.create_task(
            controller.generate_preemptible(
                [1, 2, 3],
                {"max_new_tokens": 1},
                session_id="sess-log",
                instance_id="inst-log",
                turn_id="turn-log",
                routing_key="sess-log",
            )
        )
        await asyncio.wait_for(client.generate_started.wait(), timeout=1.0)

        pause_result = await controller.pause(
            reason="weight_update",
            timeout_seconds=1.0,
        )
        request_id = client.request_id
        assert request_id is not None
        assert client.abort_calls == [
            {"request_id": request_id, "routing_key": "sess-log"}
        ]

        await controller.resume(version="v1", reason="weight_update")
        await asyncio.wait_for(generate_task, timeout=1.0)
        return pause_result, request_id

    caplog.set_level(logging.INFO, logger="dressage.proxy.generation_controller")
    pause_result, request_id = asyncio.run(run_test())

    assert pause_result["abort_attempted_request_ids"] == [request_id]
    assert pause_result["abort_succeeded_request_ids"] == [request_id]
    assert pause_result["abort_failed_request_ids"] == []
    assert f"rid={request_id}" in caplog.text
    assert f"attempted_rids=['{request_id}']" in caplog.text
    assert "http://worker.test/abort_request" in caplog.text


def test_generation_controller_skips_zero_output_preempt_routed_experts():
    class ZeroOutputPreemptClient:
        def __init__(self):
            self.calls = 0

        async def generate(
            self,
            input_ids,
            sampling_params,
            *,
            routing_key=None,
            request_id=None,
        ):
            del sampling_params, routing_key, request_id
            self.calls += 1
            if self.calls == 1:
                return SGLangResponse(
                    input_token_ids=list(input_ids),
                    input_token_logprobs_raw=[0.0] * len(input_ids),
                    input_token_texts=[""] * len(input_ids),
                    output_ids=[],
                    output_token_logprobs=[],
                    output_token_texts=[],
                    all_token_ids=list(input_ids),
                    all_logprobs=[0.0] * len(input_ids),
                    text="",
                    meta_info={
                        "finish_reason": {"type": "preempted"},
                        "routed_experts": "stale-prefix-routes",
                    },
                    finish_reason="preempted",
                    routed_experts="stale-prefix-routes",
                )
            return SGLangResponse(
                input_token_ids=list(input_ids),
                input_token_logprobs_raw=[0.0] * len(input_ids),
                input_token_texts=[""] * len(input_ids),
                output_ids=[42],
                output_token_logprobs=[-0.2],
                output_token_texts=["*"],
                output_versions=["v1"],
                all_token_ids=list(input_ids) + [42],
                all_logprobs=[0.0] * len(input_ids) + [-0.2],
                text="*",
                meta_info={
                    "finish_reason": {"type": "stop"},
                    "weight_version": "v1",
                    "routed_experts": "fresh-routes",
                },
                finish_reason="stop",
                routed_experts="fresh-routes",
            )

    async def run_test():
        client = ZeroOutputPreemptClient()
        controller = GenerationController(
            client,
            partial_rollout=True,
        )  # type: ignore[arg-type]
        return await controller.generate_preemptible(
            [1, 2, 3],
            {"max_new_tokens": 1},
            session_id="sess-r3",
            instance_id="inst-r3",
            turn_id="turn-r3",
            routing_key="sess-r3",
        )

    result = asyncio.run(run_test())

    assert result.output_ids == [42]
    assert "routed_experts" not in result.meta_info
    assert result.routed_experts_chunks == [
        {
            "data": "fresh-routes",
            "prefix_token_count": 3,
            "output_token_count": 1,
            "is_first_chunk": True,
        }
    ]
