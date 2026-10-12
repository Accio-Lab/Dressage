"""Local regressions for retry identity and partial-group training eligibility."""

import asyncio
from types import SimpleNamespace

import pytest
from agentic_rollout_contract.client import RolloutServiceClient
from agentic_rollout_contract.models import ExecutionSpec, ReserveRequest
from dressage_distroll_gpu_adapter.dynamic_sampling import (
    check_harbor_quorum_reward_nonzero_std,
)


@pytest.mark.asyncio
async def test_lost_reply_retries_the_original_request_bytes():
    request = ReserveRequest(
        client_scope="run",
        reservation_request_id="reservation",
        spec=ExecutionSpec(
            execution_id="execution", task_ref="task", model_name="model"
        ),
    )
    received = []

    async def handler(reader, writer):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in header.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            received.append(await reader.readexactly(length))
            if len(received) == 1:
                # Lost response after receiving the request, plus caller mutation.
                request.spec.task_ref = "changed-after-first-send"
                return
            body = b'{"received": true}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        async with RolloutServiceClient(
            f"http://127.0.0.1:{port}", timeout=2
        ) as client:
            result = await client._request(
                "POST", "/v4/executions:reserve", payload=request
            )
    assert result == {"received": True}
    assert len(received) == 2
    assert received[0] == received[1]
    assert b"changed-after-first-send" not in received[1]


def group(rewards):
    samples = []
    for slot in range(8):
        successful = slot < len(rewards)
        metadata = {"harbor_attempt_ordinal": slot}
        if successful:
            metadata.update(
                harbor_attempt_id=f"attempt-{slot}",
                harbor_training_committed=True,
                harbor_rewards={"reward": rewards[slot]},
            )
        else:
            metadata["harbor_partial_slot_zero_grad"] = True
        samples.append(SimpleNamespace(remove_sample=not successful, metadata=metadata))
    return samples


@pytest.mark.parametrize(
    "rewards,keep",
    [([1, 0, 1, 0, 1, 0], True), ([1, 0, 1, 0, 1], False), ([1] * 6, False)],
)
def test_quorum_counts_successes_and_requires_reward_variance(
    monkeypatch, rewards, keep
):
    monkeypatch.setenv("ROLLOUT_GROUP_MIN_SUCCESS_RATIO", "0.75")
    result = check_harbor_quorum_reward_nonzero_std(
        SimpleNamespace(n_samples_per_prompt=8), group(rewards)
    )
    assert result.keep is keep


def test_extra_segments_do_not_add_independent_trajectories(monkeypatch):
    monkeypatch.setenv("ROLLOUT_GROUP_MIN_SUCCESS_RATIO", "0.75")
    samples = group([1, 0, 1, 0, 1])
    samples.extend([samples[0], samples[1], samples[2]])
    result = check_harbor_quorum_reward_nonzero_std(
        SimpleNamespace(n_samples_per_prompt=8), samples
    )
    assert not result.keep


def test_failed_slot_requires_explicit_zero_gradient_placeholder(monkeypatch):
    monkeypatch.setenv("ROLLOUT_GROUP_MIN_SUCCESS_RATIO", "0.75")
    samples = group([1, 0, 1, 0, 1, 0])
    samples[-1].metadata.pop("harbor_partial_slot_zero_grad")
    result = check_harbor_quorum_reward_nonzero_std(
        SimpleNamespace(n_samples_per_prompt=8), samples
    )
    assert not result.keep
