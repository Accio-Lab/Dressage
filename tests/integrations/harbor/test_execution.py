from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
import json
import socket
import tempfile
import threading
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import pytest

from dressage.integrations.harbor import plugin as plugin_module
from dressage.integrations.harbor import rollout
from dressage.integrations.harbor.artifacts import HarborArtifactStore
from dressage.integrations.harbor.gateway import (
    GatewayRuntime,
    RouteConflictError,
    RouteSpec,
)
from dressage.integrations.harbor.plugin import (
    AttemptPhase,
    DressageHarborPlugin,
    RoutingPolicyError,
    TrialBinding,
)


class _Compat:
    @staticmethod
    def pending_trial_configs(job):
        return job.configs


def _plugin(tmp_path, config, runtime, proxy, *, bindings=None):
    return DressageHarborPlugin(
        config,
        gateway_runtime=runtime,
        artifact_store=HarborArtifactStore(
            tmp_path / "artifacts",
            run_id="run-a",
            require_token_versions=config.execution_mode == "training",
            fsync=False,
        ),
        proxy_client=proxy,
        compat=_Compat,
        route_spec_factory=lambda **kwargs: kwargs,
        trial_bindings=bindings,
    )


@pytest.fixture(autouse=True)
def _restricted_task_audit(monkeypatch):
    monkeypatch.setattr(
        plugin_module, "_audit_task_network_file", lambda path: ("f" * 64, "restricted")
    )


@pytest.mark.asyncio
async def test_training_attempt_runs_plugin_artifact_and_sample_pipeline(
    monkeypatch,
    tmp_path,
    integration_config_factory,
    segment_factory,
    proxy_client_factory,
    runtime_factory,
    trial_factory,
    job_factory,
    group_factory,
):
    monkeypatch.setenv("DRESSAGE_PROXY_API_KEY", "backend-secret")
    config = integration_config_factory(execution_mode="training")
    runtime = runtime_factory()
    proxy = proxy_client_factory([segment_factory(instance_id="logical-a")])
    trial, job = trial_factory(agent_name="claude-code"), None
    job = job_factory([trial])
    plugin = _plugin(
        tmp_path,
        config,
        runtime,
        proxy,
        bindings={
            "trial-a": TrialBinding(
                instance_id="logical-a", attempt_ordinal=7, expected_weight_version="v1"
            )
        },
    )
    await plugin.on_job_start(job)
    event = job.event(trial)
    for hook in ("start", "agent_start", "agent_end", "verification", "end", "end"):
        await job.hooks[hook](event)

    spec = runtime.lease.specs[0]
    assert spec["instance_id"] == "logical-a"
    assert spec["expected_version"] == "v1"
    assert spec["model_override"] == "train-model"
    assert trial.agent.env["ANTHROPIC_BASE_URL"] == runtime.lease.public_url
    assert spec["upstream_headers"] == {"Authorization": "Bearer backend-secret"}
    assert "backend-secret" not in repr(trial.agent.env)
    bundle = plugin.get_result("trial-a", "physical-1")
    assert bundle.trainable and bundle.attempt_ordinal == 7
    assert runtime.lease.handles[0].close_calls == 1
    samples = rollout._bundle_samples(
        template=group_factory(0)[0],
        bundle=bundle,
        args=SimpleNamespace(max_tokens_per_gpu=16, context_parallel_size=1),
        reward_key="reward",
    )
    assert samples[-1].reward == {"reward": 1.0}
    assert samples[-1].metadata["harbor_routing_guarantee"] == "enforced"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("public-task", RoutingPolicyError, "cannot satisfy"),
        ("missing-task", RoutingPolicyError, "cannot locate"),
        ("alternative-auth", ValueError, "would bypass"),
        ("unknown-agent", ValueError, "unsupported Harbor Agent"),
    ],
    ids=("public-task-enforced", "missing-task", "alternative-auth", "unknown-agent"),
)
async def test_plugin_fail_fast_matrix(
    case,
    error,
    message,
    monkeypatch,
    tmp_path,
    integration_config_factory,
    proxy_client_factory,
    runtime_factory,
    trial_factory,
    job_factory,
):
    config = integration_config_factory(execution_mode="training")
    monkeypatch.setenv("DRESSAGE_PROXY_API_KEY", "backend-secret")
    runtime = runtime_factory()
    trial = trial_factory()
    if case == "public-task":
        monkeypatch.setattr(
            plugin_module, "_audit_task_network_file", lambda path: ("a" * 64, "public")
        )
    elif case == "missing-task":
        trial.task = SimpleNamespace(path="missing")
    elif case == "alternative-auth":
        trial.agent.env["CODEX_AUTH_JSON_PATH"] = "/tmp/auth.json"
    else:
        trial.agent.name = "custom-agent"
    plugin = _plugin(tmp_path, config, runtime, proxy_client_factory(), bindings=None)

    with pytest.raises(error, match=message):
        await plugin.on_job_start(job_factory([trial]))
    assert runtime.secret_slots.counter == 0
    assert runtime.lease.release_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["cancel", "open-failure", "close-failure"],
    ids=("cancel-is-diagnostic", "hook-failure", "route-close-failure"),
)
async def test_plugin_attempt_failures_are_closed(
    case,
    tmp_path,
    integration_config_factory,
    segment_factory,
    proxy_client_factory,
    runtime_factory,
    trial_factory,
    job_factory,
):
    runtime = runtime_factory(
        fail_open=case == "open-failure", fail_close=case == "close-failure"
    )
    proxy = proxy_client_factory([segment_factory()])
    trial, job = trial_factory(), None
    job = job_factory([trial])
    plugin = _plugin(
        tmp_path,
        integration_config_factory(),
        runtime,
        proxy,
        bindings={"trial-a": TrialBinding(instance_id="logical-a", attempt_ordinal=0)},
    )
    await plugin.on_job_start(job)
    event = job.event(trial)
    await job.hooks["start"](event)
    await job.hooks["agent_start"](event)
    if case == "cancel":
        await job.hooks["cancel"](event)
    else:
        await job.hooks["agent_end"](event)
    await job.hooks["end"](event)

    record = plugin.attempts[("trial-a", "physical-1")]
    if case == "cancel":
        assert proxy.finalize_calls == []
        assert plugin.get_result("trial-a", "physical-1") is None
    elif case == "open-failure":
        assert record.phase is AttemptPhase.CLOSED
        assert any(f.code == "PLUGIN_HOOK_FAILED" for f in record.failures)
    else:
        assert record.bundle is not None and not record.bundle.trainable
        assert any(f.code == "ROUTE_CLOSE_FAILED" for f in record.bundle.failures)
        assert runtime.secret_slots.rotate_calls == []


@pytest.mark.asyncio
async def test_harbor_retry_rotates_token_and_reconciles_final_attempt(
    tmp_path,
    integration_config_factory,
    segment_factory,
    proxy_client_factory,
    runtime_factory,
    trial_factory,
    trial_result_factory,
    job_factory,
):
    trial, job = trial_factory(), None
    job = job_factory([trial])
    runtime, proxy = runtime_factory(), proxy_client_factory([segment_factory()])
    plugin = _plugin(
        tmp_path,
        integration_config_factory(),
        runtime,
        proxy,
        bindings={"trial-a": TrialBinding(instance_id="logical-a", attempt_ordinal=0)},
    )
    await plugin.on_job_start(job)
    for trial_id in ("physical-1", "physical-2"):
        proxy.segments = [segment_factory(trial_id)]
        event = job.event(trial, trial_id)
        for hook in ("start", "agent_start", "agent_end", "end"):
            await job.hooks[hook](event)
    await plugin.on_job_end(
        SimpleNamespace(
            id="job-a",
            trial_results=[SimpleNamespace(id="physical-2", trial_name="trial-a")],
        )
    )

    assert runtime.lease.specs[0]["token"] != runtime.lease.specs[1]["token"]
    assert plugin.get_result("trial-a", "physical-1").superseded is True
    assert plugin.get_result("trial-a", "physical-2").trainable is True
    assert runtime.lease.release_calls == 1


class _Source:
    def __init__(self, config, groups):
        self.integration_config = config
        self.groups = groups
        self.sample_offset = 0
        self.run_id = "run-a"
        self.job_config = SimpleNamespace(n_concurrent_trials=1)
        self.last_batch = None
        self.specs = {
            f"spec-{i}": SimpleNamespace(
                runtime_fingerprint="same-agent",
                task_config=SimpleNamespace(
                    source=f"dataset-{i}",
                    metadata={"instance_id": f"instance-{i}"},
                    task=SimpleNamespace(name=f"dressage/instance-{i}"),
                ),
                agent_config=SimpleNamespace(n_concurrent=1),
            )
            for i in range(len(groups))
        }

    def get_samples(self, count):
        result = self.groups[self.sample_offset : self.sample_offset + count]
        self.sample_offset += len(result)
        return result

    def resolve_spec(self, spec_id):
        return self.specs[spec_id]

    def record_batch_state(self, **state):
        self.last_batch = state


def _capabilities():
    return {
        "schema_version": "dressage.proxy.integration/v1",
        "token_build_mode": "snapshot",
        "token_build_model": "train-model",
        "tokenizer_id": "train-model",
        "chat_template_fingerprint": "abc",
        "record_token_versions": True,
        "partial_rollout": False,
        "supports_expected_version": True,
        "current_weight_version": "7",
        "weight_version_authoritative": True,
        "weight_versions_consistent": True,
    }


def _completed(work):
    samples = copy.deepcopy(work.templates)
    for attempt, sample in enumerate(samples):
        sample.tokens, sample.response_length = [1, 2], 1
        sample.loss_mask, sample.rollout_log_probs = [1], [-0.1]
        sample.reward = {"reward": float(attempt)}
        sample.metadata.update(
            {
                "parent_traj_id": f"trial-{work.position}-{attempt}",
                "segment_index": 0,
                "harbor_weight_versions": ["7"],
            }
        )
        sample.status = sample.Status.COMPLETED
    return samples


def _rollout_args(batch_size=2):
    return SimpleNamespace(
        rollout_batch_size=batch_size,
        reward_key="reward",
        custom_reward_post_process_path="dressage.training.reward_post_process.reward_post_process",
        custom_convert_samples_to_train_data_path="dressage.rollout.convert_samples.convert_samples_to_train_data",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "config_updates", "group_count", "expected"),
    [
        ("retry", {}, 2, {"groups": [0, 1], "retries": 1}),
        (
            "replace",
            {
                "failed_group_policy": "replace",
                "max_replacement_groups": 2,
                "min_live_group_ratio": 1.0,
            },
            2,
            {"groups": [1], "replacements": 1},
        ),
        ("zero-grad", {"group_max_retries": 0}, 2, {"failed": 1}),
    ],
    ids=("retry-success", "replacement-success", "zero-grad-fallback"),
)
async def test_rollout_group_policy_matrix(
    case,
    config_updates,
    group_count,
    expected,
    monkeypatch,
    integration_config_factory,
    group_factory,
):
    config = integration_config_factory(
        execution_mode="training", training=config_updates
    )
    source = _Source(config, [group_factory(i) for i in range(group_count)])
    calls = {}

    async def capabilities(config):
        return _capabilities()

    async def root(config):
        return object(), object()

    async def round_(**kwargs):
        outcomes = []
        for work in kwargs["pending"]:
            count = calls.get(work.group_index, 0)
            calls[work.group_index] = count + 1
            fail = (
                case == "retry"
                and work.group_index == 1
                and count == 0
                or case == "replace"
                and work.group_index == 0
                or case == "zero-grad"
                and work.group_index == 1
            )
            outcomes.append(
                rollout._GroupOutcome(work=work, error=RuntimeError("failed"))
                if fail
                else rollout._GroupOutcome(
                    work=work, samples=_completed(work), versions=("7",)
                )
            )
        return outcomes

    monkeypatch.setattr(rollout, "_read_proxy_capabilities", capabilities)
    monkeypatch.setattr(rollout, "_ensure_root_gateway_lease", root)
    monkeypatch.setattr(rollout, "_run_round", round_)
    groups, metrics = await rollout._run_harbor_rollout(
        _rollout_args(1 if case == "replace" else 2), 1, source, evaluation=False
    )
    if "groups" in expected:
        assert [group[0].group_index for group in groups] == expected["groups"]
    if "retries" in expected:
        assert metrics["harbor/group_retries"] == expected["retries"]
    if "replacements" in expected:
        assert metrics["harbor/replacement_groups"] == expected["replacements"]
    if "failed" in expected:
        assert metrics["harbor/failed_groups"] == expected["failed"]
        assert all(sample.remove_sample for sample in groups[1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["replacement-budget", "evaluation"],
    ids=("replacement-budget-exhausted", "evaluation-no-replace"),
)
async def test_rollout_abort_policy_matrix(
    case, monkeypatch, integration_config_factory, group_factory
):
    training = {
        "failed_group_policy": "replace",
        "max_replacement_groups": 1,
        "group_max_retries": 0,
    }
    config = integration_config_factory(execution_mode="training", training=training)
    source = _Source(config, [group_factory(0), group_factory(1)])
    source.get_eval_samples = lambda: source.groups

    async def capabilities(config):
        return _capabilities()

    async def root(config):
        return object(), object()

    async def failed(**kwargs):
        return [
            rollout._GroupOutcome(work=work, error=RuntimeError("failed"))
            for work in kwargs["pending"]
        ]

    monkeypatch.setattr(rollout, "_read_proxy_capabilities", capabilities)
    monkeypatch.setattr(rollout, "_ensure_root_gateway_lease", root)
    monkeypatch.setattr(rollout, "_run_round", failed)
    match = (
        "replacement budget exhausted"
        if case == "replacement-budget"
        else "exhausted retries"
    )
    with pytest.raises(rollout.HarborRolloutError, match=match):
        await rollout._run_harbor_rollout(
            _rollout_args(1), 1, source, evaluation=case == "evaluation"
        )


@pytest.mark.parametrize(
    "mode", ["native", "bwrap", "retry"], ids=("native", "bwrap", "retry-partition")
)
def test_temporary_job_contract(
    mode, tmp_path, integration_config_factory, group_factory
):
    updates = {}
    if mode == "bwrap":
        updates = {
            "environment": {"mode": "bwrap", "runtime_root": str(tmp_path / "runtime")},
            "gateway": {"listen_port": 39100},
            "security": {"routing_guarantee": "enforced"},
        }
    config = integration_config_factory(**updates)
    groups = [group_factory(0), group_factory(1)]
    source = _Source(config, groups)
    source.job_config.environment = SimpleNamespace(type="docker", import_path=None)
    partition = rollout._resolve_work(source, groups[:1] if mode == "retry" else groups)
    temporary = rollout._temporary_job_config(
        source_job_config=source.job_config,
        partition=partition,
        config=config,
        run_id="run-a",
        rollout_id=1,
        retry_round=int(mode == "retry"),
    )
    assert temporary.n_concurrent_trials == len(partition) * 2
    assert all(task.source is None for task in temporary.tasks)
    if mode == "bwrap":
        assert temporary.environment.import_path.endswith(":DressageEnvironment")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("record_token_versions", False, "record_token_versions"),
        ("partial_rollout", True, "partial_rollout"),
        ("weight_version_authoritative", False, "authoritative"),
    ],
    ids=("token-versions", "partial-rollout", "authoritative-version"),
)
def test_proxy_capability_contract(field, value, message, integration_config_factory):
    capabilities = _capabilities()
    capabilities[field] = value
    with pytest.raises(rollout.HarborRolloutError, match=message):
        rollout._validate_proxy_capabilities(
            SimpleNamespace(hf_checkpoint="train-model"),
            integration_config_factory(execution_mode="training"),
            capabilities,
        )


class _GatewayProxy:
    def __init__(self, **kwargs):
        self.kwargs, self.turn_id = kwargs, None
        self.errors = {
            name: None
            for name in (
                "context_overflow",
                "rollout_invalidated",
                "failed_upstream",
                "max_steps",
            )
        }

        @asynccontextmanager
        async def lifespan(app):
            yield

        self.app = FastAPI(lifespan=lifespan)

        @self.app.api_route("/{path:path}", methods=["GET", "POST"])
        async def echo(request: Request, path: str):
            payload = await request.json() if request.method == "POST" else {}
            return JSONResponse({"path": f"/{path}", "payload": payload})

    async def open_turn(self, turn_id, backend_session_id=None):
        self.turn_id = turn_id

    async def drain_turn(self, timeout=None):
        return None

    async def clear_turn(self):
        self.turn_id = None

    async def consume_context_overflow_error(self):
        return self.errors["context_overflow"]

    async def consume_rollout_invalidated_error(self):
        return self.errors["rollout_invalidated"]

    async def consume_failed_upstream_error(self):
        return self.errors["failed_upstream"]

    async def consume_max_steps_error(self):
        return self.errors["max_steps"]


@pytest.mark.asyncio
async def test_gateway_route_contract(integration_config_factory):
    runtime = GatewayRuntime(proxy_factory=_GatewayProxy)
    config = integration_config_factory(
        gateway={"listen_port": 0, "limits": {"request_body_max_bytes": 100}}
    )
    lease = await runtime.acquire(config)
    spec = RouteSpec(
        trial_name="trial-a",
        trial_id="physical-1",
        instance_id="logical-a",
        token="route-token",
        model_override="trained-model",
        expected_version="v7",
        sampling_mode="force",
        sampling_temperature=0.7,
    )
    handle = await lease.register(spec)
    async with httpx.AsyncClient(base_url=lease.public_url, trust_env=False) as client:
        inactive = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer route-token"},
            json={"model": "agent"},
        )
        assert inactive.status_code == 409
        await handle.open_turn("turn-a")
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer route-token"},
            json={"model": "agent"},
        )
        assert response.status_code == 200
        assert response.json()["payload"]["model"] == "trained-model"
        conflict = await client.post(
            "/v1/messages",
            headers={"Authorization": "Bearer route-token", "x-api-key": "different"},
            json={"model": "m"},
        )
        assert conflict.status_code == 400
        oversized = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer route-token"},
            content=json.dumps({"input": "x" * 200}),
        )
        assert oversized.status_code == 413
    await handle.quiesce()
    await handle.close()
    with pytest.raises(RouteConflictError, match="tombstoned"):
        await lease.register(spec)
    await lease.release()


@pytest.mark.asyncio
async def test_bwrap_gateway_serves_routes_over_uds(integration_config_factory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(dir="/tmp") as runtime_root:
        config = integration_config_factory(
            environment={"mode": "bwrap", "runtime_root": runtime_root},
            gateway={"listen_port": port},
            security={"routing_guarantee": "enforced"},
        )
        runtime = GatewayRuntime(proxy_factory=_GatewayProxy)
        lease = await runtime.acquire(config)
        socket_path = runtime.unix_socket_path
        assert socket_path is not None and socket_path.is_socket()
        handle = await lease.register(
            RouteSpec(
                trial_name="trial-a",
                trial_id="physical-1",
                instance_id="logical-a",
                token="route-token",
            )
        )
        await handle.open_turn("turn-a")
        transport = httpx.AsyncHTTPTransport(uds=str(socket_path))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/messages",
                headers={"x-api-key": "route-token"},
                json={"model": "m"},
            )
        assert response.status_code == 200
        await handle.quiesce()
        await handle.close()
        await lease.release()
        assert not socket_path.exists()


@pytest.mark.asyncio
async def test_gateway_call_propagates_cross_loop_cancellation(
    integration_config_factory,
):
    runtime = GatewayRuntime(proxy_factory=_GatewayProxy)
    lease = await runtime.acquire(integration_config_factory())
    started, cancelled = threading.Event(), threading.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(runtime.call(operation))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(cancelled.wait, 1)
    await lease.release()


@pytest.mark.asyncio
async def test_rollout_runtime_cleanup_is_idempotent():
    class Lease:
        calls = 0

        async def release(self):
            self.calls += 1

    lease = Lease()
    rollout._ROOT_LEASES["test"] = lease
    await rollout.close_harbor_rollout_runtime()
    await rollout.close_harbor_rollout_runtime()
    assert lease.calls == 1 and rollout._ROOT_LEASES == {}
