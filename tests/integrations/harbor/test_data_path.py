from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dressage.integrations.harbor import compat
from dressage.integrations.harbor.artifacts import (
    AttemptFailure,
    FailureStage,
    FinalizationCheckpoint,
    HarborArtifactStore,
    validate_attempt,
)
from dressage.integrations.harbor.data_source import (
    HarborDataSource,
    HarborDataSourceCheckpointError,
    HarborDataSourceConfigurationError,
)
from examples.harbor.dataset_tools.dapo.prepare_dataset import (
    DapoDatasetError,
    load_records,
    prepare_dataset,
    prepared_dataset_identity,
)


ROUTING = {"routing_guarantee": "configure_only", "task_network_class": "restricted"}
MANIFEST_ROUTING = {
    "routing_guarantee": "configure_only",
    "public_network_tasks": 0,
    "restricted_network_tasks": 1,
}
PRODUCTION_DAPO = (
    Path(__file__).resolve().parents[3] / "examples/data/dressage_dapo_prompts.jsonl"
)


def _commit_kwargs(proxy, trial_result, trial_id="physical-1", **updates):
    values = {
        **ROUTING,
        "proxy_client": proxy,
        "job_id": "job-a",
        "trial_name": "trial-a",
        "trial_id": trial_id,
        "session_id": trial_id,
        "instance_id": "logical-a",
        "attempt_ordinal": int(trial_id.endswith("2")),
        "trial_result": trial_result,
    }
    values.update(updates)
    return values


@pytest.mark.asyncio
async def test_artifact_commit_is_exactly_once_atomic_and_schema_valid(
    tmp_path, segment_factory, trial_result_factory, proxy_client_factory
):
    proxy = proxy_client_factory([segment_factory()])
    store = HarborArtifactStore(
        tmp_path / "artifacts", run_id="run-a", require_token_versions=True
    )
    kwargs = _commit_kwargs(proxy, trial_result_factory(), expected_weight_version="v1")

    first, second = await asyncio.gather(
        store.commit_attempt(**kwargs), store.commit_attempt(**kwargs)
    )

    assert first is second
    assert first.trainable and first.trainable_token_count == 2
    assert first.observed_weight_versions == ["v1"]
    assert first.finalization_checkpoint is FinalizationCheckpoint.STORE_DRAINED
    assert proxy.finalize_calls == ["physical-1"]
    assert proxy.read_calls == [("physical-1", False), ("physical-1", True)]
    payload = json.loads(store.bundle_path(first).read_text())
    assert payload["schema_version"] == "dressage.harbor.trajectory/v2"
    assert payload["routing_guarantee"] == "configure_only"
    assert store.bundle_path(first).stat().st_mode & 0o777 == 0o600
    assert (
        tmp_path / "legacy-session-payloads/logical-a/physical-1/session.json"
    ).is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_trainable", "expected_codes", "expected_reads"),
    [
        (
            "lost-finalize-response",
            True,
            set(),
            [("physical-1", False), ("physical-1", True)],
        ),
        (
            "partial-segments",
            False,
            {"FINALIZE_FAILED", "TRAJECTORY_EMPTY"},
            [("physical-1", False)],
        ),
        (
            "missing-snapshot",
            False,
            {"TRAJECTORY_READ_FAILED"},
            [("physical-1", False)],
        ),
        (
            "cancelled",
            False,
            {"HARBOR_TRIAL_EXCEPTION"},
            [("physical-1", False), ("physical-1", True)],
        ),
    ],
    ids=("finalize-recovery", "partial-segments", "missing-snapshot", "cancelled"),
)
async def test_artifact_failure_matrix_is_fail_closed(
    case,
    expected_trainable,
    expected_codes,
    expected_reads,
    tmp_path,
    segment_factory,
    trial_result_factory,
    proxy_client_factory,
):
    segments = (
        [segment_factory(segment_count=2)]
        if case == "partial-segments"
        else [segment_factory()]
    )
    proxy = proxy_client_factory(
        segments,
        finalize_error=TimeoutError("response lost")
        if "finalize" in case or case == "partial-segments"
        else None,
        read_error=RuntimeError("snapshot unavailable")
        if case == "missing-snapshot"
        else None,
    )
    exception = (
        SimpleNamespace(exception_type="AgentTimeoutError", exception_message="timeout")
        if case == "cancelled"
        else None
    )
    store = HarborArtifactStore(tmp_path / "artifacts", run_id="run-a")
    bundle = await store.commit_attempt(
        **_commit_kwargs(
            proxy,
            trial_result_factory(exception_info=exception),
            cancelled=case == "cancelled",
        )
    )

    assert bundle.trainable is expected_trainable
    assert {failure.code for failure in bundle.failures} >= expected_codes
    assert proxy.read_calls == expected_reads


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda segment: segment.update(trajectory_id="wrong"),
            "TRAJECTORY_IDENTITY_INVALID",
        ),
        (
            lambda segment: segment.update(full_logprobs=[0.0]),
            "SEGMENT_LENGTH_MISMATCH",
        ),
        (
            lambda segment: segment.update(full_versions=["input", "v2", "v2"]),
            "WEIGHT_VERSION_MISMATCH",
        ),
    ],
    ids=("identity", "array-length", "weight-version"),
)
def test_attempt_validation_matrix(segment_factory, mutation, expected_code):
    segment = segment_factory()
    mutation(segment)
    failures, _, _, _ = validate_attempt(
        rewards={"reward": 1.0},
        reward_key="reward",
        segments=[segment],
        require_token_versions=True,
        expected_weight_version="v1",
        expected_trajectory_id="physical-1",
        expected_instance_id="logical-a",
    )
    assert expected_code in {failure.code for failure in failures}


@pytest.mark.asyncio
async def test_manifest_selects_final_retry_and_abort_revokes_it(
    tmp_path, segment_factory, trial_result_factory, proxy_client_factory
):
    store = HarborArtifactStore(tmp_path / "artifacts", run_id="run-a")
    first = await store.commit_attempt(
        **_commit_kwargs(
            proxy_client_factory([segment_factory()]),
            trial_result_factory(),
            failures=[
                AttemptFailure(
                    code="UPSTREAM", stage=FailureStage.AGENT, message="retry"
                )
            ],
        )
    )
    second = await store.commit_attempt(
        **_commit_kwargs(
            proxy_client_factory([segment_factory("physical-2")]),
            trial_result_factory(),
            "physical-2",
        )
    )
    path = await store.write_job_manifest(
        SimpleNamespace(id="job-a"),
        final_keys=[("trial-a", "physical-2")],
        **MANIFEST_ROUTING,
    )
    manifest = json.loads(path.read_text())
    attempts = {item["trial_id"]: item for item in manifest["attempts"]}
    assert attempts[first.trial_id]["superseded"] is True
    assert attempts[second.trial_id]["trainable"] is True

    aborted = await store.write_job_manifest(
        SimpleNamespace(id="job-a"), final_keys=[], state="aborted", **MANIFEST_ROUTING
    )
    assert json.loads(aborted.read_text())["state"] == "aborted"
    assert second.trainable is False and second.superseded is True


@pytest.mark.asyncio
async def test_manifest_reconciliation_failure_leaves_uncommitted_authority(
    tmp_path, segment_factory, trial_result_factory, proxy_client_factory
):
    class FaultyStore(HarborArtifactStore):
        fail_reconciliation = False
        writes = 0

        def _write_model_atomic(self, path, model):
            if self.fail_reconciliation:
                self.writes += 1
                if self.writes > 1:
                    raise OSError("injected reconciliation failure")
            return super()._write_model_atomic(path, model)

    store = FaultyStore(tmp_path / "artifacts", run_id="run-a")
    first = await store.commit_attempt(
        **_commit_kwargs(
            proxy_client_factory([segment_factory()]), trial_result_factory()
        )
    )
    await store.commit_attempt(
        **_commit_kwargs(
            proxy_client_factory([segment_factory("physical-2")]),
            trial_result_factory(),
            "physical-2",
        )
    )
    store.fail_reconciliation = True
    with pytest.raises(OSError, match="injected reconciliation failure"):
        await store.write_job_manifest(
            SimpleNamespace(id="job-a"),
            final_keys=[("trial-a", "physical-1")],
            **MANIFEST_ROUTING,
        )

    manifest = json.loads((store.root / "run-a/job-a/manifest.json").read_text())
    persisted = json.loads(store.bundle_path(first).read_text())
    assert manifest["committed"] is False
    assert persisted["reconciliation_generation"] == manifest["generation"]


class _Node:
    def __init__(self, payload):
        self.payload = copy.deepcopy(payload)

    def model_dump(self, **kwargs):
        return copy.deepcopy(self.payload)

    def model_copy(self, *, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)


class _JobConfig:
    def __init__(self, payload):
        self.payload = copy.deepcopy(payload)
        self.tasks = [_Node(item) for item in payload.get("tasks", [])]
        self.agents = [_Node(item) for item in payload.get("agents", [])]

    @classmethod
    def model_validate(cls, payload):
        return cls(payload)

    def model_dump(self, **kwargs):
        value = copy.deepcopy(self.payload)
        value["tasks"] = [item.model_dump() for item in self.tasks]
        value["agents"] = [item.model_dump() for item in self.agents]
        return value

    def model_copy(self, *, deep=False):
        return copy.deepcopy(self) if deep else copy.copy(self)


def _job_payload(secret="agent-secret"):
    return {
        "tasks": [{"name": "task-a"}, {"name": "task-b"}],
        "agents": [
            {"name": "codex", "env": {"OPENAI_API_KEY": secret}},
            {"name": "claude-code", "env": {"ANTHROPIC_API_KEY": secret}},
        ],
    }


@pytest.fixture
def data_source_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dressage.integrations.harbor.data_source._harbor_job_config_type",
        lambda: _JobConfig,
    )

    async def resolve(config):
        return tuple(config.tasks)

    monkeypatch.setattr(compat, "resolve_task_configs", resolve)
    integration = tmp_path / "integration.json"
    integration.write_text(
        json.dumps(
            {
                "execution_mode": "training",
                "security": {"routing_guarantee": "enforced"},
                "artifacts": {"mode": "both", "root": str(tmp_path / "artifacts")},
                "training": {"model_override": "train-model"},
            }
        )
    )

    def create(
        *, payload=None, eval_payload=None, n_samples=2, seed=17, run_id="run-a"
    ):
        import yaml

        job = tmp_path / f"job-{run_id}.yaml"
        job.write_text(yaml.safe_dump(payload or _job_payload()))
        eval_job = None
        if eval_payload:
            eval_job = tmp_path / f"eval-{run_id}.yaml"
            eval_job.write_text(yaml.safe_dump(eval_payload))
        return HarborDataSource(
            SimpleNamespace(
                harbor_job_config=str(job),
                harbor_integration_config=str(integration),
                harbor_eval_job_config=str(eval_job) if eval_job else None,
                n_samples_per_prompt=n_samples,
                n_samples_per_eval_prompt=n_samples,
                advantage_estimator="grpo",
                rollout_seed=seed,
                rollout_shuffle=True,
                harbor_run_id=run_id,
                save=str(tmp_path / "checkpoints"),
                load=str(tmp_path / "checkpoints"),
            )
        )

    return create


def _sample_view(groups):
    return [
        [(s.group_index, s.index, s.prompt, s.metadata) for s in group]
        for group in groups
    ]


def test_data_source_expands_deterministically_without_serializing_secrets(
    data_source_factory,
):
    secret = "do-not-persist"
    first = data_source_factory(payload=_job_payload(secret), seed=91)
    second = data_source_factory(payload=_job_payload(secret), seed=91)
    random.seed(12345)
    state = random.getstate()
    groups = first.get_samples(5)
    assert random.getstate() == state
    assert _sample_view(groups) == _sample_view(second.get_samples(5))
    assert len(first.specs) == 4 and all(len(group) == 2 for group in groups)
    first.save(1)
    checkpoint = Path(first.args.save) / "rollout/harbor_data_source_state_1.json"
    assert secret not in checkpoint.read_text()


def test_data_source_checkpoint_drift_and_eval_contract(data_source_factory):
    source = data_source_factory(seed=3, run_id="resume")
    source.get_samples(3)
    source.record_batch_state("weights-7", "runtime-a")
    source.save(12)
    restored = data_source_factory(seed=3, run_id="resume")
    restored.load(12)
    assert (restored.last_batch_weight_version, restored.last_runtime_incarnation) == (
        "weights-7",
        "runtime-a",
    )
    assert _sample_view(restored.get_samples(2)) == _sample_view(source.get_samples(2))

    drift = _job_payload()
    drift["tasks"][0]["name"] = "changed"
    with pytest.raises(HarborDataSourceCheckpointError, match="drift"):
        data_source_factory(payload=drift, seed=3, run_id="resume").load(12)

    evaluation = data_source_factory(
        eval_payload={"tasks": [{"name": "eval"}], "agents": [{"name": "codex"}]},
        run_id="eval",
    )
    before = evaluation.sample_offset
    assert _sample_view(evaluation.get_eval_samples()) == _sample_view(
        evaluation.get_eval_samples()
    )
    assert evaluation.sample_offset == before


def test_data_source_rejects_single_attempt_group(data_source_factory):
    with pytest.raises(HarborDataSourceConfigurationError, match="requires.*>= 2"):
        data_source_factory(n_samples=1)


def _dapo_row(index, **updates):
    row = {
        "prompt": [{"role": "user", "content": f"What is {index}+{index}?"}],
        "label": r"\boxed{4}",
        "metadata": {"instance_id": f"dapo_math_{index:05d}"},
        "agent_mode": "blackbox",
        "blackbox_type": "opencode" if index % 2 else "claude_code",
        "reward_fn": "contains_label",
        "task_type": "math",
    }
    row.update(updates)
    return row


def _write_dapo(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [_dapo_row(0), _dapo_row(1, metadata={"instance_id": "dapo_math_00000"})],
            "duplicate",
        ),
        ([_dapo_row(0, prompt=[])], "exactly one message"),
        ([_dapo_row(0, label="")], "non-empty string"),
        ([_dapo_row(0, reward_fn="math_score")], "contains_label"),
    ],
    ids=("duplicate-id", "prompt-shape", "empty-label", "reward-function"),
)
def test_dapo_source_contract_and_validation(tmp_path, rows, message):
    production = load_records(PRODUCTION_DAPO)
    assert len(production) == 3000
    assert Counter(record.source_blackbox_type for record in production) == {
        "claude_code": 1500,
        "opencode": 1500,
    }
    with pytest.raises(DapoDatasetError, match=message):
        load_records(_write_dapo(tmp_path / "source.jsonl", rows))


def test_dapo_prepare_is_cached_concurrent_and_source_bound(tmp_path):
    source = _write_dapo(tmp_path / "source.jsonl", [_dapo_row(0), _dapo_row(1)])
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=4) as executor:
        prepared = list(
            executor.map(
                lambda _: prepare_dataset(source, cache_root=cache, limit=2), range(8)
            )
        )
    assert len({item.root for item in prepared}) == 1
    first = prepared[0]
    assert (
        prepared_dataset_identity(first.job_config_path)["fingerprint"]
        == first.fingerprint
    )
    assert len(list(first.tasks_dir.iterdir())) == 2
    _write_dapo(source, [_dapo_row(0, prompt=[{"role": "user", "content": "changed"}])])
    with pytest.raises(DapoDatasetError, match="no longer matches"):
        prepared_dataset_identity(first.job_config_path)


@pytest.mark.parametrize(
    ("events", "reward", "reason"),
    [
        ([{"type": "result", "result": r"Answer: \boxed{4}"}], 1, None),
        ([{"type": "result", "result": "wrong"}], 0, "label_not_found"),
        ([], 0, "missing_result"),
    ],
    ids=("match", "mismatch", "missing"),
)
def test_generated_dapo_verifier_contract(tmp_path, events, reward, reason):
    source = _write_dapo(tmp_path / "source.jsonl", [_dapo_row(0)])
    prepared = prepare_dataset(source, cache_root=tmp_path / "cache", limit=1)
    tests = prepared.tasks_dir / "dapo_math_00000/tests"
    stream = tmp_path / "stream.jsonl"
    stream.write_text("".join(json.dumps(event) + "\n" for event in events))
    reward_path, details_path = tmp_path / "reward.json", tmp_path / "details.json"
    subprocess.run(
        [
            sys.executable,
            str(tests / "verify.py"),
            "--stream",
            str(stream),
            "--expected",
            str(tests / "expected.json"),
            "--reward",
            str(reward_path),
            "--details",
            str(details_path),
        ],
        check=True,
    )
    assert json.loads(reward_path.read_text()) == {"reward": reward}
    assert json.loads(details_path.read_text())["mismatch_reason"] == reason
