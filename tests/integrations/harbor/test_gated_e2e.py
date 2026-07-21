from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import uuid

import pytest
import yaml


pytestmark = pytest.mark.harbor_e2e
ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples/harbor"
JOBS = EXAMPLES / "harbor_job_configs"
ROLLOUT = EXAMPLES / "run_harbor_rollout_qwen3.5_4b.sh"
TRAINING = EXAMPLES / "run_harbor_training_qwen3.5_4b.sh"


def _inputs(*names):
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.fail("selected Harbor E2E is missing: " + ", ".join(missing))
    return {name: os.environ[name] for name in names}


def _runtime_job(source, tmp_path, label):
    payload = yaml.safe_load(Path(source).read_text())
    payload.update(
        job_name=f"{label}-{uuid.uuid4().hex[:10]}",
        jobs_dir=str(tmp_path / "jobs"),
        n_attempts=1,
        n_concurrent_trials=1,
    )
    for agent in payload.get("agents", []):
        agent["n_concurrent"] = 1
    for dataset in payload.get("datasets", []):
        if dataset.get("name"):
            dataset["n_tasks"] = 1
    target = tmp_path / f"{label}.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False))
    return target


def _run(runner, profile, job, *, env=None, check=True):
    return subprocess.run(
        [str(runner)],
        cwd=ROOT,
        env={
            **os.environ,
            "DRESSAGE_HARBOR_INTEGRATION_CONFIG": str(profile),
            "DRESSAGE_HARBOR_JOB_CONFIG": str(job),
            **(env or {}),
        },
        check=check,
        capture_output=True,
        text=True,
        timeout=7200,
    )


def _assert_reward(job_path):
    from harbor.models.job.config import JobConfig
    from harbor.models.job.result import JobResult

    job = JobConfig.model_validate(yaml.safe_load(job_path.read_text()))
    result = JobResult.model_validate_json(
        (job.jobs_dir / job.job_name / "result.json").read_text()
    )
    rewards = [
        trial.verifier_result.rewards.get("reward")
        for trial in result.trial_results
        if trial.verifier_result and trial.verifier_result.rewards
    ]
    assert rewards and all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        for value in rewards
    )


@pytest.mark.parametrize(
    ("label", "job_name", "required"),
    [
        ("dapo-bwrap", None, ("DRESSAGE_HARBOR_JOB_CONFIG",)),
        ("terminal-bench", "terminal-bench-2-e2b.yaml", ("E2B_API_KEY",)),
        ("tau3", "tau3-bench-e2b.yaml", ("E2B_API_KEY",)),
    ],
    ids=("dapo-bwrap", "terminal-bench-e2b", "tau3-e2b"),
)
def test_rollout_e2e_matrix(tmp_path, label, job_name, required):
    values = _inputs("DRESSAGE_HARBOR_INTEGRATION_CONFIG", "HF_CHECKPOINT", *required)
    source = (
        Path(values["DRESSAGE_HARBOR_JOB_CONFIG"])
        if job_name is None
        else JOBS / job_name
    )
    job = _runtime_job(source, tmp_path, label)
    _run(ROLLOUT, Path(values["DRESSAGE_HARBOR_INTEGRATION_CONFIG"]), job)
    _assert_reward(job)


def _training_env(tmp_path, run_id):
    return {
        "DRESSAGE_HARBOR_RUN_ID": run_id,
        "HARBOR_TRAINING_CHECKPOINT_DIR": str(tmp_path / "checkpoints"),
        "NUM_ROLLOUT": "1",
        "ROLLOUT_BATCH_SIZE": "1",
        "N_SAMPLES_PER_PROMPT": "2",
        "GLOBAL_BATCH_SIZE": "2",
        "CKPT_LOAD": "",
    }


@pytest.mark.parametrize(
    ("label", "job_name", "required"),
    [
        ("dapo-training", None, ("DRESSAGE_HARBOR_JOB_CONFIG",)),
        ("tb2-training", "terminal-bench-2-e2b.yaml", ("E2B_API_KEY",)),
    ],
    ids=("dapo-bwrap", "terminal-bench-configure-only"),
)
def test_training_e2e_matrix(tmp_path, label, job_name, required):
    values = _inputs(
        "DRESSAGE_HARBOR_INTEGRATION_CONFIG", "HF_CHECKPOINT", "REF_LOAD", *required
    )
    source = (
        Path(values["DRESSAGE_HARBOR_JOB_CONFIG"])
        if job_name is None
        else JOBS / job_name
    )
    profile = Path(values["DRESSAGE_HARBOR_INTEGRATION_CONFIG"])
    job = _runtime_job(source, tmp_path, label)
    run_id = f"{label}-{uuid.uuid4().hex[:10]}"
    _run(TRAINING, profile, job, env=_training_env(tmp_path, run_id))
    artifact_root = Path(yaml.safe_load(profile.read_text())["artifacts"]["root"])
    assert list((artifact_root / run_id).rglob("manifest.json"))


def test_checkpoint_save_and_restore_e2e(tmp_path):
    values = _inputs(
        "DRESSAGE_HARBOR_JOB_CONFIG",
        "DRESSAGE_HARBOR_INTEGRATION_CONFIG",
        "HF_CHECKPOINT",
        "REF_LOAD",
    )
    profile = Path(values["DRESSAGE_HARBOR_INTEGRATION_CONFIG"])
    job = _runtime_job(values["DRESSAGE_HARBOR_JOB_CONFIG"], tmp_path, "checkpoint")
    env = _training_env(tmp_path, f"checkpoint-{uuid.uuid4().hex[:10]}")
    _run(TRAINING, profile, job, env=env)
    checkpoint = Path(env["HARBOR_TRAINING_CHECKPOINT_DIR"])
    assert list(checkpoint.rglob("harbor_data_source_state_*.json"))
    _run(
        TRAINING,
        profile,
        job,
        env={**env, "CKPT_LOAD": str(checkpoint), "NUM_ROLLOUT": "2"},
    )


@pytest.mark.parametrize(
    "case",
    ["wrong-run-id", "public-task"],
    ids=("wrong-run-id", "public-task-enforced"),
)
def test_e2e_failure_matrix(tmp_path, case):
    required = ["DRESSAGE_HARBOR_INTEGRATION_CONFIG", "HF_CHECKPOINT"]
    if case == "wrong-run-id":
        required += ["DRESSAGE_HARBOR_JOB_CONFIG", "REF_LOAD"]
    else:
        required += ["E2B_API_KEY"]
    values = _inputs(*required)
    profile = Path(values["DRESSAGE_HARBOR_INTEGRATION_CONFIG"])
    source = (
        values["DRESSAGE_HARBOR_JOB_CONFIG"]
        if case == "wrong-run-id"
        else JOBS / "terminal-bench-2-e2b.yaml"
    )
    job = _runtime_job(source, tmp_path, case)
    if case == "wrong-run-id":
        env = _training_env(tmp_path, f"expected-{uuid.uuid4().hex[:8]}")
        _run(TRAINING, profile, job, env=env)
        checkpoint = env["HARBOR_TRAINING_CHECKPOINT_DIR"]
        completed = _run(
            TRAINING,
            profile,
            job,
            env={
                **env,
                "DRESSAGE_HARBOR_RUN_ID": "wrong",
                "CKPT_LOAD": checkpoint,
                "NUM_ROLLOUT": "2",
            },
            check=False,
        )
        expected = ("run",)
    else:
        completed = _run(ROLLOUT, profile, job, check=False)
        expected = ("public", "enforced")
    output = (completed.stdout + completed.stderr).lower()
    assert completed.returncode != 0 and all(value in output for value in expected)
    if case == "wrong-run-id":
        assert "mismatch" in output or "does not match" in output
