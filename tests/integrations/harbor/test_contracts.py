from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
import yaml

from dressage.integrations.harbor import compat
from dressage.integrations.harbor.compat import HarborCompatibilityError
from dressage.integrations.harbor.config import HarborIntegrationConfig, load_config


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples" / "harbor"
PROFILES = EXAMPLES / "dressage_profiles"
JOBS = EXAMPLES / "harbor_job_configs"
RUNNERS = (
    EXAMPLES / "run_harbor_rollout_qwen3.5_4b.sh",
    EXAMPLES / "run_harbor_training_qwen3.5_4b.sh",
)


def test_config_defaults_loading_and_secret_resolution(tmp_path):
    config = HarborIntegrationConfig()
    assert (config.schema_version, config.execution_mode, config.environment.mode) == (
        "dressage.harbor/v1",
        "rollout",
        "native",
    )
    assert config.backend.service_headers(
        {"DRESSAGE_PROXY_API_KEY": "secret"}, required=True
    ) == {"Authorization": "Bearer secret"}
    assert "secret" not in config.model_dump_json()

    path = tmp_path / "harbor.json"
    path.write_text(json.dumps({"agent_protocol_overrides": {"custom": "openai"}}))
    assert load_config(path).agent_protocol_overrides == {"custom": "openai"}


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "execution_mode": "training",
                "training": {"model_override": "m"},
                "artifacts": {"mode": "disk"},
            },
            "artifacts.mode='both'",
        ),
        (
            {
                "environment": {"mode": "bwrap"},
                "security": {"routing_guarantee": "enforced"},
            },
            "fixed non-zero",
        ),
        (
            {
                "environment": {"mode": "bwrap"},
                "gateway": {"listen_port": 39100},
                "security": {"routing_guarantee": "configure_only"},
            },
            "bwrap requires.*enforced",
        ),
        (
            {"gateway": {"advertise_url": "http://gateway.example:39000"}},
            "must use HTTPS",
        ),
    ],
    ids=("training-artifacts", "bwrap-port", "bwrap-routing", "remote-http"),
)
def test_config_rejects_unsafe_combinations(payload, message):
    with pytest.raises(ValidationError, match=message):
        HarborIntegrationConfig.model_validate(payload)


@dataclass
class _Trial:
    task: object
    agent: object
    trial_name: str


def _compat_job():
    tasks, agents = [object(), object()], [object(), object()]
    configs = [
        _Trial(task, agent, f"trial-{index}")
        for index, (task, agent) in enumerate(
            (task, agent) for _ in range(2) for task in tasks for agent in agents
        )
    ]
    return SimpleNamespace(
        is_resuming=False,
        _remaining_trial_configs=configs,
        _task_configs=tasks,
        config=SimpleNamespace(agents=agents, n_attempts=2),
    )


def test_harbor_018_runtime_and_trial_plan_contract(monkeypatch):
    with pytest.raises(HarborCompatibilityError, match="Python >=3.12"):
        compat.require_harbor_runtime(
            python_version=(3, 10, 18), harbor_version="0.18.0"
        )
    with pytest.raises(HarborCompatibilityError, match="expected exactly"):
        compat.require_harbor_runtime(
            python_version=(3, 12, 0), harbor_version="0.18.1"
        )

    monkeypatch.setattr(compat, "require_harbor_runtime", lambda: None)
    job = _compat_job()
    assert len(compat.pending_trial_configs(job)) == 8
    plan = compat.build_trial_plan(job)
    assert [(p.attempt_index, p.task_index, p.agent_index) for p in plan] == [
        (attempt, task, agent)
        for attempt in range(2)
        for task in range(2)
        for agent in range(2)
    ]
    named = compat.assign_trial_names(
        job, lambda p: f"a{p.attempt_index}-{p.task_index}-{p.agent_index}"
    )
    assert len({item.trial_name for item in named}) == 8


def test_harbor_018_task_resolver_boundary(monkeypatch):
    pytest.importorskip("harbor")
    monkeypatch.setattr(compat, "require_harbor_runtime", lambda: None)
    copied = object()
    config = SimpleNamespace(
        tasks=[SimpleNamespace(model_copy=lambda deep: copied)], datasets=[]
    )
    assert __import__("asyncio").run(compat.resolve_task_configs(config)) == (copied,)


@pytest.mark.parametrize(
    ("name", "execution", "environment"),
    [
        ("rollout-native-local.yaml", "rollout", "native"),
        ("rollout-native-remote.yaml", "rollout", "native"),
        ("rollout-bwrap.yaml", "rollout", "bwrap"),
        ("training-native-local.yaml", "training", "native"),
        ("training-native-remote.yaml", "training", "native"),
        ("training-bwrap.yaml", "training", "bwrap"),
    ],
)
def test_profiles_are_valid_public_configs(name, execution, environment):
    config = HarborIntegrationConfig.model_validate(
        yaml.safe_load((PROFILES / name).read_text())
    )
    assert (config.execution_mode, config.environment.mode) == (execution, environment)
    assert config.security.routing_guarantee == (
        "enforced" if environment == "bwrap" else "configure_only"
    )


def test_official_jobs_validate_with_harbor_018():
    pytest.importorskip("harbor")
    from harbor.models.job.config import JobConfig

    assert importlib.metadata.version("harbor") == "0.18.0"
    for name, dataset in {
        "terminal-bench-2-e2b.yaml": "terminal-bench/terminal-bench-2",
        "tau3-bench-e2b.yaml": "sierra-research/tau3-bench",
    }.items():
        job = JobConfig.model_validate(yaml.safe_load((JOBS / name).read_text()))
        assert job.datasets[0].name == dataset
        assert job.agents[0].model_name == "Qwen/Qwen3.5-4B"
        assert getattr(job.environment.type, "value", job.environment.type) == "e2b"


def _executable(path: Path, source: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("runner", "exit_code", "cleanup_tokens"),
    [
        (RUNNERS[0], 23, ("sglang_router.launch_router", "proxy.server")),
        (RUNNERS[1], 29, ("proxy.server", "ray-stop")),
    ],
    ids=("rollout", "training"),
)
def test_runner_contract_and_cleanup(tmp_path, runner, exit_code, cleanup_tokens):
    subprocess.run(["bash", "-n", str(runner)], check=True)
    bin_dir, capture = tmp_path / "bin", tmp_path / "commands.log"
    slime = tmp_path / "slime"
    model_args = slime / "scripts" / "models" / "qwen3.5-4B.sh"
    model_args.parent.mkdir(parents=True)
    model_args.write_text("MODEL_ARGS=(--fake-model)\n")
    _executable(
        bin_dir / "python",
        "#!/usr/bin/env bash\ntrap 'printf '%s\\n' \"$*\" >>\"${COMMAND_CAPTURE}\"; exit 0' TERM INT\nwhile true; do /bin/sleep 1; done\n",
    )
    _executable(bin_dir / "curl", "#!/usr/bin/env bash\n/bin/sleep 0.05\n")
    _executable(bin_dir / "openssl", "#!/usr/bin/env bash\nprintf '%096d\\n' 0\n")
    _executable(bin_dir / "harbor", '#!/usr/bin/env bash\nexit "${RUNNER_EXIT_CODE}"\n')
    _executable(
        bin_dir / "ray",
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >>"${COMMAND_CAPTURE}"\n'
        '[[ "${1:-} ${2:-}" == "job submit" ]] && exit "${RUNNER_EXIT_CODE}"\n'
        '[[ "${1:-}" == "stop" ]] && printf \'ray-stop\\n\' >>"${COMMAND_CAPTURE}"\nexit 0\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHON_BIN": str(bin_dir / "python"),
        "REPO_ROOT": str(ROOT),
        "SLIME_ROOT": str(slime),
        "MEGATRON_ROOT": str(tmp_path / "megatron"),
        "DRESSAGE_HARBOR_JOB_CONFIG": str(tmp_path / "job.yaml"),
        "MASTER_ADDR": "127.0.0.1",
        "SLIME_HOST_IP": "127.0.0.1",
        "HARBOR_LOG_DIR": str(tmp_path / "logs"),
        "HARBOR_DEBUG_DIR": str(tmp_path / "debug"),
        "HARBOR_TRAINING_CHECKPOINT_DIR": str(tmp_path / "checkpoints"),
        "COMMAND_CAPTURE": str(capture),
        "RUNNER_EXIT_CODE": str(exit_code),
    }
    completed = subprocess.run(["bash", str(runner)], cwd=tmp_path, env=env, timeout=20)
    assert completed.returncode == exit_code
    commands = capture.read_text()
    assert all(token in commands for token in cleanup_tokens)
    if runner == RUNNERS[1]:
        assert "--max-output-tokens" in commands


def test_examples_contain_no_literal_credentials():
    sources = "\n".join(
        path.read_text()
        for path in EXAMPLES.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}
    )
    assert "wandb_v1_" not in sources
    assert "dressage-local-example-key" not in sources
