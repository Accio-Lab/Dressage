from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest


pytest.importorskip("harbor")

from harbor.environments.base import BaseEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial

from dressage.integrations.harbor import environment as environment_module


def _contains(command, values):
    width = len(values)
    return any(
        command[index : index + width] == values for index in range(len(command))
    )


def _configure_bwrap(monkeypatch, tmp_path):
    config = tmp_path / "integration.json"
    config.write_text(
        json.dumps(
            {
                "environment": {
                    "mode": "bwrap",
                    "runtime_root": str(tmp_path / "runtime"),
                },
                "gateway": {"listen_port": 39100},
                "security": {"routing_guarantee": "enforced"},
            }
        )
    )
    monkeypatch.setenv("DRESSAGE_HARBOR_INTEGRATION_CONFIG", str(config))
    monkeypatch.setattr(
        environment_module, "_required_binary", lambda name: shutil.which(name) or f"/usr/bin/{name}"
    )


def _environment(monkeypatch, tmp_path, session_id, mode, *, workdir="/app"):
    _configure_bwrap(monkeypatch, tmp_path)
    verifier = "__verifier__" in session_id and not session_id.endswith("__env")
    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    paths = TrialPaths(tmp_path / f"trial-{session_id}")
    paths.mkdir()
    auto_mounts = None
    if verifier:
        auto_mounts = [
            {"type": "bind", "source": str(paths.verifier_dir), "target": "/logs/verifier"}
        ]
    policy = NetworkPolicy(
        network_mode=mode,
        allowed_hosts=["127.0.0.1"] if mode == NetworkMode.ALLOWLIST else [],
    )
    return environment_module.DressageEnvironment(
        environment_dir=environment_dir,
        environment_name="test",
        session_id=session_id,
        trial_paths=paths,
        task_env_config=EnvironmentConfig(
            network_mode=mode,
            allowed_hosts=policy.allowed_hosts,
            gpus=0,
            workdir=workdir,
        ),
        network_policy=policy,
        auto_mounts=auto_mounts,
    )


@pytest.mark.asyncio
async def test_agent_and_verifier_are_network_and_filesystem_isolated(
    monkeypatch, tmp_path
):
    agent = _environment(monkeypatch, tmp_path, "trial__env", NetworkMode.ALLOWLIST)
    verifier = _environment(
        monkeypatch, tmp_path, "trial__verifier__trial", NetworkMode.PUBLIC
    )
    assert {mount["target"] for mount in agent._mounts} == {
        "/logs/agent",
        "/logs/verifier",
        "/logs/artifacts",
    }
    assert {mount["target"] for mount in verifier._mounts} == {"/logs/verifier"}
    (verifier.environment_dir / "agent-input.txt").write_text("ordinary environment input\n")
    await agent.start(False)
    await verifier.start(False)
    try:
        socket_dir = agent._runtime_root / "gateway"
        socket_dir.mkdir(parents=True)
        (socket_dir / "gateway.sock").touch()
        agent_command = agent._bwrap_command("true", cwd="/app")
        verifier_command = verifier._bwrap_command("true", cwd="/app")
        assert agent._session_root != verifier._session_root
        assert (
            "--unshare-net" in agent_command and "--unshare-net" not in verifier_command
        )
        assert "--unshare-pid" in agent_command
        assert "--unshare-pid" in verifier_command
        assert _contains(agent_command, ["--bind", str(agent._rootfs), "/"])
        assert not _contains(agent_command, ["--ro-bind", "/", "/"])
        assert list((agent._rootfs / "tests").iterdir()) == []
        assert list((verifier._rootfs / "tests").iterdir()) == []
        script = tmp_path / "test.sh"
        script.write_bytes(b"#!/bin/sh\nexit 0\n")
        await verifier.upload_file(script, "/tests/test.sh")
        assert (verifier._rootfs / "tests/test.sh").read_bytes() == script.read_bytes()
        assert list((agent._rootfs / "tests").iterdir()) == []
        artifact = agent._rootfs / "app/result.txt"
        artifact.write_text("result")
        exported = tmp_path / "result.txt"
        await agent.download_file("/app/result.txt", exported)
        await verifier.upload_file(exported, "/app/result.txt")
        assert (verifier._rootfs / "app/result.txt").read_text() == "result"
    finally:
        await agent.stop(True)
        await verifier.stop(True)


@pytest.mark.asyncio
async def test_paths_mounts_and_arbitrary_artifacts_remain_safe(monkeypatch, tmp_path):
    environment = _environment(
        monkeypatch,
        tmp_path,
        "trial__env",
        NetworkMode.NO_NETWORK,
        workdir="/workspace/project",
    )
    await environment.start(False)
    try:
        with pytest.raises(ValueError, match="without '..'"):
            environment._host_path("/app/../etc/passwd")
        with pytest.raises(PermissionError):
            environment._host_path("/etc/passwd", writable=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (environment._rootfs / "app/escape").symlink_to(
            outside, target_is_directory=True
        )
        with pytest.raises(ValueError, match="escapes rootfs"):
            environment._host_path("/app/escape/file", writable=True)

        source = tmp_path / "artifact.txt"
        source.write_text("artifact")
        await environment.upload_file(source, "/var/results/artifact.txt")
        assert (
            environment._rootfs / "var/results/artifact.txt"
        ).read_text() == "artifact"

        usr = tmp_path / "usr"
        (usr / "bin").mkdir(parents=True)
        link = tmp_path / "bin"
        link.symlink_to("usr/bin", target_is_directory=True)
        mounts = environment_module._readonly_runtime_mounts(
            (), system_roots=(usr, link), home=tmp_path / "home"
        )
        rootfs = tmp_path / "mount-rootfs"
        environment_module._initialize_rootfs(rootfs, mounts, workdir="/workspace")
        assert rootfs.joinpath(*link.parts[1:]).is_symlink()
    finally:
        await environment.stop(True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auto_mounting", "external_mount", "expected_downloads"),
    [(True, False, 0), (False, False, 1), (False, True, 1)],
    ids=["automatic", "sandbox-only", "explicit-external"],
)
async def test_trial_recovers_agent_logs(
    monkeypatch, tmp_path, auto_mounting, external_mount, expected_downloads
):
    _configure_bwrap(monkeypatch, tmp_path)
    task = tmp_path / "task"
    for name in ("environment", "solution", "tests"):
        (task / name).mkdir(parents=True)
    (task / "task.toml").write_text('schema_version = "1.3"\n[environment]\ngpus = 0\n')
    (task / "instruction.md").write_text("Write a log.")
    (task / "solution/solve.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")

    external = tmp_path / "external-agent-logs"
    backend = {
        "import_path": "dressage.integrations.harbor.environment:DressageEnvironment",
        "kwargs": {"auto_mounting": auto_mounting},
    }
    if external_mount:
        backend["mounts"] = [
            {"type": "bind", "source": str(external), "target": "/logs/agent"}
        ]
    config = TrialConfig.model_validate(
        {
            "task": {"path": task},
            "agent": {"name": "oracle"},
            "environment": backend,
            "trial_name": "logs",
            "trials_dir": tmp_path / "trials",
        }
    )
    trial = await Trial.create(config)
    env = trial.agent_environment
    try:
        await env.start(False)
        payload = tmp_path / "probe.log"
        payload.write_bytes(b"one log entry\n")
        await env.upload_file(payload, "/logs/agent/probe.log")
        target = trial.paths.agent_dir / "probe.log"
        assert target.exists() == auto_mounting
        if external_mount:
            assert (external / "probe.log").read_bytes() == payload.read_bytes()

        download = AsyncMock(wraps=env.download_dir)
        monkeypatch.setattr(env, "download_dir", download)
        await trial._download_agent_logs()
        assert download.await_count == expected_downloads
        assert target.read_bytes() == payload.read_bytes()
        assert "Failed to download logs" not in trial.paths.log_path.read_text()
    finally:
        await env.stop(delete=True)
        trial._close_logger_handler()


@pytest.fixture(scope="module")
def bwrap_network_namespace():
    bwrap = shutil.which("bwrap")
    if sys.platform != "linux" or bwrap is None:
        pytest.skip("requires Linux and bwrap")

    # Exercise bwrap's loopback setup before running any Harbor or scoring code.
    probe = subprocess.run(
        [bwrap, "--die-with-parent", "--unshare-user", "--unshare-net",
         "--ro-bind", "/", "/", "--", "/bin/true"],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        timeout=10,
    )
    if probe.returncode == 0:
        return

    error = probe.stderr.strip()
    permission_denied = error.startswith("bwrap: No permissions to create new namespace") or (
        error.startswith(("bwrap: Creating new namespace failed", "bwrap: loopback:"))
        and error.endswith(("Operation not permitted", "Permission denied"))
    )
    namespace_limit_exceeded = error == (
        "bwrap: Creating new namespace failed: nesting depth or "
        "/proc/sys/user/max_*_namespaces exceeded (ENOSPC)"
    )
    if permission_denied or namespace_limit_exceeded:
        pytest.skip(f"bwrap cannot create an isolated network namespace: {error}")
    pytest.fail(f"bwrap network namespace probe exited {probe.returncode}: {error}")


@pytest.mark.usefixtures("bwrap_network_namespace")
@pytest.mark.asyncio
@pytest.mark.parametrize("envdir", ["missing", "empty", "populated"])
async def test_trial_uploads_host_tests_to_separate_verifier(
    monkeypatch, tmp_path, envdir
):
    _configure_bwrap(monkeypatch, tmp_path)
    task = tmp_path / "task"
    tests_dir = task / "tests"
    (tests_dir / "fixtures").mkdir(parents=True)
    (task / "instruction.md").write_text("What is 2 + 2?")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n[environment]\ngpus = 0\n'
        '[verifier]\nenvironment_mode = "separate"\n'
        '[verifier.environment]\nnetwork_mode = "no-network"\ngpus = 0\n'
    )
    if envdir != "missing":
        (task / "environment").mkdir()
    if envdir == "populated":
        (task / "environment/agent-input.txt").write_bytes(b"ordinary environment input\n")

    (tests_dir / "fixtures/answer.txt").write_bytes(b"4\n")
    (tests_dir / "test.sh").write_text(
        """#!/bin/sh
set -eu
cp -R /tests /logs/verifier/received
test "$(cat /tests/fixtures/answer.txt)" = 4
echo '{"reward": 1}' > /logs/verifier/reward.json
"""
    )
    config = TrialConfig.model_validate(
        {
            "task": {"path": task},
            "agent": {"name": "oracle"},
            "environment": {
                "import_path": "dressage.integrations.harbor.environment:DressageEnvironment",
                "kwargs": {"auto_mounting": True},
            },
            "trial_name": "grade",
            "trials_dir": tmp_path / "trials",
        }
    )
    trial = await Trial.create(config)
    try:
        result = await trial._run_separate_verifier(
            key="verify", timeout_sec=15, user=None, artifacts_dir=trial.paths.artifacts_dir
        )
        # Compare the files seen by the running verifier with their host originals.
        received = trial.paths.verifier_dir / "received"
        actual_files = {
            path.relative_to(received).as_posix(): path.read_bytes()
            for path in received.rglob("*")
            if path.is_file()
        }
        assert actual_files == {
            "test.sh": (tests_dir / "test.sh").read_bytes(),
            "fixtures/answer.txt": (tests_dir / "fixtures/answer.txt").read_bytes(),
        }
        assert json.loads(trial.paths.reward_json_path.read_text()) == {"reward": 1}
        assert result.rewards == {"reward": 1}
    finally:
        trial._close_logger_handler()


@pytest.mark.usefixtures("bwrap_network_namespace")
@pytest.mark.asyncio
async def test_oracle_trial_delivers_logs_with_default_mounts(monkeypatch, tmp_path):
    _configure_bwrap(monkeypatch, tmp_path)
    task = tmp_path / "task"
    for name in ("environment", "solution", "tests"):
        (task / name).mkdir(parents=True)
    (task / "instruction.md").write_text("Write 4 to /app/answer.txt.")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n[environment]\ngpus = 0\nnetwork_mode = "no-network"\n'
    )
    (task / "solution/solve.sh").write_text(
        "#!/bin/sh\nset -eu\ntest -d /logs/artifacts\necho 4 > /app/answer.txt\necho solved\n"
    )
    (task / "tests/test.sh").write_text(
        """#!/bin/sh
set -eu
test "$(cat /app/answer.txt)" = 4
echo '{"reward": 1}' > /logs/verifier/reward.json
"""
    )
    config = TrialConfig.model_validate(
        {
            "task": {"path": task},
            "agent": {"name": "oracle"},
            "environment": {
                "import_path": "dressage.integrations.harbor.environment:DressageEnvironment"
            },
            "trial_name": "oracle-logs",
            "trials_dir": tmp_path / "trials",
        }
    )
    trial = await Trial.create(config)
    result = await trial.run()

    assert "solved" in (trial.paths.agent_dir / "oracle.txt").read_text().splitlines()
    assert result.verifier_result is not None
    assert result.verifier_result.rewards == {"reward": 1}


def test_harbor_preflight_signature():
    # Flag changes to the Harbor hook separately from the factory behavior tests.
    parameters = inspect.signature(BaseEnvironment.preflight).parameters.values()
    assert [(param.name, param.kind, param.default) for param in parameters] == [
        ("kwargs", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ]


def test_preflight_uses_container_compatible_proc(monkeypatch, tmp_path):
    config = tmp_path / "integration.json"
    config.write_text(
        json.dumps(
            {
                "environment": {
                    "mode": "bwrap",
                    "runtime_root": str(tmp_path / "runtime"),
                },
                "gateway": {"listen_port": 39100},
                "security": {"routing_guarantee": "enforced"},
            }
        )
    )
    monkeypatch.setenv("DRESSAGE_HARBOR_INTEGRATION_CONFIG", str(config))
    monkeypatch.setattr(environment_module.sys, "platform", "linux")
    monkeypatch.setattr(
        environment_module, "_required_binary", lambda name: f"/usr/bin/{name}"
    )
    commands = []
    monkeypatch.setattr(
        environment_module.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command)
        or subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    environment_module.DressageEnvironment.preflight()
    assert len(commands) == 1
    assert "--unshare-pid" in commands[0]
    assert _contains(commands[0], ["--ro-bind", "/proc", "/proc"])
    assert "--proc" not in commands[0]


@pytest.mark.parametrize(
    "factory_arguments",
    [
        {},
        {"kwargs": None},
        {"kwargs": {}},
        {"kwargs": {"test_marker": "keep"}},
    ],
    ids=["default", "explicit-none", "empty-options", "nonempty-options"],
)
def test_preflight_through_harbor_factory(factory_arguments, monkeypatch):
    class BackendCheckReached(Exception):
        """Stop at the first bwrap config lookup."""

    backend = environment_module.DressageEnvironment
    parse_options = Mock(wraps=backend.parse_options)
    backend_check = Mock(side_effect=BackendCheckReached)
    monkeypatch.setattr(backend, "parse_options", parse_options)
    monkeypatch.setattr(environment_module.sys, "platform", "linux")
    monkeypatch.setattr(environment_module, "_load_bwrap_integration_config", backend_check)

    with pytest.raises(BackendCheckReached):
        EnvironmentFactory.run_preflight(
            type=None,
            import_path="dressage.integrations.harbor.environment:DressageEnvironment",
            **factory_arguments,
        )

    parse_options.assert_called_once_with(factory_arguments.get("kwargs"))
    backend_check.assert_called_once_with()


@pytest.mark.skipif(sys.platform != "linux", reason="bubblewrap requires Linux")
@pytest.mark.asyncio
async def test_real_bwrap_persists_workspace(monkeypatch, tmp_path):
    missing = [
        name
        for name in ("bwrap", "claude", "curl", "python3")
        if not shutil.which(name)
    ]
    if missing:
        pytest.skip("missing bwrap runtime: " + ", ".join(missing))
    environment = _environment(
        monkeypatch,
        tmp_path,
        "trial__verifier__trial",
        NetworkMode.PUBLIC,
        workdir="/workspace",
    )
    await environment.start(False)
    try:
        first = await environment.exec("printf persistent > /workspace/state.txt")
        second = await environment.exec("cat /workspace/state.txt")
        assert first.return_code == 0 and second.stdout == "persistent"
    finally:
        await environment.stop(True)
