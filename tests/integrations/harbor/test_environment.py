from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest


pytest.importorskip("harbor")

from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths

from dressage.integrations.harbor import environment as environment_module


def _contains(command, values):
    width = len(values)
    return any(
        command[index : index + width] == values for index in range(len(command))
    )


def _environment(monkeypatch, tmp_path, session_id, mode, *, workdir="/app"):
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
        environment_module, "_required_binary", lambda name: f"/usr/bin/{name}"
    )
    verifier = "__verifier__" in session_id and not session_id.endswith("__env")
    environment_dir = tmp_path / "task" / ("tests" if verifier else "environment")
    environment_dir.mkdir(parents=True, exist_ok=True)
    paths = TrialPaths(tmp_path / f"trial-{session_id}")
    paths.mkdir()
    mounts = [
        {
            "type": "bind",
            "source": str(paths.verifier_dir if verifier else paths.agent_dir),
            "target": "/logs/verifier" if verifier else "/logs/agent",
        }
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
        mounts=mounts,
    )


@pytest.mark.asyncio
async def test_agent_and_verifier_are_network_and_filesystem_isolated(
    monkeypatch, tmp_path
):
    agent = _environment(monkeypatch, tmp_path, "trial__env", NetworkMode.ALLOWLIST)
    verifier = _environment(
        monkeypatch, tmp_path, "trial__verifier__trial", NetworkMode.PUBLIC
    )
    (verifier.environment_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n")
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
        assert _contains(agent_command, ["--bind", str(agent._rootfs), "/"])
        assert not _contains(agent_command, ["--ro-bind", "/", "/"])
        assert list((agent._rootfs / "tests").iterdir()) == []
        assert (verifier._rootfs / "tests/test.sh").is_file()
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
    assert _contains(commands[0], ["--ro-bind", "/proc", "/proc"])
    assert "--proc" not in commands[0]


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
