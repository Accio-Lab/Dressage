from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal

import pytest

from dressage.sandbox.local.bwrap.runner import (
    LocalSandboxRunner,
    LocalSandboxRunnerConfig,
)
from dressage.sandbox.local.bwrap.slot import SlotConfig, SlotRuntime


def _slot(tmp_path: Path) -> SlotRuntime:
    return SlotRuntime(
        SlotConfig(
            slot_id=3,
            port=31003,
            bind_host="0.0.0.0",
            advertise_host="10.0.0.12",
            base_dir=tmp_path,
            blackbox_type="opencode",
        )
    )


def test_runner_defaults_to_bwrap_and_blackbox_server_main(tmp_path):
    runner = LocalSandboxRunner(
        LocalSandboxRunnerConfig(
            python_bin="/usr/bin/python3",
            readonly_mounts=(Path("/usr"),),
            extra_bubblewrap_args=("--tmpfs", "/run", "--ro-bind", "/proc", "/proc"),
        )
    )
    slot = _slot(tmp_path)

    command = runner.build_command(slot)
    env = runner.build_env(slot)

    assert command[0] == "bwrap"
    assert command[-4:] == ["--", "/usr/bin/python3", "-m", "blackbox_server.main"]
    assert "--unshare-user" in command
    assert "--unshare-ipc" in command
    assert "--unshare-uts" in command
    assert "--unshare-pid" not in command
    assert "--unshare-net" not in command
    assert "--disable-userns" not in command
    assert "--proc" not in command
    assert _has_subsequence(command, ["--tmpfs", "/run"])
    assert _has_subsequence(command, ["--ro-bind", "/proc", "/proc"])
    assert _has_subsequence(command, ["--ro-bind", "/usr", "/usr"])
    assert _has_subsequence(command, ["--hostname", "dressage-bb-0003"])
    assert env["HOME"] == "/home/blackbox"
    assert env["TMPDIR"] == "/tmp"
    assert env["BBS_HOST"] == "0.0.0.0"
    assert env["BBS_PORT"] == "31003"
    assert env["BBS_RUNTIME_ROOT"] == "/workspace_sandbox/blackbox_server_runtime"
    assert env["DRESSAGE_BLACKBOX_RUNTIME_ROOT"] == env["BBS_RUNTIME_ROOT"]
    assert env["DRESSAGE_BLACKBOX_SLOT_ID"] == "3"
    assert env["DRESSAGE_BLACKBOX_SLOT_DIR"] == str(slot.config.slot_dir)
    assert env["DRESSAGE_BLACKBOX_SLOT_GENERATION"] == "0"
    assert env["DRESSAGE_BLACKBOX_SLOT_PORT"] == "31003"
    assert env["DRESSAGE_BLACKBOX_SLOT_TOKEN"]


def test_runner_uses_host_paths_for_direct_mode_env(tmp_path):
    runner = LocalSandboxRunner(LocalSandboxRunnerConfig(mode="direct"))
    slot = _slot(tmp_path)

    command = runner.build_command(slot)
    env = runner.build_env(slot)

    assert command[-3:] == [runner.config.python_bin, "-m", "blackbox_server.main"]
    assert command[0] != "bwrap"
    assert env["HOME"] == str(slot.config.home_dir)
    assert env["TMPDIR"] == str(slot.config.tmp_dir)
    assert env["BBS_RUNTIME_ROOT"] == str(slot.config.runtime_dir)


def test_config_from_env_defaults_to_bwrap_with_container_safe_options(monkeypatch):
    monkeypatch.delenv("DRESSAGE_BLACKBOX_RUNNER_MODE", raising=False)
    monkeypatch.delenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", raising=False)
    monkeypatch.delenv("DRESSAGE_BLACKBOX_BWRAP_EXTRA_ARGS", raising=False)
    monkeypatch.delenv("DRESSAGE_BLACKBOX_BWRAP_UNSHARE_PID", raising=False)
    monkeypatch.delenv("DRESSAGE_BLACKBOX_BWRAP_DISABLE_USERNS", raising=False)
    monkeypatch.delenv("DRESSAGE_BLACKBOX_BWRAP_UNSHARE_NET", raising=False)

    config = LocalSandboxRunnerConfig.from_env()
    runner = LocalSandboxRunner(config)
    command = runner.build_command(_slot(Path("/tmp/slots")))

    assert config.mode == "bwrap"
    assert config.slot_uid == os.geteuid()
    assert config.slot_gid == os.getegid()
    assert config.disable_proc is True
    assert config.bubblewrap_unshare_pid is False
    assert config.bubblewrap_unshare_net is False
    assert config.bubblewrap_disable_userns is False
    assert _has_subsequence(command, ["--tmpfs", "/run"])
    if Path("/proc").exists():
        assert _has_subsequence(command, ["--ro-bind", "/proc", "/proc"])
    else:
        assert not _has_subsequence(command, ["--ro-bind", "/proc", "/proc"])
    assert "--proc" not in command
    assert "--unshare-pid" not in command
    assert "--disable-userns" not in command


def test_config_from_env_allows_empty_readonly_mounts_and_extra_args(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", "")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BWRAP_EXTRA_ARGS", "")

    config = LocalSandboxRunnerConfig.from_env()
    runner = LocalSandboxRunner(config)
    command = runner.build_command(_slot(Path("/tmp/slots")))

    assert config.readonly_mounts == ()
    assert config.extra_bubblewrap_args == ()
    assert not _has_subsequence(command, ["--ro-bind", "/proc", "/proc"])


def test_config_from_env_supports_bubblewrap_options(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_RUNNER_MODE", "bubblewrap")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BWRAP_BIN", "bubblewrap")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BWRAP_EXTRA_ARGS", "--tmpfs /custom-run")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BWRAP_UNSHARE_NET", "1")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BWRAP_UNSHARE_PID", "1")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_BWRAP_DISABLE_USERNS", "1")
    monkeypatch.setenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", "")

    config = LocalSandboxRunnerConfig.from_env()
    runner = LocalSandboxRunner(config)
    command = runner.build_command(_slot(Path("/tmp/slots")))

    assert config.mode == "bwrap"
    assert config.bubblewrap_bin == "bubblewrap"
    assert config.extra_bubblewrap_args == ("--tmpfs", "/custom-run")
    assert config.bubblewrap_unshare_net is True
    assert config.bubblewrap_unshare_pid is True
    assert config.bubblewrap_disable_userns is True
    assert command[0] == "bubblewrap"
    assert "--unshare-net" in command
    assert "--unshare-pid" in command
    assert "--disable-userns" in command


def test_config_from_env_can_enable_fresh_proc_mount(monkeypatch):
    monkeypatch.setenv("DRESSAGE_BLACKBOX_DISABLE_PROC", "0")
    monkeypatch.delenv("DRESSAGE_BLACKBOX_BWRAP_EXTRA_ARGS", raising=False)
    monkeypatch.setenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", "")

    config = LocalSandboxRunnerConfig.from_env()
    runner = LocalSandboxRunner(config)
    command = runner.build_command(_slot(Path("/tmp/slots")))

    assert config.disable_proc is False
    assert _has_subsequence(command, ["--proc", "/proc"])
    assert not _has_subsequence(command, ["--ro-bind", "/proc", "/proc"])


def test_unsupported_nsjail_mode_is_rejected():
    runner = LocalSandboxRunner(LocalSandboxRunnerConfig(mode="nsjail"))

    with pytest.raises(ValueError, match="expected 'bwrap', 'bubblewrap', or 'direct'"):
        runner.build_command(_slot(Path("/tmp/slots")))


def test_default_readonly_mounts_include_pythonpath(monkeypatch, tmp_path):
    repo = tmp_path / "Dressage"
    slime = repo / "slime"
    slime.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", f"{repo}:{slime}:")
    monkeypatch.delenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", raising=False)

    config = LocalSandboxRunnerConfig.from_env()

    assert repo in config.readonly_mounts
    assert slime in config.readonly_mounts


def test_default_readonly_mounts_include_common_user_tool_roots(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    (fake_home / ".opencode").mkdir(parents=True)
    (fake_home / ".openclaw").mkdir()
    (fake_home / ".local").mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", raising=False)

    config = LocalSandboxRunnerConfig.from_env()

    assert fake_home / ".opencode" in config.readonly_mounts
    assert fake_home / ".openclaw" in config.readonly_mounts
    assert fake_home / ".local" in config.readonly_mounts


def test_default_readonly_mounts_include_codex_bin_symlink_target(
    monkeypatch,
    tmp_path,
):
    usr_bin = tmp_path / "usr" / "local" / "bin"
    codex_root = tmp_path / "opt" / "codex"
    codex_bin = codex_root / "bin" / "codex"
    usr_bin.mkdir(parents=True)
    codex_bin.parent.mkdir(parents=True)
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    (usr_bin / "codex").symlink_to(codex_bin)
    monkeypatch.setenv("CODEX_BIN", str(usr_bin / "codex"))
    monkeypatch.delenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", raising=False)

    config = LocalSandboxRunnerConfig.from_env()

    assert tmp_path / "usr" / "local" in config.readonly_mounts
    assert codex_root in config.readonly_mounts


def test_default_readonly_mounts_include_claude_code_bin_target(
    monkeypatch,
    tmp_path,
):
    fake_home = tmp_path / "home"
    claude_bin = fake_home / ".claude" / "local" / "claude"
    claude_bin.parent.mkdir(parents=True)
    claude_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(claude_bin))
    monkeypatch.delenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", raising=False)

    config = LocalSandboxRunnerConfig.from_env()

    assert fake_home / ".claude" / "local" in config.readonly_mounts


def test_default_readonly_mounts_include_backend_binary_found_on_path(
    monkeypatch,
    tmp_path,
):
    usr_bin = tmp_path / "usr" / "local" / "bin"
    codex_root = tmp_path / "opt" / "codex"
    codex_bin = codex_root / "bin" / "codex"
    usr_bin.mkdir(parents=True)
    codex_bin.parent.mkdir(parents=True)
    codex_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    codex_bin.chmod(0o755)
    (usr_bin / "codex").symlink_to(codex_bin)
    monkeypatch.setenv("PATH", str(usr_bin))
    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.delenv("DRESSAGE_BLACKBOX_READONLY_MOUNTS", raising=False)

    config = LocalSandboxRunnerConfig.from_env()

    assert codex_root in config.readonly_mounts


def test_bubblewrap_readonly_mount_preserves_symlink(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to("real")
    runner = LocalSandboxRunner(
        LocalSandboxRunnerConfig(mode="bubblewrap", readonly_mounts=(link,))
    )

    command = runner.build_command(_slot(tmp_path / "slots"))

    assert _has_subsequence(command, ["--symlink", "real", str(link)])
    assert not _has_subsequence(command, ["--ro-bind", str(link), str(link)])


def test_runner_extra_env_cannot_override_slot_markers(tmp_path):
    runner = LocalSandboxRunner(
        LocalSandboxRunnerConfig(
            mode="direct",
            extra_env={
                "DRESSAGE_BLACKBOX_SLOT_ID": "wrong",
                "DRESSAGE_BLACKBOX_SLOT_TOKEN": "wrong-token",
            },
        )
    )
    slot = _slot(tmp_path)

    env = runner.build_env(slot)

    assert env["DRESSAGE_BLACKBOX_SLOT_ID"] == "3"
    assert env["DRESSAGE_BLACKBOX_SLOT_TOKEN"] == slot.cleanup_token
    assert env["DRESSAGE_BLACKBOX_SLOT_TOKEN"] != "wrong-token"


def test_stop_signals_process_group_before_residual_cleanup(monkeypatch, tmp_path):
    class FakeProcess:
        pid = 12345
        returncode = None

        def terminate(self):
            raise AssertionError("process group should be terminated before proc")

        def kill(self):
            raise AssertionError("process group should be killed before proc")

        async def wait(self):
            self.returncode = 0
            return 0

    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "getpgid", lambda pid: 54321)
    monkeypatch.setattr(os, "getpgrp", lambda: 111)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    runner = LocalSandboxRunner(
        LocalSandboxRunnerConfig(
            mode="direct",
            cleanup_residual_processes=False,
        )
    )
    slot = _slot(tmp_path)
    slot.process = FakeProcess()

    asyncio.run(runner.stop(slot, timeout_sec=0.01))

    assert calls == [(54321, signal.SIGTERM)]


def test_stop_scans_residual_processes_even_when_top_process_exited(
    monkeypatch, tmp_path
):
    runner = LocalSandboxRunner(LocalSandboxRunnerConfig(mode="direct"))
    called = False

    async def fake_cleanup(slot):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "_cleanup_residual_processes", fake_cleanup)
    slot = _slot(tmp_path)
    slot.process = type("ExitedProcess", (), {"returncode": 0})()

    asyncio.run(runner.stop(slot, timeout_sec=0.01))

    assert called is True


def test_residual_cleanup_matches_only_current_slot_token(monkeypatch, tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    slot = _slot(tmp_path / "slots")
    slot.generation = 7
    slot.cleanup_token = "token-current"
    slot.supervisor_run_id = "run-current"

    _write_fake_environ(
        proc_root / "101" / "environ",
        {
            "DRESSAGE_BLACKBOX_SLOT_ID": "3",
            "DRESSAGE_BLACKBOX_SLOT_DIR": str(slot.config.slot_dir),
            "DRESSAGE_BLACKBOX_SLOT_GENERATION": "7",
            "DRESSAGE_BLACKBOX_SLOT_PORT": "31003",
            "DRESSAGE_BLACKBOX_SLOT_TOKEN": "token-current",
            "DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID": "run-current",
        },
    )
    _write_fake_environ(
        proc_root / "102" / "environ",
        {
            "DRESSAGE_BLACKBOX_SLOT_ID": "3",
            "DRESSAGE_BLACKBOX_SLOT_DIR": str(slot.config.slot_dir),
            "DRESSAGE_BLACKBOX_SLOT_GENERATION": "7",
            "DRESSAGE_BLACKBOX_SLOT_PORT": "31003",
            "DRESSAGE_BLACKBOX_SLOT_TOKEN": "other-token",
            "DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID": "run-current",
        },
    )
    _write_fake_environ(
        proc_root / "103" / "environ",
        {"PATH": "/usr/bin", "CMD": "opencode serve"},
    )

    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    runner = LocalSandboxRunner(
        LocalSandboxRunnerConfig(
            mode="direct",
            residual_proc_root=proc_root,
            residual_cleanup_timeout_sec=0,
        )
    )

    asyncio.run(runner.stop(slot, timeout_sec=0.01))

    assert killed == [(101, signal.SIGTERM), (101, signal.SIGKILL)]


def _write_fake_environ(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\0".join(f"{key}={value}".encode() for key, value in values.items())
    path.write_bytes(payload + b"\0")


def _has_subsequence(values: list[str], expected: list[str]) -> bool:
    width = len(expected)
    return any(values[index : index + width] == expected for index in range(len(values)))
