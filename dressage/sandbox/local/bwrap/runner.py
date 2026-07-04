"""Process runner for local bwrap slots.

The local runner supports both blackbox server processes and command-only
one-shot tool commands. It intentionally supports only two execution modes:

* ``bwrap``/``bubblewrap`` (default): run inside a lightweight bubblewrap
  sandbox.
* ``direct``: run directly on host paths, mainly for debugging.

The bubblewrap defaults are tuned for the in-container Dressage blackbox setup:
keep network and PID namespace visible to the outer supervisor/runtime monitor,
do not use ``--disable-userns`` in containers with read-only ``/proc/sys``, bind
common runtime/source/tool paths automatically, and provide a read-only ``/proc``
view without requiring bubblewrap to mount a fresh procfs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import shutil
import shlex
import signal
import sys
from typing import Any

from dressage.sandbox.local.bwrap.slot import SlotRuntime


BWRAP_MODES = {"bwrap", "bubblewrap"}
DIRECT_MODES = {"direct"}
SUPPORTED_MODES = BWRAP_MODES | DIRECT_MODES

logger = logging.getLogger(__name__)

_SLOT_MARKER_KEYS = (
    "DRESSAGE_BLACKBOX_SLOT_ID",
    "DRESSAGE_BLACKBOX_SLOT_DIR",
    "DRESSAGE_BLACKBOX_SLOT_GENERATION",
    "DRESSAGE_BLACKBOX_SLOT_PORT",
)

_BLACKBOX_BACKEND_BINARY_ENV_KEYS = (
    "OPENCODE_BIN",
    "OPENCLAW_BIN",
    "CLAUDE_CODE_BIN",
    "CODEX_BIN",
)

_BLACKBOX_BACKEND_BINARIES = (
    "opencode",
    "openclaw",
    "claude",
    "codex",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if parsed < min_value:
        return default
    return parsed


def _env_float(name: str, default: float, *, min_value: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if parsed < min_value:
        return default
    return parsed


def _path_is_under(path: Path, parents: tuple[Path, ...]) -> bool:
    try:
        return any(path.is_relative_to(parent) for parent in parents)
    except ValueError:
        return False


def _append_existing_mount_root(
    candidates: list[Path],
    path_text: str | None,
    *,
    prefer_bin_parent: bool = True,
) -> None:
    if not path_text:
        return
    path = Path(path_text)
    if not path.is_absolute():
        return
    for candidate in (path, path.resolve()):
        if not candidate.exists():
            continue
        if candidate.is_file():
            if prefer_bin_parent and candidate.parent.name == "bin":
                candidates.append(candidate.parent.parent)
            else:
                candidates.append(candidate.parent)
        else:
            candidates.append(candidate)


def _pythonpath_mounts() -> list[Path]:
    mounts: list[Path] = []
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        if path.is_absolute() and path.exists():
            mounts.append(path)
    return mounts


def _common_user_tool_roots() -> tuple[Path, ...]:
    """Common local tool dirs used by blackbox backend installations."""
    home = Path.home()
    return (
        home / ".opencode",
        home / ".openclaw",
        home / ".local",
    )


def _default_readonly_mounts(python_bin: str | None = None) -> tuple[Path, ...]:
    """Host paths needed for Python, source modules, and installed tools.

    bubblewrap starts from an empty root. Besides common system directories,
    include current Python, PYTHONPATH entries, resolved blackbox backend CLI
    launcher roots, and common user install roots when they exist. This makes
    the successful bwrap runtime shape the default rather than something every
    training script must re-export manually.
    """
    common_system_roots = (
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc"),
    )
    candidates = list(common_system_roots)

    _append_existing_mount_root(candidates, python_bin or sys.executable)

    candidates.extend(_pythonpath_mounts())
    candidates.extend(_common_user_tool_roots())

    for env_name in _BLACKBOX_BACKEND_BINARY_ENV_KEYS:
        _append_existing_mount_root(candidates, os.environ.get(env_name))

    for binary in _BLACKBOX_BACKEND_BINARIES:
        resolved = shutil.which(binary)
        if not resolved:
            continue
        path = Path(resolved)
        real_path = path.resolve()
        if _path_is_under(path, common_system_roots) and _path_is_under(
            real_path, common_system_roots
        ):
            continue
        _append_existing_mount_root(candidates, str(path))
        if real_path != path:
            _append_existing_mount_root(candidates, str(real_path))

    mounts: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        mounts.append(path)
    return tuple(mounts)


def _default_bwrap_extra_args(*, bind_host_proc: bool) -> tuple[str, ...]:
    args: list[str] = ["--tmpfs", "/run"]
    if bind_host_proc and Path("/proc").exists():
        args.extend(["--ro-bind", "/proc", "/proc"])
    return tuple(args)


def _normalize_mode(mode: str) -> str:
    normalized = (mode or "bwrap").strip().lower()
    if normalized in BWRAP_MODES:
        return "bwrap"
    if normalized in DIRECT_MODES:
        return "direct"
    raise ValueError(
        "unsupported DRESSAGE_BLACKBOX_RUNNER_MODE="
        f"{mode!r}; expected 'bwrap', 'bubblewrap', or 'direct'"
    )


def _normalize_sandbox_cwd(cwd: str | None) -> str:
    if not cwd:
        return "/workspace"
    text = str(cwd).strip()
    if not text:
        return "/workspace"
    if text.startswith("/"):
        return text
    return "/workspace/" + text


@dataclass(slots=True)
class LocalSandboxRunnerConfig:
    """Configuration used to build a slot server process command."""

    mode: str = "bwrap"
    bubblewrap_bin: str = "bwrap"
    python_bin: str = sys.executable
    server_module: str = "blackbox_server.main"
    server_command: list[str] | None = None
    rootfs: Path | None = None
    runtime_prefix: Path = Path("/opt/dressage-blackbox")
    readonly_mounts: tuple[Path, ...] = ()
    extra_bubblewrap_args: tuple[str, ...] = ()
    disable_proc: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)
    use_systemd_scope: bool = False
    systemd_run_bin: str = "systemd-run"
    slot_uid: int = field(default_factory=os.geteuid)
    slot_gid: int = field(default_factory=os.getegid)
    bubblewrap_unshare_net: bool = False
    bubblewrap_unshare_pid: bool = False
    bubblewrap_disable_userns: bool = False
    stop_timeout_sec: float = 10.0
    cleanup_residual_processes: bool = True
    residual_cleanup_timeout_sec: float = 5.0
    residual_proc_root: Path = Path("/proc")

    @classmethod
    def from_env(cls) -> "LocalSandboxRunnerConfig":
        server_command = None
        command_text = os.environ.get("DRESSAGE_BLACKBOX_SERVER_COMMAND")
        if command_text:
            server_command = shlex.split(command_text)

        rootfs = os.environ.get("DRESSAGE_BLACKBOX_ROOTFS")
        python_bin = os.environ.get("DRESSAGE_BLACKBOX_PYTHON_BIN", sys.executable)
        mounts_text = os.environ.get("DRESSAGE_BLACKBOX_READONLY_MOUNTS")
        if mounts_text is None:
            readonly_mounts = _default_readonly_mounts(python_bin)
        else:
            readonly_mounts = tuple(Path(item) for item in mounts_text.split(":") if item)

        disable_proc = _env_bool("DRESSAGE_BLACKBOX_DISABLE_PROC", True)
        extra_bwrap_text = os.environ.get("DRESSAGE_BLACKBOX_BWRAP_EXTRA_ARGS")
        if extra_bwrap_text is None:
            extra_bubblewrap_args = _default_bwrap_extra_args(bind_host_proc=disable_proc)
        else:
            extra_bubblewrap_args = tuple(shlex.split(extra_bwrap_text))

        return cls(
            mode=_normalize_mode(os.environ.get("DRESSAGE_BLACKBOX_RUNNER_MODE", "bwrap")),
            bubblewrap_bin=os.environ.get(
                "DRESSAGE_BLACKBOX_BWRAP_BIN",
                os.environ.get("DRESSAGE_BLACKBOX_BUBBLEWRAP_BIN", "bwrap"),
            ),
            python_bin=python_bin,
            server_module=os.environ.get(
                "DRESSAGE_BLACKBOX_SERVER_MODULE", "blackbox_server.main"
            ),
            server_command=server_command,
            rootfs=Path(rootfs) if rootfs else None,
            runtime_prefix=Path(
                os.environ.get("DRESSAGE_BLACKBOX_RUNTIME_PREFIX", "/opt/dressage-blackbox")
            ),
            readonly_mounts=readonly_mounts,
            extra_bubblewrap_args=extra_bubblewrap_args,
            disable_proc=disable_proc,
            use_systemd_scope=_env_bool("DRESSAGE_BLACKBOX_USE_SYSTEMD_SCOPE", False),
            systemd_run_bin=os.environ.get(
                "DRESSAGE_BLACKBOX_SYSTEMD_RUN_BIN", "systemd-run"
            ),
            slot_uid=_env_int("DRESSAGE_BLACKBOX_SLOT_UID", os.geteuid()),
            slot_gid=_env_int("DRESSAGE_BLACKBOX_SLOT_GID", os.getegid()),
            bubblewrap_unshare_net=_env_bool(
                "DRESSAGE_BLACKBOX_BWRAP_UNSHARE_NET", False
            ),
            bubblewrap_unshare_pid=_env_bool(
                "DRESSAGE_BLACKBOX_BWRAP_UNSHARE_PID", False
            ),
            bubblewrap_disable_userns=_env_bool(
                "DRESSAGE_BLACKBOX_BWRAP_DISABLE_USERNS", False
            ),
            stop_timeout_sec=_env_float(
                "DRESSAGE_BLACKBOX_STOP_TIMEOUT_SEC", 10.0, min_value=0.1
            ),
            cleanup_residual_processes=_env_bool(
                "DRESSAGE_BLACKBOX_CLEANUP_RESIDUAL_PROCESSES", True
            ),
            residual_cleanup_timeout_sec=_env_float(
                "DRESSAGE_BLACKBOX_RESIDUAL_CLEANUP_TIMEOUT_SEC",
                5.0,
                min_value=0.0,
            ),
        )


class LocalSandboxRunner:
    """Start and stop blackbox server processes for supervisor slots."""

    def __init__(self, config: LocalSandboxRunnerConfig | None = None) -> None:
        self.config = config or LocalSandboxRunnerConfig.from_env()

    async def start(self, slot: SlotRuntime) -> asyncio.subprocess.Process:
        slot.config.ensure_dirs()
        cmd = self.build_command(slot)
        env = self.build_env(slot)
        stdout_path = slot.config.log_dir / f"server-{slot.generation}.out"
        stderr_path = slot.config.log_dir / f"server-{slot.generation}.err"
        stdout_file = stdout_path.open("ab")
        stderr_file = stderr_path.open("ab")
        try:
            stderr_file.write(self._startup_log(slot, cmd, env).encode())
            stderr_file.flush()
            return await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        finally:
            stdout_file.close()
            stderr_file.close()

    async def stop(self, slot: SlotRuntime, *, timeout_sec: float | None = None) -> None:
        proc = slot.process
        timeout = self.config.stop_timeout_sec if timeout_sec is None else timeout_sec
        if proc is not None and getattr(proc, "returncode", None) is None:
            await self._terminate_process_group(proc, timeout=timeout)

        # A top-level bwrap/server process can exit while leaving a child that
        # re-parented itself or created its own session.  Residual cleanup is
        # intentionally independent from the top-level process state.
        if self.config.cleanup_residual_processes:
            await self._cleanup_residual_processes(slot)

    async def _terminate_process_group(
        self,
        proc: asyncio.subprocess.Process,
        *,
        timeout: float,
    ) -> None:
        pid = getattr(proc, "pid", None)
        sent_group_signal = False
        if pid is not None:
            sent_group_signal = self._signal_process_group(pid, signal.SIGTERM)

        if not sent_group_signal:
            try:
                proc.terminate()
            except ProcessLookupError:
                return

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            return
        except (ProcessLookupError, asyncio.TimeoutError):
            pass

        if pid is not None:
            sent_group_signal = self._signal_process_group(pid, signal.SIGKILL)
        else:
            sent_group_signal = False
        if not sent_group_signal:
            try:
                proc.kill()
            except ProcessLookupError:
                return
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except (ProcessLookupError, asyncio.TimeoutError):
            pass

    def _signal_process_group(self, pid: int, sig: signal.Signals) -> bool:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return False
        except OSError as exc:
            logger.debug("failed to resolve process group for pid=%s: %s", pid, exc)
            return False

        if pgid in {0, os.getpgrp()}:
            logger.warning(
                "refusing to signal unsafe process group pgid=%s for pid=%s", pgid, pid
            )
            return False
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            logger.warning(
                "failed to signal blackbox process group pgid=%s pid=%s sig=%s: %s",
                pgid,
                pid,
                sig,
                exc,
            )
            return False
        return True

    async def _cleanup_residual_processes(self, slot: SlotRuntime) -> None:
        timeout = self.config.residual_cleanup_timeout_sec
        pids = self._find_residual_pids(slot)
        if not pids:
            return
        self._signal_pids(pids, signal.SIGTERM)
        if timeout > 0:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while True:
                await asyncio.sleep(min(0.1, max(0.0, deadline - loop.time())))
                pids = self._find_residual_pids(slot)
                if not pids or loop.time() >= deadline:
                    break
        pids = self._find_residual_pids(slot)
        if pids:
            self._signal_pids(pids, signal.SIGKILL)

    def _find_residual_pids(self, slot: SlotRuntime) -> list[int]:
        expected = self._slot_cleanup_markers(slot)
        if not expected.get("DRESSAGE_BLACKBOX_SLOT_TOKEN"):
            return []
        proc_root = self.config.residual_proc_root
        try:
            entries = list(proc_root.iterdir())
        except OSError:
            return []
        pids: list[int] = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                pid = int(entry.name)
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            env = self._read_proc_environ(entry / "environ")
            if not env:
                continue
            if self._matches_slot_markers(env, expected):
                pids.append(pid)
        return sorted(set(pids))

    def _slot_cleanup_markers(self, slot: SlotRuntime) -> dict[str, str]:
        markers = {
            "DRESSAGE_BLACKBOX_SLOT_ID": str(slot.config.slot_id),
            "DRESSAGE_BLACKBOX_SLOT_DIR": str(slot.config.slot_dir),
            "DRESSAGE_BLACKBOX_SLOT_GENERATION": str(slot.generation),
            "DRESSAGE_BLACKBOX_SLOT_PORT": str(slot.config.port),
        }
        if slot.cleanup_token:
            markers["DRESSAGE_BLACKBOX_SLOT_TOKEN"] = slot.cleanup_token
        if slot.supervisor_run_id:
            markers["DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID"] = slot.supervisor_run_id
        return markers

    def _read_proc_environ(self, path: Path) -> dict[str, str]:
        try:
            data = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            return {}
        env: dict[str, str] = {}
        for item in data.split(b"\0"):
            if not item or b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            env[key.decode(errors="replace")] = value.decode(errors="replace")
        return env

    def _matches_slot_markers(
        self,
        env: dict[str, str],
        expected: dict[str, str],
    ) -> bool:
        if env.get("DRESSAGE_BLACKBOX_SLOT_TOKEN") != expected.get(
            "DRESSAGE_BLACKBOX_SLOT_TOKEN"
        ):
            return False
        if expected.get("DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID") and env.get(
            "DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID"
        ) != expected.get("DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID"):
            return False
        return all(env.get(key) == expected.get(key) for key in _SLOT_MARKER_KEYS)

    def _signal_pids(self, pids: list[int], sig: signal.Signals) -> None:
        for pid in pids:
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                continue

    def build_command(self, slot: SlotRuntime) -> list[str]:
        server_cmd = self._server_command(slot)
        mode = _normalize_mode(self.config.mode)
        if mode == "direct":
            return server_cmd
        return self._build_bubblewrap_command(slot, server_cmd)

    def build_tool_command(
        self,
        slot: SlotRuntime,
        command: str | list[str],
        *,
        cwd: str | None = None,
    ) -> list[str]:
        """Build a one-shot command that executes inside the slot sandbox.

        The command reuses the same bubblewrap mount shape as the blackbox
        server.  Relative ``cwd`` values are resolved under ``/workspace``.
        """

        command_text = command if isinstance(command, str) else shlex.join(command)
        sandbox_cwd = _normalize_sandbox_cwd(cwd)
        shell_cmd = f"cd {shlex.quote(sandbox_cwd)} && {command_text}"
        tool_cmd = ["/bin/sh", "-lc", shell_cmd]
        mode = _normalize_mode(self.config.mode)
        if mode == "direct":
            return tool_cmd
        return self._build_bubblewrap_command(slot, tool_cmd)

    def build_tool_env(
        self,
        slot: SlotRuntime,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        env = self.build_env(slot)
        if extra_env:
            env.update({str(key): str(value) for key, value in extra_env.items()})
        return env

    def _build_bubblewrap_command(
        self,
        slot: SlotRuntime,
        server_cmd: list[str],
    ) -> list[str]:
        bwrap_cmd = [
            self.config.bubblewrap_bin,
            "--unshare-user",
            "--uid",
            str(self.config.slot_uid),
            "--gid",
            str(self.config.slot_gid),
            "--unshare-ipc",
            "--unshare-uts",
            "--hostname",
            f"dressage-bb-{slot.config.slot_id:04d}",
            "--new-session",
            "--die-with-parent",
        ]
        if self.config.bubblewrap_unshare_pid:
            bwrap_cmd.append("--unshare-pid")
        if self.config.bubblewrap_disable_userns:
            bwrap_cmd.append("--disable-userns")
        if self.config.bubblewrap_unshare_net:
            bwrap_cmd.append("--unshare-net")

        if self.config.rootfs is not None:
            bwrap_cmd.extend(["--ro-bind", str(self.config.rootfs), "/"])
        else:
            for path in self.config.readonly_mounts:
                self._append_bubblewrap_readonly_mount(bwrap_cmd, path)

        bwrap_cmd.extend(["--dev", "/dev"])
        if not self.config.disable_proc:
            bwrap_cmd.extend(["--proc", "/proc"])

        bwrap_cmd.extend(
            [
                "--bind",
                str(slot.config.home_dir),
                "/home/blackbox",
                "--bind",
                str(slot.config.work_dir),
                "/workspace",
                "--bind",
                str(slot.config.runtime_dir),
                "/workspace_sandbox/blackbox_server_runtime",
                "--bind",
                str(slot.config.tmp_dir),
                "/tmp",
                "--chdir",
                "/workspace",
            ]
        )
        bwrap_cmd.extend(self.config.extra_bubblewrap_args)
        bwrap_cmd.extend(["--", *server_cmd])

        return self._maybe_wrap_systemd_scope(slot, bwrap_cmd)

    def _append_bubblewrap_readonly_mount(
        self,
        bwrap_cmd: list[str],
        path: Path,
    ) -> None:
        """Append a readonly mount while preserving merged-/usr symlinks."""
        if path.is_symlink():
            target = os.readlink(path)
            if target.startswith("/"):
                target = target[1:]
            bwrap_cmd.extend(["--symlink", target, str(path)])
            return
        bwrap_cmd.extend(["--ro-bind", str(path), str(path)])

    def _maybe_wrap_systemd_scope(
        self,
        slot: SlotRuntime,
        sandbox_cmd: list[str],
    ) -> list[str]:
        if not self.config.use_systemd_scope:
            return sandbox_cmd
        return [
            self.config.systemd_run_bin,
            "--scope",
            "-p",
            f"MemoryHigh={slot.config.memory_high_bytes}",
            "-p",
            f"MemoryMax={slot.config.memory_max_bytes}",
            "-p",
            f"TasksMax={slot.config.pids_max}",
            "-p",
            "CPUWeight=20",
            *sandbox_cmd,
        ]

    def build_env(self, slot: SlotRuntime) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.extra_env)
        direct_mode = _normalize_mode(self.config.mode) == "direct"
        home = str(slot.config.home_dir) if direct_mode else "/home/blackbox"
        tmp = str(slot.config.tmp_dir) if direct_mode else "/tmp"
        runtime_root = (
            str(slot.config.runtime_dir)
            if direct_mode
            else "/workspace_sandbox/blackbox_server_runtime"
        )
        if not slot.cleanup_token:
            slot.rotate_cleanup_token(supervisor_run_id=slot.supervisor_run_id)
        env.update(
            {
                "HOME": home,
                "XDG_CONFIG_HOME": f"{home}/.config",
                "XDG_CACHE_HOME": f"{home}/.cache",
                "TMPDIR": tmp,
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "void",
                "BBS_HOST": slot.config.bind_host,
                "BBS_PORT": str(slot.config.port),
                "BBS_RUNTIME_ROOT": runtime_root,
                "DRESSAGE_BLACKBOX_SLOT_ID": str(slot.config.slot_id),
                "DRESSAGE_BLACKBOX_SLOT_DIR": str(slot.config.slot_dir),
                "DRESSAGE_BLACKBOX_SLOT_GENERATION": str(slot.generation),
                "DRESSAGE_BLACKBOX_SLOT_PORT": str(slot.config.port),
                "DRESSAGE_BLACKBOX_SLOT_TOKEN": slot.cleanup_token or "",
                "DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID": slot.supervisor_run_id or "",
                "DRESSAGE_BLACKBOX_RUNTIME_ROOT": runtime_root,
            }
        )
        return env

    def _server_command(self, slot: SlotRuntime) -> list[str]:
        command = self.config.server_command
        if command is None:
            command = [
                self.config.python_bin,
                "-m",
                self.config.server_module,
            ]
        values: dict[str, Any] = {
            "bind_host": slot.config.bind_host,
            "advertise_host": slot.config.advertise_host,
            "port": slot.config.port,
            "slot_id": slot.config.slot_id,
            "blackbox_type": slot.config.blackbox_type,
            "home_dir": slot.config.home_dir,
            "work_dir": slot.config.work_dir,
            "runtime_dir": slot.config.runtime_dir,
            "tmp_dir": slot.config.tmp_dir,
            "runtime_prefix": self.config.runtime_prefix,
        }
        return [str(part).format(**values) for part in command]

    def _startup_log(
        self,
        slot: SlotRuntime,
        cmd: list[str],
        env: dict[str, str],
    ) -> str:
        keys = (
            "BBS_HOST",
            "BBS_PORT",
            "BBS_RUNTIME_ROOT",
            "HOME",
            "TMPDIR",
            "PYTHONPATH",
            "PATH",
            "OPENCODE_BIN",
            "OPENCLAW_BIN",
            "DRESSAGE_BLACKBOX_SLOT_ID",
            "DRESSAGE_BLACKBOX_SLOT_DIR",
            "DRESSAGE_BLACKBOX_SLOT_GENERATION",
            "DRESSAGE_BLACKBOX_SLOT_PORT",
            "DRESSAGE_BLACKBOX_SLOT_TOKEN",
            "DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID",
        )
        env_lines = "\n".join(
            f"# env {key}={env.get(key, '')}" for key in keys if key in env
        )
        mounts = ":".join(str(path) for path in self.config.readonly_mounts)
        return (
            f"# dressage blackbox slot {slot.config.slot_id} generation "
            f"{slot.generation}\n"
            f"# command {shlex.join(cmd)}\n"
            f"# readonly_mounts {mounts}\n"
            f"{env_lines}\n"
        )
