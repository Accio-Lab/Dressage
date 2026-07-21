"""Minimal local bubblewrap environment for Harbor 0.18.

The agent and verifier receive separate filesystem roots.  Agent sandboxes use
an isolated network namespace with a single loopback relay to the Dressage
Gateway; verifier sandboxes use Harbor's separate-environment mode and the
parent container network.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any
import uuid

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy, TaskOS

from dressage.integrations.harbor.config import (
    BwrapEnvironmentConfig,
    HarborIntegrationConfig,
    load_config,
)


_CONFIG_ENV = "DRESSAGE_HARBOR_INTEGRATION_CONFIG"
_GATEWAY_SOCKET_DIR = PurePosixPath("/run/dressage-harbor")
_GATEWAY_SOCKET = _GATEWAY_SOCKET_DIR / "gateway.sock"
_HOME = PurePosixPath("/home/dressage")
_LOGS = PurePosixPath("/logs")
_PROTECTED_TRANSFER_ROOTS = (
    PurePosixPath("/proc"),
    PurePosixPath("/dev"),
    _GATEWAY_SOCKET_DIR,
    PurePosixPath("/run/dressage-relay.py"),
)
_STANDARD_ROOTFS_DIRS = (
    PurePosixPath("/app"),
    PurePosixPath("/workspace"),
    _LOGS,
    PurePosixPath("/tests"),
    PurePosixPath("/solution"),
    PurePosixPath("/harbor/skills"),
    PurePosixPath("/tmp"),
    PurePosixPath("/run"),
    PurePosixPath("/proc"),
    PurePosixPath("/dev"),
    PurePosixPath("/root"),
    _HOME,
)
_SYSTEM_RUNTIME_ROOTS = tuple(
    Path(path) for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")
)

_RELAY_SOURCE = r"""#!/usr/bin/env python3
import asyncio
import fcntl
from pathlib import Path
import socket
import struct
import sys


def enable_loopback():
    interface = struct.pack("16sH", b"lo", 0)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
        flags = struct.unpack("16sH", fcntl.ioctl(control, 0x8913, interface))[1]
        fcntl.ioctl(
            control,
            0x8914,
            struct.pack("16sH", b"lo", flags | 0x1 | 0x40),
        )


async def copy(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def handle(client_reader, client_writer, socket_path):
    try:
        gateway_reader, gateway_writer = await asyncio.open_unix_connection(socket_path)
    except Exception:
        client_writer.close()
        await client_writer.wait_closed()
        return
    tasks = {
        asyncio.create_task(copy(client_reader, gateway_writer)),
        asyncio.create_task(copy(gateway_reader, client_writer)),
    }
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    socket_path, port, ready_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    enable_loopback()
    server = await asyncio.start_server(
        lambda reader, writer: handle(reader, writer, socket_path),
        "127.0.0.1",
        port,
    )
    ready_path.touch()
    async with server:
        await server.serve_forever()


asyncio.run(main())
"""


def _base_bwrap_args(bwrap: str) -> list[str]:
    return [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--uid",
        "0",
        "--gid",
        "0",
        "--unshare-uts",
        "--unshare-ipc",
    ]


def _readonly_runtime_mounts(
    executables: tuple[str, ...],
    *,
    system_roots: tuple[Path, ...] = _SYSTEM_RUNTIME_ROOTS,
    home: Path | None = None,
) -> tuple[Path, ...]:
    """Return the minimal host roots needed by commands inside bwrap."""

    candidates = [*system_roots, (home or Path.home()) / ".local"]
    for executable in executables:
        path = Path(executable)
        for resolved in (path, path.resolve()):
            if not resolved.is_absolute() or not resolved.exists():
                continue
            candidates.append(
                resolved.parent.parent
                if resolved.parent.name == "bin"
                else resolved.parent
            )

    mounts: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_absolute() or (
            not candidate.exists() and not candidate.is_symlink()
        ):
            continue
        key = str(candidate)
        if key in seen:
            continue
        if not candidate.is_symlink() and any(
            not mounted.is_symlink() and candidate.is_relative_to(mounted)
            for mounted in mounts
        ):
            continue
        seen.add(key)
        mounts.append(candidate)
    return tuple(mounts)


def _append_readonly_mounts(args: list[str], mounts: tuple[Path, ...]) -> None:
    """Bind runtime directories after rootfs symlinks have been materialized."""

    for path in mounts:
        if not path.is_symlink():
            args.extend(["--ro-bind", str(path), str(path)])


def _rootfs_path(rootfs: Path, environment_path: PurePosixPath) -> Path:
    return rootfs.joinpath(*environment_path.parts[1:])


def _initialize_rootfs(
    rootfs: Path,
    runtime_mounts: tuple[Path, ...],
    *,
    workdir: str | PurePosixPath,
) -> None:
    """Create the persistent writable tree and stable runtime mount targets."""

    rootfs.mkdir(parents=True, exist_ok=True, mode=0o700)
    directories = {*_STANDARD_ROOTFS_DIRS, _environment_path(workdir)}
    for path in sorted(directories, key=lambda item: len(item.parts)):
        _rootfs_path(rootfs, path).mkdir(parents=True, exist_ok=True)

    for source in runtime_mounts:
        target = _rootfs_path(rootfs, PurePosixPath(str(source)))
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            link_target = os.readlink(source)
            if target.is_symlink():
                if os.readlink(target) != link_target:
                    raise RuntimeError(f"rootfs symlink target mismatch: {target}")
                continue
            if target.exists():
                raise RuntimeError(f"rootfs runtime target is not a symlink: {target}")
            target.symlink_to(link_target, target_is_directory=True)
        else:
            target.mkdir(parents=True, exist_ok=True)


def _is_within(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or path.is_relative_to(root)


def _sandbox_process_env() -> dict[str, str]:
    return {
        "HOME": str(_HOME),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "SHELL": "/bin/bash",
        "TMPDIR": "/tmp",
        "USER": "dressage",
    }


def _is_separate_verifier_session(
    session_id: str,
    mounts: list[dict[str, Any]],
) -> bool:
    """Recognize Harbor 0.18's separate-verifier constructor contract."""

    targets = {
        str(mount.get("target")) for mount in mounts if mount.get("type") == "bind"
    }
    separate_mount_shape = "/logs/verifier" in targets and not targets.intersection(
        {"/logs/agent", "/logs/artifacts"}
    )
    if separate_mount_shape:
        return True
    if "__verifier__" in session_id and not session_id.endswith("__env"):
        raise ValueError(
            "separate verifier does not have Harbor's verifier mount shape"
        )
    return False


class DressageEnvironment(BaseEnvironment):
    """A strict bwrap command environment with no image/build path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = _load_bwrap_integration_config()
        self._bwrap = _required_binary("bwrap")
        self._python = _required_binary("python3")
        self._claude = _required_binary("claude")
        self._curl = _required_binary("curl")
        self._runtime_mounts = _readonly_runtime_mounts(
            (self._python, self._claude, self._curl)
        )
        self._runtime_root = config.environment.runtime_root
        self._gateway_port = config.gateway.listen_port
        session_key = (
            f"{kwargs.get('session_id', 'harbor')}\0"
            f"{getattr(kwargs.get('trial_paths'), 'trial_dir', '')}"
        )
        session_digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:20]
        self._session_root = self._runtime_root / "environments" / session_digest
        self._rootfs = self._session_root / "rootfs"
        self._started = False
        self._processes: set[asyncio.subprocess.Process] = set()
        super().__init__(*args, **kwargs)
        self._harbor_mounts = self._build_harbor_mounts()
        self._separate_verifier = _is_separate_verifier_session(
            self.session_id, self._mounts
        )

    @staticmethod
    def type() -> str:
        return "dressage-bwrap"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
            network_allowlist_ipv4_addresses=True,
            mounted=True,
        )

    @classmethod
    def preflight(cls) -> None:
        if sys.platform != "linux":
            raise RuntimeError("Dressage bwrap requires Linux")
        _load_bwrap_integration_config()
        binaries = {
            name: _required_binary(name)
            for name in ("bwrap", "curl", "claude", "python3")
        }
        if not Path("/bin/bash").is_file():
            raise RuntimeError("required executable is unavailable: /bin/bash")
        runtime_mounts = _readonly_runtime_mounts(
            (binaries["python3"], binaries["claude"], binaries["curl"])
        )
        with tempfile.TemporaryDirectory(prefix="dressage-harbor-preflight-") as root:
            rootfs = Path(root) / "rootfs"
            _initialize_rootfs(rootfs, runtime_mounts, workdir="/app")
            args = _base_bwrap_args(binaries["bwrap"])
            args.extend(
                [
                    "--bind",
                    str(rootfs),
                    "/",
                ]
            )
            _append_readonly_mounts(args, runtime_mounts)
            args.extend(
                [
                    "--ro-bind",
                    "/proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--chdir",
                    "/app",
                    "--",
                    "/bin/bash",
                    "-c",
                    "python3 -c 'import sys' && "
                    "claude --version >/dev/null && curl --version >/dev/null",
                ]
            )
            completed = subprocess.run(
                args,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
                env=_sandbox_process_env(),
            )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"bubblewrap runtime preflight failed: {detail}")

    def _validate_definition(self) -> None:
        if self.task_env_config.os != TaskOS.LINUX:
            raise ValueError("Dressage bwrap supports Linux tasks only")
        if self.task_env_config.docker_image:
            raise ValueError("Dressage bwrap does not accept task docker_image")
        for filename in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml"):
            if (self.environment_dir / filename).exists():
                raise ValueError(f"Dressage bwrap does not accept {filename}")
        workdir = _environment_path(self.task_env_config.workdir or "/app")
        for runtime_root in self._runtime_mounts:
            target = PurePosixPath(str(runtime_root))
            if _is_within(workdir, target):
                raise ValueError(
                    f"Dressage bwrap workdir is under a read-only runtime: {workdir}"
                )
        if any(_is_within(workdir, root) for root in _PROTECTED_TRANSFER_ROOTS):
            raise ValueError(f"Dressage bwrap workdir is protected: {workdir}")

    def validate_network_policy_support(
        self, network_policy: NetworkPolicy | None = None
    ) -> None:
        policy = network_policy or self.network_policy
        super().validate_network_policy_support(policy)
        if policy.network_mode == NetworkMode.ALLOWLIST and policy.allowed_hosts != [
            "127.0.0.1"
        ]:
            raise ValueError("Dressage bwrap allowlist must contain only 127.0.0.1")

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Dressage bwrap does not build environments")
        if self._started:
            return
        self._runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._runtime_root.chmod(0o700)
        if self._session_root.exists():
            await asyncio.to_thread(shutil.rmtree, self._session_root)
        _initialize_rootfs(
            self._rootfs,
            self._runtime_mounts,
            workdir=self.task_env_config.workdir or "/app",
        )
        for host_path, read_only in self._harbor_mounts.values():
            if read_only:
                if not host_path.is_dir():
                    raise NotADirectoryError(host_path)
            else:
                host_path.mkdir(parents=True, exist_ok=True)
        for target in self._harbor_mounts:
            _rootfs_path(self._rootfs, target).mkdir(parents=True, exist_ok=True)
        _rootfs_path(self._rootfs, _GATEWAY_SOCKET_DIR).mkdir(
            parents=True, exist_ok=True
        )
        _rootfs_path(self._rootfs, PurePosixPath("/run/dressage-relay.py")).touch()
        if self._separate_verifier:
            if not self.environment_dir.is_dir():
                raise NotADirectoryError(self.environment_dir)
            await asyncio.to_thread(
                shutil.copytree,
                self.environment_dir,
                _rootfs_path(self._rootfs, PurePosixPath("/tests")),
                dirs_exist_ok=True,
                symlinks=True,
            )
        relay_path = self._session_root / "relay.py"
        relay_path.write_text(_RELAY_SOURCE, encoding="utf-8")
        relay_path.chmod(0o500)
        self._started = True

    async def stop(self, delete: bool) -> None:
        del delete
        processes = tuple(self._processes)
        if processes:
            await asyncio.gather(
                *(_terminate_process(process) for process in processes),
                return_exceptions=True,
            )
        self._processes.clear()
        if self._session_root.exists():
            await asyncio.to_thread(shutil.rmtree, self._session_root)
        self._started = False

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self._host_path(target_path, writable=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise NotADirectoryError(source)
        target = self._host_path(target_dir, writable=True)
        target.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            shutil.copytree,
            source,
            target,
            dirs_exist_ok=True,
            symlinks=True,
        )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self._host_path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = self._host_path(source_dir)
        if not source.is_dir():
            raise NotADirectoryError(source_dir)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            shutil.copytree,
            source,
            target,
            dirs_exist_ok=True,
            symlinks=True,
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if not self._started:
            raise RuntimeError("Dressage bwrap environment is not started")
        effective_user = self._resolve_user(user)
        if effective_user not in (None, 0, "0", "root"):
            raise ValueError("Dressage bwrap supports only the root sandbox user")

        command_line = self._bwrap_command(
            self._with_relay(command),
            cwd=cwd or self.task_env_config.workdir or "/app",
        )
        process_env = _sandbox_process_env()
        process_env.update(self._merge_env(env) or {})
        process = await asyncio.create_subprocess_exec(
            *command_line,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_env,
            start_new_session=True,
        )
        self._processes.add(process)
        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                await _terminate_process(process)
                raise
            except asyncio.CancelledError:
                await _terminate_process(process)
                raise
        finally:
            self._processes.discard(process)

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        callback = self._output_callback()
        if callback is not None:
            if stdout_text:
                await callback(stdout_text, "stdout")
            if stderr_text:
                await callback(stderr_text, "stderr")
        return ExecResult(
            stdout=stdout_text,
            stderr=stderr_text,
            return_code=int(process.returncode or 0),
        )

    def _build_harbor_mounts(self) -> dict[PurePosixPath, tuple[Path, bool]]:
        mounts: dict[PurePosixPath, tuple[Path, bool]] = {}
        for mount in self._mounts:
            if mount.get("type") != "bind":
                raise ValueError("Dressage bwrap supports bind mounts only")
            target = _environment_path(mount["target"])
            if not _is_within(target, _LOGS):
                raise ValueError("Dressage bwrap only accepts Harbor /logs mounts")
            mounts[target] = (
                Path(mount["source"]).expanduser().resolve(),
                bool(mount.get("read_only", False)),
            )
        return mounts

    def _host_path(self, environment_path: str, *, writable: bool = False) -> Path:
        path = _environment_path(environment_path)
        if any(_is_within(path, root) for root in _PROTECTED_TRANSFER_ROOTS):
            raise ValueError(f"protected environment path: {environment_path}")

        for target, (source, read_only) in sorted(
            self._harbor_mounts.items(),
            key=lambda item: len(item[0].parts),
            reverse=True,
        ):
            if path == target or path.is_relative_to(target):
                if writable and read_only:
                    raise PermissionError(environment_path)
                relative = path.relative_to(target)
                candidate = source.joinpath(*relative.parts)
                resolved_source = source.resolve()
                resolved_candidate = candidate.resolve(strict=False)
                if not resolved_candidate.is_relative_to(resolved_source):
                    raise ValueError(
                        f"environment path escapes its mount: {environment_path}"
                    )
                return candidate

        for runtime_root in sorted(
            self._runtime_mounts, key=lambda item: len(item.parts), reverse=True
        ):
            target = PurePosixPath(str(runtime_root))
            if not _is_within(path, target):
                continue
            if writable:
                raise PermissionError(environment_path)
            relative = path.relative_to(target)
            candidate = runtime_root.joinpath(*relative.parts)
            resolved_source = runtime_root.resolve()
            resolved_candidate = candidate.resolve(strict=False)
            if not resolved_candidate.is_relative_to(resolved_source):
                raise ValueError(
                    f"environment path escapes its mount: {environment_path}"
                )
            return candidate

        candidate = _rootfs_path(self._rootfs, path)
        resolved_rootfs = self._rootfs.resolve()
        resolved_candidate = candidate.resolve(strict=False)
        if not resolved_candidate.is_relative_to(resolved_rootfs):
            raise ValueError(f"environment path escapes rootfs: {environment_path}")
        return candidate

    def _bwrap_command(self, command: str, *, cwd: str) -> list[str]:
        args = _base_bwrap_args(self._bwrap)
        if self.network_policy.network_mode != NetworkMode.PUBLIC:
            args.append("--unshare-net")
        args.extend(
            [
                "--bind",
                str(self._rootfs),
                "/",
            ]
        )
        _append_readonly_mounts(args, self._runtime_mounts)
        args.extend(
            [
                "--ro-bind",
                "/proc",
                "/proc",
                "--dev",
                "/dev",
            ]
        )
        for target, (source, read_only) in sorted(
            self._harbor_mounts.items(), key=lambda item: len(item[0].parts)
        ):
            args.extend(
                ["--ro-bind" if read_only else "--bind", str(source), str(target)]
            )
        args.extend(
            [
                "--ro-bind",
                str(self._session_root / "relay.py"),
                "/run/dressage-relay.py",
            ]
        )
        if self.network_policy.network_mode == NetworkMode.ALLOWLIST:
            socket_dir = self._runtime_root / "gateway"
            if not (socket_dir / "gateway.sock").exists():
                raise RuntimeError("Dressage Gateway Unix socket is not available")
            args.extend(["--ro-bind", str(socket_dir), str(_GATEWAY_SOCKET_DIR)])
        args.extend(
            [
                "--chdir",
                str(_environment_path(cwd)),
                "--",
                "/bin/bash",
                "-c",
                command,
            ]
        )
        return args

    def _with_relay(self, command: str) -> str:
        if self.network_policy.network_mode != NetworkMode.ALLOWLIST:
            return command
        ready = f"/tmp/relay-{uuid.uuid4().hex}.ready"
        return (
            f"rm -f {ready}; "
            f"{self._python} /run/dressage-relay.py {_GATEWAY_SOCKET} "
            f"{self._gateway_port} {ready} & "
            "relay_pid=$!; "
            "trap 'kill $relay_pid 2>/dev/null || true; wait $relay_pid 2>/dev/null || true' EXIT; "
            f"for _ in {{1..200}}; do [ -e {ready} ] && break; "
            "kill -0 $relay_pid 2>/dev/null || exit 70; sleep 0.01; done; "
            f"[ -e {ready} ] || exit 70; {command}"
        )


def _load_bwrap_integration_config() -> HarborIntegrationConfig:
    config_path = os.environ.get(_CONFIG_ENV)
    if not config_path:
        raise RuntimeError(f"{_CONFIG_ENV} is required for Dressage bwrap")
    config = load_config(config_path)
    environment = config.environment
    if not isinstance(environment, BwrapEnvironmentConfig):
        raise RuntimeError("DressageEnvironment requires environment.mode='bwrap'")
    return config


def _required_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required executable is not on PATH: {name}")
    return path


def _environment_path(value: str | PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"environment path must be absolute without '..': {value}")
    return path


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()


__all__ = ["DressageEnvironment"]
