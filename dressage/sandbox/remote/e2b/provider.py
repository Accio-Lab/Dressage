"""E2B implementation of the Dressage sandbox provider interface."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable

from dressage.sandbox.types import (
    CommandResult,
    SandboxEndpoint,
    SandboxLease,
    SandboxSpec,
)

_SANDBOX_CMD_STDIO_METADATA_LIMIT = 4096


class E2BSandboxProvider:
    """Remote sandbox provider backed by E2B.

    The provider deliberately does not install blackbox runtime dependencies at
    create time.  ``sandbox_image`` or ``DRESSAGE_SANDBOX_DEFAULT_IMAGE`` must
    point at a pre-built template that already starts the blackbox server when
    blackbox mode is used.
    """

    name = "e2b"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        template: str | None = None,
        timeout_sec: float | None = None,
        blackbox_port: int | None = None,
        sandbox_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DRESSAGE_E2B_API_KEY")
        self.default_template = template
        self.timeout_sec = _env_float("DRESSAGE_E2B_TIMEOUT_SEC", timeout_sec or 3600.0)
        self.blackbox_port = _env_int(
            "DRESSAGE_E2B_BLACKBOX_PORT",
            31000 if blackbox_port is None else blackbox_port,
        )
        self._sandbox_factory = sandbox_factory
        self._sandboxes: dict[str, Any] = {}

    async def create(self, spec: SandboxSpec) -> SandboxLease:
        factory = self._sandbox_factory or _load_async_sandbox_create()
        extra_params = _sandbox_extra_params(spec.env_args)
        template = _template_for_spec(spec, default_template=self.default_template)
        sandbox_cmds = _normalize_sandbox_cmds(spec.env_args.get("sandbox_cmd"))
        envs = dict(spec.env)
        envs.update(_dict_extra_param(extra_params, "e2b_envs"))
        metadata = {
            "trajectory_id": spec.trajectory_id,
            **{str(k): str(v) for k, v in spec.metadata.items()},
            **{str(k): str(v) for k, v in _dict_extra_param(extra_params, "e2b_metadata").items()},
        }
        timeout = int(spec.timeout_sec or self.timeout_sec)
        sandbox = await _maybe_await(
            _call_create(
                factory,
                template=template,
                timeout=timeout,
                metadata=metadata,
                envs=envs,
                api_key=self.api_key,
            )
        )
        sandbox_id = _sandbox_id(sandbox)
        lease = SandboxLease(
            trajectory_id=spec.trajectory_id,
            provider=self.name,
            sandbox_id=sandbox_id,
            capabilities=(
                {"command", "file", "public_url"}
                if spec.services
                else {"command", "file"}
            ),
            metadata={
                "template": template,
                "timeout_sec": timeout,
            },
            raw=sandbox,
        )
        try:
            if sandbox_cmds:
                sandbox_cmd_results = await self._run_sandbox_cmds(
                    lease,
                    sandbox_cmds,
                    timeout=timeout,
                )
                lease.metadata["sandbox_cmd_results"] = sandbox_cmd_results
                lease.metadata["sandbox_cmd_result"] = sandbox_cmd_results[-1]
            for service in spec.services:
                try:
                    lease.endpoints[service.name] = await self.get_public_url(
                        lease,
                        port=service.port,
                        service_name=service.name,
                    )
                except Exception:
                    # Keep create usable for pure whitebox templates even when no
                    # port exposure is available in a mock or older SDK.
                    if service.name == "blackbox":
                        raise
        except BaseException:
            await _best_effort_kill(sandbox)
            raise
        self._sandboxes[spec.trajectory_id] = sandbox
        return lease

    async def terminate(self, lease: SandboxLease | str) -> dict[str, Any]:
        trajectory_id = lease if isinstance(lease, str) else lease.trajectory_id
        sandbox = None if isinstance(lease, str) else lease.raw
        sandbox = sandbox or self._sandboxes.pop(trajectory_id, None)
        if sandbox is None:
            return {"terminated": False, "trajectory_id": trajectory_id, "missing": True}
        self._sandboxes.pop(trajectory_id, None)
        kill = getattr(sandbox, "kill", None)
        if kill is None:
            return {"terminated": False, "trajectory_id": trajectory_id, "missing_kill": True}
        killed = await _maybe_await(kill())
        return {"terminated": bool(killed) if killed is not None else True, "trajectory_id": trajectory_id}

    async def get_public_url(
        self,
        lease: SandboxLease,
        *,
        port: int,
        service_name: str | None = None,
    ) -> SandboxEndpoint:
        del service_name
        sandbox = _require_sandbox(lease)
        host = await _call_get_host(sandbox, port)
        url = str(host)
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        return SandboxEndpoint(url=url.rstrip("/"), headers={})

    async def run_command(
        self,
        lease: SandboxLease,
        command: str | list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: str | bytes | None = None,
    ) -> CommandResult:
        sandbox = _require_sandbox(lease)
        cmd = _command_to_text(command)
        if stdin not in (None, b"", ""):
            raise NotImplementedError("E2BSandboxProvider.run_command does not support stdin yet")
        commands = getattr(sandbox, "commands", None)
        run = getattr(commands, "run", None) if commands is not None else None
        if run is None:
            raise RuntimeError("E2B sandbox object does not expose commands.run")
        result = await _maybe_await(_call_commands_run(run, cmd, cwd=cwd, env=env, timeout=timeout))
        return _command_result_from_e2b(command, result)

    async def read_file(
        self,
        lease: SandboxLease,
        path: str,
        *,
        encoding: str | None = "utf-8",
        max_bytes: int | None = None,
    ) -> str | bytes:
        sandbox = _require_sandbox(lease)
        files = getattr(sandbox, "files", None)
        read = getattr(files, "read", None) if files is not None else None
        if read is None:
            raise RuntimeError("E2B sandbox object does not expose files.read")
        if encoding is None:
            data = await _maybe_await(_call_file_read(read, path, format="bytes"))
            raw = bytes(data)
            return raw if max_bytes is None else raw[:max_bytes]
        text = await _maybe_await(_call_file_read(read, path, format="text"))
        if not isinstance(text, str):
            text = bytes(text).decode(encoding)
        return text if max_bytes is None else text[:max_bytes]

    async def write_file(
        self,
        lease: SandboxLease,
        path: str,
        content: str | bytes,
        *,
        encoding: str | None = "utf-8",
        append: bool = False,
    ) -> dict[str, Any]:
        sandbox = _require_sandbox(lease)
        files = getattr(sandbox, "files", None)
        write = getattr(files, "write", None) if files is not None else None
        if write is None:
            raise RuntimeError("E2B sandbox object does not expose files.write")
        payload: str | bytes = content
        if append:
            try:
                existing = await self.read_file(
                    lease,
                    path,
                    encoding=None if isinstance(content, bytes) else encoding,
                )
            except Exception:
                existing = b"" if isinstance(content, bytes) else ""
            payload = existing + content  # type: ignore[operator]
        result = await _maybe_await(write(path, payload))
        return {"path": path, "written": True, "raw": result}

    async def _run_sandbox_cmds(
        self,
        lease: SandboxLease,
        commands: tuple[str, ...],
        *,
        timeout: float,
    ) -> list[dict[str, Any]]:
        results = []
        for command in commands:
            result = await self.run_command(lease, command, timeout=timeout)
            summary = _sandbox_cmd_result_summary(result)
            results.append(summary)
            if result.returncode != 0 or result.timed_out:
                raise RuntimeError(
                    "sandbox_cmd failed: "
                    f"cmd={command!r} returncode={result.returncode} "
                    f"timed_out={result.timed_out}"
                )
        return results


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _sandbox_extra_params(env_args: dict[str, Any]) -> dict[str, Any]:
    extra_params = env_args.get("sandbox_extra_params")
    if extra_params is None:
        return {}
    if not isinstance(extra_params, dict):
        raise ValueError("sandbox_extra_params must be a dict for E2BSandboxProvider")
    allowed = {"e2b_envs", "e2b_metadata"}
    unknown = sorted(str(key) for key in extra_params if key not in allowed)
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"unsupported e2b sandbox_extra_params key(s): {keys}")
    return dict(extra_params)


def _dict_extra_param(extra_params: dict[str, Any], key: str) -> dict[str, Any]:
    value = extra_params.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"sandbox_extra_params.{key} must be a dict")
    return dict(value)


def _normalize_sandbox_cmds(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        command = value.strip()
        return (command,) if command else ()
    if isinstance(value, list):
        commands: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("sandbox_cmd must be a string or list of strings")
            command = item.strip()
            if command:
                commands.append(command)
        return tuple(commands)
    raise ValueError("sandbox_cmd must be a string or list of strings")


def _template_for_spec(spec: SandboxSpec, *, default_template: str | None) -> str:
    template = spec.env_args.get("sandbox_image") or default_template or os.environ.get(
        "DRESSAGE_SANDBOX_DEFAULT_IMAGE"
    )
    if template is not None and str(template).strip():
        return str(template).strip()
    raise ValueError(
        "E2BSandboxProvider requires sample metadata.sandbox_image or "
        "DRESSAGE_SANDBOX_DEFAULT_IMAGE; use a pre-built E2B template instead "
        "of runtime installation"
    )


def _load_async_sandbox_create() -> Callable[..., Any]:
    try:
        from e2b import AsyncSandbox  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "E2BSandboxProvider requires the e2b Python SDK; install e2b and set "
            "sample metadata.sandbox_image or DRESSAGE_SANDBOX_DEFAULT_IMAGE"
        ) from exc
    return AsyncSandbox.create


def _call_create(factory: Callable[..., Any], **kwargs: Any) -> Any:
    # The E2B SDK has had small keyword differences across versions.  Try the
    # current SDK shape first, then fall back to positional template creation.
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    try:
        return factory(**filtered)
    except TypeError:
        template = filtered.pop("template")
        try:
            return factory(template, **filtered)
        except TypeError:
            filtered.pop("envs", None)
            filtered.pop("metadata", None)
            return factory(template, **filtered)


def _sandbox_id(sandbox: Any) -> str | None:
    for attr in ("sandbox_id", "sandboxId", "id"):
        value = getattr(sandbox, attr, None)
        if value:
            return str(value)
    return None


def _require_sandbox(lease: SandboxLease) -> Any:
    if lease.raw is None:
        raise ValueError(f"sandbox lease {lease.trajectory_id!r} has no live SDK object")
    return lease.raw


async def _call_get_host(sandbox: Any, port: int) -> str:
    for name in ("get_host", "getHost"):
        method = getattr(sandbox, name, None)
        if method is None:
            continue
        return str(await _maybe_await(method(port)))
    raise RuntimeError("E2B sandbox object does not expose get_host/getHost")


def _command_to_text(command: str | list[str]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


async def _best_effort_kill(sandbox: Any) -> None:
    kill = getattr(sandbox, "kill", None)
    if kill is None:
        return
    try:
        await _maybe_await(kill())
    except Exception:
        return


def _sandbox_cmd_result_summary(result: CommandResult) -> dict[str, Any]:
    stdout, stdout_truncated = _truncated_metadata_text(result.stdout)
    stderr, stderr_truncated = _truncated_metadata_text(result.stderr)
    return {
        "cmd": result.cmd,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
    }


def _truncated_metadata_text(value: Any) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if len(text) <= _SANDBOX_CMD_STDIO_METADATA_LIMIT:
        return text, False
    return text[:_SANDBOX_CMD_STDIO_METADATA_LIMIT], True


def _call_commands_run(
    run: Callable[..., Any],
    cmd: str,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: float | None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if cwd is not None:
        kwargs["cwd"] = cwd
    if env:
        kwargs["envs"] = env
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return run(cmd, **kwargs)
    except TypeError:
        kwargs.pop("timeout", None)
        if timeout is not None:
            kwargs["request_timeout"] = timeout
        try:
            return run(cmd, **kwargs)
        except TypeError:
            return run(cmd)


def _call_file_read(read: Callable[..., Any], path: str, *, format: str) -> Any:
    try:
        return read(path, format=format)
    except TypeError:
        return read(path)


def _command_result_from_e2b(command: str | list[str], result: Any) -> CommandResult:
    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    if stdout is None and hasattr(result, "logs"):
        logs = getattr(result, "logs")
        stdout = getattr(logs, "stdout", None) or str(logs)
    returncode = None
    for attr in ("returncode", "return_code", "exit_code", "exitCode"):
        value = getattr(result, attr, None)
        if value is not None:
            try:
                returncode = int(value)
            except (TypeError, ValueError):
                returncode = None
            break
    return CommandResult(
        cmd=command,
        stdout="" if stdout is None else str(stdout),
        stderr="" if stderr is None else str(stderr),
        returncode=returncode,
        timed_out=bool(getattr(result, "timed_out", False)),
        metadata={"raw_type": type(result).__name__},
    )
