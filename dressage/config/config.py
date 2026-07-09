"""Shared Dressage runtime defaults.

This module keeps open-source defaults in one place so example scripts and
runtime entrypoints do not each reinvent the same environment variable logic.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROXY_PORT = 8800
DEFAULT_SGLANG_ROUTER_PORT = 8000
DEFAULT_PADDOCK_MODE = "blackbox"
DEFAULT_SANDBOX_PROVIDER = "local_bwrap"
DEFAULT_LOCAL_BWRAP_NAMESPACE = "dressage"
DEFAULT_LOCAL_BWRAP_MANAGER_NAME = "dressage_local_bwrap_manager"
DEFAULT_TOKEN_BUILD_MODEL = "qwen3_5"


@dataclass(frozen=True)
class TokenBuildDefaults:
    model_mask_type: str | None
    model_tool_call_type: str | None
    model_reasoning_type: str | None
    tito_model: str | None


def repo_root() -> Path:
    value = os.environ.get("REPO_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def log_dir() -> Path:
    value = os.environ.get("LOG_DIR")
    if value:
        return Path(value).expanduser()
    return repo_root() / "log"


def run_name(default: str = "dressage") -> str:
    return os.environ.get("DRESSAGE_RUN_NAME") or default


def proxy_port() -> int:
    return _env_int("PROXY_PORT", DEFAULT_PROXY_PORT)


def master_addr() -> str:
    value = os.environ.get("MASTER_ADDR")
    if value:
        return value
    try:
        resolved = socket.gethostbyname(socket.gethostname())
    except OSError:
        resolved = ""
    return resolved or "127.0.0.1"


def proxy_local_url() -> str:
    return proxy_url()


def proxy_url() -> str:
    value = os.environ.get("DRESSAGE_PROXY_URL")
    if value:
        return value
    host = os.environ.get("PROXY_PUBLIC_HOST") or master_addr()
    return f"http://{host}:{proxy_port()}"


def proxy_public_url() -> str:
    return proxy_url()


def sglang_router_url() -> str:
    value = os.environ.get("SGLANG_ROUTER_URL")
    if value:
        return value
    host = os.environ.get("SGLANG_ROUTER_HOST") or master_addr()
    port = _env_int("SGLANG_ROUTER_PORT", DEFAULT_SGLANG_ROUTER_PORT)
    return f"http://{host}:{port}"


def paddock_mode() -> str:
    return _env_choice("DRESSAGE_PADDOCK_MODE", DEFAULT_PADDOCK_MODE)


def sandbox_provider() -> str:
    return _env_choice("DRESSAGE_SANDBOX_PROVIDER", DEFAULT_SANDBOX_PROVIDER)


def local_bwrap_pool_mode(*, mode: str | None = None) -> str:
    value = os.environ.get("DRESSAGE_LOCAL_BWRAP_POOL_MODE")
    if value:
        return value.strip().lower()
    selected_mode = (mode or paddock_mode()).strip().lower()
    return "command_only" if selected_mode == "whitebox" else "blackbox"


def local_bwrap_namespace() -> str:
    return os.environ.get("DRESSAGE_LOCAL_BWRAP_RAY_NAMESPACE") or DEFAULT_LOCAL_BWRAP_NAMESPACE


def local_bwrap_manager_name() -> str:
    return os.environ.get("DRESSAGE_LOCAL_BWRAP_MANAGER_NAME") or DEFAULT_LOCAL_BWRAP_MANAGER_NAME


def trajectory_payload_log_dir(*, name: str | None = None) -> Path:
    value = os.environ.get("DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR")
    if value:
        return Path(value).expanduser()
    return log_dir() / "traj_payload" / (name or run_name())


def trajectory_error_log_dir(*, name: str | None = None) -> Path:
    value = os.environ.get("DRESSAGE_TRAJECTORY_ERROR_LOG_DIR")
    if value:
        return Path(value).expanduser()
    return log_dir() / "traj_err" / (name or run_name())


def proxy_log_file(*, name: str | None = None) -> Path:
    value = os.environ.get("PROXY_LOG_FILE")
    if value:
        return Path(value).expanduser()
    return log_dir() / "proxy" / f"{name or run_name()}.log"


def proxy_pid_file(*, name: str | None = None) -> Path:
    value = os.environ.get("PROXY_PID_FILE")
    if value:
        return Path(value).expanduser()
    return log_dir() / "proxy" / f"{name or run_name()}.pid"


def token_build_defaults(
    *,
    token_build_mode: str,
    token_build_model: str | None = None,
) -> TokenBuildDefaults:
    model = (token_build_model or DEFAULT_TOKEN_BUILD_MODEL).strip().lower()
    if model != "qwen3_5":
        raise ValueError(
            "unsupported token_build_model="
            f"{token_build_model!r}; expected 'qwen3_5'"
        )
    tito_model = "qwen3_5" if token_build_mode == "tito" else None
    return TokenBuildDefaults(
        model_mask_type="qwen3_5",
        model_tool_call_type="qwen3_5",
        model_reasoning_type="qwen3",
        tito_model=tito_model,
    )


def _env_choice(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
