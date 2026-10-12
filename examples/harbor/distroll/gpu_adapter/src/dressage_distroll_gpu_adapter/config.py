from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicySettings(ConfigModel):
    control_plane_url: str = "http://127.0.0.1:19443"
    listen_host: str = "0.0.0.0"
    listen_port: int = 18443
    internal_gateway_host: str = "127.0.0.1"
    internal_gateway_port: int = 18081
    dressage_proxy_url: str = "http://127.0.0.1:8800"
    model_name: str = "qwen3-30b-a3b"
    max_model_request_bytes: int = Field(default=32 * 1024 * 1024, ge=1024)
    max_model_response_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    ledger_max_bytes: int = Field(default=4 * 1024 * 1024 * 1024, ge=1024)
    ledger_max_entries: int = Field(default=8192, ge=1)
    ledger_max_entries_per_route: int = Field(default=256, ge=1)
    max_actions_per_route: int = Field(default=512, ge=1)
    closed_route_ttl_sec: float = Field(default=3600.0, ge=0)
    max_closed_routes: int = Field(default=4096, ge=1)
    proxy_retry_attempts: int = Field(default=4, ge=1, le=20)
    proxy_retry_base_sec: float = Field(default=0.25, ge=0)
    proxy_retry_max_sec: float = Field(default=2.0, ge=0)
    proxy_discard_timeout_sec: float = Field(default=60.0, gt=0, le=600)
    route_cleanup_timeout_sec: float = Field(default=150.0, gt=0, le=600)
    route_drain_timeout_sec: float = Field(default=180.0, gt=0, le=600)

    @model_validator(mode="after")
    def validate_capacity(self) -> PolicySettings:
        if self.ledger_max_bytes < self.max_model_response_bytes:
            raise ValueError(
                "ledger_max_bytes must hold at least one maximum model response"
            )
        if self.proxy_retry_max_sec < self.proxy_retry_base_sec:
            raise ValueError("proxy retry max delay must be >= base delay")
        return self


class RemoteRolloutSettings(ConfigModel):
    control_plane_url: str
    policy_service_url: str
    dressage_proxy_url: str
    run_id: str
    control_outage_timeout_sec: float = Field(default=0.0, ge=0)
    wait_long_poll_sec: float = Field(default=20.0, gt=0, le=30)
    max_concurrent_executions: int = Field(default=64, ge=1)
    admission_timeout_sec: float = Field(default=3900, gt=0)
    agent_max_turns: int = Field(default=80, ge=1)
    accept_max_turns: bool = False
    agent_timeout_sec: float | None = Field(default=None, gt=0)
    claude_auto_compact_pct: int | None = Field(default=None, ge=1, le=100)
    claude_file_read_max_output_tokens: int = Field(default=1024, ge=1)
    claude_code_version: str = "2.1.207"
    model_name: str = "qwen3-30b-a3b"
    control_retry_attempts: int = Field(default=20, ge=1, le=20)
    proxy_retry_attempts: int = Field(default=4, ge=1, le=20)
    retry_base_sec: float = Field(default=0.25, ge=0)
    retry_max_sec: float = Field(default=2.0, ge=0)
    group_max_retries: int = Field(default=1, ge=0, le=20)
    group_min_success_ratio: float = Field(default=1.0, gt=0, le=1)
    dynamic_sampling_early_stop: bool = False
    speculative_cancel_timeout_sec: float = Field(default=300.0, gt=0, le=600)
    failed_group_policy: Literal["zero_grad", "abort_batch"] = "zero_grad"
    min_live_group_ratio: float = Field(default=0.9, ge=0, le=1)
    diagnostics_dir: Path | None = None

    @model_validator(mode="after")
    def validate_retry_delays(self) -> RemoteRolloutSettings:
        if self.retry_max_sec < self.retry_base_sec:
            raise ValueError("rollout retry max delay must be >= base delay")
        for name, value in (
            ("control_plane_url", self.control_plane_url),
            ("policy_service_url", self.policy_service_url),
        ):
            if urlsplit(value).scheme != "http":
                raise ValueError(f"{name} must use http in the lite deployment")
        return self


def policy_settings_from_env() -> PolicySettings:
    return PolicySettings(
        control_plane_url=os.environ.get(
            "ROLLOUT_CONTROL_URL", "http://127.0.0.1:19443"
        ),
        listen_host=os.environ.get("POLICY_SERVICE_HOST", "0.0.0.0"),
        listen_port=int(os.environ.get("POLICY_SERVICE_PORT", "18443")),
        internal_gateway_host=os.environ.get("POLICY_INTERNAL_HOST", "127.0.0.1"),
        internal_gateway_port=int(os.environ.get("POLICY_INTERNAL_PORT", "18081")),
        dressage_proxy_url=os.environ.get(
            "DRESSAGE_PROXY_URL", "http://127.0.0.1:8800"
        ),
        model_name=os.environ.get("ROLLOUT_MODEL_NAME", "qwen3-30b-a3b"),
        max_model_request_bytes=int(
            os.environ.get("POLICY_MAX_MODEL_REQUEST_BYTES", str(32 * 1024 * 1024))
        ),
        max_model_response_bytes=int(
            os.environ.get("POLICY_MAX_MODEL_RESPONSE_BYTES", str(8 * 1024 * 1024))
        ),
        ledger_max_bytes=int(
            os.environ.get("POLICY_LEDGER_MAX_BYTES", str(4 * 1024 * 1024 * 1024))
        ),
        ledger_max_entries=int(os.environ.get("POLICY_LEDGER_MAX_ENTRIES", "8192")),
        ledger_max_entries_per_route=int(
            os.environ.get("POLICY_LEDGER_MAX_ENTRIES_PER_ROUTE", "256")
        ),
        max_actions_per_route=int(
            os.environ.get("POLICY_MAX_ACTIONS_PER_ROUTE", "512")
        ),
        closed_route_ttl_sec=float(
            os.environ.get("POLICY_CLOSED_ROUTE_TTL_SEC", "3600")
        ),
        max_closed_routes=int(os.environ.get("POLICY_MAX_CLOSED_ROUTES", "4096")),
        proxy_retry_attempts=int(os.environ.get("POLICY_PROXY_RETRY_ATTEMPTS", "4")),
        proxy_retry_base_sec=float(
            os.environ.get("POLICY_PROXY_RETRY_BASE_SEC", "0.25")
        ),
        proxy_retry_max_sec=float(os.environ.get("POLICY_PROXY_RETRY_MAX_SEC", "2")),
        proxy_discard_timeout_sec=float(
            os.environ.get("POLICY_PROXY_DISCARD_TIMEOUT_SEC", "60")
        ),
        route_cleanup_timeout_sec=float(
            os.environ.get("POLICY_ROUTE_CLEANUP_TIMEOUT_SEC", "150")
        ),
        route_drain_timeout_sec=float(
            os.environ.get("POLICY_ROUTE_DRAIN_TIMEOUT_SEC", "180")
        ),
    )


def remote_rollout_settings_from_env() -> RemoteRolloutSettings:
    return RemoteRolloutSettings(
        control_plane_url=_http_service_url(os.environ["ROLLOUT_CONTROL_URL"]),
        policy_service_url=_http_service_url(os.environ["POLICY_SERVICE_URL"]),
        dressage_proxy_url=os.environ.get(
            "DRESSAGE_PROXY_URL", "http://127.0.0.1:8800"
        ).rstrip("/"),
        run_id=os.environ["DRESSAGE_DISTROLL_RUN_ID"],
        control_outage_timeout_sec=float(
            os.environ.get("ROLLOUT_CONTROL_OUTAGE_TIMEOUT_SEC", "0")
        ),
        wait_long_poll_sec=float(os.environ.get("ROLLOUT_WAIT_LONG_POLL_SEC", "20")),
        max_concurrent_executions=int(
            os.environ.get("ROLLOUT_MAX_CONCURRENT_EXECUTIONS", "64")
        ),
        admission_timeout_sec=float(
            os.environ.get("ROLLOUT_ADMISSION_TIMEOUT_SEC", "3900")
        ),
        agent_max_turns=int(os.environ.get("CLAUDE_CODE_MAX_TURNS", "80")),
        accept_max_turns=_env_flag("ROLLOUT_ACCEPT_MAX_TURNS", False),
        agent_timeout_sec=(
            float(value)
            if (value := os.environ.get("CLAUDE_CODE_AGENT_TIMEOUT_SEC", "").strip())
            else None
        ),
        claude_auto_compact_pct=(
            int(value)
            if (value := os.environ.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", "").strip())
            else None
        ),
        claude_code_version=os.environ.get("CLAUDE_CODE_VERSION", "2.1.207"),
        claude_file_read_max_output_tokens=int(
            os.environ.get("CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS", "1024")
        ),
        model_name=os.environ.get("ROLLOUT_MODEL_NAME", "qwen3-30b-a3b"),
        control_retry_attempts=int(
            os.environ.get("ROLLOUT_CONTROL_RETRY_ATTEMPTS", "20")
        ),
        proxy_retry_attempts=int(os.environ.get("ROLLOUT_PROXY_RETRY_ATTEMPTS", "4")),
        retry_base_sec=float(os.environ.get("ROLLOUT_RETRY_BASE_SEC", "0.25")),
        retry_max_sec=float(os.environ.get("ROLLOUT_RETRY_MAX_SEC", "2")),
        group_max_retries=int(os.environ.get("ROLLOUT_GROUP_MAX_RETRIES", "1")),
        group_min_success_ratio=float(
            os.environ.get("ROLLOUT_GROUP_MIN_SUCCESS_RATIO", "1.0")
        ),
        dynamic_sampling_early_stop=_env_flag("DYNAMIC_SAMPLING_EARLY_STOP", False),
        speculative_cancel_timeout_sec=float(
            os.environ.get("ROLLOUT_SPECULATIVE_CANCEL_TIMEOUT_SEC", "300")
        ),
        failed_group_policy=os.environ.get("ROLLOUT_FAILED_GROUP_POLICY", "zero_grad"),
        min_live_group_ratio=float(
            os.environ.get("ROLLOUT_MIN_LIVE_GROUP_RATIO", "0.9")
        ),
        diagnostics_dir=(
            Path(value).expanduser()
            if (value := os.environ.get("ROLLOUT_DIAGNOSTICS_DIR", "").strip())
            else None
        ),
    )


def _http_service_url(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme != "http" or not parsed.netloc:
        raise ValueError(f"service URL must use http: {value}")
    return value.rstrip("/")


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")
