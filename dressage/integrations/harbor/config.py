"""Validated, versioned configuration for the Harbor integration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal, Mapping
from urllib.parse import urlsplit

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


HARBOR_INTEGRATION_SCHEMA_VERSION = "dressage.harbor/v1"


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class NativeEnvironmentConfig(_ConfigModel):
    """Use one of Harbor's native environment providers."""

    mode: Literal["native"] = "native"


class BwrapEnvironmentConfig(_ConfigModel):
    """Run Harbor agent and verifier commands in local bubblewrap sandboxes."""

    mode: Literal["bwrap"] = "bwrap"
    runtime_root: Path = Path("/tmp/dressage-harbor")

    @field_validator("runtime_root")
    @classmethod
    def validate_runtime_root(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("environment.runtime_root must be an absolute path")
        return path


EnvironmentConfig = Annotated[
    NativeEnvironmentConfig | BwrapEnvironmentConfig,
    Field(discriminator="mode"),
]


class GatewayLimitsConfig(_ConfigModel):
    max_active_routes: int = Field(default=64, ge=1)
    max_inflight_global: int = Field(default=128, ge=1)
    max_inflight_per_route: int = Field(default=8, ge=1)
    queue_timeout_sec: float = Field(default=30.0, gt=0)
    drain_timeout_sec: float = Field(default=60.0, gt=0)
    request_body_max_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    request_header_max_bytes: int = Field(default=64 * 1024, ge=1)
    sse_idle_timeout_sec: float = Field(default=300.0, gt=0)
    route_ttl_sec: float = Field(default=7200.0, gt=0)
    tombstone_ttl_sec: float = Field(default=300.0, gt=0)
    max_tombstones: int = Field(default=4096, ge=1)


class GatewayConfig(_ConfigModel):
    listen_host: str = Field(default="127.0.0.1", min_length=1)
    listen_port: int = Field(default=0, ge=0, le=65535)
    advertise_url: AnyHttpUrl | None = None
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "warning"
    limits: GatewayLimitsConfig = Field(default_factory=GatewayLimitsConfig)

    @model_validator(mode="after")
    def validate_advertise_address(self) -> "GatewayConfig":
        if self.listen_host in {"0.0.0.0", "::", "[::]"} and self.advertise_url is None:
            raise ValueError(
                "gateway.advertise_url is required for a wildcard listen_host"
            )
        return self


class BackendConfig(_ConfigModel):
    dressage_proxy_url: AnyHttpUrl = "http://127.0.0.1:8800"
    router_api_path: str = "/v1"
    sticky_header_name: str = "X-SMG-Routing-Key"
    service_api_key_env: str | None = "DRESSAGE_PROXY_API_KEY"
    service_api_key_header: str = "Authorization"
    service_api_key_scheme: str | None = "Bearer"
    verify_tls: bool = True

    @model_validator(mode="after")
    def validate_backend(self) -> "BackendConfig":
        if not self.router_api_path.startswith("/"):
            raise ValueError("backend.router_api_path must start with '/'")
        if not self.service_api_key_header.strip():
            raise ValueError("backend.service_api_key_header must not be empty")
        return self

    def service_headers(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        required: bool,
    ) -> dict[str, str]:
        """Resolve the backend credential without ever serializing it in config."""

        environ = os.environ if environ is None else environ
        env_name = self.service_api_key_env
        if env_name is None:
            if required:
                raise ValueError(
                    "backend.service_api_key_env is required in this execution mode"
                )
            return {}
        value = environ.get(env_name)
        if not value:
            if required:
                raise ValueError(
                    f"backend service credential environment variable {env_name!r} is unset"
                )
            return {}
        scheme = (
            self.service_api_key_scheme.strip() if self.service_api_key_scheme else ""
        )
        prefix = f"{scheme} " if scheme else ""
        return {self.service_api_key_header: f"{prefix}{value}"}


class SecurityConfig(_ConfigModel):
    routing_guarantee: Literal["configure_only", "enforced"] = "configure_only"
    allow_model_listing: bool = False
    require_tls_for_non_loopback: bool = True
    additional_agent_egress_hosts: tuple[str, ...] = ()


class TrajectoryConfig(_ConfigModel):
    max_steps: int | None = Field(default=100, ge=1)
    default_temperature: float | None = Field(default=None, ge=0)
    require_trainable_tokens: bool = True


class ArtifactConfig(_ConfigModel):
    mode: Literal["memory", "disk", "both"] = "both"
    root: Path = Path("harbor-artifacts")
    fsync: bool = True
    file_mode: int = Field(default=0o600, ge=0, le=0o777)
    dir_mode: int = Field(default=0o700, ge=0, le=0o777)


class SamplingConfig(_ConfigModel):
    mode: Literal["fill_missing", "force"] = "fill_missing"
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed_policy: Literal["agent", "per_attempt_request"] = "agent"


class TrainingConfig(_ConfigModel):
    model_override: str = Field(min_length=1)
    reward_key: str = Field(default="reward", min_length=1)
    group_max_retries: int = Field(default=2, ge=0)
    failed_group_policy: Literal["zero_grad", "abort_batch", "replace"] = "zero_grad"
    max_replacement_groups: int = Field(default=8, ge=0)
    replacement_exhausted_policy: Literal["abort_batch"] = "abort_batch"
    min_live_group_ratio: float = Field(default=0.8, ge=0, le=1)
    require_single_weight_version: bool = True
    sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(
            mode="force",
            temperature=0.8,
            top_p=0.95,
            seed_policy="per_attempt_request",
        )
    )


class HarborIntegrationConfig(_ConfigModel):
    schema_version: Literal["dressage.harbor/v1"] = HARBOR_INTEGRATION_SCHEMA_VERSION
    execution_mode: Literal["rollout", "training"] = "rollout"
    environment: EnvironmentConfig = Field(
        default_factory=NativeEnvironmentConfig
    )
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    agent_protocol_overrides: dict[str, Literal["openai", "anthropic", "both"]] = Field(
        default_factory=dict
    )
    training: TrainingConfig | None = None

    @model_validator(mode="after")
    def validate_mode_invariants(self) -> "HarborIntegrationConfig":
        if self.environment.mode == "bwrap":
            if self.gateway.listen_host != "127.0.0.1":
                raise ValueError(
                    "bwrap requires gateway.listen_host='127.0.0.1'"
                )
            if self.gateway.listen_port == 0:
                raise ValueError("bwrap requires a fixed non-zero gateway.listen_port")
            if self.gateway.advertise_url is not None:
                raise ValueError("bwrap does not accept gateway.advertise_url")
            if self.security.routing_guarantee != "enforced":
                raise ValueError(
                    "bwrap requires security.routing_guarantee='enforced'"
                )
            if self.security.additional_agent_egress_hosts:
                raise ValueError(
                    "bwrap does not support additional_agent_egress_hosts"
                )

        if self.execution_mode == "training":
            if self.training is None:
                raise ValueError(
                    "training configuration is required when execution_mode='training'"
                )
            if self.artifacts.mode != "both":
                raise ValueError("training requires artifacts.mode='both'")
            if not self.trajectory.require_trainable_tokens:
                raise ValueError(
                    "training requires trajectory.require_trainable_tokens=true"
                )
        elif self.training is not None:
            raise ValueError(
                "training configuration is only valid when execution_mode='training'"
            )

        advertise_url = self.gateway.advertise_url
        if advertise_url is not None and self.security.require_tls_for_non_loopback:
            parsed = urlsplit(str(advertise_url))
            if (
                parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                and parsed.scheme != "https"
            ):
                raise ValueError("a non-loopback gateway.advertise_url must use HTTPS")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def gateway_fingerprint(self) -> str:
        payload = {
            "environment": self.environment.model_dump(mode="json"),
            "gateway": self.gateway.model_dump(mode="json", exclude_none=True),
            "backend": self.backend.model_dump(mode="json", exclude_none=True),
            "security": self.security.model_dump(mode="json", exclude_none=True),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> HarborIntegrationConfig:
    """Load JSON or YAML while keeping the YAML dependency optional."""

    config_path = Path(path).expanduser()
    raw = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised without the optional extra
            raise RuntimeError(
                "PyYAML is required to load Harbor YAML configuration"
            ) from exc
        payload = yaml.safe_load(raw)
    else:
        raise ValueError("Harbor integration config must use .json, .yaml, or .yml")
    if not isinstance(payload, dict):
        raise ValueError("Harbor integration config must contain a mapping at its root")
    return HarborIntegrationConfig.model_validate(payload)


__all__ = [
    "HARBOR_INTEGRATION_SCHEMA_VERSION",
    "ArtifactConfig",
    "BackendConfig",
    "BwrapEnvironmentConfig",
    "EnvironmentConfig",
    "GatewayConfig",
    "GatewayLimitsConfig",
    "HarborIntegrationConfig",
    "NativeEnvironmentConfig",
    "SamplingConfig",
    "SecurityConfig",
    "TrainingConfig",
    "TrajectoryConfig",
    "load_config",
]
