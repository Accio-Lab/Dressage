"""Paddock implementation for blackbox agent rollouts."""

from __future__ import annotations

import os
from typing import Any

from dressage.paddock.blackbox.client import BlackboxServerClient
from dressage.paddock.blackbox.common.defaults import (
    DEFAULT_BLACKBOX_TYPE,
    merge_backend_options,
    normalize_blackbox_type,
    server_config_for,
)
from dressage.paddock.blackbox.common.state import SandboxState
from dressage.paddock.blackbox.common.utils import (
    _require_public_proxy_url,
    _validate_public_proxy_url,
)
from dressage.paddock.interface import BlackboxPaddock
from dressage.sandbox import SandboxEndpoint, SandboxLease, SandboxServiceSpec, SandboxSpec
from dressage.sandbox.factory import create_sandbox_provider_from_env
from dressage.sandbox.provider import SandboxProvider


class BlackboxAgentPaddock(BlackboxPaddock):
    """Run blackbox agents through a provider-created sandbox service."""

    def __init__(
        self,
        *,
        provider: SandboxProvider | None = None,
        blackbox_client: BlackboxServerClient | None = None,
        proxy_public_url: str | None = None,
        blackbox_port: int | None = None,
        wait_health: bool | None = None,
    ) -> None:
        self._provider = provider or create_sandbox_provider_from_env()
        self._client = blackbox_client or BlackboxServerClient()
        self._proxy_public_url = _require_public_proxy_url(proxy_public_url)
        self._blackbox_port = int(
            blackbox_port
            or os.environ.get("DRESSAGE_BLACKBOX_PORT")
            or _provider_blackbox_port_env(getattr(self._provider, "name", ""))
            or "31000"
        )
        if wait_health is None:
            wait_health = os.environ.get("DRESSAGE_BLACKBOX_SKIP_HEALTHCHECK") not in {
                "1",
                "true",
                "TRUE",
                "yes",
            }
        self._wait_health = wait_health
        self._leases: dict[str, SandboxLease] = {}
        self._states: dict[str, SandboxState] = {}

    async def init(
        self,
        traj_id: str,
        env_type: str | None = None,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SandboxState:
        env_args = {**(env_args or {}), **kwargs}
        spec = SandboxSpec(
            trajectory_id=traj_id,
            env_type=env_type,
            env_args=env_args,
            services=(
                SandboxServiceSpec(
                    name="blackbox",
                    port=int(env_args.get("blackbox_port") or self._blackbox_port),
                    health_path="/health",
                ),
            ),
            timeout_sec=env_args.get("sandbox_timeout_sec"),
            metadata={"paddock_mode": "blackbox"},
        )
        lease = await self._provider.create(spec)
        endpoint = lease.endpoints.get("blackbox")
        if endpoint is None:
            endpoint = await self._provider.get_public_url(
                lease,
                port=spec.services[0].port,
                service_name="blackbox",
            )
            lease.endpoints["blackbox"] = endpoint
        endpoint = endpoint.normalized()
        if self._wait_health:
            await self._client.health(endpoint)
        self._leases[traj_id] = lease
        state = SandboxState(
            trajectory_id=traj_id,
            sandbox_url=endpoint.url,
            sandbox_id=lease.sandbox_id,
            raw_register_response={
                "provider": lease.provider,
                "sandbox_id": lease.sandbox_id,
                "metadata": lease.metadata,
                "endpoints": {
                    name: endpoint.url for name, endpoint in lease.endpoints.items()
                },
            },
        )
        self._states[traj_id] = state
        return state

    async def register_agent(
        self,
        state: SandboxState | str,
        *,
        instance_id: str,
        session_id: str,
        router_url: str | None = None,
        blackbox_type: str = DEFAULT_BLACKBOX_TYPE,
        backend_options: Any = None,
        router_api_path: str = "/v1",
    ) -> dict[str, Any]:
        state = self._resolve_state(state)
        lease = self._leases.get(state.trajectory_id)
        endpoint = self._endpoint_for_state(state, lease)
        router = _validate_public_proxy_url(router_url or self._proxy_public_url)
        blackbox_type = normalize_blackbox_type(blackbox_type)
        return await self._client.register_agent(
            endpoint,
            trajectory_id=state.trajectory_id,
            instance_id=instance_id,
            session_id=session_id,
            router_url=router,
            blackbox_type=blackbox_type,
            backend_options=merge_backend_options(blackbox_type, backend_options),
            server_config=_server_config_for_provider(self._provider.name, blackbox_type),
            router_api_path=router_api_path,
        )

    async def call_agent(
        self,
        state: SandboxState | str,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        state = self._resolve_state(state)
        return await self._client.call_agent(
            self._endpoint_for_state(state, self._leases.get(state.trajectory_id)),
            trajectory_id=state.trajectory_id,
            session_id=session_id,
            messages=messages,
            metadata=metadata,
            turn_id=turn_id,
        )

    async def execute_cmd(
        self,
        state: SandboxState | str,
        *,
        session_id: str,
        cmd: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        state = self._resolve_state(state)
        return await self._client.execute_cmd(
            self._endpoint_for_state(state, self._leases.get(state.trajectory_id)),
            session_id=session_id,
            cmd=cmd,
            timeout=timeout,
        )

    async def pause(
        self,
        traj_id: str | None = None,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        states = self._select_states(traj_id)
        results = {
            tid: await self._client.pause(
                self._endpoint_for_state(state, self._leases.get(tid)),
                reason=reason,
                timeout_seconds=timeout_seconds,
            )
            for tid, state in states.items()
        }
        return {
            "status": "paused",
            "reason": reason,
            "quiesced": all(bool(result.get("quiesced", True)) for result in results.values()),
            "results": results,
        }

    async def resume(
        self,
        traj_id: str | None = None,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        states = self._select_states(traj_id)
        results = {
            tid: await self._client.resume(
                self._endpoint_for_state(state, self._leases.get(tid)),
                version=version,
                reason=reason,
            )
            for tid, state in states.items()
        }
        return {"status": "resumed", "reason": reason, "version": version, "results": results}

    async def terminate(
        self,
        traj_id: str,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del env_args, kwargs
        state = self._states.pop(traj_id, None)
        lease = self._leases.pop(traj_id, None)
        if lease is None:
            if state is None:
                return {"terminated": False, "trajectory_id": traj_id, "missing": True}
            return await self._provider.terminate(state.trajectory_id)
        return await self._provider.terminate(lease)

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    def _resolve_state(self, state: SandboxState | str) -> SandboxState:
        if isinstance(state, SandboxState):
            return state
        if state not in self._states:
            raise KeyError(f"sandbox state not found for trajectory_id={state}")
        return self._states[state]

    def _select_states(self, traj_id: str | None) -> dict[str, SandboxState]:
        if traj_id is None:
            return dict(self._states)
        return {traj_id: self._resolve_state(traj_id)}

    def _endpoint_for_state(
        self,
        state: SandboxState,
        lease: SandboxLease | None,
    ) -> SandboxEndpoint:
        if lease is not None and "blackbox" in lease.endpoints:
            return lease.endpoints["blackbox"].normalized()
        return SandboxEndpoint(url=state.sandbox_url, headers={})


def _provider_blackbox_port_env(provider_name: str) -> str | None:
    if provider_name == "e2b":
        return os.environ.get("DRESSAGE_E2B_BLACKBOX_PORT")
    if provider_name == "local_bwrap":
        return os.environ.get("DRESSAGE_LOCAL_BWRAP_BLACKBOX_PORT")
    return None


def _server_config_for_provider(provider_name: str, blackbox_type: str) -> dict[str, Any]:
    config = server_config_for(blackbox_type)
    if provider_name == "local_bwrap":
        # Local blackbox servers are launched by the node supervisor, so runtime
        # root comes from BBS_RUNTIME_ROOT in the server process environment.
        config.pop("runtime_root", None)
    return config
