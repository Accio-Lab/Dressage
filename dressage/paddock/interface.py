"""Paddock interfaces for blackbox agents and whitebox tools."""

from __future__ import annotations

import abc
from typing import Any


class Paddock(abc.ABC):
    """Common lifecycle interface for execution environments."""

    @abc.abstractmethod
    async def init(
        self,
        traj_id: str,
        env_type: str | None = None,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Initialize an environment instance bound to traj_id."""

    @abc.abstractmethod
    async def terminate(
        self,
        traj_id: str,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tear down the environment for traj_id and reclaim resources."""


class BlackboxPaddock(Paddock):
    """Capabilities required by blackbox agent rollouts."""

    @abc.abstractmethod
    async def register_agent(
        self,
        state: Any,
        *,
        instance_id: str,
        session_id: str,
        router_url: str | None = None,
        blackbox_type: str = "opencode",
        backend_options: Any = None,
        router_api_path: str = "/v1",
    ) -> dict[str, Any]:
        """Register an agent process inside the sandbox service."""

    @abc.abstractmethod
    async def call_agent(
        self,
        state: Any,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Delegate to a blackbox agent and return its response payload."""

    @abc.abstractmethod
    async def execute_cmd(
        self,
        state: Any,
        *,
        session_id: str,
        cmd: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Execute a shell command in a blackbox session."""

    @abc.abstractmethod
    async def pause(
        self,
        traj_id: str | None = None,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Pause rollout generation and make the model side quiesced."""

    @abc.abstractmethod
    async def resume(
        self,
        traj_id: str | None = None,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        """Resume rollout generation after a weight update."""


class WhiteboxPaddock(Paddock):
    """Capabilities required by whitebox tool rollouts."""

    @abc.abstractmethod
    async def tool_call(
        self,
        traj_id: str,
        tool_id: str,
        tool_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Execute a tool and return (tool_response, metadata)."""
