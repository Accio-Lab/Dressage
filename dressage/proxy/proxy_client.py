"""HTTP client for interacting with the Dressage proxy."""

from __future__ import annotations

from typing import Any

import httpx


class ProxyClient:
    """Thin async client used by rollout code to talk to the proxy."""

    def __init__(
        self,
        proxy_url: str,
        *,
        timeout: httpx.Timeout | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self._proxy_url = proxy_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout or httpx.Timeout(None), trust_env=False
        )

    async def chat_completions(
        self,
        body: dict[str, Any],
        *,
        session_id: str,
        instance_id: str | None = None,
        turn_id: str | None = None,
    ) -> dict:
        headers = {"X-Session-Id": session_id}
        if instance_id is not None:
            headers["X-Instance-Id"] = instance_id
        if turn_id is not None:
            headers["X-Turn-Id"] = turn_id

        response = await self._client.post(
            f"{self._proxy_url}/v1/chat/completions", json=body, headers=headers
        )
        response.raise_for_status()
        return response.json()

    async def finalize_session(
        self,
        session_id: str,
        *,
        instance_id: str | None = None,
        label: Any | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"session_id": session_id}
        if instance_id is not None:
            payload["instance_id"] = instance_id
        if label is not None:
            payload["label"] = label
        response = await self._client.post(
            f"{self._proxy_url}/session/finalize", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def read_trajectory(
        self,
        *,
        trajectory_id: str | None = None,
        session_id: str | None = None,
        instance_id: str | None = None,
        max_groups: int | None = None,
        segment_view: str | None = None,
        drain: bool = False,
    ) -> dict:
        payload: dict[str, Any] = {
            "drain": drain,
        }
        if trajectory_id is not None:
            payload["trajectory_id"] = trajectory_id
        if session_id is not None:
            payload["session_id"] = session_id
        if instance_id is not None:
            payload["instance_id"] = instance_id
        if max_groups is not None:
            payload["max_groups"] = max_groups
        if segment_view is not None:
            payload["segment_view"] = segment_view

        response = await self._client.post(
            f"{self._proxy_url}/trajectory/read", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def pause_rollout(
        self,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"reason": reason}
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        response = await self._client.post(
            f"{self._proxy_url}/v1/rollout/pause", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def resume_rollout(
        self,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict:
        payload: dict[str, Any] = {"reason": reason}
        if version is not None:
            payload["version"] = version
        response = await self._client.post(
            f"{self._proxy_url}/v1/rollout/resume", json=payload
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
