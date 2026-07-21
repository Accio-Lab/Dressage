"""HTTP client for the Dressage blackbox server protocol."""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from uuid import uuid4

import httpx

from dressage.paddock.blackbox.common.command import build_execute_cmd_payload
from dressage.paddock.blackbox.common.http_retry import (
    get_json_with_retry,
    post_json_with_retry,
)
from dressage.paddock.blackbox.common.utils import _env_float, _env_int
from dressage.sandbox.types import SandboxEndpoint

logger = logging.getLogger(__name__)

# Terminal turn states reported by the async turn status endpoint.
_TURN_COMMITTED = "committed"
_TURN_ERROR_STATES = frozenset({"failed", "unknown", "cancelled"})
_TURN_ACTIVE_STATES = frozenset({"queued", "inflight"})

# Client-side /messages submission mode. "async" submits + long-polls (default);
# "sync" issues a request-bound call and returns the result directly.
_CALL_MODE_ENV = "DRESSAGE_BLACKBOX_AGENT_CALL_MODE"
_DEFAULT_CALL_MODE = "async"
_VALID_CALL_MODES = frozenset({"sync", "async"})


class BlackboxServerClient:
    """Client for the blackbox server HTTP API.

    This class is intentionally provider-agnostic.  It only needs a
    ``SandboxEndpoint`` and does not know whether the endpoint came from E2B or
    a local Ray/bubblewrap lease.
    """

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(None), trust_env=False
        )

    async def health(self, endpoint: SandboxEndpoint) -> dict[str, Any]:
        response = await self._client.get(
            f"{endpoint.url.rstrip('/')}/health",
            headers=endpoint.headers,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"ok": True, "text": response.text}

    async def register_agent(
        self,
        endpoint: SandboxEndpoint,
        *,
        trajectory_id: str,
        instance_id: str,
        session_id: str,
        router_url: str,
        blackbox_type: str,
        backend_options: Any,
        server_config: dict[str, Any],
        router_api_path: str = "/v1",
    ) -> dict[str, Any]:
        payload = {
            "blackbox_type": blackbox_type,
            "router": router_url,
            "router_api_path": router_api_path,
            "bound_instance_id": instance_id,
            "bound_session_id": session_id,
            "backend_options": backend_options,
            "server_config": server_config,
        }
        response = await self._post_agent_with_retry(
            endpoint,
            "/v1/rollout/register",
            json=payload,
            operation="register_agent",
            trajectory_id=trajectory_id,
        )
        return response.json()

    async def call_agent(
        self,
        endpoint: SandboxEndpoint,
        *,
        trajectory_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit an agent turn asynchronously and poll until it settles.

        A stable ``turn_id`` is generated once (when the caller does not supply
        one) and reused across the whole submit retry cycle, making the submit
        POST idempotent: a retried submit re-attaches to the same server-side
        turn instead of starting a second execution.
        """
        if turn_id is None:
            turn_id = f"turn-{uuid4().hex}"
        call_mode = self._resolve_call_mode()
        submit_response = await self._post_agent_with_retry(
            endpoint,
            f"/v1/sessions/{session_id}/messages",
            json={
                "turn_id": turn_id,
                "mode": call_mode,
                "messages": messages,
                "metadata": metadata or {},
            },
            operation="call_agent",
            trajectory_id=trajectory_id,
        )
        if call_mode == "sync":
            # The server blocks until completion and returns the full result
            # (or raises the semantic HTTP error), matching the legacy body.
            return submit_response.json()
        submit_data = submit_response.json()
        idempotent_replay = bool(submit_data.get("idempotent_replay", False))
        return await self._poll_turn(
            endpoint,
            trajectory_id=trajectory_id,
            session_id=session_id,
            turn_id=turn_id,
            idempotent_replay=idempotent_replay,
        )

    @staticmethod
    def _resolve_call_mode() -> str:
        raw = os.environ.get(_CALL_MODE_ENV)
        if raw is None or raw == "":
            return _DEFAULT_CALL_MODE
        mode = raw.strip().lower()
        if mode not in _VALID_CALL_MODES:
            logger.warning(
                "invalid %s=%r; falling back to %r",
                _CALL_MODE_ENV,
                raw,
                _DEFAULT_CALL_MODE,
            )
            return _DEFAULT_CALL_MODE
        return mode

    async def _poll_turn(
        self,
        endpoint: SandboxEndpoint,
        *,
        trajectory_id: str,
        session_id: str,
        turn_id: str,
        idempotent_replay: bool,
    ) -> dict[str, Any]:
        poll_wait = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_POLL_WAIT_SEC", 30.0, min_value=0.0
        )
        total_timeout = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_POLL_TOTAL_TIMEOUT_SEC", 0.0, min_value=0.0
        )
        deadline = time.monotonic() + total_timeout if total_timeout > 0 else None
        url = f"{endpoint.url.rstrip('/')}/v1/sessions/{session_id}/turns/{turn_id}"
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"blackbox turn poll exceeded total timeout for "
                    f"session_id={session_id} turn_id={turn_id}"
                )
            response = await self._get_turn_with_retry(
                url,
                params={"wait": poll_wait},
                operation="poll_turn",
                trajectory_id=trajectory_id,
                headers=endpoint.headers,
            )
            data = response.json()
            status = str(data.get("status") or "")
            if status == _TURN_COMMITTED:
                return self._committed_call_payload(data, idempotent_replay=idempotent_replay)
            if status in _TURN_ERROR_STATES:
                self._raise_turn_error(data, url)
            # queued / inflight (or unknown transient) -> keep polling.

    @staticmethod
    def _committed_call_payload(
        data: dict[str, Any],
        *,
        idempotent_replay: bool,
    ) -> dict[str, Any]:
        """Assemble a payload matching the legacy synchronous ``/messages`` body."""
        return {
            "request_id": data.get("request_id", ""),
            "session_id": data.get("session_id"),
            "instance_id": data.get("instance_id"),
            "turn_id": data.get("turn_id"),
            "state": data.get("state"),
            "idempotent_replay": idempotent_replay,
            "outputs": data.get("outputs") or [],
            "backend": data.get("backend"),
            "usage": data.get("usage"),
        }

    @staticmethod
    def _raise_turn_error(data: dict[str, Any], url: str) -> None:
        """Reconstruct the equivalent HTTP error for a terminal-failed turn."""
        error = data.get("error")
        if not isinstance(error, dict):
            error = {}
        http_status = int(error.get("http_status") or 502)
        body = {
            "error": error.get("error", "backend_error"),
            "message": error.get("message", "blackbox turn failed"),
            "details": error.get("details") if isinstance(error.get("details"), dict) else {},
        }
        request = httpx.Request("GET", url)
        response = httpx.Response(http_status, json=body, request=request)
        raise httpx.HTTPStatusError(
            f"blackbox turn {body['error']} (HTTP {http_status})",
            request=request,
            response=response,
        )

    async def execute_cmd(
        self,
        endpoint: SandboxEndpoint,
        *,
        session_id: str,
        cmd: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload = build_execute_cmd_payload(cmd=cmd, timeout=timeout)
        response = await self._client.post(
            f"{endpoint.url.rstrip('/')}/v1/sessions/{session_id}/execute_cmd",
            json=payload,
            headers=endpoint.headers,
        )
        response.raise_for_status()
        return response.json()

    async def pause(
        self,
        endpoint: SandboxEndpoint,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": reason}
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        response = await self._client.post(
            f"{endpoint.url.rstrip('/')}/v1/rollout/pause",
            json=payload,
            headers=endpoint.headers,
        )
        response.raise_for_status()
        return response.json()

    async def resume(
        self,
        endpoint: SandboxEndpoint,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        response = await self._client.post(
            f"{endpoint.url.rstrip('/')}/v1/rollout/resume",
            json={"reason": reason, "version": version},
            headers=endpoint.headers,
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _retry_config() -> tuple[int, float, float, float]:
        max_attempts = _env_int(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_MAX_ATTEMPTS",
            _env_int(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_MAX_ATTEMPTS",
                6,
                min_value=1,
            ),
            min_value=1,
        )
        initial_delay = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_INITIAL_DELAY_SEC",
            _env_float(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_INITIAL_DELAY_SEC",
                1.0,
                min_value=0.0,
            ),
            min_value=0.0,
        )
        max_delay = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_MAX_DELAY_SEC",
            _env_float(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_MAX_DELAY_SEC",
                10.0,
                min_value=0.0,
            ),
            min_value=0.0,
        )
        jitter_fraction = _env_float(
            "DRESSAGE_BLACKBOX_AGENT_REQUEST_JITTER_FRACTION",
            _env_float(
                "DRESSAGE_SANDBOX_AGENT_REQUEST_JITTER_FRACTION",
                0.2,
                min_value=0.0,
            ),
            min_value=0.0,
        )
        return max_attempts, initial_delay, max_delay, jitter_fraction

    async def _post_agent_with_retry(
        self,
        endpoint: SandboxEndpoint,
        path: str,
        *,
        json: dict[str, Any],
        operation: str,
        trajectory_id: str,
    ) -> httpx.Response:
        max_attempts, initial_delay, max_delay, jitter_fraction = self._retry_config()
        return await post_json_with_retry(
            self._client,
            f"{endpoint.url.rstrip('/')}{path}",
            json=json,
            operation=operation,
            trajectory_id=trajectory_id,
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            jitter_fraction=jitter_fraction,
            log_prefix="blackbox server",
            logger=logger,
            headers=endpoint.headers,
        )

    async def _get_turn_with_retry(
        self,
        url: str,
        *,
        params: dict[str, Any],
        operation: str,
        trajectory_id: str,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        max_attempts, initial_delay, max_delay, jitter_fraction = self._retry_config()
        return await get_json_with_retry(
            self._client,
            url,
            params=params,
            operation=operation,
            trajectory_id=trajectory_id,
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            jitter_fraction=jitter_fraction,
            log_prefix="blackbox server",
            logger=logger,
            headers=headers,
        )
