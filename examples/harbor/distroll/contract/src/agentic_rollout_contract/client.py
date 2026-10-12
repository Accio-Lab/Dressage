from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from .models import (
    ActivateRequest,
    CancelRequest,
    CapacityView,
    CleanupReport,
    ContextBindRequest,
    ContextPrepareRequest,
    ExecutionClaim,
    ExecutionClaimRequest,
    ExecutionIdentity,
    ExecutionResult,
    ExecutionTransition,
    ExecutionView,
    RecoveryProof,
    ReserveRequest,
    ResultAck,
    RouteControlRequest,
    RouteView,
    WorkerDrainRequest,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerView,
)

LOG = logging.getLogger(__name__)


class _Client:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: httpx.Timeout | float | None = None,
        limits: httpx.Limits | None = None,
        event_hooks: dict[str, list[Any]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout or httpx.Timeout(3600.0, connect=15.0),
            transport=httpx.AsyncHTTPTransport(
                limits=limits
                or httpx.Limits(
                    max_connections=512,
                    max_keepalive_connections=512,
                    keepalive_expiry=5,
                ),
                trust_env=False,
            ),
            event_hooks=event_hooks,
            trust_env=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: BaseModel | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        # All mutations use stable IDs/CAS at their owner; transport retries do
        # not create a second operation identity.
        frozen_payload = payload.model_dump(mode="json") if payload else None
        frozen_params = dict(params) if params else None
        for attempt in range(2):
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=frozen_payload,
                    params=frozen_params,
                )
                response.raise_for_status()
                return response.json() if response.content else None
            except httpx.TransportError as exc:
                LOG.warning(
                    "Control transport failed attempt=%d/2 type=%s",
                    attempt + 1,
                    type(exc).__name__,
                )
                if attempt:
                    raise
                await asyncio.sleep(0.2)


def _id(value: str) -> str:
    return quote(value, safe="")


class RolloutServiceClient(_Client):
    async def capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/v4/capabilities")

    async def reserve(self, request: ReserveRequest) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request("POST", "/v4/executions:reserve", payload=request)
        )

    async def activate(
        self, execution_id: str, request: ActivateRequest
    ) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request(
                "POST", f"/v4/executions/{_id(execution_id)}:activate", payload=request
            )
        )

    async def get_execution(
        self, execution_id: str, *, after_version: int = 0, wait: float = 0
    ) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request(
                "GET",
                f"/v4/executions/{_id(execution_id)}",
                params={"after_version": after_version, "wait": wait},
            )
        )

    async def list_executions(
        self, scope: str, *, after_id: str = "", limit: int = 256
    ) -> list[ExecutionView]:
        rows = await self._request(
            "GET",
            "/v4/executions",
            params={"client_scope": scope, "after_id": after_id, "limit": limit},
        )
        return [ExecutionView.model_validate(row) for row in rows]

    async def cancel(self, execution_id: str, request: CancelRequest) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request(
                "POST", f"/v4/executions/{_id(execution_id)}:cancel", payload=request
            )
        )

    async def ack(self, execution_id: str, request: ResultAck) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request(
                "POST", f"/v4/executions/{_id(execution_id)}:ack", payload=request
            )
        )

    async def capacity(self) -> CapacityView:
        return CapacityView.model_validate(await self._request("GET", "/v4/capacity"))

    async def pending_cleanup(
        self, worker_id: str, incarnation_id: str
    ) -> list[ExecutionView]:
        rows = await self._request(
            "GET",
            f"/internal/v4/workers/{_id(worker_id)}/cleanup",
            params={"incarnation_id": incarnation_id},
        )
        return [ExecutionView.model_validate(row) for row in rows]

    async def recover_execution(
        self, execution_id: str, request: RecoveryProof
    ) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request(
                "POST",
                f"/internal/v4/executions/{_id(execution_id)}:recover",
                payload=request,
            )
        )

    async def abandon_execution(
        self, execution_id: str, request: ExecutionIdentity
    ) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request(
                "POST",
                f"/internal/v4/executions/{_id(execution_id)}:abandon",
                payload=request,
            )
        )

    async def register_worker(self, request: WorkerRegistration) -> WorkerView:
        return WorkerView.model_validate(
            await self._request(
                "POST", "/internal/v4/workers:register", payload=request
            )
        )

    async def heartbeat(self, request: WorkerHeartbeat) -> WorkerView:
        return WorkerView.model_validate(
            await self._request(
                "POST", "/internal/v4/workers:heartbeat", payload=request
            )
        )

    async def drain_worker(self, request: WorkerDrainRequest) -> WorkerView:
        return WorkerView.model_validate(
            await self._request("POST", "/internal/v4/workers:drain", payload=request)
        )

    async def claim(self, request: ExecutionClaimRequest) -> ExecutionClaim | None:
        data = await self._request(
            "POST", "/internal/v4/executions:claim", payload=request
        )
        return ExecutionClaim.model_validate(data) if data else None

    async def transition(self, execution_id: str, request: ExecutionTransition) -> None:
        await self._request(
            "POST",
            f"/internal/v4/executions/{_id(execution_id)}:transition",
            payload=request,
        )

    async def complete(
        self, execution_id: str, request: ExecutionResult
    ) -> ExecutionView:
        return ExecutionView.model_validate(
            await self._request(
                "POST",
                f"/internal/v4/executions/{_id(execution_id)}:complete",
                payload=request,
            )
        )

    async def cleanup(self, execution_id: str, request: CleanupReport) -> None:
        await self._request(
            "POST",
            f"/internal/v4/executions/{_id(execution_id)}:cleanup",
            payload=request,
        )


class PolicyServiceClient(_Client):
    async def prepare(self, request: ContextPrepareRequest) -> RouteView:
        return RouteView.model_validate(
            await self._request("POST", "/v4/contexts", payload=request)
        )

    async def bind(self, context_id: str, request: ContextBindRequest) -> RouteView:
        return RouteView.model_validate(
            await self._request(
                "POST", f"/v4/contexts/{_id(context_id)}:bind", payload=request
            )
        )

    async def get_context(self, context_id: str) -> RouteView:
        return RouteView.model_validate(
            await self._request("GET", f"/v4/contexts/{_id(context_id)}")
        )

    async def control(
        self, context_id: str, action: str, request: RouteControlRequest
    ) -> RouteView:
        return RouteView.model_validate(
            await self._request(
                "POST", f"/v4/contexts/{_id(context_id)}:{action}", payload=request
            )
        )

    async def revoke(
        self, context_id: str, action_id: str, execution_id: str
    ) -> RouteView:
        return await self.control(
            context_id,
            "revoke",
            RouteControlRequest(action_id=action_id, execution_id=execution_id),
        )
