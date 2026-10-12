from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import asyncpg
from agentic_rollout_contract.models import (
    CONTRACT_VERSION,
    ActivateRequest,
    CancelRequest,
    CapacityView,
    CleanupReport,
    ExecutionClaim,
    ExecutionClaimRequest,
    ExecutionIdentity,
    ExecutionResult,
    ExecutionTransition,
    ExecutionView,
    RecoveryProof,
    ReserveRequest,
    ResultAck,
    WorkerDrainRequest,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerView,
)
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from .config import ControlSettings, Inventory
from .store import (
    CapacityError,
    ConflictError,
    NotFoundError,
    PostgresStore,
    StoreError,
)
from .task_resource_index import load_inventory_task_resource_index


def create_control_app(
    settings: ControlSettings,
    inventory: Inventory,
    *,
    store: PostgresStore | None = None,
) -> FastAPI:
    selected = store or PostgresStore(
        inventory,
        settings.database_url,
        resource_index=load_inventory_task_resource_index(inventory),
        pool_min_size=settings.database_pool_min_size,
        pool_max_size=settings.database_pool_max_size,
        admission=settings.admission,
    )

    async def sweep() -> None:
        while True:
            await asyncio.sleep(settings.admission.sweep_sec)
            try:
                await selected.reconcile()
            except (asyncpg.PostgresError, OSError, TimeoutError):
                logging.getLogger(__name__).exception(
                    "execution reconcile temporarily unavailable"
                )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await selected.initialize()
        app.state.store = selected
        task = asyncio.create_task(sweep(), name="execution-reconciler")
        app.state.reconciler = task
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await selected.close()

    app = FastAPI(
        title="CPU Execution Service", version=CONTRACT_VERSION, lifespan=lifespan
    )

    @app.exception_handler(StoreError)
    async def store_error(_request, exc):
        status = 404 if isinstance(exc, NotFoundError) else 409
        if isinstance(exc, CapacityError):
            return JSONResponse(
                status_code=429,
                content={
                    "code": "CAPACITY_UNAVAILABLE",
                    "message": str(exc),
                    "retryable": True,
                },
                headers={"Retry-After": str(settings.admission.retry_after_sec)},
            )
        if not isinstance(exc, (NotFoundError, ConflictError)):
            status = 503
        return JSONResponse(
            status_code=status,
            content={
                "code": type(exc).__name__,
                "message": str(exc),
                "retryable": status >= 500,
            },
        )

    @app.exception_handler(asyncpg.PostgresConnectionError)
    async def database_error(_request, _exc):
        return JSONResponse(
            status_code=503, content={"code": "DATABASE_UNAVAILABLE", "retryable": True}
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "contract_version": CONTRACT_VERSION}

    @app.get("/readyz")
    async def readyz():
        task = getattr(app.state, "reconciler", None)
        if task is None or task.done():
            raise HTTPException(503, "execution reconciler is not running")
        return {"status": "ready"}

    @app.get("/v4/capabilities")
    async def capabilities():
        return {
            "contract_version": CONTRACT_VERSION,
            "pause": False,
            "capabilities": [
                "execution-cancel-v4",
                "atomic-reservation-v4",
                "execution-quarantine-v4",
                "fixed-execution-deadline-v1",
            ],
        }

    @app.post("/v4/executions:reserve", response_model=ExecutionView)
    async def reserve(request: ReserveRequest):
        return await selected.reserve(request)

    @app.post("/v4/executions/{execution_id}:activate", response_model=ExecutionView)
    async def activate(execution_id: str, request: ActivateRequest):
        return await selected.activate(execution_id, request)

    @app.get("/v4/executions", response_model=list[ExecutionView])
    async def executions(
        client_scope: str,
        after_id: str = "",
        limit: Annotated[int, Query(ge=1, le=512)] = 256,
    ):
        return await selected.list_executions(client_scope, after_id, limit)

    @app.get("/v4/executions/{execution_id}", response_model=ExecutionView)
    async def execution(
        execution_id: str,
        after_version: int = 0,
        wait: Annotated[float, Query(ge=0, le=30)] = 0,
    ):
        deadline = asyncio.get_running_loop().time() + wait
        while True:
            view = await selected.get_execution(execution_id)
            if (
                view.state_version != after_version
                or asyncio.get_running_loop().time() >= deadline
            ):
                return view
            await asyncio.sleep(0.25)

    @app.post("/v4/executions/{execution_id}:cancel", response_model=ExecutionView)
    async def cancel(execution_id: str, request: CancelRequest):
        return await selected.cancel(execution_id, request)

    @app.post("/v4/executions/{execution_id}:ack", response_model=ExecutionView)
    async def ack(execution_id: str, request: ResultAck):
        return await selected.ack(execution_id, request)

    @app.get("/v4/capacity", response_model=CapacityView)
    async def capacity():
        return await selected.capacity()

    @app.post("/internal/v4/workers:register", response_model=WorkerView)
    async def register(request: WorkerRegistration):
        return await selected.register_worker(request)

    @app.post("/internal/v4/workers:heartbeat", response_model=WorkerView)
    async def heartbeat(request: WorkerHeartbeat):
        return await selected.heartbeat(request)

    @app.post("/internal/v4/workers:drain", response_model=WorkerView)
    async def drain(request: WorkerDrainRequest):
        return await selected.drain_worker(request)

    @app.post("/internal/v4/executions:claim", response_model=ExecutionClaim | None)
    async def claim(request: ExecutionClaimRequest):
        return await selected.claim(request)

    @app.get(
        "/internal/v4/workers/{worker_id}/cleanup", response_model=list[ExecutionView]
    )
    async def pending_cleanup(worker_id: str, incarnation_id: str):
        return await selected.pending_cleanup(worker_id, incarnation_id)

    @app.post(
        "/internal/v4/executions/{execution_id}:recover", response_model=ExecutionView
    )
    async def recover(execution_id: str, request: RecoveryProof):
        return await selected.recover_execution(execution_id, request)

    @app.post(
        "/internal/v4/executions/{execution_id}:abandon", response_model=ExecutionView
    )
    async def abandon(execution_id: str, request: ExecutionIdentity):
        return await selected.abandon_execution(execution_id, request)

    @app.post("/internal/v4/executions/{execution_id}:transition", status_code=204)
    async def transition(execution_id: str, request: ExecutionTransition):
        await selected.transition(execution_id, request)
        return Response(status_code=204)

    @app.post(
        "/internal/v4/executions/{execution_id}:complete", response_model=ExecutionView
    )
    async def complete(execution_id: str, request: ExecutionResult):
        return await selected.complete(execution_id, request)

    @app.post("/internal/v4/executions/{execution_id}:cleanup", status_code=204)
    async def cleanup(execution_id: str, request: CleanupReport):
        await selected.cleanup(execution_id, request)
        return Response(status_code=204)

    return app
