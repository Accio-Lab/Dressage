from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from typing import Any
from uuid import uuid4

import asyncpg
from agentic_rollout_contract.canonical import canonical_digest
from agentic_rollout_contract.models import (
    TERMINAL_EXECUTION_STATES,
    ActivateRequest,
    AllocatedCharge,
    CancelRequest,
    CapacityView,
    CapacityWorker,
    CleanupReport,
    ExecutionClaim,
    ExecutionClaimRequest,
    ExecutionIdentity,
    ExecutionResult,
    ExecutionState,
    ExecutionTransition,
    ExecutionView,
    RecoveryProof,
    ReserveRequest,
    ResultAck,
    WorkerDrainRequest,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerState,
    WorkerView,
)

from .config import AdmissionSettings, Inventory
from .scheduler import add_charge, available_capacity, dominant_load
from .task_resource_index import TaskResourceIndex

E = ExecutionState
_TRANSITIONS = {
    E.PREPARING: {E.WAITING_ROUTE},
    E.WAITING_ROUTE: {E.AGENT_RUNNING},
    E.AGENT_RUNNING: {E.VERIFYING},
}


class StoreError(RuntimeError):
    pass


class ConflictError(StoreError):
    pass


class FencedError(ConflictError):
    pass


class NotFoundError(StoreError):
    pass


class CapacityError(StoreError):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _view(row: asyncpg.Record) -> ExecutionView:
    return ExecutionView.model_validate(_json(row["body"]))


class PostgresStore:
    """Single-execution authority. No training/group semantics enter this store.

    Mutations serialize through one short PostgreSQL advisory transaction lock.
    Capacity, reservation expiry, claim fencing and cancellation are thus decided
    together. Network waits and task execution never happen inside that lock.
    """

    def __init__(
        self,
        inventory: Inventory,
        database_url: str,
        *,
        resource_index: TaskResourceIndex,
        admission: AdmissionSettings | None = None,
        pool_min_size: int = 1,
        pool_max_size: int = 16,
    ) -> None:
        self.inventory = inventory
        self.resource_index = resource_index
        self.database_url = database_url
        self.admission = admission or AdmissionSettings()
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            command_timeout=15,
        )
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(83002403)")
            if await connection.fetchval("SELECT to_regclass('rollout_v3.executions')"):
                pending = await connection.fetchval("""SELECT count(*) FROM rollout_v3.executions
                    WHERE NOT cleanup_verified OR (body->'result' IS DISTINCT FROM 'null'::jsonb
                    AND NOT (body->>'result_acknowledged')::boolean)""")
                if pending:
                    raise StoreError(
                        "drain the deployed V3 ownership before installing V4"
                    )
            if await connection.fetchval("SELECT to_regclass('rollout_v4.executions')"):
                pending = await connection.fetchval("""SELECT count(*) FROM rollout_v4.executions
                    WHERE claim IS NOT NULL AND NOT cleanup_verified
                    AND NOT (claim ? 'execution_deadline_at')""")
                if pending:
                    raise StoreError("drain renewable executions before upgrading")
            await connection.execute(
                files("rollout_service")
                .joinpath("migrations/001_executions_v4.sql")
                .read_text()
            )

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    @asynccontextmanager
    async def _transaction(self):
        if self.pool is None:
            raise StoreError("store is not initialized")
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(83002403)")
            yield connection

    async def _execution(self, conn, execution_id: str) -> asyncpg.Record:
        row = await conn.fetchrow(
            "SELECT * FROM rollout_v4.executions WHERE execution_id=$1", execution_id
        )
        if row is None:
            raise NotFoundError("execution does not exist")
        return row

    async def _write(self, conn, view: ExecutionView) -> ExecutionView:
        view.state_version += 1
        view.updated_at = _now()
        await conn.execute(
            """UPDATE rollout_v4.executions
               SET state=$2,worker_id=$3,cleanup_verified=$4,body=$5::jsonb
               WHERE execution_id=$1""",
            view.execution_id,
            view.state.value,
            view.worker_id,
            view.cleanup_verified,
            view.model_dump_json(),
        )
        return view

    async def _reconcile(self, conn) -> None:
        now = _now()
        rows = await conn.fetch(
            "SELECT * FROM rollout_v4.executions WHERE NOT cleanup_verified"
        )
        for row in rows:
            view = _view(row)
            if view.state == E.RESERVED and view.reservation_expires_at <= now:
                view.state = E.EXPIRED
                view.cleanup_verified = True
                await self._write(conn, view)
            elif (
                row["claim"] is not None
                and row["lease_expires_at"] <= now
                and not view.failure_code
            ):
                await self._fail_uncertain(conn, view)
            elif (
                view.state == E.ACCEPTED
                and (now - view.updated_at).total_seconds()
                > self.inventory.assignment_ttl_sec
            ):
                # No claim means no Worker has permission to run this execution.
                view.state = E.CANCELLED
                view.cleanup_verified = True
                await self._write(conn, view)
        await conn.execute(
            """UPDATE rollout_v4.workers SET state='LOST'
               WHERE last_heartbeat < $1 AND state <> 'LOST'""",
            now - timedelta(seconds=self.inventory.heartbeat_timeout_sec),
        )
        lost = await conn.fetch(
            """SELECT e.body FROM rollout_v4.executions e JOIN rollout_v4.workers w USING(worker_id)
               WHERE w.state='LOST' AND e.claim IS NOT NULL AND NOT e.cleanup_verified
               AND e.body->>'failure_code' IS NULL"""
        )
        for row in lost:
            await self._fail_uncertain(conn, _view(row))
        unstarted = await conn.fetch(
            """SELECT e.body FROM rollout_v4.executions e JOIN rollout_v4.workers w USING(worker_id)
               WHERE w.state IN ('LOST','QUARANTINED') AND e.claim IS NULL AND NOT e.cleanup_verified"""
        )
        for row in unstarted:
            view = _view(row)
            view.state = E.EXPIRED
            view.cleanup_verified = True
            await self._write(conn, view)

    async def _fail_uncertain(self, conn, view):
        view.state = E.FAILED
        view.failure_code = "EXECUTION_UNCERTAIN"
        view.resource_quarantined = True
        await conn.execute(
            "UPDATE rollout_v4.workers SET state='QUARANTINED' WHERE worker_id=$1 AND state<>'DRAINING'",
            view.worker_id,
        )
        return await self._write(conn, view)

    async def reconcile(self) -> None:
        async with self._transaction() as conn:
            await self._reconcile(conn)

    async def _allocations(self, conn) -> dict[str, AllocatedCharge]:
        allocated: dict[str, AllocatedCharge] = {}
        for row in await conn.fetch(
            "SELECT body FROM rollout_v4.executions WHERE NOT cleanup_verified"
        ):
            item = _view(row)
            if item.worker_id and item.resolved_resources:
                allocated[item.worker_id] = add_charge(
                    allocated.get(item.worker_id, AllocatedCharge()),
                    item.resolved_resources.charge,
                )
        return allocated

    async def reserve(self, request: ReserveRequest) -> ExecutionView:
        spec = request.spec
        digest = canonical_digest(spec)
        try:
            resources = self.resource_index.resolve(spec.task_ref)
        except ValueError as exc:
            raise ConflictError("task is not in the frozen resource index") from exc
        if not any(
            resources.charge.fits(w.virtual_capacity) for w in self.inventory.workers
        ):
            raise ConflictError("execution exceeds every configured Worker capacity")
        async with self._transaction() as conn:
            await self._reconcile(conn)
            row = await conn.fetchrow(
                "SELECT * FROM rollout_v4.executions WHERE execution_id=$1",
                spec.execution_id,
            )
            if row:
                view = _view(row)
                if view.client_scope != request.client_scope:
                    raise ConflictError("execution belongs to another client")
                if view.spec is None:  # cancel-before-submit tombstone
                    return view
                if view.request_digest != digest:
                    raise ConflictError("execution identity payload conflict")
                if view.reservation_request_id == request.reservation_request_id:
                    return view
                if request.reservation_request_id in row["reservation_keys"]:
                    raise ConflictError("reservation request was superseded")
                if view.state != E.EXPIRED:
                    raise ConflictError("execution cannot be reserved again")
            allocations = await self._allocations(conn)
            workers = await conn.fetch(
                "SELECT * FROM rollout_v4.workers WHERE state='READY'"
            )
            candidates = []
            for worker in workers:
                node = self.inventory.worker(worker["worker_id"])
                used = allocations.get(node.worker_id, AllocatedCharge())
                if resources.charge.fits(
                    available_capacity(node.virtual_capacity, used)
                ):
                    candidates.append(
                        (
                            dominant_load(node.virtual_capacity, used),
                            node.worker_id,
                            worker,
                        )
                    )
            if not candidates:
                raise CapacityError("no READY Worker can fit this execution")
            _, worker_id, worker = min(candidates, key=lambda item: item[:2])
            registration = WorkerRegistration.model_validate(
                _json(worker["registration"])
            )
            now = _now()
            view = ExecutionView(
                execution_id=spec.execution_id,
                client_scope=request.client_scope,
                spec=spec,
                state=E.RESERVED,
                request_digest=digest,
                reservation_request_id=request.reservation_request_id,
                reservation_id=uuid4().hex,
                reservation_expires_at=now
                + timedelta(seconds=self.admission.reservation_ttl_sec),
                resolved_resources=resources,
                resource_catalog_digest=self.resource_index.digest,
                worker_id=worker_id,
                worker_incarnation=registration.incarnation_id,
                state_version=(_view(row).state_version + 1 if row else 1),
                created_at=(_view(row).created_at if row else now),
                updated_at=now,
            )
            await conn.execute(
                """INSERT INTO rollout_v4.executions(execution_id,client_scope,state,worker_id,body,reservation_keys)
                   VALUES($1,$2,$3,$4,$5::jsonb,ARRAY[$6]::text[])
                   ON CONFLICT(execution_id) DO UPDATE SET state=$3,worker_id=$4,
                   body=$5::jsonb,cleanup_verified=false,
                   reservation_keys=rollout_v4.executions.reservation_keys || $6::text""",
                spec.execution_id,
                request.client_scope,
                E.RESERVED.value,
                worker_id,
                view.model_dump_json(),
                request.reservation_request_id,
            )
            return view

    async def activate(
        self, execution_id: str, request: ActivateRequest
    ) -> ExecutionView:
        async with self._transaction() as conn:
            await self._reconcile(conn)
            view = _view(await self._execution(conn, execution_id))
            self._scope(view, request)
            if view.reservation_id != request.reservation_id:
                raise ConflictError("reservation identity conflict")
            if view.context_id is not None:
                if view.context_id != request.context_id:
                    raise ConflictError("execution context is immutable")
                return view
            if view.state != E.EXPIRED:
                if view.state != E.RESERVED:
                    raise ConflictError("reservation is cancelled")
                worker, _ = await self._worker(
                    conn, view.worker_id, view.worker_incarnation
                )
                if worker["state"] != "READY":
                    raise ConflictError("reserved Worker is no longer READY")
                view.context_id = request.context_id
                view.state = E.ACCEPTED
                return await self._write(conn, view)
        # Commit expiry before rejecting the late activation. Raising inside
        # the transaction would undo its tombstone and resource release.
        raise ConflictError("reservation is expired")

    @staticmethod
    def _scope(view: ExecutionView, request: Any) -> None:
        if view.client_scope != request.client_scope:
            raise ConflictError("execution scope mismatch")

    async def get_execution(self, execution_id: str) -> ExecutionView:
        if self.pool is None:
            raise StoreError("store is not initialized")
        async with self.pool.acquire() as conn:
            row = await self._execution(conn, execution_id)
            view = _view(row)
            # Polling must not serialize 64 clients behind the write lock.
            # Expired ownership is still immediately visible, before the next sweep.
            if (
                not view.cleanup_verified
                and not view.failure_code
                and row["claim"]
                and row["lease_expires_at"] <= _now()
            ):
                view.state = E.FAILED
                view.failure_code = "EXECUTION_UNCERTAIN"
                view.resource_quarantined = False
            return view

    async def list_executions(
        self, scope: str, after_id: str = "", limit: int = 256
    ) -> list[ExecutionView]:
        if self.pool is None:
            raise StoreError("store is not initialized")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT body FROM rollout_v4.executions
                   WHERE client_scope=$1 AND execution_id>$2 ORDER BY execution_id LIMIT $3""",
                scope,
                after_id,
                limit,
            )
            return [_view(row) for row in rows]

    async def cancel(self, execution_id: str, request: CancelRequest) -> ExecutionView:
        async with self._transaction() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM rollout_v4.executions WHERE execution_id=$1",
                execution_id,
            )
            if row is None:
                now = _now()
                view = ExecutionView(
                    execution_id=execution_id,
                    client_scope=request.client_scope,
                    state=E.CANCELLED,
                    cleanup_verified=True,
                    created_at=now,
                    updated_at=now,
                )
                await conn.execute(
                    """INSERT INTO rollout_v4.executions(execution_id,client_scope,state,cleanup_verified,body)
                       VALUES($1,$2,'CANCELLED',true,$3::jsonb)""",
                    execution_id,
                    request.client_scope,
                    view.model_dump_json(),
                )
                return view
            view = _view(row)
            self._scope(view, request)
            if (
                request.quarantine
                and row["claim"] is not None
                and not view.cleanup_verified
            ):
                return await self._fail_uncertain(conn, view)
            if view.state in {
                E.SUCCEEDED,
                E.FAILED,
                E.CANCELLED,
                E.CANCELLING,
            }:
                return view
            view.state = E.CANCELLED if row["claim"] is None else E.CANCELLING
            view.cleanup_verified = row["claim"] is None
            return await self._write(conn, view)

    async def ack(self, execution_id: str, request: ResultAck) -> ExecutionView:
        async with self._transaction() as conn:
            view = _view(await self._execution(conn, execution_id))
            self._scope(view, request)
            if not view.result_digest or view.result_digest != request.result_digest:
                raise ConflictError("result ACK digest mismatch")
            if not view.result_acknowledged:
                view.result_acknowledged = True
                await self._write(conn, view)
            return view

    async def _worker(self, conn, worker_id: str, incarnation: str):
        row = await conn.fetchrow(
            "SELECT * FROM rollout_v4.workers WHERE worker_id=$1", worker_id
        )
        if not row:
            raise NotFoundError("Worker is not registered")
        registration = WorkerRegistration.model_validate(_json(row["registration"]))
        if registration.incarnation_id != incarnation:
            raise FencedError("Worker incarnation is stale")
        if (
            registration.config_digest != self.inventory.digest()
            or registration.task_resource_catalog_digest != self.resource_index.digest
        ):
            raise FencedError(
                "Worker frozen configuration changed; drain and redeploy required"
            )
        return row, registration

    async def register_worker(self, request: WorkerRegistration) -> WorkerView:
        node = self.inventory.worker(request.worker_id)
        if (
            request.config_digest != self.inventory.digest()
            or request.virtual_capacity != node.virtual_capacity
        ):
            raise ConflictError("Worker inventory mismatch")
        if request.task_resource_catalog_digest != self.resource_index.digest:
            raise ConflictError("Worker resource catalog mismatch")
        if "execution-cancel-v4" not in request.capabilities:
            raise ConflictError("Worker must support V4 cooperative cancellation")
        if "fixed-execution-deadline-v1" not in request.capabilities:
            raise ConflictError("Worker must support fixed execution deadlines")
        async with self._transaction() as conn:
            previous = await conn.fetchrow(
                "SELECT * FROM rollout_v4.workers WHERE worker_id=$1", request.worker_id
            )
            if (
                previous
                and _json(previous["registration"])["incarnation_id"]
                == request.incarnation_id
            ):
                if _json(previous["registration"]) != request.model_dump(mode="json"):
                    raise ConflictError("Worker registration identity conflict")
                return WorkerView(
                    worker_id=request.worker_id,
                    incarnation_id=request.incarnation_id,
                    state=previous["state"],
                    virtual_capacity=node.virtual_capacity,
                )
            if (
                previous
                and _json(previous["registration"])["incarnation_id"]
                != request.incarnation_id
            ):
                rows = await conn.fetch(
                    "SELECT * FROM rollout_v4.executions WHERE worker_id=$1 AND NOT cleanup_verified",
                    request.worker_id,
                )
                for row in rows:
                    view = _view(row)
                    if row["claim"]:
                        await self._fail_uncertain(conn, view)
                    else:
                        view.state = E.CANCELLED
                        view.cleanup_verified = True
                        await self._write(conn, view)
            await conn.execute(
                """INSERT INTO rollout_v4.workers(worker_id,registration,state,last_heartbeat)
                   VALUES($1,$2::jsonb,'READY',$3) ON CONFLICT(worker_id) DO UPDATE
                   SET registration=$2::jsonb,state='READY',last_heartbeat=$3""",
                request.worker_id,
                request.model_dump_json(),
                _now(),
            )
            pending = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM rollout_v4.executions WHERE worker_id=$1 AND NOT cleanup_verified AND body->>'failure_code'='EXECUTION_UNCERTAIN')",
                request.worker_id,
            )
            state = WorkerState.QUARANTINED if pending else WorkerState.READY
            await conn.execute(
                "UPDATE rollout_v4.workers SET state=$2 WHERE worker_id=$1",
                request.worker_id,
                state.value,
            )
            return WorkerView(
                worker_id=request.worker_id,
                incarnation_id=request.incarnation_id,
                state=state,
                virtual_capacity=node.virtual_capacity,
            )

    async def heartbeat(self, request: WorkerHeartbeat) -> WorkerView:
        async with self._transaction() as conn:
            row, registration = await self._worker(
                conn, request.worker_id, request.incarnation_id
            )
            if len(request.active_execution_ids) != len(
                set(request.active_execution_ids)
            ):
                raise ConflictError("duplicate active executions")
            for execution_id in request.active_execution_ids:
                execution = await self._execution(conn, execution_id)
                claim = _json(execution["claim"])
                if (
                    not claim
                    or claim["worker_id"] != request.worker_id
                    or claim["worker_incarnation"] != request.incarnation_id
                ):
                    raise FencedError("heartbeat execution ownership mismatch")
            state = (
                WorkerState.DRAINING
                if row["state"] == "DRAINING"
                else WorkerState.QUARANTINED
                if row["state"] == "QUARANTINED"
                else WorkerState.READY
            )
            await conn.execute(
                "UPDATE rollout_v4.workers SET state=$2,last_heartbeat=$3,active_execution_ids=$4::jsonb WHERE worker_id=$1",
                request.worker_id,
                state.value,
                _now(),
                json.dumps(request.active_execution_ids),
            )
            cancelled = await conn.fetch(
                """SELECT execution_id FROM rollout_v4.executions
                   WHERE worker_id=$1 AND NOT cleanup_verified
                   AND (state='CANCELLING' OR body->>'failure_code'='EXECUTION_UNCERTAIN')""",
                request.worker_id,
            )
            return WorkerView(
                worker_id=request.worker_id,
                incarnation_id=request.incarnation_id,
                state=state,
                virtual_capacity=registration.virtual_capacity,
                cancel_execution_ids=[item["execution_id"] for item in cancelled],
            )

    async def drain_worker(self, request: WorkerDrainRequest) -> WorkerView:
        async with self._transaction() as conn:
            _, registration = await self._worker(
                conn, request.worker_id, request.incarnation_id
            )
            await conn.execute(
                "UPDATE rollout_v4.workers SET state='DRAINING' WHERE worker_id=$1",
                request.worker_id,
            )
            return WorkerView(
                worker_id=request.worker_id,
                incarnation_id=request.incarnation_id,
                state=WorkerState.DRAINING,
                virtual_capacity=registration.virtual_capacity,
            )

    async def claim(self, request: ExecutionClaimRequest) -> ExecutionClaim | None:
        async with self._transaction() as conn:
            await self._reconcile(conn)
            worker, _ = await self._worker(
                conn, request.worker_id, request.incarnation_id
            )
            repeated = await conn.fetchrow(
                """SELECT claim,state,cleanup_verified,lease_expires_at FROM rollout_v4.executions WHERE worker_id=$1
                   AND claim->>'worker_incarnation'=$2 AND claim_request_id=$3""",
                request.worker_id,
                request.incarnation_id,
                request.claim_request_id,
            )
            if repeated:
                if (
                    repeated["cleanup_verified"]
                    or repeated["state"]
                    in {E.FAILED.value, E.CANCELLED.value, E.CANCELLING.value}
                    or repeated["lease_expires_at"] <= _now()
                ):
                    raise FencedError("replayed claim no longer authorizes execution")
                return ExecutionClaim.model_validate(_json(repeated["claim"]))
            if worker["state"] != "READY":
                return None
            row = await conn.fetchrow(
                """SELECT * FROM rollout_v4.executions WHERE worker_id=$1
                   AND state='ACCEPTED' ORDER BY sequence_no LIMIT 1""",
                request.worker_id,
            )
            if row is None:
                return None
            view = _view(row)
            if view.worker_incarnation != request.incarnation_id:
                raise FencedError("assignment Worker incarnation changed")
            execution_deadline = _now() + timedelta(
                seconds=self.inventory.execution_timeout_sec
            )
            claim = ExecutionClaim(
                execution=view.spec,
                client_scope=view.client_scope,
                context_id=view.context_id,
                reservation_id=view.reservation_id,
                resolved_resources=view.resolved_resources,
                resource_catalog_digest=view.resource_catalog_digest,
                worker_id=request.worker_id,
                worker_incarnation=request.incarnation_id,
                lease_token=uuid4().hex,
                execution_deadline_at=execution_deadline,
                lease_expires_at=execution_deadline
                + timedelta(seconds=self.inventory.settlement_timeout_sec),
            )
            await conn.execute(
                """UPDATE rollout_v4.executions SET claim=$2::jsonb,claim_request_id=$3,lease_expires_at=$4
                   WHERE execution_id=$1""",
                view.execution_id,
                claim.model_dump_json(),
                request.claim_request_id,
                claim.lease_expires_at,
            )
            view.state = E.PREPARING
            view.claim_digest = canonical_digest(
                ExecutionIdentity(
                    worker_id=claim.worker_id,
                    worker_incarnation=claim.worker_incarnation,
                    lease_token=claim.lease_token,
                )
            )
            await self._write(conn, view)
            return claim

    async def _lease(
        self,
        conn,
        execution_id: str,
        request: ExecutionIdentity,
    ):
        row = await self._execution(conn, execution_id)
        claim = _json(row["claim"])
        if claim is None or any(
            claim[key] != getattr(request, key)
            for key in ("worker_id", "worker_incarnation", "lease_token")
        ):
            raise FencedError("execution lease identity mismatch")
        view = _view(row)
        await self._worker(conn, request.worker_id, request.worker_incarnation)
        if row["lease_expires_at"] <= _now() or view.failure_code:
            raise FencedError("execution lease expired or ownership unknown")
        return row, view

    async def transition(self, execution_id: str, request: ExecutionTransition) -> None:
        async with self._transaction() as conn:
            _, view = await self._lease(conn, execution_id, request)
            if view.state != request.state and request.state not in _TRANSITIONS.get(
                view.state, set()
            ):
                raise ConflictError("illegal execution transition")
            if view.trial_id and request.trial_id and view.trial_id != request.trial_id:
                raise ConflictError("trial identity changed")
            view.state = request.state
            view.trial_id = request.trial_id or view.trial_id
            view.trial_name = request.trial_name or view.trial_name
            await self._write(conn, view)

    async def complete(
        self, execution_id: str, request: ExecutionResult
    ) -> ExecutionView:
        async with self._transaction() as conn:
            row = await self._execution(conn, execution_id)
            view = _view(row)
            digest = canonical_digest(request)
            if view.failure_code:
                raise FencedError(
                    "execution has been abandoned; discard late completion"
                )
            if view.result is not None:
                if view.result_digest != digest:
                    raise ConflictError("completion result identity conflict")
                return view
            _, view = await self._lease(conn, execution_id, request)
            if view.state in {E.CANCELLING, E.CANCELLED}:
                raise ConflictError("execution was cancelled before completion")
            if request.state == E.SUCCEEDED and (
                view.state != E.VERIFYING
                or view.trial_id != request.trial_id
                or "reward" not in request.rewards
            ):
                raise ConflictError("successful result lacks verifier/trial proof")
            view.state = request.state
            view.trial_id = request.trial_id
            view.trial_name = request.trial_name
            view.result = request
            view.result_digest = digest
            return await self._write(conn, view)

    async def cleanup(self, execution_id: str, request: CleanupReport) -> None:
        async with self._transaction() as conn:
            row = await self._execution(conn, execution_id)
            claim = _json(row["claim"])
            if not claim or any(
                claim[k] != getattr(request, k)
                for k in ("worker_id", "worker_incarnation", "lease_token")
            ):
                raise FencedError("cleanup claim identity mismatch")
            view = _view(row)
            await self._worker(conn, request.worker_id, request.worker_incarnation)
            if view.cleanup_verified:
                return
            if view.state == E.CANCELLING:
                view.state = E.CANCELLED
            if view.state not in TERMINAL_EXECUTION_STATES:
                raise ConflictError("execution is not settled")
            view.cleanup_verified = True
            view.resource_quarantined = False
            await self._write(conn, view)
            await self._unquarantine(conn, view.worker_id)

    async def capacity(self) -> CapacityView:
        async with self._transaction() as conn:
            await self._reconcile(conn)
            allocations = await self._allocations(conn)
            workers = {
                row["worker_id"]: row
                for row in await conn.fetch("SELECT * FROM rollout_v4.workers")
            }
            counts = {
                row["state"]: row["n"]
                for row in await conn.fetch(
                    "SELECT state,count(*) n FROM rollout_v4.executions WHERE NOT cleanup_verified GROUP BY state"
                )
            }
            return CapacityView(
                workers=[
                    CapacityWorker(
                        worker_id=node.worker_id,
                        state=workers[node.worker_id]["state"]
                        if node.worker_id in workers
                        else WorkerState.NOT_READY,
                        labels=node.labels,
                        virtual_capacity=node.virtual_capacity,
                        allocated_charge=allocations.get(
                            node.worker_id, AllocatedCharge()
                        ),
                    )
                    for node in self.inventory.workers
                ],
                reserved_executions=counts.get("RESERVED", 0),
                queued_executions=counts.get("ACCEPTED", 0),
                cleanup_pending_executions=await conn.fetchval(
                    "SELECT count(*) FROM rollout_v4.executions WHERE NOT cleanup_verified AND body->>'failure_code'='EXECUTION_UNCERTAIN'"
                ),
            )

    async def pending_cleanup(self, worker_id: str, incarnation_id: str):
        async with self._transaction() as conn:
            await self._worker(conn, worker_id, incarnation_id)
            await self._reconcile(conn)
            rows = await conn.fetch(
                """SELECT body FROM rollout_v4.executions WHERE worker_id=$1
                   AND NOT cleanup_verified AND body->>'failure_code'='EXECUTION_UNCERTAIN'
                   ORDER BY sequence_no LIMIT 512""",
                worker_id,
            )
            return [_view(row) for row in rows]

    async def abandon_execution(self, execution_id: str, request: ExecutionIdentity):
        async with self._transaction() as conn:
            await self._worker(conn, request.worker_id, request.worker_incarnation)
            row = await self._execution(conn, execution_id)
            claim = _json(row["claim"])
            if not claim or any(
                claim[k] != getattr(request, k)
                for k in ("worker_id", "worker_incarnation", "lease_token")
            ):
                raise FencedError("abandon claim identity mismatch")
            view = _view(row)
            if view.cleanup_verified or view.failure_code:
                return view
            return await self._fail_uncertain(conn, view)

    async def _unquarantine(self, conn, worker_id):
        await conn.execute(
            """UPDATE rollout_v4.workers SET state='READY'
               WHERE worker_id=$1 AND state='QUARANTINED' AND last_heartbeat>$2
               AND NOT EXISTS(SELECT 1 FROM rollout_v4.executions WHERE worker_id=$1
                 AND NOT cleanup_verified AND body->>'failure_code'='EXECUTION_UNCERTAIN')""",
            worker_id,
            _now() - timedelta(seconds=self.inventory.heartbeat_timeout_sec),
        )

    async def recover_execution(
        self, execution_id: str, proof: RecoveryProof
    ) -> ExecutionView:
        async with self._transaction() as conn:
            await self._worker(conn, proof.worker_id, proof.incarnation_id)
            row = await self._execution(conn, execution_id)
            view = _view(row)
            if (
                view.worker_id != proof.worker_id
                or view.worker_incarnation != proof.worker_incarnation
            ):
                raise ConflictError("cleanup recovery identity mismatch")
            if view.cleanup_verified:
                return view
            if view.failure_code != "EXECUTION_UNCERTAIN":
                raise ConflictError("execution is not quarantined")
            view.cleanup_verified = True
            view.resource_quarantined = False
            await conn.execute(
                "INSERT INTO rollout_v4.recovery_audit(execution_id,evidence_digest) VALUES($1,$2) ON CONFLICT DO NOTHING",
                execution_id,
                proof.evidence_digest,
            )
            await self._write(conn, view)
            await self._unquarantine(conn, view.worker_id)
            return view
