from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from agentic_rollout_contract.canonical import canonical_digest
from agentic_rollout_contract.client import RolloutServiceClient
from agentic_rollout_contract.models import (
    ContextBindRequest,
    ContextPrepareRequest,
    ExecutionIdentity,
    ExecutionState,
    RouteControlRequest,
    RouteView,
)
from fastapi import FastAPI, HTTPException, Request, Response

from .config import PolicySettings
from .runtime_debug import event

logger = logging.getLogger(__name__)


@dataclass
class _RouteEntry:
    request: ContextPrepareRequest
    request_digest: str
    handle: Any = None
    binding: ContextBindRequest | None = None
    state: str = "PREPARED"
    actions: dict[str, _ActionRecord] = field(default_factory=dict)
    finalize_task: asyncio.Task[Any] | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed_at: float | None = None
    gateway_close_failed: bool = False
    revoking: bool = False
    trajectory_retained: bool = False
    cleanup_phase: str = "idle"


@dataclass
class _ActionRecord:
    digest: str
    task: asyncio.Task[RouteView]


@dataclass(frozen=True)
class _BufferedResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes


@dataclass
class _LedgerRecord:
    digest: str
    task: asyncio.Task[_BufferedResponse]
    accounted_bytes: int


class _LedgerCapacityError(RuntimeError):
    pass


class _ResponseTooLargeError(RuntimeError):
    pass


def _verify_gateway_source(module: Any) -> None:
    """Refuse a private-binding drain proof against an unpinned Gateway build."""

    lock_path = Path(
        os.environ.get(
            "GPU_RUNTIME_LOCK",
            str(Path(__file__).resolve().parents[3] / "configs/gpu-runtime-lock.json"),
        )
    )
    source_path = Path(module.__file__).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = lock["source_hashes"]["dressage"][
        "dressage/integrations/harbor/gateway.py"
    ]
    actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("Dressage Gateway source differs from pinned drain adapter")


class _RequestLedger:
    def __init__(
        self,
        *,
        max_entries: int = 8192,
        max_entries_per_route: int = 256,
        max_bytes: int = 4 * 1024 * 1024 * 1024,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[tuple[str, str], _LedgerRecord] = {}
        self._route_counts: dict[str, int] = {}
        self._accounted_bytes = 0
        self._max_entries = max_entries
        self._max_entries_per_route = max_entries_per_route
        self._max_bytes = max_bytes
        self._max_response_bytes = max_response_bytes

    async def acquire(
        self,
        route_id: str,
        request_id: str,
        digest: str,
        operation: Callable[[], Awaitable[_BufferedResponse]],
    ) -> asyncio.Task[_BufferedResponse]:
        key = (route_id, request_id)
        async with self._lock:
            existing = self._tasks.get(key)
            if existing is not None:
                if existing.digest != digest:
                    raise ValueError("model request id digest conflict")
                return existing.task
            route_count = self._route_counts.get(route_id, 0)
            if len(self._tasks) >= self._max_entries:
                raise _LedgerCapacityError("model request ledger entry limit reached")
            if route_count >= self._max_entries_per_route:
                raise _LedgerCapacityError(
                    "model request ledger per-route entry limit reached"
                )
            if self._accounted_bytes + self._max_response_bytes > self._max_bytes:
                raise _LedgerCapacityError("model request ledger byte limit reached")
            self._accounted_bytes += self._max_response_bytes
            self._route_counts[route_id] = route_count + 1
            task = asyncio.create_task(
                self._run(key, operation), name=f"policy-{route_id}-{request_id[:10]}"
            )
            self._tasks[key] = _LedgerRecord(
                digest=digest,
                task=task,
                accounted_bytes=self._max_response_bytes,
            )
            return task

    async def execute(
        self,
        route_id: str,
        request_id: str,
        digest: str,
        operation: Callable[[], Awaitable[_BufferedResponse]],
    ) -> _BufferedResponse:
        task = await self.acquire(route_id, request_id, digest, operation)
        return await asyncio.shield(task)

    async def _run(
        self,
        key: tuple[str, str],
        operation: Callable[[], Awaitable[_BufferedResponse]],
    ) -> _BufferedResponse:
        actual_bytes = 0
        try:
            response = await operation()
            actual_bytes = len(response.body)
            if actual_bytes > self._max_response_bytes:
                raise _ResponseTooLargeError(
                    f"model response exceeds {self._max_response_bytes} bytes"
                )
            return response
        finally:
            async with self._lock:
                record = self._tasks.get(key)
                if record is not None and record.task is asyncio.current_task():
                    self._accounted_bytes += actual_bytes - record.accounted_bytes
                    record.accounted_bytes = actual_bytes

    async def wait_route_idle(self, route_id: str) -> list[BaseException]:
        async with self._lock:
            tasks = [
                record.task for key, record in self._tasks.items() if key[0] == route_id
            ]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result for result in results if isinstance(result, BaseException)]

    async def clear_route(self, route_id: str) -> None:
        tasks: list[asyncio.Task[_BufferedResponse]] = []
        async with self._lock:
            for key in [key for key in self._tasks if key[0] == route_id]:
                record = self._tasks.pop(key)
                self._accounted_bytes -= record.accounted_bytes
                tasks.append(record.task)
            self._route_counts.pop(route_id, None)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        async with self._lock:
            route_ids = set(self._route_counts)
        for route_id in route_ids:
            await self.clear_route(route_id)


class PolicyRuntime:
    def __init__(self, settings: PolicySettings) -> None:
        self.settings = settings
        self.instance_id = str(uuid4())
        self.retired: dict[str, tuple[str, RouteView]] = {}
        self.control_client = RolloutServiceClient(
            settings.control_plane_url,
            timeout=httpx.Timeout(30.0, connect=15.0),
        )
        self.gateway_runtime: Any = None
        self.gateway_lease: Any = None
        self.integration_config: Any = None
        self.routes: dict[str, _RouteEntry] = {}
        self.lock = asyncio.Lock()
        self.ledger = _RequestLedger(
            max_entries=settings.ledger_max_entries,
            max_entries_per_route=settings.ledger_max_entries_per_route,
            max_bytes=settings.ledger_max_bytes,
            max_response_bytes=settings.max_model_response_bytes,
        )
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(7200.0, connect=15.0), trust_env=False
        )
        self.proxy_client: Any = None

    async def start(self) -> None:
        from dressage.integrations.harbor import gateway as gateway_module
        from dressage.integrations.harbor.config import (
            BackendConfig,
            GatewayConfig,
            GatewayLimitsConfig,
            HarborIntegrationConfig,
            SecurityConfig,
        )
        from dressage.integrations.harbor.gateway import GatewayRuntime
        from dressage.proxy.proxy_client import ProxyClient

        from .protocol_proxy import AgentProtocolProxy

        _verify_gateway_source(gateway_module)

        config = HarborIntegrationConfig(
            execution_mode="rollout",
            gateway=GatewayConfig(
                listen_host=self.settings.internal_gateway_host,
                listen_port=self.settings.internal_gateway_port,
                limits=GatewayLimitsConfig(
                    max_active_routes=256,
                    max_inflight_global=512,
                    max_inflight_per_route=8,
                    route_ttl_sec=14400,
                    tombstone_ttl_sec=3600,
                    sse_idle_timeout_sec=1800,
                ),
            ),
            backend=BackendConfig(
                dressage_proxy_url=self.settings.dressage_proxy_url,
            ),
            security=SecurityConfig(
                routing_guarantee="configure_only",
                require_tls_for_non_loopback=True,
            ),
        )
        # Supplying the factory explicitly avoids Dressage's optional Harbor 0.18 check.
        self.gateway_runtime = GatewayRuntime(proxy_factory=AgentProtocolProxy)
        self.integration_config = config
        self.gateway_lease = await self.gateway_runtime.acquire(config)
        self.proxy_client = ProxyClient(self.settings.dressage_proxy_url)
        event(
            "policy_protocol_ready",
            policy_instance_id=self.instance_id,
            adapter="AgentProtocolProxy",
            cleanup_timeout_sec=self.settings.route_cleanup_timeout_sec,
        )

    async def close(self) -> None:
        async with self.lock:
            entries = list(self.routes.values())
            self.routes.clear()
        await self.ledger.close()
        action_tasks = [
            record.task for entry in entries for record in entry.actions.values()
        ]
        if action_tasks:
            await asyncio.gather(*action_tasks, return_exceptions=True)
        finalize_tasks = [
            entry.finalize_task for entry in entries if entry.finalize_task is not None
        ]
        if finalize_tasks:
            await asyncio.gather(*finalize_tasks, return_exceptions=True)
        await asyncio.gather(
            *(
                entry.handle.close(tombstone=True)
                for entry in entries
                if entry.handle is not None
            ),
            return_exceptions=True,
        )
        if self.proxy_client is not None:
            await asyncio.gather(
                *(self._discard_session(entry) for entry in entries),
                return_exceptions=True,
            )
        if self.gateway_lease is not None:
            await self.gateway_lease.release()
        if self.proxy_client is not None:
            await self.proxy_client.close()
        await self.client.aclose()
        await self.control_client.close()

    async def prepare(self, request: ContextPrepareRequest) -> RouteView:
        if request.policy_instance_id != self.instance_id:
            raise ValueError(
                "Policy incarnation changed; reconcile before resubmission"
            )
        digest = canonical_digest(request)
        async with self.lock:
            self._prune_closed_routes_locked()
            existing = self.routes.get(request.context_id)
            if existing is not None:
                if existing.request_digest != digest:
                    raise ValueError("context idempotency conflict")
                return self._view(existing)
            if request.context_id in self.retired:
                previous_digest, view = self.retired[request.context_id]
                if previous_digest != digest:
                    raise ValueError("retired context idempotency conflict")
                return view
            if len(self.retired) >= 1_000_000:
                raise _LedgerCapacityError("context tombstone capacity exhausted")
            if sum(entry.state != "CLOSED" for entry in self.routes.values()) >= 256:
                raise _LedgerCapacityError("context capacity exhausted")
            if (
                sum(entry.trajectory_retained for entry in self.routes.values())
                >= self.settings.max_closed_routes
            ):
                raise _LedgerCapacityError(
                    "retained trajectory capacity exhausted; release old batch"
                )
            entry = _RouteEntry(request=request, request_digest=digest)
            self.routes[request.context_id] = entry
            return self._view(entry)

    async def bind(self, context_id: str, binding: ContextBindRequest) -> RouteView:
        from dressage.integrations.harbor.gateway import RouteSpec

        entry = self.routes.get(context_id)
        if entry is None:
            raise KeyError(context_id)
        async with entry.operation_lock:
            if entry.state in {"CLOSED", "REVOKING", "BROKEN"}:
                raise ValueError("context is no longer bindable")
            if entry.binding is not None:
                if entry.binding != binding:
                    raise ValueError("context trial/lease binding conflict")
                return self._view(entry)
            request = entry.request
            if (
                binding.execution_id != request.execution_id
                or binding.reservation_id != request.reservation_id
            ):
                raise ValueError("context execution/reservation mismatch")
            execution = await self.control_client.get_execution(binding.execution_id)
            proof = ExecutionIdentity(
                worker_id=binding.worker_id,
                worker_incarnation=binding.worker_incarnation,
                lease_token=binding.lease_token,
            )
            if (
                execution.context_id != context_id
                or execution.reservation_id != request.reservation_id
                or execution.client_scope != request.client_scope
                or execution.state
                not in {ExecutionState.PREPARING, ExecutionState.WAITING_ROUTE}
                or execution.claim_digest != canonical_digest(proof)
                or execution.trial_id != binding.trial_id
            ):
                raise ValueError(
                    "CPU did not confirm an active matching execution lease"
                )
            spec = RouteSpec(
                trial_name=binding.trial_name,
                trial_id=binding.trial_id,
                instance_id=request.instance_id,
                token=context_id,
                upstream_origin=self.settings.dressage_proxy_url,
                model_override=request.model_override,
                expected_version=request.expected_weight_version,
                sampling_mode="force",
                sampling_temperature=request.sampling_temperature,
                sampling_top_p=request.sampling_top_p,
                sampling_seed_base=request.sampling_seed_base,
                max_steps=200,
                upstream_headers=self.integration_config.backend.service_headers(
                    required=False
                ),
            )
            entry.handle = await self.gateway_lease.register(spec)
            entry.binding = binding
            entry.state = "READY"
            return self._view(entry)

    async def control(
        self, route_id: str, action: str, request: RouteControlRequest
    ) -> RouteView:
        if not request.action_id or len(request.action_id) > 256:
            raise ValueError("route action_id must contain 1 to 256 characters")
        action_digest = canonical_digest(
            {"action": action, "request": request.model_dump(mode="json")}
        )
        async with self.lock:
            self._prune_closed_routes_locked()
            entry = self.routes.get(route_id)
            if entry is None:
                retired = self.retired.get(route_id)
                if (
                    retired
                    and action == "revoke"
                    and retired[1].execution_id == request.execution_id
                ):
                    return retired[1]
                raise KeyError(route_id)
            if request.execution_id != entry.request.execution_id:
                raise ValueError("route control attempt identity mismatch")
            existing = entry.actions.get(request.action_id)
            if existing is not None:
                if existing.digest != action_digest:
                    raise ValueError("route action idempotency conflict")
                # A revoke can safely resume only by re-proving Gateway idle and
                # repeating the idempotent close/discard sequence. Keep its
                # action digest so a different request cannot reuse this ID.
                if action in {"revoke", "detach"} and existing.task.done():
                    failed = existing.task.cancelled() or (
                        existing.task.exception() is not None
                    )
                    if failed:
                        existing.task = asyncio.create_task(
                            self._run_control_action(entry, route_id, action, request),
                            name=f"route-action-{route_id}-{request.action_id[:12]}",
                        )
                task = existing.task
            else:
                if len(entry.actions) >= self.settings.max_actions_per_route:
                    raise _LedgerCapacityError("route action ledger limit reached")
                task = asyncio.create_task(
                    self._run_control_action(entry, route_id, action, request),
                    name=f"route-action-{route_id}-{request.action_id[:12]}",
                )
                entry.actions[request.action_id] = _ActionRecord(
                    digest=action_digest, task=task
                )
        return await asyncio.shield(task)

    async def _run_control_action(
        self,
        entry: _RouteEntry,
        route_id: str,
        action: str,
        request: RouteControlRequest,
    ) -> RouteView:
        started = time.monotonic()
        identity = {
            "context_id": route_id,
            "execution_id": entry.request.execution_id,
            "action": action,
            "action_id": request.action_id,
        }
        event("route_action_start", **identity, state=entry.state)
        try:
            budget = (
                self.settings.route_cleanup_timeout_sec
                if action in {"revoke", "detach"}
                else None
            )
            async with asyncio.timeout(budget):
                result = await self._perform_control_action(
                    entry, route_id, action, request
                )
            event(
                "route_action_end",
                **identity,
                state=entry.state,
                elapsed_sec=round(time.monotonic() - started, 3),
            )
            return result
        except BaseException as exc:
            if action in {"revoke", "detach"} and entry.state != "CLOSED":
                entry.state = "BROKEN"
            event(
                "route_action_failed",
                **identity,
                state=entry.state,
                phase=entry.cleanup_phase,
                error_type=type(exc).__name__,
                elapsed_sec=round(time.monotonic() - started, 3),
            )
            raise

    def _cleanup_phase(self, entry: _RouteEntry, phase: str) -> None:
        entry.cleanup_phase = phase
        event(
            "route_cleanup_phase",
            context_id=entry.request.context_id,
            execution_id=entry.request.execution_id,
            phase=phase,
            state=entry.state,
        )

    async def _perform_control_action(
        self,
        entry: _RouteEntry,
        route_id: str,
        action: str,
        request: RouteControlRequest,
    ) -> RouteView:
        if (
            action in {"revoke", "detach"}
            and entry.state != "CLOSED"
            and not entry.gateway_close_failed
        ):
            # A seal may own operation_lock while waiting for accepted model
            # requests. Cancel those requests before waiting for that lock;
            # otherwise early-stop could sit behind the very work it cancels.
            entry.revoking = True
            self._cleanup_phase(entry, "cancel_ledger_before_lock")
            await self.ledger.clear_route(route_id)
        if action in {"revoke", "detach"}:
            self._cleanup_phase(entry, "operation_lock")
        async with entry.operation_lock:
            if action == "open-turn":
                await self._open_turn(entry, request)
            elif action == "quiesce":
                await self._quiesce(entry, route_id)
            elif action == "seal":
                await self._seal(entry, route_id)
            elif action == "revoke":
                await self._revoke(entry, route_id)
            elif action == "detach":
                await self._revoke(entry, route_id, retain=True)
            else:
                raise ValueError(f"unsupported route action {action!r}")
            return self._view(entry)

    async def _open_turn(
        self, entry: _RouteEntry, request: RouteControlRequest
    ) -> None:
        if entry.state != "READY" or entry.revoking:
            raise ValueError(f"cannot open turn while route is {entry.state}")
        if not request.turn_id:
            raise ValueError("turn_id is required")
        try:
            await entry.handle.open_turn(
                request.turn_id, backend_session_id=request.backend_session_id
            )
        except BaseException:
            entry.state = "BROKEN"
            raise
        entry.state = "ACTIVE"

    async def _quiesce(self, entry: _RouteEntry, route_id: str) -> None:
        if entry.state == "READY":
            await self.ledger.clear_route(route_id)
            return
        if entry.state != "ACTIVE":
            raise ValueError(f"cannot quiesce while route is {entry.state}")
        entry.state = "QUIESCING"
        ledger_errors = await self.ledger.wait_route_idle(route_id)
        if ledger_errors:
            entry.state = "BROKEN"
            event(
                "route_quiesce_rejected",
                context_id=route_id,
                reason="request_failure",
                error_types=[type(e).__name__ for e in ledger_errors],
            )
            raise ValueError(
                f"{len(ledger_errors)} accepted model request(s) failed before quiesce"
            )
        try:
            errors = await entry.handle.quiesce()
        except BaseException:
            entry.state = "BROKEN"
            raise
        if errors:
            entry.state = "BROKEN"
            event(
                "route_quiesce_rejected",
                context_id=route_id,
                reason="gateway_failure",
                error_types=sorted(str(name) for name in errors),
            )
            raise ValueError(
                "Dressage route quiesce reported fatal errors: "
                + ", ".join(sorted(str(name) for name in errors))
            )
        entry.state = "READY"
        await self.ledger.clear_route(route_id)

    async def _seal(self, entry: _RouteEntry, route_id: str) -> None:
        if entry.state == "ACTIVE":
            await self._quiesce(entry, route_id)
        elif entry.state == "READY":
            await self.ledger.clear_route(route_id)
        elif entry.state != "SEALED":
            raise ValueError(f"cannot seal while route is {entry.state}")
        entry.state = "SEALED"
        self._ensure_finalize_task(entry, route_id)

    async def _revoke(
        self, entry: _RouteEntry, route_id: str, *, retain: bool = False
    ) -> None:
        if entry.state == "CLOSED":
            if retain and not entry.trajectory_retained:
                raise ValueError("discarded trajectory cannot be retained again")
            if not retain and entry.trajectory_retained:
                await self._discard_session(entry, strict=True)
                entry.trajectory_retained = False
            return
        if entry.handle is None:
            entry.state = "CLOSED"
            entry.closed_at = time.monotonic()
            return
        if entry.gateway_close_failed:
            raise RuntimeError(
                "Gateway close previously failed; route cannot be safely reopened"
            )
        if retain:
            if entry.binding is None:
                raise ValueError("cannot retain an unbound trajectory")
            self._cleanup_phase(entry, "finalize_retained_trajectory")
            finalized = await self._proxy_call_with_retry(
                "finalize",
                lambda: self.proxy_client.finalize_session(
                    entry.binding.trial_id, instance_id=entry.request.instance_id
                ),
            )
            if finalized.get("success") is not True:
                raise RuntimeError(
                    "trajectory retention requires successful finalization"
                )
        entry.state = "REVOKING"
        self._cleanup_phase(entry, "cancel_ledger")
        await self.ledger.clear_route(route_id)
        if entry.finalize_task is not None:
            self._cleanup_phase(entry, "background_finalize")
            results = await asyncio.gather(
                asyncio.shield(entry.finalize_task), return_exceptions=True
            )
            if isinstance(results[0], BaseException):
                logger.warning(
                    "background finalize failed before route revoke: route=%s error=%r",
                    route_id,
                    results[0],
                )
        try:
            self._cleanup_phase(entry, "gateway_idle")
            await self._wait_gateway_route_idle(entry)
            try:
                self._cleanup_phase(entry, "gateway_close")
                await entry.handle.close(tombstone=True)
            except BaseException:
                # Gateway may have marked its binding CLOSED before a failed
                # lifespan exit. A later close() could become a no-op; never
                # turn that ambiguity into a successful cancellation barrier.
                entry.gateway_close_failed = True
                raise
            if retain:
                # Close generation ownership without deleting finalized TQ fields.
                # RL later revokes this exact context after consumption/rejection.
                entry.trajectory_retained = True
            else:
                self._cleanup_phase(entry, "proxy_discard")
                await self._discard_session(entry, strict=True)
        except BaseException:
            entry.state = "BROKEN"
            raise
        entry.state = "CLOSED"
        entry.closed_at = time.monotonic()
        self._cleanup_phase(entry, "closed")

    async def _wait_gateway_route_idle(self, entry: _RouteEntry) -> None:
        """Read the pinned Gateway binding only on its owning event loop."""

        binding = getattr(entry.handle, "_binding", None)
        zero = getattr(binding, "inflight_zero", None)
        if (
            not isinstance(zero, asyncio.Event)
            or not isinstance(getattr(binding, "inflight", None), int)
            or self.gateway_runtime is None
            or not callable(getattr(self.gateway_runtime, "call", None))
        ):
            raise RuntimeError("pinned Gateway route idle proof is unavailable")

        async def on_gateway_loop() -> int:
            await asyncio.wait_for(
                zero.wait(), timeout=self.settings.route_drain_timeout_sec
            )
            inflight = binding.inflight
            if inflight != 0 or not zero.is_set():
                raise RuntimeError("Gateway route still has inflight requests")
            return inflight

        started = time.monotonic()
        try:
            remaining = await self.gateway_runtime.call(on_gateway_loop)
        except TimeoutError:
            logger.error(
                "Gateway route drain timed out: route=%s elapsed_sec=%.3f",
                entry.request.context_id,
                time.monotonic() - started,
            )
            raise
        logger.info(
            "Gateway route idle proved: route=%s inflight=%d elapsed_sec=%.3f",
            entry.request.context_id,
            remaining,
            time.monotonic() - started,
        )

    def _ensure_finalize_task(self, entry: _RouteEntry, route_id: str) -> None:
        if entry.finalize_task is not None:
            if not entry.finalize_task.done():
                return
            if not entry.finalize_task.cancelled():
                try:
                    if entry.finalize_task.exception() is None:
                        return
                except asyncio.CancelledError:
                    pass
        entry.finalize_task = asyncio.create_task(
            self._proxy_call_with_retry(
                "finalize",
                lambda: self.proxy_client.finalize_session(
                    entry.binding.trial_id,
                    instance_id=entry.request.instance_id,
                ),
            ),
            name=f"finalize-{route_id}",
        )

    async def _discard_session(
        self, entry: _RouteEntry, *, strict: bool = False
    ) -> None:
        if entry.binding is None:
            return
        try:
            result = await asyncio.wait_for(
                self._proxy_call_with_retry(
                    "discard",
                    lambda: self.proxy_client.discard_session(entry.binding.trial_id),
                ),
                timeout=self.settings.proxy_discard_timeout_sec,
            )
            if not isinstance(result, dict) or result.get("success") is not True:
                raise RuntimeError("Dressage Proxy did not confirm session discard")
        except Exception:
            logger.exception(
                "failed to discard Dressage session while revoking route %s",
                entry.request.context_id,
            )
            if strict:
                raise

    async def _proxy_call_with_retry(
        self, name: str, operation: Callable[[], Awaitable[Any]]
    ) -> Any:
        delay = self.settings.proxy_retry_base_sec
        for attempt in range(self.settings.proxy_retry_attempts):
            try:
                return await operation()
            except httpx.HTTPError as exc:
                if (
                    attempt + 1 >= self.settings.proxy_retry_attempts
                    or not _retryable_http_error(exc)
                ):
                    raise
                logger.warning(
                    "Dressage proxy %s failed; retrying (%d/%d)",
                    name,
                    attempt + 1,
                    self.settings.proxy_retry_attempts,
                )
                if delay:
                    await asyncio.sleep(delay)
                delay = min(
                    max(delay * 2, self.settings.proxy_retry_base_sec),
                    self.settings.proxy_retry_max_sec,
                )

    def _prune_closed_routes_locked(self) -> None:
        now = time.monotonic()
        closed = sorted(
            (
                (entry.closed_at, route_id)
                for route_id, entry in self.routes.items()
                if entry.closed_at is not None and not entry.trajectory_retained
            ),
            key=lambda item: item[0],
        )
        expired = {
            route_id
            for closed_at, route_id in closed
            if now - closed_at >= self.settings.closed_route_ttl_sec
        }
        excess = max(0, len(closed) - self.settings.max_closed_routes)
        expired.update(route_id for _, route_id in closed[:excess])
        for route_id in expired:
            entry = self.routes.pop(route_id)
            self.retired[route_id] = (entry.request_digest, self._view(entry))

    async def forward(self, route_id: str, path: str, request: Request) -> Response:
        async with self.lock:
            entry = self.routes.get(route_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="unknown route")
        normalized = path.strip("/")
        if normalized not in {"v1/messages", "v1/chat/completions", "v1/responses"}:
            raise HTTPException(status_code=404, detail="unsupported model path")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="invalid Content-Length"
                ) from exc
            if declared_length > self.settings.max_model_request_bytes:
                raise HTTPException(
                    status_code=413, detail="model request is too large"
                )
        body = await request.body()
        if len(body) > self.settings.max_model_request_bytes:
            raise HTTPException(status_code=413, detail="model request is too large")
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        supplied_digest = request.headers.get("x-agentic-rollout-request-digest")
        if not supplied_digest:
            raise HTTPException(
                status_code=400, detail="stable model request digest is required"
            )
        if supplied_digest != digest:
            raise HTTPException(status_code=409, detail="model request digest mismatch")
        request_id = request.headers.get("x-agentic-rollout-request-id")
        if not request_id:
            raise HTTPException(
                status_code=400, detail="stable model request id is required"
            )
        if len(request_id) > 256:
            raise HTTPException(status_code=400, detail="model request id is too long")

        ledger_digest = canonical_digest(
            {
                "path": normalized,
                "body_digest": digest,
                "content_type": request.headers.get("content-type"),
                "anthropic_version": request.headers.get("anthropic-version"),
            }
        )

        async def operation() -> _BufferedResponse:
            event(
                "model_request_accepted",
                execution_id=entry.request.execution_id,
                session_id=entry.binding.trial_id if entry.binding else None,
                request_id=request_id,
                body_sha256=digest.removeprefix("sha256:"),
            )
            headers = {
                key: value
                for key, value in request.headers.items()
                if key.lower()
                not in {
                    "authorization",
                    "x-api-key",
                    "host",
                    "content-length",
                    "transfer-encoding",
                }
            }
            headers["Authorization"] = f"Bearer {entry.request.context_id}"
            async with self.client.stream(
                "POST",
                f"{self.gateway_runtime.advertise_url}/{normalized}",
                content=body,
                headers=headers,
            ) as response:
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.settings.max_model_response_bytes:
                        raise _ResponseTooLargeError(
                            "model response exceeds configured buffer limit"
                        )
                    chunks.append(chunk)
                return _BufferedResponse(
                    status_code=response.status_code,
                    headers={
                        key: value
                        for key, value in response.headers.items()
                        if key.lower()
                        in {"content-type", "anthropic-ratelimit-tokens-remaining"}
                    },
                    body=b"".join(chunks),
                )

        try:
            async with entry.operation_lock:
                if entry.state != "ACTIVE" or entry.revoking:
                    raise HTTPException(
                        status_code=409, detail=f"route is {entry.state}"
                    )
                task = await self.ledger.acquire(
                    route_id, request_id, ledger_digest, operation
                )
            buffered = await asyncio.shield(task)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except _LedgerCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc
        except _ResponseTooLargeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return Response(
            content=buffered.body,
            status_code=buffered.status_code,
            headers=buffered.headers,
        )

    def _view(self, entry: _RouteEntry) -> RouteView:
        return RouteView(
            context_id=entry.request.context_id,
            execution_id=entry.request.execution_id,
            trial_id=entry.binding.trial_id if entry.binding else None,
            policy_instance_id=self.instance_id,
            instance_id=entry.request.instance_id,
            state=entry.state,
            trajectory_retained=entry.trajectory_retained,
        )


def create_policy_app(settings: PolicySettings) -> FastAPI:
    runtime = PolicyRuntime(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.start()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Dressage DistRoll GPU Policy Service", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "contract_version": "agentic-rollout/v4",
            "policy_instance_id": runtime.instance_id,
            "capabilities": ["trajectory-detach-v1"],
        }

    @app.post("/v4/contexts", response_model=RouteView)
    async def prepare(request: ContextPrepareRequest) -> RouteView:
        try:
            return await runtime.prepare(request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except _LedgerCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc

    @app.post("/v4/contexts/{context_id}:bind", response_model=RouteView)
    async def bind(context_id: str, request: ContextBindRequest) -> RouteView:
        try:
            return await runtime.bind(context_id, request)
        except KeyError as exc:
            raise HTTPException(404, "unknown context") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/v4/contexts/{context_id}", response_model=RouteView)
    async def context(context_id: str) -> RouteView:
        entry = runtime.routes.get(context_id)
        if entry is None:
            if context_id in runtime.retired:
                return runtime.retired[context_id][1]
            raise HTTPException(404, "unknown or retired context")
        return runtime._view(entry)

    @app.post("/v4/contexts/{route_id}:{action}", response_model=RouteView)
    async def control(
        route_id: str, action: str, request: RouteControlRequest
    ) -> RouteView:
        try:
            return await runtime.control(route_id, action, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown route") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except _LedgerCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc

    @app.post("/routes/{route_id}/{path:path}", response_class=Response)
    async def model(route_id: str, path: str, request: Request) -> Response:
        return await runtime.forward(route_id, path, request)

    return app


def _retryable_http_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return (
            exc.response.status_code in {408, 425, 429}
            or exc.response.status_code >= 500
        )
    return False
