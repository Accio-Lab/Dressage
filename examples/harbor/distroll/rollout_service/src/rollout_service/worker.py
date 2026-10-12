from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from agentic_rollout_contract.client import RolloutServiceClient
from agentic_rollout_contract.models import (
    CleanupReport,
    ExecutionClaim,
    ExecutionClaimRequest,
    ExecutionIdentity,
    ExecutionResult,
    ExecutionState,
    RecoveryProof,
    WorkerDrainRequest,
    WorkerHeartbeat,
    WorkerRegistration,
)
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import Inventory, WorkerNode, WorkerSettings
from .harbor_runner import HarborOutcome, HarborRunner
from .image_cache import ImageCacheError, run_command
from .policy_relay import PolicyRelayRegistry, install_relay_routes
from .resource_telemetry import capture_host_resource_snapshot
from .sandbox_guard import SandboxGuard, current_guard
from .task_resource_index import TaskResourceIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailureClassification:
    failure_class: str
    failure_stage: str
    retryable: bool


_TASK_SETUP_ERROR_CODES = frozenset(
    {
        "AgentAuthenticationError",
        "ConfigError",
        "ConfigurationError",
        "FileNotFoundError",
        "NotADirectoryError",
        "PermissionError",
        "TaskNotFoundError",
        "TaskVersionNotFoundError",
        "TypeError",
        "ValidationError",
        "ValueError",
    }
)
_RETRYABLE_INFRA_ERROR_CODES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "ApiConnectionClosedError",
        "ApiInternalServerError",
        "ApiOverloadedError",
        "ApiRateLimitError",
        "ConnectError",
        "ConnectionError",
        "DockerAPIError",
        "DockerException",
        "EnvironmentStartTimeoutError",
        "NetworkConnectionError",
        "OSError",
        "ReadError",
        "ReadTimeout",
        "SandboxNotFoundError",
        "TimeoutError",
    }
)


def classify_runner_exception(exc: BaseException) -> FailureClassification:
    """Classify failures raised before Harbor can return a durable trial result."""

    if isinstance(exc, ImageCacheError):
        return FailureClassification(
            failure_class=exc.failure_class,
            failure_stage=exc.failure_stage,
            retryable=exc.retryable,
        )
    if isinstance(
        exc,
        (FileNotFoundError, NotADirectoryError, PermissionError, TypeError, ValueError),
    ):
        return FailureClassification(
            failure_class="TASK_SETUP",
            failure_stage="PREPARING",
            retryable=False,
        )
    return FailureClassification(
        failure_class="RETRYABLE_INFRA",
        failure_stage="EXECUTION",
        retryable=True,
    )


def classify_harbor_failure(error_code: str | None) -> FailureClassification:
    """Classify a failed Harbor trial from its stable exception type name."""

    if error_code in _TASK_SETUP_ERROR_CODES:
        return FailureClassification(
            failure_class="TASK_SETUP",
            failure_stage="PREPARING",
            retryable=False,
        )
    if error_code in _RETRYABLE_INFRA_ERROR_CODES:
        return FailureClassification(
            failure_class="RETRYABLE_INFRA",
            failure_stage="EXECUTION",
            retryable=True,
        )
    return FailureClassification(
        failure_class="AGENT_RUNTIME",
        failure_stage="AGENT_OR_VERIFIER",
        retryable=True,
    )


def execution_result_from_outcome(
    *,
    worker_id: str,
    worker_incarnation: str,
    lease_token: str,
    outcome: HarborOutcome,
    failure: FailureClassification | None = None,
) -> ExecutionResult:
    if outcome.succeeded:
        failure = None
    elif failure is None:
        failure = classify_harbor_failure(outcome.error_code)
    return ExecutionResult(
        worker_id=worker_id,
        worker_incarnation=worker_incarnation,
        lease_token=lease_token,
        state=ExecutionState.SUCCEEDED if outcome.succeeded else ExecutionState.FAILED,
        trial_id=outcome.trial_id,
        trial_name=outcome.trial_name,
        rewards=outcome.rewards,
        reward=outcome.reward,
        agent_terminal_reason=outcome.agent_terminal_reason,
        error_code=outcome.error_code,
        error_message=outcome.error_message,
        failure_class=failure.failure_class if failure is not None else None,
        failure_stage=failure.failure_stage if failure is not None else None,
        retryable=failure.retryable if failure is not None else False,
    )


@dataclass(frozen=True)
class PendingCompletion:
    path: Path
    execution_id: str
    result: ExecutionResult
    lease_valid_until: datetime


class CompletionOutbox:
    """Small process-persistent buffer for results not yet accepted by Control."""

    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir.expanduser() / "completion-outbox"
        self.rejected_path = state_dir.expanduser() / "completion-rejected"
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.rejected_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path, 0o700)
        os.chmod(self.rejected_path, 0o700)

    async def put(
        self,
        execution_id: str,
        result: ExecutionResult,
        lease_valid_until: datetime,
    ) -> PendingCompletion:
        name = hashlib.sha256(execution_id.encode("utf-8")).hexdigest() + ".json"
        selected = self.path / name
        payload = {
            "schema_version": 3,
            "execution_id": execution_id,
            "result": result.model_dump(mode="json", exclude_none=True),
            "lease_valid_until": lease_valid_until.isoformat(),
        }
        await asyncio.to_thread(_atomic_json_write, selected, payload)
        return PendingCompletion(selected, execution_id, result, lease_valid_until)

    async def pending(self) -> list[PendingCompletion]:
        return await asyncio.to_thread(self._pending_sync)

    async def has_pending(self, execution_id: str) -> bool:
        name = hashlib.sha256(execution_id.encode("utf-8")).hexdigest() + ".json"
        return await asyncio.to_thread((self.path / name).exists)

    def _pending_sync(self) -> list[PendingCompletion]:
        entries: list[PendingCompletion] = []
        for selected in sorted(self.path.glob("*.json")):
            try:
                payload = json.loads(selected.read_text(encoding="utf-8"))
                if payload.get("schema_version") != 3:
                    raise ValueError("unsupported completion outbox schema")
                entries.append(
                    PendingCompletion(
                        path=selected,
                        execution_id=str(payload["execution_id"]),
                        result=ExecutionResult.model_validate(payload["result"]),
                        lease_valid_until=datetime.fromisoformat(
                            payload["lease_valid_until"]
                        ),
                    )
                )
            except Exception as exc:
                logger.exception(
                    "invalid completion outbox requires operator reconciliation: %s",
                    selected,
                )
                raise RuntimeError(
                    "invalid completion outbox; refusing Worker startup"
                ) from exc
        return entries

    async def delivered(self, entry: PendingCompletion) -> None:
        await asyncio.to_thread(entry.path.unlink, missing_ok=True)

    async def reject(self, entry: PendingCompletion, reason: str) -> None:
        await asyncio.to_thread(self._reject_sync, entry.path, reason)

    def _reject_sync(self, selected: Path, reason: str) -> None:
        if not selected.exists():
            return
        suffix = "".join(char if char.isalnum() else "-" for char in reason)[:40]
        target = self.rejected_path / f"{selected.stem}.{suffix or 'rejected'}.json"
        os.replace(selected, target)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class WorkerDaemon:
    def __init__(
        self,
        settings: WorkerSettings,
        inventory: Inventory,
        node: WorkerNode,
        relay_registry: PolicyRelayRegistry,
        resource_index: TaskResourceIndex,
    ) -> None:
        self.settings = settings
        self.inventory = inventory
        self.node = node
        self.relay_registry = relay_registry
        self.resource_index = resource_index
        self.incarnation_id = uuid.uuid4().hex
        self.outbox = CompletionOutbox(node.state_dir)
        self.control_client = RolloutServiceClient(settings.control_plane_url)
        self.runner = HarborRunner(
            node,
            control_client=self.control_client,
            relay_registry=relay_registry,
            resource_index=resource_index,
            policy_url=inventory.policy.url,
        )
        self._closed = asyncio.Event()
        self._draining = asyncio.Event()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_claims: dict[str, ExecutionClaim] = {}
        self._cancel_requested: set[str] = set()
        self._cancel_cleanup_gate = asyncio.Semaphore(2)
        self._claim_heartbeat_lock = asyncio.Lock()
        self._registered = False
        self._quarantined = False
        self._claim_loop_running = False
        self._last_control_ok: float | None = None
        self._fatal_error: str | None = None
        self.guard = SandboxGuard(
            settings.sandbox_guard,
            node.state_dir,
            node.worker_id,
            self.incarnation_id,
            _atomic_json_write,
            on_trigger=self._start_guard_cleanup,
        )
        self._guard_runners: dict[str, asyncio.Task] = {}
        self._guard_pending: dict[str, Any] = {}
        self._cancelled_tasks: set[asyncio.Task] = set()

    @property
    def ready(self) -> bool:
        if (
            not self._registered
            or self._quarantined
            or not self._claim_loop_running
            or self._draining.is_set()
            or self._fatal_error is not None
            or not self.guard.healthy
            or self._last_control_ok is None
        ):
            return False
        stale_after = max(
            self.inventory.heartbeat_timeout_sec,
            self.inventory.heartbeat_interval_sec * 2,
        )
        return asyncio.get_running_loop().time() - self._last_control_ok < stale_after

    @property
    def status(self) -> dict[str, Any]:
        return {
            "worker_id": self.node.worker_id,
            "incarnation_id": self.incarnation_id,
            "registered": self._registered,
            "claim_loop_running": self._claim_loop_running,
            "draining": self._draining.is_set(),
            "quarantined": self._quarantined,
            "active_execution_ids": sorted(self._active_claims),
            "fatal_error": self._fatal_error,
            "sandbox_guard_mode": self.settings.sandbox_guard.mode,
            "sandbox_guard_healthy": self.guard.healthy,
            "sandbox_guard_settings": self.settings.sandbox_guard.model_dump(
                mode="json"
            ),
        }

    async def run(self) -> None:
        heartbeat: asyncio.Task[None] | None = None
        guard_task: asyncio.Task | None = None
        instance_lock = (self.node.state_dir / "worker.lock").open("a")
        try:
            self.guard.check_support()
            fcntl.flock(instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            instance_lock.close()
            raise
        try:
            _atomic_json_write(
                self.node.state_dir / "worker-owner.json",
                {
                    "worker_id": self.node.worker_id,
                    "incarnation_id": self.incarnation_id,
                    "pid": os.getpid(),
                },
            )
            logger.info(
                "worker configuration worker_id=%s cluster_id=%s "
                "inventory_digest=%s policy_url=%s",
                self.node.worker_id,
                self.inventory.cluster_id,
                self.inventory.digest(),
                self.inventory.policy.url,
            )
            # Old-incarnation results must be replayed before registration fences them.
            await self._replay_pending_completions()
            if not await self._register_until_draining():
                return
            self._registered = True
            self._mark_control_ok()
            if self._draining.is_set():
                await self._publish_drain()
            heartbeat = asyncio.create_task(
                self._heartbeat_loop(), name="worker-heartbeat"
            )
            guard_task = asyncio.create_task(self.guard.run(), name="sandbox-guard")

            def guard_stopped(task):
                if not task.cancelled():
                    self.guard.healthy = False
                    logger.error("sandbox guard loop stopped: %r", task.exception())

            guard_task.add_done_callback(guard_stopped)
            await self._claim_loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fatal_error = f"{type(exc).__name__}: {exc}"
            logger.exception("worker daemon stopped unexpectedly")
            raise
        finally:
            self._draining.set()
            if self._registered:
                await self._publish_drain()
            if self._tasks:
                await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            if self._cleanup_tasks:
                await asyncio.gather(
                    *self._cleanup_tasks.values(), return_exceptions=True
                )
            if self._guard_runners:
                await asyncio.shield(
                    asyncio.gather(
                        *self._guard_runners.values(), return_exceptions=True
                    )
                )
            if guard_task is not None:
                guard_task.cancel()
                await asyncio.gather(guard_task, return_exceptions=True)
            self._closed.set()
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            self._registered = False
            await self.control_client.close()
            instance_lock.close()

    async def _register_until_draining(self) -> bool:
        request = WorkerRegistration(
            worker_id=self.node.worker_id,
            incarnation_id=self.incarnation_id,
            config_digest=self.inventory.digest(),
            address=self.node.address,
            relay_base_url=self.node.relay_base_url,
            labels=self.node.labels,
            virtual_capacity=self.node.virtual_capacity,
            task_resource_catalog_digest=self.resource_index.digest,
            capabilities=[
                "harbor-0.22",
                "claude-code",
                "guarded-docker",
                "anthropic-messages",
                "execution-cancel-v4",
                "fixed-execution-deadline-v1",
            ],
        )
        backoff = self.settings.poll_interval_sec
        while not self._draining.is_set():
            try:
                async with asyncio.timeout(self.settings.control_request_timeout_sec):
                    response = await self.control_client.register_worker(request)
                    self._quarantined = response.state == "QUARANTINED"
                return True
            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    raise
                logger.exception("worker registration failed; retrying")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.settings.retry_max_sec)
            except Exception:
                logger.exception("worker registration failed; retrying")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.settings.retry_max_sec)
        return False

    async def stop(self) -> None:
        self._draining.set()
        if self._registered:
            deadline = asyncio.get_running_loop().time() + self.settings.retry_max_sec
            while not await self._publish_drain():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    logger.error(
                        "worker drain could not be published; heartbeat will expire"
                    )
                    break
                await asyncio.sleep(min(self.settings.poll_interval_sec, remaining))

    async def _claim_loop(self) -> None:
        backoff = self.settings.poll_interval_sec
        pending_request: ExecutionClaimRequest | None = None
        self._claim_loop_running = True
        try:
            while True:
                if self._draining.is_set():
                    return
                self._collect_finished_tasks()
                if self._quarantined or not self.guard.healthy or self._guard_pending:
                    await asyncio.sleep(self.settings.poll_interval_sec)
                    continue
                if pending_request is None:
                    pending_request = ExecutionClaimRequest(
                        worker_id=self.node.worker_id,
                        incarnation_id=self.incarnation_id,
                    )
                duplicate_claim = False
                try:
                    async with self._claim_heartbeat_lock:
                        async with asyncio.timeout(
                            self.settings.control_request_timeout_sec
                        ):
                            claim = await self.control_client.claim(pending_request)
                        self._mark_control_ok()
                        if claim is not None:
                            execution_id = claim.execution.execution_id
                            if execution_id in self._active_claims:
                                duplicate_claim = True
                            else:
                                self._active_claims[execution_id] = claim
                                task = asyncio.create_task(
                                    self._run_claim(claim),
                                    name=f"attempt-{execution_id}",
                                )
                                self._tasks[execution_id] = task
                    backoff = self.settings.poll_interval_sec
                except httpx.HTTPStatusError as exc:
                    if 400 <= exc.response.status_code < 500:
                        pending_request = None
                        if exc.response.status_code in {401, 403}:
                            raise
                        logger.error(
                            "claim request was rejected with HTTP %d; starting a new request",
                            exc.response.status_code,
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, self.settings.retry_max_sec)
                        continue
                    logger.exception("worker claim failed; retrying")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.settings.retry_max_sec)
                    continue
                except Exception:
                    logger.exception("worker claim failed; retrying")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.settings.retry_max_sec)
                    continue
                pending_request = None
                if claim is None:
                    if self._draining.is_set():
                        return
                    await asyncio.sleep(self.settings.poll_interval_sec)
                    continue
                if duplicate_claim:
                    logger.error(
                        "Control returned duplicate active claim %s",
                        claim.execution.execution_id,
                    )
                    await asyncio.sleep(self.settings.poll_interval_sec)
                    continue
                await asyncio.sleep(0)
        finally:
            self._claim_loop_running = False

    def _collect_finished_tasks(self) -> None:
        for execution_id, task in list(self._tasks.items()):
            if not task.done():
                continue
            self._tasks.pop(execution_id, None)
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "attempt task %s escaped its execution boundary", execution_id
                )

    def _cancel_once(self, execution_id, task):
        if task is not None and not task.done() and task not in self._cancelled_tasks:
            self._cancel_requested.add(execution_id)
            self._cancelled_tasks.add(task)
            task.add_done_callback(self._cancelled_tasks.discard)
            task.cancel()

    async def _guard_cleanup(self, watch, runner_task):
        watch.record("cleanup_started")
        try:
            async with asyncio.timeout(
                self.settings.sandbox_guard.local_cleanup_timeout_sec
            ):
                # Kill before waiting for Harbor's hooks or network cleanup.
                while True:
                    await _remove_execution_resources(
                        watch.execution_id,
                        terminate_grace_sec=self.settings.sandbox_guard.terminate_grace_sec,
                        worker_id=self.node.worker_id,
                        event=watch.record,
                    )
                    if watch.creating == 0:
                        break
                    await asyncio.sleep(0.1)
                await _verify_execution_resources_absent(watch.execution_id)
                await asyncio.shield(
                    asyncio.gather(runner_task, return_exceptions=True)
                )
                # Catch a resource created while Harbor was still unwinding.
                await _verify_execution_resources_absent(watch.execution_id)
        except Exception:
            self._quarantined = True
            self._guard_pending[watch.execution_id] = watch
            self._retain_guard_runner(watch.execution_id, runner_task)
            watch.record("cleanup_unconfirmed")
            raise
        finally:
            watch.record("local_cleanup_finished", runner_finished=runner_task.done())

    def _start_guard_cleanup(self, watch):
        # Start from the monitor, even while Worker is waiting inside ordinary
        # Harbor/network cleanup. The awaiter only settles the resulting proof.
        if watch.cleanup_task is None:
            self._cancel_once(watch.execution_id, watch.runner_task)
            watch.cleanup_task = asyncio.create_task(
                self._guard_cleanup(watch, watch.runner_task)
            )
        return watch.cleanup_task

    def _retain_guard_runner(self, execution_id, runner_task):
        self._guard_runners[execution_id] = runner_task

        def finished(task):
            self._guard_runners.pop(execution_id, None)
            if not task.cancelled():
                task.exception()

        runner_task.add_done_callback(finished)

    async def _run_claim(self, claim: ExecutionClaim) -> None:
        execution_id = claim.execution.execution_id
        loop = asyncio.get_running_loop()
        remaining = max(
            (claim.lease_expires_at - datetime.now(UTC)).total_seconds(),
            0.0,
        )
        settlement_deadline = loop.time() + remaining
        watchdog: asyncio.Task | None = None
        runner_task: asyncio.Task[HarborOutcome] | None = None
        watch = self.guard.register(
            execution_id, claim.execution.task_ref, claim.client_scope
        )
        guard_waiter = asyncio.create_task(watch.triggered.wait())
        guard_cleanup: asyncio.Task | None = None
        guard_cleanup_failed = False
        logger.info(
            "attempt resource start execution_id=%s resources=%s host=%s",
            execution_id,
            claim.resolved_resources.model_dump(mode="json"),
            capture_host_resource_snapshot(),
        )
        try:
            if claim.execution_deadline_at <= datetime.now(UTC):
                logger.error("execution expired before launch: %s", execution_id)
                async with asyncio.timeout(self.settings.control_request_timeout_sec):
                    await self.control_client.abandon_execution(
                        execution_id,
                        ExecutionIdentity(
                            worker_id=claim.worker_id,
                            worker_incarnation=claim.worker_incarnation,
                            lease_token=claim.lease_token,
                        ),
                    )
                return
            # Deadline stop-loss owns creation tracking even with OOM sampling off.
            token = current_guard.set(watch)
            try:
                runner_task = asyncio.create_task(
                    self.runner.run(claim), name=f"harbor-{execution_id}"
                )
                watch.runner_task = runner_task
                watchdog = asyncio.create_task(
                    self._execution_watchdog(claim, watch),
                    name=f"deadline-{execution_id}",
                )
            finally:
                current_guard.reset(token)
            await asyncio.wait(
                {runner_task, guard_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if watch.triggered.is_set():
                guard_cleanup = self._start_guard_cleanup(watch)
                try:
                    await asyncio.shield(guard_cleanup)
                except (OSError, RuntimeError, TimeoutError):
                    return
            # Close the local environment before freezing the completion. A
            # guard racing a Harbor success may never publish a successful digest.
            if guard_cleanup is None:
                local_finish = asyncio.create_task(
                    _wait_execution_resources_absent(execution_id)
                )
                try:
                    await asyncio.wait(
                        {local_finish, guard_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if watch.triggered.is_set():
                        guard_cleanup = self._start_guard_cleanup(watch)
                        try:
                            await asyncio.shield(guard_cleanup)
                        except (OSError, RuntimeError, TimeoutError):
                            return
                    else:
                        await local_finish
                finally:
                    local_finish.cancel()
                    await asyncio.gather(local_finish, return_exceptions=True)
            await self.guard.finish(watch)
            if watch.triggered.is_set():
                failure = FailureClassification("AGENT_RUNTIME", "EXECUTION", True)
                outcome = HarborOutcome(
                    succeeded=False,
                    trial_id=execution_id,
                    trial_name=f"failed-{execution_id}",
                    rewards={},
                    reward=0.0,
                    error_code="EXECUTION_TIMEOUT"
                    if watch.reason.get("rule") == "execution_timeout"
                    else "SANDBOX_RESOURCE_GUARD",
                    error_message=json.dumps(watch.reason, separators=(",", ":")),
                )
            else:
                try:
                    outcome = runner_task.result()
                except Exception as exc:
                    logger.exception("attempt %s failed", execution_id)
                    failure = classify_runner_exception(exc)
                    outcome = HarborOutcome(
                        succeeded=False,
                        trial_id=execution_id,
                        trial_name=f"failed-{execution_id}",
                        rewards={},
                        reward=0.0,
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                    )
                else:
                    failure = None
            result = execution_result_from_outcome(
                worker_id=self.node.worker_id,
                worker_incarnation=self.incarnation_id,
                lease_token=claim.lease_token,
                outcome=outcome,
                failure=failure,
            )
            logger.info(
                "attempt outcome execution_id=%s trial_id=%s trial_name=%s "
                "succeeded=%s error_code=%s failure_class=%s "
                "failure_stage=%s retryable=%s reward=%s",
                execution_id,
                outcome.trial_id,
                outcome.trial_name,
                outcome.succeeded,
                outcome.error_code,
                result.failure_class,
                result.failure_stage,
                result.retryable,
                outcome.reward,
            )
            persist_task = asyncio.create_task(
                self.outbox.put(
                    execution_id,
                    result,
                    claim.lease_expires_at,
                ),
                name=f"persist-result-{execution_id}",
            )
            try:
                # Persist before the first network delivery, including the fast path.
                await asyncio.shield(persist_task)
                try:
                    async with asyncio.timeout(
                        self.settings.control_request_timeout_sec
                    ):
                        await self.control_client.complete(execution_id, result)
                    self._mark_control_ok()
                except (httpx.HTTPError, OSError, TimeoutError):
                    entry = await asyncio.shield(persist_task)
                    await self._deliver_completion(entry)
                else:
                    entry = await asyncio.shield(persist_task)
                    await self.outbox.delivered(entry)
                if watch.reason:
                    watch.record(
                        "result_delivery_finished",
                        pending=await self.outbox.has_pending(execution_id),
                    )
            except asyncio.CancelledError:
                entry = await asyncio.shield(persist_task)
                if execution_id in self._cancel_requested:
                    await self.outbox.reject(entry, "speculative-cancelled")
                raise
        finally:
            if (
                watch.triggered.is_set()
                and guard_cleanup is None
                and runner_task is not None
            ):
                guard_cleanup = self._start_guard_cleanup(watch)
            if (
                runner_task is not None
                and not runner_task.done()
                and guard_cleanup is None
            ):
                self._cancel_once(execution_id, runner_task)
                await asyncio.wait(
                    {runner_task, guard_waiter}, return_when=asyncio.FIRST_COMPLETED
                )
                if watch.triggered.is_set():
                    guard_cleanup = self._start_guard_cleanup(watch)
            if guard_cleanup is not None:
                # Shielded local stop-loss must survive an overlapping RL cancel.
                await asyncio.gather(guard_cleanup, return_exceptions=True)
                if guard_cleanup.cancelled() or guard_cleanup.exception() is not None:
                    guard_cleanup_failed = True
                    if execution_id not in self._guard_pending:
                        self._quarantined = True
                        self._guard_pending[execution_id] = watch
                        self._retain_guard_runner(execution_id, runner_task)
            if (
                runner_task is not None
                and not runner_task.done()
                and not guard_cleanup_failed
            ):
                self._cancel_once(execution_id, runner_task)
                await asyncio.gather(runner_task, return_exceptions=True)
            cleanup_verified = True
            try:
                if guard_cleanup_failed:
                    raise RuntimeError("resource guard cleanup was not confirmed")
                async with asyncio.timeout_at(settlement_deadline):
                    async with self._cancel_cleanup_gate:
                        await _wait_execution_resources_absent(execution_id)
                    if await self.outbox.has_pending(execution_id):
                        raise RuntimeError("execution still has a pending completion")
                    while True:
                        try:
                            async with asyncio.timeout(
                                self.settings.control_request_timeout_sec
                            ):
                                await self.control_client.cleanup(
                                    execution_id,
                                    CleanupReport(
                                        worker_id=self.node.worker_id,
                                        worker_incarnation=self.incarnation_id,
                                        lease_token=claim.lease_token,
                                    ),
                                )
                            if watch.reason:
                                watch.record("cleanup_receipt_confirmed")
                            break
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code < 500:
                                raise
                        except (httpx.TransportError, TimeoutError):
                            pass
                        await asyncio.sleep(self.settings.poll_interval_sec)
            except (OSError, RuntimeError, TimeoutError, httpx.HTTPError):
                cleanup_verified = False
                self._quarantined = True
                logger.exception("execution cleanup pending: %s", execution_id)
                try:
                    async with asyncio.timeout(
                        self.settings.control_request_timeout_sec
                    ):
                        await self.control_client.abandon_execution(
                            execution_id,
                            ExecutionIdentity(
                                **claim.model_dump(
                                    include={
                                        "worker_id",
                                        "worker_incarnation",
                                        "lease_token",
                                    }
                                )
                            ),
                        )
                except (httpx.HTTPError, TimeoutError):
                    logger.exception(
                        "could not publish execution quarantine: %s", execution_id
                    )
            finally:
                if watch.reason:
                    watch.record("attempt_finished", cleanup_verified=cleanup_verified)
                    await watch.flush()
                watch.sealed = True
                self.guard.attempts.pop(execution_id, None)
                guard_waiter.cancel()
                await asyncio.gather(guard_waiter, return_exceptions=True)
                if watchdog is not None:
                    watchdog.cancel()
                    await asyncio.gather(watchdog, return_exceptions=True)
            logger.info(
                "attempt resource terminal execution_id=%s host=%s",
                execution_id,
                capture_host_resource_snapshot(),
            )
            if cleanup_verified:
                self._active_claims.pop(execution_id, None)
                self._cancel_requested.discard(execution_id)

    async def _execution_watchdog(self, claim: ExecutionClaim, watch) -> None:
        remaining = (claim.execution_deadline_at - datetime.now(UTC)).total_seconds()
        await asyncio.sleep(max(remaining, 0.0))
        if watch.runner_task.done() or watch.triggered.is_set():
            return
        watch.reason = {"rule": "execution_timeout"}
        watch.triggered.set()
        watch.record("execution_timeout")
        logger.error("execution total deadline reached: %s", watch.execution_id)
        self._start_guard_cleanup(watch)

    async def _deliver_completion(self, entry: PendingCompletion) -> bool:
        loop = asyncio.get_running_loop()
        backoff = self.settings.poll_interval_sec
        deadline = loop.time() + max(
            (entry.lease_valid_until - datetime.now(UTC)).total_seconds(), 0.0
        )
        while True:
            try:
                async with asyncio.timeout(
                    min(
                        self.settings.control_request_timeout_sec,
                        max(deadline - loop.time(), 0.01),
                    )
                ):
                    await self.control_client.complete(entry.execution_id, entry.result)
                self._mark_control_ok()
                await self.outbox.delivered(entry)
                return True
            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    logger.error(
                        "completion %s was definitively rejected with HTTP %d "
                        "state=%s error_code=%s failure_class=%s "
                        "failure_stage=%s trial_id=%s",
                        entry.execution_id,
                        exc.response.status_code,
                        entry.result.state,
                        entry.result.error_code,
                        entry.result.failure_class,
                        entry.result.failure_stage,
                        entry.result.trial_id,
                    )
                    await self.outbox.reject(entry, f"http-{exc.response.status_code}")
                    return False
                logger.exception(
                    "completion %s delivery failed; retrying", entry.execution_id
                )
            except Exception:
                logger.exception(
                    "completion %s delivery failed; retrying", entry.execution_id
                )

            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.error(
                    "completion %s lease expired before delivery", entry.execution_id
                )
                await self.outbox.reject(entry, "lease-expired")
                return False
            await asyncio.sleep(min(backoff, remaining))
            backoff = min(backoff * 2, self.settings.retry_max_sec)

    async def _replay_pending_completions(self) -> None:
        for entry in await self.outbox.pending():
            logger.warning("replaying pending completion %s", entry.execution_id)
            await self._deliver_completion(entry)

    async def _heartbeat_loop(self) -> None:
        while not self._closed.is_set():
            await self._publish_heartbeat()
            await asyncio.sleep(self.inventory.heartbeat_interval_sec)

    async def _publish_heartbeat(self) -> bool:
        try:
            async with self._claim_heartbeat_lock:
                async with asyncio.timeout(self.settings.control_request_timeout_sec):
                    response = await self.control_client.heartbeat(
                        WorkerHeartbeat(
                            worker_id=self.node.worker_id,
                            incarnation_id=self.incarnation_id,
                            active_execution_ids=sorted(self._active_claims),
                        )
                    )
            for execution_id in getattr(response, "cancel_execution_ids", []):
                task = self._tasks.get(execution_id)
                if (
                    task is None
                    or task.done()
                    or execution_id in self._cancel_requested
                ):
                    continue
                logger.info(
                    "cancelling attempt at Control request execution_id=%s",
                    execution_id,
                )
                self._cancel_once(execution_id, task)
            self._quarantined = response.state == "QUARANTINED" or bool(
                self._guard_pending
            )
            # Local stop-loss cannot depend on a successful abandonment RPC.
            # Retry publishing quarantine after Control returns; do not let a
            # READY heartbeat reopen local admission in that gap.
            for execution_id in list(self._guard_pending):
                claim = self._active_claims[execution_id]
                async with asyncio.timeout(self.settings.control_request_timeout_sec):
                    await self.control_client.abandon_execution(
                        execution_id,
                        ExecutionIdentity(
                            **claim.model_dump(
                                include={
                                    "worker_id",
                                    "worker_incarnation",
                                    "lease_token",
                                }
                            )
                        ),
                    )
            for view in await self.control_client.pending_cleanup(
                self.node.worker_id, self.incarnation_id
            ):
                task = self._tasks.get(view.execution_id)
                if task is not None and not task.done():
                    continue
                if view.execution_id in self._guard_runners:
                    continue
                old = self._cleanup_tasks.get(view.execution_id)
                if old is None or old.done():
                    cleanup = asyncio.create_task(self._recover_local(view))
                    self._cleanup_tasks[view.execution_id] = cleanup
                    cleanup.add_done_callback(
                        lambda task, eid=view.execution_id: (
                            self._cleanup_tasks.pop(eid, None)
                            if self._cleanup_tasks.get(eid) is task
                            else None
                        )
                    )
            self._mark_control_ok()
            return True
        except Exception:
            logger.exception("worker heartbeat failed")
            return False

    async def _recover_local(self, view):
        try:
            async with self._cancel_cleanup_gate:
                await _remove_execution_resources(view.execution_id)
                for entry in await self.outbox.pending():
                    if entry.execution_id == view.execution_id:
                        await self.outbox.reject(entry, "execution-uncertain")
                await _verify_execution_resources_absent(view.execution_id)
                evidence = hashlib.sha256(
                    f"{self.node.worker_id}:{self.incarnation_id}:{view.execution_id}:{view.worker_incarnation}:absent".encode()
                ).hexdigest()
                await self.control_client.recover_execution(
                    view.execution_id,
                    RecoveryProof(
                        worker_id=self.node.worker_id,
                        incarnation_id=self.incarnation_id,
                        worker_incarnation=view.worker_incarnation,
                        evidence_digest="sha256:" + evidence,
                        sandbox_absent=True,
                        outbox_empty=True,
                    ),
                )
                self._active_claims.pop(view.execution_id, None)
                self._cancel_requested.discard(view.execution_id)
                watch = self._guard_pending.pop(view.execution_id, None)
                if watch is not None:
                    watch.record("recovery_receipt_confirmed", cleanup_verified=True)
                    await watch.flush()
        except (OSError, RuntimeError, TimeoutError, httpx.HTTPError):
            logger.exception(
                "quarantined execution still requires cleanup: %s", view.execution_id
            )

    async def _publish_drain(self) -> bool:
        try:
            async with asyncio.timeout(self.settings.control_request_timeout_sec):
                await self.control_client.drain_worker(
                    WorkerDrainRequest(
                        worker_id=self.node.worker_id,
                        incarnation_id=self.incarnation_id,
                    )
                )
            self._mark_control_ok()
            return True
        except Exception:
            logger.exception("worker drain request failed")
            return False

    def _mark_control_ok(self) -> None:
        self._last_control_ok = asyncio.get_running_loop().time()


def _execution_project(project: str, execution_id: str) -> bool:
    return project == f"remote-{execution_id}__env" or project.startswith(
        f"remote-{execution_id}__verifier__"
    )


async def _remove_execution_resources(
    execution_id: str,
    *,
    terminate_grace_sec: int = 10,
    worker_id: str | None = None,
    event=None,
) -> None:
    for kind in ("container", "network", "volume"):
        command = ["docker", kind, "ls"] + (["--all"] if kind == "container" else [])
        if kind in {"container", "network"}:
            command += ["--no-trunc"]
        command += ["--format", "{{json .}}"]
        result = await run_command(command, timeout_sec=20)
        if result.returncode:
            raise RuntimeError("could not inspect quarantined resources")
        for line in result.stdout.splitlines():
            item = json.loads(line)
            labels = dict(
                v.split("=", 1) for v in item.get("Labels", "").split(",") if "=" in v
            )
            project = labels.get("com.docker.compose.project", "")
            if not _execution_project(project, execution_id):
                continue
            identifier = item.get("ID") or item.get("Name")
            if not identifier:
                raise RuntimeError("Docker resource has no precise identifier")
            if kind == "container":
                if (
                    worker_id is not None
                    and labels.get("com.docker.compose.service") == "main"
                    and (
                        labels.get("io.dressage.rollout.managed") != "true"
                        or labels.get("io.dressage.rollout.attempt_id") != execution_id
                        or labels.get("io.dressage.rollout.worker_id") != worker_id
                    )
                ):
                    raise RuntimeError("resource guard refuses foreign container")
                try:
                    stopped = await run_command(
                        [
                            "docker",
                            "container",
                            "stop",
                            "--time",
                            str(terminate_grace_sec),
                            identifier,
                        ],
                        timeout_sec=terminate_grace_sec + 5,
                    )
                    stop_failed = stopped.returncode != 0
                except TimeoutError:
                    stop_failed = True
                if stop_failed:
                    # Docker stop itself sends SIGKILL after its grace. If it
                    # failed while Harbor concurrently removed this container,
                    # only a fresh listing can prove absence.
                    fresh = await run_command(command, timeout_sec=5)
                    if fresh.returncode:
                        raise RuntimeError("could not confirm container disappearance")
                    if not any(
                        json.loads(line).get("ID") == identifier
                        for line in fresh.stdout.splitlines()
                    ):
                        continue
                    if worker_id is None:
                        raise RuntimeError("quarantined container did not stop")
                    killed = await run_command(
                        ["docker", "container", "kill", identifier], timeout_sec=5
                    )
                    if killed.returncode:
                        raise RuntimeError("resource guard could not kill container")
                if event is not None:
                    event(
                        "container_stopped",
                        container_id=identifier,
                        terminate_grace_sec=terminate_grace_sec,
                        explicit_kill=stop_failed,
                    )
            removed = await run_command(
                ["docker", kind, "rm", identifier], timeout_sec=30
            )
            if removed.returncode:
                fresh = await run_command(command, timeout_sec=5)
                if fresh.returncode or any(
                    (json.loads(line).get("ID") or json.loads(line).get("Name"))
                    == identifier
                    for line in fresh.stdout.splitlines()
                ):
                    raise RuntimeError("quarantined resource did not disappear")
            if event is not None:
                event("resource_removed", kind=kind, identifier=identifier)


async def _verify_execution_resources_absent(execution_id: str) -> None:
    """Check only this exact Harbor Compose project; never delete resources."""
    for kind in ("container", "network", "volume"):
        command = ["docker", kind, "ls"] + (["--all"] if kind == "container" else [])
        command += ["--format", '{{.Label "com.docker.compose.project"}}']
        result = await run_command(command, timeout_sec=20.0)
        if result.returncode != 0:
            raise RuntimeError("could not inspect execution Docker resources")
        if any(
            _execution_project(project, execution_id)
            for project in result.stdout.splitlines()
        ):
            raise RuntimeError("execution still owns Docker resources")


async def _wait_execution_resources_absent(
    execution_id: str, *, timeout_sec: float = 180.0
) -> None:
    """Wait for Harbor's shielded Docker cleanup, then fail closed if it stalls."""

    deadline = asyncio.get_running_loop().time() + timeout_sec
    while True:
        try:
            await _verify_execution_resources_absent(execution_id)
            return
        except RuntimeError as exc:
            if str(exc) != "execution still owns Docker resources":
                raise
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError(
                    f"cancelled attempt {execution_id} Docker cleanup timed out"
                ) from exc
            await asyncio.sleep(min(1.0, remaining))


def create_worker_app(
    settings: WorkerSettings,
    inventory: Inventory,
    *,
    resource_index: TaskResourceIndex,
) -> FastAPI:
    node = inventory.worker(settings.worker_id)
    relay_registry = PolicyRelayRegistry(
        inventory.policy.url,
        transport_retry_attempts=settings.policy_transport_retry_attempts,
        transport_retry_base_sec=settings.policy_transport_retry_base_sec,
        transport_retry_max_sec=settings.policy_transport_retry_max_sec,
    )
    daemon = WorkerDaemon(
        settings,
        inventory,
        node,
        relay_registry,
        resource_index,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker_task = asyncio.create_task(daemon.run(), name="worker-daemon")
        app.state.worker_task = worker_task
        app.state.worker_daemon = daemon
        try:
            yield
        finally:
            await daemon.stop()
            try:
                await asyncio.wait_for(asyncio.shield(worker_task), timeout=300.0)
            except TimeoutError:
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
            await relay_registry.close()

    app = FastAPI(title=f"Rollout Worker {node.worker_id}", lifespan=lifespan)
    install_relay_routes(app, relay_registry)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        worker_task = getattr(app.state, "worker_task", None)
        healthy = worker_task is not None and not worker_task.done()
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ok" if healthy else "failed", **daemon.status},
        )

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        return JSONResponse(
            status_code=200 if daemon.ready else 503,
            content={
                "status": "ready" if daemon.ready else "not-ready",
                **daemon.status,
            },
        )

    return app
