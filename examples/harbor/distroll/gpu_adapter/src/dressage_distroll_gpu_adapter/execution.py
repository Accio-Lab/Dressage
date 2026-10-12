"""RL-owned execution queue. CPU knows execution IDs, never GRPO groups."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import httpx
from agentic_rollout_contract.canonical import canonical_digest
from agentic_rollout_contract.client import PolicyServiceClient, RolloutServiceClient
from agentic_rollout_contract.models import (
    ActivateRequest,
    CancelRequest,
    ContextPrepareRequest,
    ExecutionSpec,
    ExecutionState,
    ReserveRequest,
    ResultAck,
    RouteControlRequest,
)

E = ExecutionState
PROCESS_OWNER = uuid.uuid4().hex
logger = logging.getLogger(__name__)
_current_execution = ContextVar("rl_execution", default=None)


def identity(*parts: Any) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, ":".join(map(str, parts))).hex


def write_artifact(path: Path, data: bytes) -> str:
    """Atomic and durable before acknowledging data to its producer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with os.fdopen(
            os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
        ) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_artifact(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"artifact digest mismatch: {path.name}")
    return data


class ExecutionQueue:
    """One owner, one bounded pool, one opaque CAS checkpoint per RL run.

    A slot's finite revisions stay in the same coroutine. A revision starts
    only after its predecessor settles; no group-level semaphore is used.
    """

    def __init__(self, settings, *, root: Path, proxy: Any, materialize):
        self.settings = settings
        self.root = root
        self.proxy = proxy
        self.materialize = materialize
        self.artifacts: dict[str, dict] = {}
        self.control = RolloutServiceClient(
            settings.control_plane_url,
            timeout=httpx.Timeout(30, connect=15),
        )
        self.policy = PolicyServiceClient(
            settings.policy_service_url,
            timeout=httpx.Timeout(settings.speculative_cancel_timeout_sec, connect=15),
        )
        self.lock = asyncio.Lock()
        self.stop = asyncio.Event()
        self.payload: dict[str, Any] = {}
        self.policy_instance = ""
        self._writer_fd: int | None = None
        self._stop_error: Exception | None = None

    def _load_state(self, path: Path | None = None) -> dict[str, Any] | None:
        path = path or self.root / "state.json"
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_bytes())
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("durable RL state is invalid") from exc
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("run_id") != self.settings.run_id
            or not isinstance(state.get("payload"), dict)
            or state.get("digest") != canonical_digest(state["payload"])
        ):
            raise RuntimeError("durable RL state is invalid")
        return state["payload"]

    @contextmanager
    def commit_guard(self):
        """Serialize an operator stop request against the final commit."""
        descriptor = os.open(self.root / "commit.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    async def call(self, operation, *, attempts=None, retry_backpressure=True):
        delay = self.settings.retry_base_sec
        attempts = attempts or self.settings.control_retry_attempts
        started = time.monotonic()
        for attempt in range(attempts):
            try:
                budget = self.settings.control_outage_timeout_sec
                remaining = (
                    max(0.0, budget - (time.monotonic() - started)) if budget else None
                )
                async with asyncio.timeout(remaining):
                    return await operation()
            except httpx.HTTPError as exc:
                if (
                    not retry_backpressure
                    and isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code == 429
                ):
                    raise
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and (
                        exc.response.status_code in {408, 425, 429}
                        or exc.response.status_code >= 500
                    )
                )
                if not retryable or attempt + 1 == attempts:
                    raise
                status = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                kind = (
                    "backpressure"
                    if status == 429
                    else "transport"
                    if status is None
                    else "upstream_http"
                )
                logger.warning(
                    "rl_rpc_retry run=%s step=%s execution_id=%s kind=%s retry=%d/%d error=%s status=%s",
                    self.settings.run_id,
                    self.payload.get("step"),
                    _current_execution.get(),
                    kind,
                    attempt + 1,
                    attempts,
                    type(exc).__name__,
                    status,
                )
                await asyncio.sleep(delay)
                delay = min(
                    max(delay * 2, self.settings.retry_base_sec),
                    self.settings.retry_max_sec,
                )

    async def open(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self._writer_fd = os.open(
            self.root / "owner.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            fcntl.flock(self._writer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._writer_fd)
            self._writer_fd = None
            raise RuntimeError("another RL process owns this run state") from None
        if self._load_state() is not None:
            raise RuntimeError(
                "this step already ran; trajectories are not recoverable; use a new run ID"
            )
        capabilities = await self.call(self.control.capabilities)
        if capabilities.get("contract_version") != "agentic-rollout/v4":
            raise RuntimeError("CPU execution protocol must be V4")
        if "fixed-execution-deadline-v1" not in capabilities.get("capabilities", []):
            raise RuntimeError("CPU must support fixed execution deadlines")
        self.payload = {}
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.get(self.settings.policy_service_url + "/healthz")
            response.raise_for_status()
            health = response.json()
            if "trajectory-detach-v1" not in health.get("capabilities", []):
                raise RuntimeError(
                    "Policy must support trajectory-detach-v1 before submission"
                )
            self.policy_instance = health["policy_instance_id"]

    async def close(self):
        try:
            await self.policy.close()
            await self.control.close()
        finally:
            if self._writer_fd is not None:
                fcntl.flock(self._writer_fd, fcntl.LOCK_UN)
                os.close(self._writer_fd)
                self._writer_fd = None

    async def save(self):
        """Durably record RL state before any external side effect."""
        state = {
            "version": 1,
            "run_id": self.settings.run_id,
            "payload": self.payload,
            "digest": canonical_digest(self.payload),
        }
        write_artifact(
            self.root / "state.json",
            json.dumps(state, allow_nan=False, separators=(",", ":")).encode(),
        )

    async def initialize_step(self, step: int, manifest: dict, plan_digest: str):
        async with self.lock:
            previous = self.payload or (
                self._load_state(self.root.parent / f"step-{step - 1}" / "state.json")
                if step > 0
                else None
            )
            if previous and previous["step"] == step:
                raise RuntimeError(
                    "step replay is forbidden; use a new run after drain"
                )
            if previous and (
                not previous.get("committed") or step != previous["step"] + 1
            ):
                raise RuntimeError(
                    "previous RL step is not committed, or step cursor is discontinuous"
                )
            self.payload = {
                "step": step,
                "process_owner": PROCESS_OWNER,
                "manifest": manifest,
                "plan_digest": plan_digest,
                "policy_instance": self.policy_instance,
                "executions": {},
                "completed_slots": {},
                "accepted": [],
                "committed": False,
                "stopped": False,
            }
            await self.save()

    async def _record(self, execution_id: str, **changes):
        async with self.lock:
            self.payload["executions"].setdefault(execution_id, {}).update(changes)
            await self.save()

    async def _close_context(self, context_id, action, execution_id, *, retain=False):
        if retain:
            await self._record(execution_id, trajectory_retention_intent=True)
        closed = await self.call(
            lambda: (
                self.policy.control(
                    context_id,
                    "detach",
                    RouteControlRequest(action_id=action, execution_id=execution_id),
                )
                if retain
                else self.policy.revoke(context_id, action, execution_id)
            )
        )
        if (
            closed.state != "CLOSED"
            or closed.context_id != context_id
            or closed.execution_id != execution_id
            or closed.policy_instance_id != self.policy_instance
            or closed.trajectory_retained != retain
        ):
            raise RuntimeError("Policy did not confirm the exact context closure")
        await self._record(execution_id, policy_closed=True, trajectory_retained=retain)

    async def _reserve(self, spec, record):
        while True:
            if self.stop.is_set():
                return await self.call(
                    lambda: self.control.cancel(
                        spec.execution_id,
                        CancelRequest(
                            client_scope=self.settings.run_id,
                            reason="RL stopped before admission",
                        ),
                    )
                )
            if time.time() >= record["admission_deadline"]:
                raise TimeoutError("finite CPU admission deadline exceeded")
            request = ReserveRequest(
                client_scope=self.settings.run_id,
                spec=spec,
                reservation_request_id=record["reservation_request_id"],
            )
            try:
                view = await self.call(
                    lambda request=request: self.control.reserve(request),
                    retry_backpressure=False,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429:
                    raise
                logger.info(
                    "admission_wait run=%s execution_id=%s",
                    self.settings.run_id,
                    spec.execution_id,
                )
                try:
                    await asyncio.wait_for(
                        self.stop.wait(),
                        timeout=min(
                            2.0, max(0.01, record["admission_deadline"] - time.time())
                        ),
                    )
                except TimeoutError:
                    pass
                continue
            if view.state != E.EXPIRED or self.stop.is_set():
                return view
            generation = record.get("reservation_expiries", 0) + 1
            if generation > self.settings.control_retry_attempts:
                raise RuntimeError("reservation expiry budget exhausted")
            if record.get("prepare"):
                prepared = ContextPrepareRequest.model_validate(record["prepare"])
                await self.call(lambda prepared=prepared: self.policy.prepare(prepared))
                await self._close_context(
                    prepared.context_id, "expired", spec.execution_id
                )
            # A proven unactivated expiry is not a new rollout attempt.
            await self._record(
                spec.execution_id,
                reservation_expiries=generation,
                reservation_request_id=identity(
                    spec.execution_id, "reserve", generation
                ),
                context_id=None,
                prepare=None,
            )

    async def execute(
        self,
        *,
        group: int,
        slot: int,
        revision: int,
        spec: ExecutionSpec,
        context: dict,
    ) -> dict | None:
        execution_id = spec.execution_id
        _current_execution.set(execution_id)
        record = self.payload["executions"].get(execution_id)
        if record and record.get("settled"):
            return record.get("artifact")
        if not record:
            if self.stop.is_set():
                return None
            await self._record(
                execution_id,
                group=group,
                slot=slot,
                revision=revision,
                spec=spec.model_dump(mode="json"),
                reservation_request_id=identity(execution_id, "reserve", 0),
                admission_deadline=time.time() + self.settings.admission_timeout_sec,
            )
            record = self.payload["executions"][execution_id]
        if record["spec"] != spec.model_dump(mode="json"):
            raise RuntimeError("execution intent changed on replay")
        try:
            while True:
                # 429 retries never create Policy state. Ambiguous replies reuse ID.
                view = await self._reserve(spec, record)
                if view.failure_code:
                    await self._settle_cancel(execution_id)
                    return None
                if self.stop.is_set() or view.state == E.CANCELLED:
                    await self._settle_cancel(execution_id)
                    return None
                context_id = identity(execution_id, view.reservation_id, "context")
                prepared = ContextPrepareRequest(
                    policy_instance_id=self.policy_instance,
                    context_id=context_id,
                    execution_id=execution_id,
                    client_scope=self.settings.run_id,
                    reservation_id=view.reservation_id,
                    **context,
                )
                await self._record(
                    execution_id,
                    context_id=context_id,
                    reservation_id=view.reservation_id,
                    prepare=prepared.model_dump(mode="json"),
                    policy_closed=False,
                )
                await self.call(lambda prepared=prepared: self.policy.prepare(prepared))
                if self.stop.is_set():
                    await self._settle_cancel(execution_id)
                    return None
                try:
                    view = await self.call(
                        lambda reservation_id=view.reservation_id, context_id=context_id: (
                            self.control.activate(
                                execution_id,
                                ActivateRequest(
                                    client_scope=self.settings.run_id,
                                    reservation_id=reservation_id,
                                    context_id=context_id,
                                ),
                            )
                        )
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 409:
                        raise
                    latest = await self.call(
                        lambda: self.control.get_execution(execution_id)
                    )
                    if latest.state != E.EXPIRED:
                        raise
                    # _reserve() proves and closes the expired context before
                    # retrying this same logical execution, without recursion.
                    continue
                break
            while True:
                if self.stop.is_set():
                    await self._settle_cancel(execution_id)
                    return None
                if (
                    view.failure_code
                    or view.result is not None
                    or view.state in {E.CANCELLED, E.EXPIRED}
                ):
                    break
                version = view.state_version
                view = await self.call(
                    lambda version=version: self.control.get_execution(
                        execution_id,
                        after_version=version,
                        wait=self.settings.wait_long_poll_sec,
                    )
                )
            artifact = record.get("artifact")
            if (
                not view.failure_code
                and view.result
                and view.result.state == E.SUCCEEDED
                and artifact is None
            ):
                data = await self.materialize(view, context)
                artifact = await self.store_trajectory(execution_id, data)
                await self._record(
                    execution_id, artifact=artifact, tq_refs=data.get("tq_refs", [])
                )
            if view.result_digest:
                await self.call(
                    lambda: self.control.ack(
                        execution_id,
                        ResultAck(
                            client_scope=self.settings.run_id,
                            result_digest=view.result_digest,
                        ),
                    )
                )
            await self._close_context(
                context_id, "settle", execution_id, retain=artifact is not None
            )
            cleaned = await self._wait_cleanup(execution_id)
            if cleaned.failure_code:
                artifact = None
                self.artifacts.pop(execution_id, None)
                await self._close_context(context_id, "discard-uncertain", execution_id)
            await self._record(
                execution_id,
                settled=True,
                artifact=artifact,
                terminal_state=cleaned.state.value,
                result_digest=view.result_digest,
                failure_class="RETRYABLE_INFRA"
                if cleaned.failure_code
                else view.result.failure_class
                if view.result
                else None,
                failure_stage=view.result.failure_stage if view.result else None,
                error_code=cleaned.failure_code
                or (view.result.error_code if view.result else None),
                retryable=True
                if cleaned.failure_code
                else view.result.retryable
                if view.result
                else False,
                cpu_cleanup_verified=cleaned.cleanup_verified,
                cpu_quarantined=cleaned.resource_quarantined
                and not cleaned.cleanup_verified,
                activated=view.context_id is not None,
            )
            return artifact
        except BaseException as exc:
            # Do not mask an unresolved execution by immediately starting its
            # revision. Cleanup failure propagates and blocks parameter update.
            logging.getLogger(__name__).error(
                "execution_fatal_before_cleanup execution=%s error_type=%s frames=%s",
                execution_id,
                type(exc).__name__,
                [
                    (Path(f.filename).name, f.name, f.lineno)
                    for f in traceback.extract_tb(exc.__traceback__)
                ],
            )
            task = asyncio.create_task(self._settle_cancel(execution_id))
            await asyncio.shield(task)
            raise

    async def store_trajectory(self, execution_id, data):
        self.artifacts[execution_id] = data
        digest = canonical_digest(data)
        return {
            "name": f"{execution_id}.json",
            "digest": digest,
            "execution_id": execution_id,
            "result_digest": data["result_digest"],
        }

    def read_trajectory(self, artifact):
        try:
            return self.artifacts[artifact["execution_id"]]
        except KeyError:
            raise RuntimeError(
                "trajectory memory lost; do not replay this step"
            ) from None

    async def _wait_cleanup(self, execution_id: str):
        deadline = time.monotonic() + self.settings.speculative_cancel_timeout_sec
        while True:
            view = await self.call(lambda: self.control.get_execution(execution_id))
            if view.failure_code == "EXECUTION_UNCERTAIN" and view.resource_quarantined:
                return view
            if view.cleanup_verified:
                return view
            if time.monotonic() >= deadline:
                view = await self.call(
                    lambda: self.control.cancel(
                        execution_id,
                        CancelRequest(
                            client_scope=self.settings.run_id,
                            reason="CPU cleanup deadline exceeded",
                            quarantine=True,
                        ),
                    )
                )
                if view.cleanup_verified or (
                    view.failure_code == "EXECUTION_UNCERTAIN"
                    and view.resource_quarantined
                ):
                    return view
                raise RuntimeError(
                    "CPU neither confirmed cleanup nor quarantined the execution"
                )
            await asyncio.sleep(0.5)

    async def _settle_cancel(self, execution_id: str):
        await self._cancel_and_close(execution_id)

    async def _cancel_and_close(self, execution_id: str):
        _current_execution.set(execution_id)
        view = await self.call(
            lambda: self.control.cancel(
                execution_id,
                CancelRequest(
                    client_scope=self.settings.run_id, reason="RL cancellation"
                ),
            )
        )
        record = self.payload["executions"].get(execution_id, {})
        context_id = record.get("context_id") or view.context_id
        if context_id:
            try:
                if record.get("prepare"):
                    # Resolve a lost prepare reply before revoke. Prepared
                    # contexts are lightweight and cannot start a sandbox.
                    prepared = ContextPrepareRequest.model_validate(record["prepare"])
                    await self.call(lambda: self.policy.prepare(prepared))
                await self._close_context(context_id, "cancel", execution_id)
            except httpx.HTTPStatusError as exc:
                # A missing context is not a cancellation proof: the prepare
                # response itself may have been lost. Fail closed.
                raise RuntimeError("Policy cancellation was not confirmed") from exc
        view = await self._wait_cleanup(execution_id)
        if view.result_digest:
            # A result may have won the completion/cancel race. Acknowledge
            # delivery, but never turn a discarded surplus result into data.
            await self.call(
                lambda: self.control.ack(
                    execution_id,
                    ResultAck(
                        client_scope=self.settings.run_id,
                        result_digest=view.result_digest,
                    ),
                )
            )
        await self._record(
            execution_id,
            settled=True,
            cancelled=not bool(view.failure_code),
            artifact=None,
            result_digest=view.result_digest,
            terminal_state=view.state.value,
            activated=view.context_id is not None,
            error_code=view.failure_code,
            retryable=bool(view.failure_code),
            cpu_cleanup_verified=view.cleanup_verified,
            cpu_quarantined=view.resource_quarantined and not view.cleanup_verified,
        )

    async def _check_stop_request(self) -> bool:
        path = self.root / "stop-request.json"
        if not path.exists():
            return False
        request = json.loads(path.read_text())
        if request != {
            "run_id": self.settings.run_id,
            "step": self.payload["step"],
        }:
            raise RuntimeError("operator stop request does not match this RL step")
        async with self.lock:
            if not self.payload.get("operator_stop"):
                self.payload.update(operator_stop=True, stopped=True)
                await self.save()
        self.stop.set()
        return True

    async def _watch_stop_request(self) -> None:
        try:
            while not self.stop.is_set():
                if await self._check_stop_request():
                    return
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed on any invalid stop request
            self._stop_error = exc
            self.stop.set()

    async def run_slots(
        self,
        jobs: list[tuple[int, int]],
        operation,
        group_ready,
        *,
        width: int,
        target: int,
        early_stop: bool,
    ):
        """All slots are queued immediately; only execution work consumes C."""
        queue: asyncio.Queue = asyncio.Queue()
        finished = self.payload["completed_slots"]
        accepted = self.payload["accepted"]
        outstanding = {
            (item["group"], item["slot"])
            for item in self.payload["executions"].values()
            if not item.get("settled")
        }
        if len(outstanding) > self.settings.max_concurrent_executions:
            raise RuntimeError("restored executions exceed configured concurrency")
        await self._check_stop_request()
        # Restore existing permits first; never submit C new tasks beside C old ones.
        for group, slot in sorted(jobs, key=lambda item: item not in outstanding):
            if f"{group}:{slot}" not in finished:
                queue.put_nowait((group, slot))
        if self.payload.get("stopped"):
            self.stop.set()
            # Resume a durable early-stop decision by finishing its barrier,
            # never by skipping the already-owned executions.
            pending = [
                key
                for key, item in self.payload["executions"].items()
                if not item.get("settled")
            ]
            cleanup = await asyncio.gather(
                *(self._settle_cancel(key) for key in pending), return_exceptions=True
            )
            failures = [item for item in cleanup if isinstance(item, BaseException)]
            if failures:
                raise RuntimeError("resumed cancellation barrier failed") from failures[
                    0
                ]
            if any(
                not item.get("settled") for item in self.payload["executions"].values()
            ):
                raise RuntimeError(
                    "resumed cancellation barrier left unsettled ownership"
                )
            if self.payload.get("operator_stop"):
                raise RuntimeError(
                    "operator stopped rollout after cancellation barrier"
                )
            return list(accepted)

        async def worker():
            while not self.stop.is_set():
                try:
                    group, slot = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    outcome = await operation(group, slot)
                    async with self.lock:
                        finished[f"{group}:{slot}"] = outcome
                        if all(
                            f"{group}:{index}" in finished for index in range(width)
                        ):
                            if group not in accepted and group_ready(group, finished):
                                accepted.append(group)
                            if early_stop and len(accepted) >= target:
                                self.payload["stopped"] = True
                        await self.save()
                        if self.payload["stopped"]:
                            self.stop.set()
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self.settings.max_concurrent_executions, queue.qsize()))
        ]
        watcher = asyncio.create_task(self._watch_stop_request())
        try:
            # On one fatal error stop submission, then let every other worker
            # perform cooperative cleanup; do not cancel cleanup coroutines.
            await asyncio.gather(*workers)
        except BaseException:
            self.stop.set()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if self._stop_error is not None:
            raise RuntimeError("operator stop monitoring failed") from self._stop_error
        unresolved = [
            key
            for key, item in self.payload["executions"].items()
            if not item.get("settled")
        ]
        if unresolved:
            raise RuntimeError(f"{len(unresolved)} executions remain unsettled")
        if self.payload.get("operator_stop"):
            raise RuntimeError("operator stopped rollout after cancellation barrier")
        return list(accepted)
