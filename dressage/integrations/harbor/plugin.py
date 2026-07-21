"""Harbor JobPlugin lifecycle integration.

The module avoids importing Harbor at import time so ``import dressage`` keeps
working when the optional, Python-3.12-only Harbor dependency is absent.
Concrete Harbor objects are received through the JobPlugin protocol and are
accessed through their public attributes.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable, Mapping, Protocol
from urllib.parse import urlparse
import uuid

from .artifacts import (
    AttemptFailure,
    FailureStage,
    HarborArtifactStore,
    HarborTrajectoryBundle,
    ProxyClientProtocol,
)


logger = logging.getLogger(__name__)


class RoutingPolicyError(RuntimeError):
    """A Harbor task cannot provide the requested no-bypass guarantee."""


class AttemptPhase(str, Enum):
    STARTING = "starting"
    READY = "ready"
    ACTIVE = "active"
    DRAINING = "draining"
    FINALIZING = "finalizing"
    BROKEN = "broken"
    CLOSED = "closed"


class AttemptHandleProtocol(Protocol):
    proxy: Any

    async def open_turn(
        self, turn_id: str, backend_session_id: str | None = None
    ) -> None: ...

    async def quiesce(self, timeout: float | None = None) -> Mapping[str, Any]: ...

    async def mark_broken(self, reason: str) -> None: ...

    async def close(self, *, tombstone: bool = True) -> None: ...


class GatewayLeaseProtocol(Protocol):
    async def register(self, spec: Any) -> AttemptHandleProtocol: ...

    async def release(self) -> None: ...


class SecretSlotRegistryProtocol(Protocol):
    def create(
        self, *, job_id: str, slot_id: str, env_name: str | None = None
    ) -> str: ...

    def current(self, slot_id: str) -> str: ...

    def fingerprint(self, slot_id: str) -> str: ...

    def rotate(self, slot_id: str) -> str: ...

    def delete(self, slot_id: str) -> None: ...

    def close_job(self, job_id: str) -> None: ...


class GatewayRuntimeProtocol(Protocol):
    secret_slots: SecretSlotRegistryProtocol

    async def acquire(self, config: Any) -> GatewayLeaseProtocol: ...


@dataclass(frozen=True)
class TrialBinding:
    """Programmatic training identity assigned before a Harbor Job runs."""

    instance_id: str
    attempt_ordinal: int
    expected_weight_version: str | None = None

    @classmethod
    def from_value(cls, value: "TrialBinding | Mapping[str, Any]") -> "TrialBinding":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("trial binding must be a TrialBinding or mapping")
        unknown = set(value) - {
            "instance_id",
            "attempt_ordinal",
            "expected_weight_version",
        }
        if unknown:
            raise ValueError(
                "unknown trial binding fields: " + ", ".join(sorted(unknown))
            )
        instance_id = str(value.get("instance_id") or "").strip()
        if not instance_id:
            raise ValueError("trial binding instance_id must not be empty")
        attempt_ordinal = value.get("attempt_ordinal")
        if isinstance(attempt_ordinal, bool) or not isinstance(attempt_ordinal, int):
            raise TypeError("trial binding attempt_ordinal must be an integer")
        if attempt_ordinal < 0:
            raise ValueError("trial binding attempt_ordinal must be non-negative")
        expected = value.get("expected_weight_version")
        return cls(
            instance_id=instance_id,
            attempt_ordinal=attempt_ordinal,
            expected_weight_version=None if expected is None else str(expected),
        )


@dataclass
class SlotRecord:
    slot_id: str
    env_name: str
    trial_name: str
    instance_id: str
    agent_fingerprint: str
    resolved_task_sha256: str
    task_network_class: str
    bound_attempt_ordinal: int | None = None
    expected_weight_version: str | None = None
    attempt_ids: list[str] = field(default_factory=list)
    previous_token_fingerprint: str | None = None


@dataclass
class AttemptRecord:
    trial_name: str
    trial_id: str
    session_id: str
    instance_id: str
    attempt_ordinal: int
    token_fingerprint: str
    resolved_task_sha256: str
    task_network_class: str
    expected_weight_version: str | None = None
    handle: AttemptHandleProtocol | None = None
    phase: AttemptPhase = AttemptPhase.STARTING
    turn_sequence: int = 0
    cancel_requested: bool = False
    poisoned: bool = False
    failures: list[AttemptFailure] = field(default_factory=list)
    finalize_task: asyncio.Task[HarborTrajectoryBundle | None] | None = None
    bundle: HarborTrajectoryBundle | None = None


class DressageHarborPlugin:
    """Bridge Harbor Trial hooks to the shared Dressage Gateway.

    Dependencies are injectable to make lifecycle tests independent from both
    Harbor and a live Uvicorn server.  The no-argument/kwargs constructor is
    still suitable for Harbor's entry-point loader.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        config_path: str | Path | None = None,
        gateway_runtime: GatewayRuntimeProtocol | Any | None = None,
        artifact_store: HarborArtifactStore | None = None,
        proxy_client: ProxyClientProtocol | None = None,
        compat: ModuleType | Any | None = None,
        route_spec_factory: Callable[..., Any] | None = None,
        trial_bindings: Mapping[str, TrialBinding | Mapping[str, Any]] | None = None,
        defer_final_selection: bool = False,
        **_: Any,
    ) -> None:
        if isinstance(config, (str, Path)):
            if config_path is not None:
                raise ValueError("pass only one of config or config_path when using a path")
            config_path = config
            config = None
        self.config = config if config is not None else _load_config(config_path)
        self._runtime = gateway_runtime
        self._artifact_store = artifact_store
        self._proxy_client = proxy_client
        self._owns_proxy_client = proxy_client is None
        self._compat = compat
        self._route_spec_factory = route_spec_factory
        self._trial_bindings = {
            str(name): TrialBinding.from_value(value)
            for name, value in (trial_bindings or {}).items()
        }
        self._defer_final_selection = bool(defer_final_selection)

        self._lease: GatewayLeaseProtocol | None = None
        self._job_id: str | None = None
        self._slots: dict[str, SlotRecord] = {}
        self._attempts: dict[tuple[str, str], AttemptRecord] = {}
        self._results: dict[tuple[str, str], HarborTrajectoryBundle] = {}
        self._network_audits: dict[str, str] = {}
        self._task_dir_audits: dict[Path, tuple[str, str]] = {}
        self._started = False
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._background_close_tasks: set[asyncio.Task[Any]] = set()

    @property
    def results(self) -> Mapping[tuple[str, str], HarborTrajectoryBundle]:
        return dict(self._results)

    @property
    def attempts(self) -> Mapping[tuple[str, str], AttemptRecord]:
        return dict(self._attempts)

    @property
    def routing_summary(self) -> Mapping[str, int | str]:
        """Actual unique resolved-task network classes seen by this Job."""

        return {
            "routing_guarantee": str(
                _config_get(
                    self.config,
                    "security.routing_guarantee",
                    "configure_only",
                )
            ),
            "public_network_tasks": sum(
                value == "public" for value in self._network_audits.values()
            ),
            "restricted_network_tasks": sum(
                value == "restricted" for value in self._network_audits.values()
            ),
        }

    @property
    def resolved_task_network_classes(self) -> Mapping[str, str]:
        """Stable resolved-task digest to actual network class."""

        return dict(self._network_audits)

    def get_result(
        self, trial_name: str, trial_id: str | uuid.UUID
    ) -> HarborTrajectoryBundle | None:
        return self._results.get((str(trial_name), str(trial_id)))

    async def on_job_start(self, job: Any) -> None:
        """Validate/mutate pending configs and register all Trial hooks."""

        if self._started:
            raise RuntimeError("DressageHarborPlugin is already attached to a job")
        if bool(getattr(job, "is_resuming", False)):
            raise ValueError("Harbor job resume is not supported by this integration")

        self._job_id = str(getattr(job, "id"))
        self._compat = self._compat or _load_compat()
        pending = list(_pending_trial_configs(self._compat, job))
        if not pending:
            raise ValueError("Harbor job has no pending TrialConfigs")
        pending_names = {str(getattr(item, "trial_name")) for item in pending}
        if "" in pending_names or len(pending_names) != len(pending):
            raise ValueError("pending Harbor TrialConfigs must have unique non-empty names")
        unknown_bindings = sorted(set(self._trial_bindings) - pending_names)
        if unknown_bindings:
            raise ValueError(
                "trial_bindings contains unknown Harbor trial names: "
                + ", ".join(unknown_bindings)
            )
        # Resolve all fail-fast policy decisions before allocating a listener
        # or creating the first secret slot.  A CLI construction error must
        # never leave a usable route credential behind.
        _backend_headers(self.config)
        trial_network: dict[str, tuple[str, str]] = {}
        for trial_config in pending:
            trial_name = str(getattr(trial_config, "trial_name"))
            candidate_agent = getattr(trial_config, "agent")
            protocol = _agent_protocol(candidate_agent, self.config)
            _reject_alternative_auth(candidate_agent, protocol)
            task_dir = _existing_task_directory(getattr(trial_config, "task", None))
            if task_dir is None:
                raise RoutingPolicyError(
                    "cannot locate resolved Harbor task.toml before allocating "
                    "Gateway or route credentials"
                )
            cached = self._task_dir_audits.get(task_dir)
            if cached is None:
                cached = _audit_task_network_file(task_dir)
                self._task_dir_audits[task_dir] = cached
            digest, network_class = cached
            existing_class = self._network_audits.setdefault(digest, network_class)
            if existing_class != network_class:
                raise RoutingPolicyError(
                    f"resolved task {digest} produced inconsistent network classes"
                )
            trial_network[trial_name] = (digest, network_class)
            if network_class == "public" and _routing_is_enforced(self.config):
                raise RoutingPolicyError(
                    f"resolved Harbor task {digest} uses public network access and "
                    "cannot satisfy security.routing_guarantee='enforced'"
                )

        public_count = sum(value == "public" for value in self._network_audits.values())
        if public_count and not _routing_is_enforced(self.config):
            logger.warning(
                "Harbor Job %s contains %d unique public-network task(s); "
                "configure_only injects Dressage routing but does not claim isolation",
                self._job_id,
                public_count,
            )

        runtime = self._runtime or _default_gateway_runtime()
        self._runtime = runtime
        self._lease = await _maybe_await(runtime.acquire(self.config))
        registry = runtime.secret_slots

        try:
            gateway_url = _gateway_public_url(self.config, self._lease)
        except Exception:
            await self.aclose()
            raise
        seen_names: set[str] = set()
        for trial_config in pending:
            trial_name = str(getattr(trial_config, "trial_name"))
            if not trial_name or trial_name in seen_names:
                raise ValueError(
                    f"pending Harbor TrialConfig has duplicate/empty trial_name {trial_name!r}"
                )
            seen_names.add(trial_name)

            original_agent = getattr(trial_config, "agent")
            original_key = str(
                getattr(original_agent, "concurrency_key", _stable_dump(original_agent))
            )
            agent = _deep_copy_model(original_agent)
            if (
                getattr(agent, "n_concurrent", None) is not None
                and getattr(agent, "concurrency_group", None) is None
            ):
                agent.concurrency_group = (
                    "dressage:" + hashlib.sha256(original_key.encode()).hexdigest()[:24]
                )

            protocol = _agent_protocol(agent, self.config)
            _reject_alternative_auth(agent, protocol)
            task_fingerprint = _stable_dump(getattr(trial_config, "task", None))
            agent_fingerprint = hashlib.sha256(original_key.encode()).hexdigest()
            instance_id = _logical_instance_id(
                self._job_id,
                task_fingerprint,
                agent_fingerprint,
            )
            binding = self._trial_bindings.get(trial_name)
            if binding is not None:
                instance_id = binding.instance_id
            resolved_task_sha256, task_network_class = trial_network[trial_name]
            slot_id = f"{self._job_id}:{trial_name}"
            env_name = "DRESSAGE_HARBOR_ROUTE_" + hashlib.sha256(
                slot_id.encode()
            ).hexdigest()[:24].upper()
            try:
                token = registry.create(
                    job_id=self._job_id,
                    slot_id=slot_id,
                    env_name=env_name,
                )
                if inspect.isawaitable(token):
                    raise TypeError("SecretSlotRegistry.create must be synchronous")
                token = registry.current(slot_id)
                os.environ[env_name] = token
                _inject_agent_routing(agent, self.config, env_name, gateway_url)
                trial_config.agent = agent
                self._slots[trial_name] = SlotRecord(
                    slot_id=slot_id,
                    env_name=env_name,
                    trial_name=trial_name,
                    instance_id=instance_id,
                    agent_fingerprint=agent_fingerprint,
                    resolved_task_sha256=resolved_task_sha256,
                    task_network_class=task_network_class,
                    bound_attempt_ordinal=(
                        binding.attempt_ordinal if binding is not None else None
                    ),
                    expected_weight_version=(
                        binding.expected_weight_version if binding is not None else None
                    ),
                )
            except Exception:
                await self.aclose()
                raise

        try:
            self._ensure_artifact_dependencies()
            job.on_trial_started(self._safe(self._on_start, "START"))
            job.on_agent_started(self._safe(self._on_agent_start, "AGENT_START"))
            job.on_agent_ended(self._safe(self._on_agent_end, "AGENT_END"))
            job.on_verification_started(
                self._safe(self._on_verification_start, "VERIFICATION_START")
            )
            job.on_trial_cancelled(self._safe(self._on_cancel, "CANCEL"))
            job.on_trial_ended(self._safe(self._on_end, "END"))
        except Exception:
            await self.aclose()
            raise
        self._started = True

    async def on_job_end(self, job_result: Any) -> None:
        """Reconcile Harbor's final attempt IDs and release all resources."""

        final_keys = {
            (str(getattr(result, "trial_name")), str(getattr(result, "id")))
            for result in (getattr(job_result, "trial_results", None) or [])
        }
        try:
            if self._artifact_store is not None and not self._defer_final_selection:
                summary = self.routing_summary
                await self._artifact_store.write_job_manifest(
                    job_result,
                    final_keys=sorted(final_keys),
                    routing_guarantee=str(summary["routing_guarantee"]),
                    public_network_tasks=int(summary["public_network_tasks"]),
                    restricted_network_tasks=int(
                        summary["restricted_network_tasks"]
                    ),
                )
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Idempotently close routes, secrets, clients, and the Gateway lease."""

        async with self._close_lock:
            if self._closed:
                return
            if self._close_task is None:
                self._close_task = asyncio.create_task(
                    self._aclose_once(),
                    name=f"harbor-plugin-close-{self._job_id or 'unstarted'}",
                )
            close_task = self._close_task
        # Caller cancellation must not interrupt cleanup half-way through.
        await asyncio.shield(close_task)

    def _track_background_close(self, task: asyncio.Task[Any], label: str) -> None:
        self._background_close_tasks.add(task)

        def completed(done: asyncio.Task[Any]) -> None:
            self._background_close_tasks.discard(done)
            if done.cancelled():
                return
            try:
                done.result()
            except Exception:
                logger.warning("background Harbor %s failed", label, exc_info=True)

        task.add_done_callback(completed)

    async def _wait_close_tasks(
        self,
        tasks: set[asyncio.Task[Any]],
        *,
        timeout: float,
        label: str,
    ) -> set[asyncio.Task[Any]]:
        if not tasks:
            return set()
        done, pending = await asyncio.wait(tasks, timeout=max(timeout, 0.0))
        for task in done:
            if task.cancelled():
                continue
            try:
                task.result()
            except Exception:
                logger.warning("Harbor %s failed during close", label, exc_info=True)
        return set(pending)

    async def _aclose_once(self) -> None:
        """Run bounded cleanup once; slow operations may finish in background."""

        drain_timeout = max(
            float(_config_get(self.config, "gateway.limits.drain_timeout_sec", 60.0)),
            0.1,
        )
        total_timeout = max(drain_timeout * 2.0, 5.0)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + total_timeout
        try:
            # END commits are shielded from Harbor callback cancellation.  Give
            # them most of the deadline, then cancel their coordinators so
            # shutdown cannot wait forever on an unhealthy proxy/filesystem.
            finalizers = {
                record.finalize_task
                for record in self._attempts.values()
                if record.finalize_task is not None
            }
            pending_finalizers = await self._wait_close_tasks(
                finalizers,
                timeout=total_timeout * 0.65,
                label="END finalizer",
            )
            if pending_finalizers:
                logger.warning(
                    "timed out waiting for %d Harbor END finalizer(s)",
                    len(pending_finalizers),
                )
                for record in self._attempts.values():
                    if record.finalize_task in pending_finalizers:
                        record.poisoned = True
                        record.failures.append(
                            AttemptFailure(
                                code="FINALIZER_CLOSE_TIMEOUT",
                                stage=FailureStage.CLEANUP,
                                message="END finalizer exceeded the plugin close deadline",
                                fatal=True,
                                retryable=True,
                            )
                        )
                for task in pending_finalizers:
                    task.cancel()
                    self._track_background_close(task, "END finalizer cancellation")
                if self._artifact_store is not None:
                    cancel_commits = asyncio.create_task(
                        self._artifact_store.cancel_pending_commits(),
                        name="harbor-cancel-pending-artifacts",
                    )
                    pending_cancel = await self._wait_close_tasks(
                        {cancel_commits},
                        timeout=max(deadline - loop.time(), 0.0) * 0.2,
                        label="artifact cancellation",
                    )
                    for task in pending_cancel:
                        self._track_background_close(task, "artifact cancellation")

            route_tasks = {
                asyncio.create_task(
                    self._close_attempt(record),
                    name=f"harbor-close-route-{record.trial_id}",
                )
                for record in self._attempts.values()
            }
            pending_routes = await self._wait_close_tasks(
                route_tasks,
                timeout=max(deadline - loop.time(), 0.0) * 0.55,
                label="route close",
            )
            for task in pending_routes:
                self._track_background_close(task, "route close")

            final_tasks: set[asyncio.Task[Any]] = set()
            runtime = self._runtime
            if runtime is not None and self._job_id is not None:
                try:
                    value = runtime.secret_slots.close_job(self._job_id)
                    if inspect.isawaitable(value):
                        final_tasks.add(
                            asyncio.ensure_future(value)
                        )
                except Exception:
                    logger.warning("failed to close Harbor secret slots", exc_info=True)
            for slot in self._slots.values():
                os.environ.pop(slot.env_name, None)

            lease = self._lease
            self._lease = None
            if lease is not None:
                final_tasks.add(
                    asyncio.create_task(
                        lease.release(),
                        name="harbor-release-gateway-lease",
                    )
                )
            if self._owns_proxy_client and self._proxy_client is not None:
                close = getattr(self._proxy_client, "close", None)
                if close is not None:
                    final_tasks.add(
                        asyncio.create_task(
                            _maybe_await(close()),
                            name="harbor-close-proxy-client",
                        )
                    )
            pending_final = await self._wait_close_tasks(
                final_tasks,
                timeout=max(deadline - loop.time(), 0.0),
                label="resource release",
            )
            for task in pending_final:
                self._track_background_close(task, "resource release")
        finally:
            self._closed = True

    def _safe(
        self,
        callback: Callable[[Any], Awaitable[None]],
        event_name: str,
    ) -> Callable[[Any], Awaitable[None]]:
        async def wrapper(event: Any) -> None:
            try:
                await callback(event)
            except Exception as exc:
                logger.warning(
                    "Dressage Harbor %s hook failed for trial %s/%s",
                    event_name,
                    getattr(event, "trial_name", "unknown"),
                    getattr(event, "trial_id", "unknown"),
                    exc_info=True,
                )
                record = self._record_for_event(event)
                if record is not None:
                    record.poisoned = True
                    record.failures.append(
                        AttemptFailure.from_exception(
                            "PLUGIN_HOOK_FAILED",
                            FailureStage.LIFECYCLE,
                            exc,
                            details={"event": event_name},
                        )
                    )
                    await self._mark_broken(record, f"{event_name}: {exc}")
                if isinstance(exc, RoutingPolicyError):
                    # Continuing after a policy failure would run the Agent with
                    # public egress and turn "enforced" into configure-only.
                    raise

        return wrapper

    async def _on_start(self, event: Any) -> None:
        trial_name, trial_id = _event_key(event)
        slot = self._slots.get(trial_name)
        if slot is None:
            raise KeyError(f"no Dressage slot for Harbor trial {trial_name!r}")
        if (trial_name, trial_id) in self._attempts:
            return
        if self._runtime is None or self._lease is None:
            raise RuntimeError("Gateway lease is not active")

        registry = self._runtime.secret_slots
        token = registry.current(slot.slot_id)
        fingerprint = registry.fingerprint(slot.slot_id)
        record = AttemptRecord(
            trial_name=trial_name,
            trial_id=trial_id,
            session_id=trial_id,
            instance_id=slot.instance_id,
            attempt_ordinal=len(slot.attempt_ids),
            token_fingerprint=fingerprint,
            resolved_task_sha256=slot.resolved_task_sha256,
            task_network_class=slot.task_network_class,
            expected_weight_version=slot.expected_weight_version,
        )
        if slot.bound_attempt_ordinal is not None:
            record.attempt_ordinal = slot.bound_attempt_ordinal
        self._attempts[(trial_name, trial_id)] = record
        slot.attempt_ids.append(trial_id)

        if (
            slot.previous_token_fingerprint is not None
            and fingerprint == slot.previous_token_fingerprint
        ):
            record.poisoned = True
            record.phase = AttemptPhase.BROKEN
            record.failures.append(
                AttemptFailure(
                    code="ROUTE_TOKEN_NOT_ROTATED",
                    stage=FailureStage.START,
                    message="Harbor retry reused the previous attempt route token",
                    fatal=True,
                )
            )
            return

        spec = self._make_route_spec(
            trial_name=trial_name,
            trial_id=trial_id,
            instance_id=slot.instance_id,
            token=token,
            expected_version=slot.expected_weight_version,
        )
        record.handle = await self._lease.register(spec)
        record.phase = AttemptPhase.READY

    async def _on_agent_start(self, event: Any) -> None:
        record = self._require_record(event)
        if record.phase is AttemptPhase.ACTIVE:
            return
        if record.phase is not AttemptPhase.READY or record.handle is None:
            raise RuntimeError(
                f"AGENT_START is invalid while attempt is {record.phase.value}"
            )
        turn_id = f"{record.trial_id}:{record.turn_sequence}"
        await record.handle.open_turn(turn_id)
        record.turn_sequence += 1
        record.phase = AttemptPhase.ACTIVE

    async def _on_agent_end(self, event: Any) -> None:
        record = self._require_record(event)
        if record.phase is AttemptPhase.READY:
            return
        if record.phase in {AttemptPhase.BROKEN, AttemptPhase.CLOSED}:
            return
        if record.phase is not AttemptPhase.ACTIVE:
            raise RuntimeError(f"AGENT_END is invalid while attempt is {record.phase.value}")
        await self._quiesce(record)

    async def _on_verification_start(self, event: Any) -> None:
        record = self._require_record(event)
        if record.phase is AttemptPhase.ACTIVE:
            record.poisoned = True
            record.failures.append(
                AttemptFailure(
                    code="VERIFICATION_WHILE_ROUTE_ACTIVE",
                    stage=FailureStage.LIFECYCLE,
                    message="verification began before AGENT_END quiesced the model route",
                    fatal=True,
                )
            )
            await self._quiesce(record)

    async def _on_cancel(self, event: Any) -> None:
        record = self._record_for_event(event)
        if record is None:
            return
        record.cancel_requested = True
        record.poisoned = True
        if record.phase is AttemptPhase.ACTIVE:
            await self._quiesce(record)
        # Deliberately do not finalize/read/close/rotate here.  Harbor emits END
        # from Trial._finalize() after CANCEL.

    async def _on_end(self, event: Any) -> None:
        record = self._require_record(event)
        if record.finalize_task is None:
            record.finalize_task = asyncio.create_task(
                self._finalize_attempt(record, event),
                name=f"harbor-end-{record.trial_id}",
            )
        await asyncio.shield(record.finalize_task)

    async def _finalize_attempt(
        self,
        record: AttemptRecord,
        event: Any,
    ) -> HarborTrajectoryBundle | None:
        try:
            if record.phase is AttemptPhase.ACTIVE:
                await self._quiesce(record)
            if record.cancel_requested:
                # CANCEL is followed by END in Harbor.  END is still the sole
                # cleanup owner, but cancelled sessions are diagnostics-only
                # and must never be finalized/read into a training bundle.
                return None
            if record.phase is not AttemptPhase.BROKEN:
                record.phase = AttemptPhase.FINALIZING

            if self._artifact_store is None or self._proxy_client is None:
                raise RuntimeError("artifact dependencies were not initialized")
            trial_dir = _trial_dir(event)
            bundle = await self._artifact_store.commit_attempt(
                proxy_client=self._proxy_client,
                job_id=self._job_id or "unknown",
                trial_name=record.trial_name,
                trial_id=record.trial_id,
                session_id=record.session_id,
                instance_id=record.instance_id,
                attempt_ordinal=record.attempt_ordinal,
                trial_result=getattr(event, "result", None),
                failures=record.failures,
                cancelled=record.cancel_requested,
                expected_weight_version=(
                    record.expected_weight_version
                    if record.expected_weight_version is not None
                    else _config_get(
                        self.config, "training.expected_weight_version", None
                    )
                ),
                routing_guarantee=str(
                    _config_get(
                        self.config,
                        "security.routing_guarantee",
                        "configure_only",
                    )
                ),
                task_network_class=record.task_network_class,
                label=None,
                trial_dir=trial_dir,
            )
            record.bundle = bundle
            self._results[(record.trial_name, record.trial_id)] = bundle
            return bundle
        finally:
            closed = await self._close_attempt(record)
            if closed:
                self._rotate_slot_after_end(record)
            else:
                self._results.pop((record.trial_name, record.trial_id), None)
                if record.bundle is not None and self._artifact_store is not None:
                    failure = record.failures[-1]
                    try:
                        await self._artifact_store.invalidate_bundle(
                            record.bundle,
                            failure,
                        )
                    except Exception:
                        logger.warning(
                            "failed to invalidate bundle after route close failure",
                            exc_info=True,
                        )

    async def _quiesce(self, record: AttemptRecord) -> None:
        if record.handle is None:
            record.phase = AttemptPhase.BROKEN
            return
        record.phase = AttemptPhase.DRAINING
        errors = await record.handle.quiesce(
            timeout=float(
                _config_get(self.config, "gateway.limits.drain_timeout_sec", 60.0)
            )
        )
        for name, payload in dict(errors or {}).items():
            if payload is None:
                continue
            record.poisoned = True
            record.failures.append(
                AttemptFailure(
                    code=f"PROXY_{str(name).upper()}",
                    stage=FailureStage.AGENT,
                    message=f"RolloutLLMProxy reported {name}",
                    fatal=True,
                    retryable=name in {"failed_upstream", "drain_timeout"},
                    details={"payload": _sanitize_details(payload)},
                )
            )
        record.phase = AttemptPhase.READY

    async def _mark_broken(self, record: AttemptRecord, reason: str) -> None:
        if record.phase is AttemptPhase.CLOSED:
            return
        record.phase = AttemptPhase.BROKEN
        if record.handle is not None:
            try:
                await record.handle.mark_broken(reason)
            except Exception:
                logger.warning("failed to mark Gateway route broken", exc_info=True)

    async def _close_attempt(self, record: AttemptRecord) -> bool:
        if record.phase is AttemptPhase.CLOSED:
            return True
        if record.handle is not None:
            try:
                await record.handle.close(tombstone=True)
            except Exception as exc:
                logger.warning(
                    "failed to close Harbor route for %s", record.trial_id, exc_info=True
                )
                record.phase = AttemptPhase.BROKEN
                record.poisoned = True
                record.failures.append(
                    AttemptFailure.from_exception(
                        "ROUTE_CLOSE_FAILED",
                        FailureStage.CLEANUP,
                        exc,
                        retryable=True,
                    )
                )
                return False
        record.phase = AttemptPhase.CLOSED
        return True

    def _rotate_slot_after_end(self, record: AttemptRecord) -> None:
        slot = self._slots.get(record.trial_name)
        if slot is None or self._runtime is None:
            return
        slot.previous_token_fingerprint = record.token_fingerprint
        rotated = self._runtime.secret_slots.rotate(slot.slot_id)
        if inspect.isawaitable(rotated):
            raise TypeError("SecretSlotRegistry.rotate must be synchronous")
        os.environ[slot.env_name] = self._runtime.secret_slots.current(slot.slot_id)

    def _record_for_event(self, event: Any) -> AttemptRecord | None:
        return self._attempts.get(_event_key(event))

    def _require_record(self, event: Any) -> AttemptRecord:
        record = self._record_for_event(event)
        if record is None:
            trial_name, trial_id = _event_key(event)
            raise KeyError(f"unknown Harbor attempt {trial_name}/{trial_id}")
        return record

    def _make_route_spec(self, **identity: Any) -> Any:
        factory = self._route_spec_factory or _default_route_spec_factory()
        upstream_headers = _backend_headers(self.config)
        expected_version = identity.pop("expected_version", None)
        sampling = _config_get(self.config, "training.sampling", None)
        seed_policy = getattr(sampling, "seed_policy", "per_attempt_request")
        seed_material = (
            f"{identity.get('instance_id', '')}\0{identity.get('trial_id', '')}".encode()
        )
        return factory(
            **identity,
            upstream_origin=_backend_url(self.config),
            router_api_path=str(
                _config_first(
                    self.config,
                    ("backend.router_api_path", "gateway.backend.router_api_path"),
                    "/v1",
                )
            ),
            sticky_header_name=str(
                _config_get(self.config, "backend.sticky_header_name", "X-Session-Id")
            ),
            model_override=_config_get(self.config, "training.model_override", None),
            expected_version=(
                expected_version
                if expected_version is not None
                else _config_get(
                    self.config, "training.expected_weight_version", None
                )
            ),
            upstream_headers=upstream_headers,
            sampling_mode=getattr(sampling, "mode", None),
            sampling_temperature=getattr(sampling, "temperature", None),
            sampling_top_p=getattr(sampling, "top_p", None),
            sampling_seed_base=(
                int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
                if seed_policy == "per_attempt_request"
                else None
            ),
            verify_tls=bool(_config_get(self.config, "backend.verify_tls", True)),
            max_steps=_config_get(self.config, "trajectory.max_steps", 100),
            default_temperature=_config_get(
                self.config, "trajectory.default_temperature", None
            ),
            debug_log_dir=_config_get(self.config, "artifacts.debug_log_dir", None),
        )

    def _ensure_artifact_dependencies(self) -> None:
        if self._proxy_client is None:
            # Keep this import lazy: loading the Harbor entry point should not
            # import the full Dressage proxy server (and tokenizer stack).
            from dressage.proxy.proxy_client import ProxyClient

            drain_timeout = float(
                _config_get(self.config, "gateway.limits.drain_timeout_sec", 60.0)
            )
            self._proxy_client = ProxyClient(
                _backend_url(self.config),
                timeout=max(min(drain_timeout / 4.0, 30.0), 1.0),
                default_headers=_backend_headers(self.config),
                verify=bool(_config_get(self.config, "backend.verify_tls", True)),
            )
        if self._artifact_store is None:
            root = _config_get(self.config, "artifacts.root", None)
            if root is None:
                root = Path.cwd() / ".dressage" / "harbor-artifacts"
            file_mode = _parse_mode(_config_get(self.config, "artifacts.file_mode", 0o600))
            dir_mode = _parse_mode(_config_get(self.config, "artifacts.dir_mode", 0o700))
            self._artifact_store = HarborArtifactStore(
                root,
                run_id=_config_get(self.config, "run_id", None),
                reward_key=str(_config_get(self.config, "training.reward_key", "reward")),
                mode=str(_config_get(self.config, "artifacts.mode", "both")),
                require_token_versions=bool(
                    _config_get(
                        self.config,
                        "training.require_single_weight_version",
                        False,
                    )
                ),
                require_trainable_tokens=bool(
                    _config_get(
                        self.config,
                        "trajectory.require_trainable_tokens",
                        True,
                    )
                ),
                file_mode=file_mode,
                dir_mode=dir_mode,
                fsync=bool(_config_get(self.config, "artifacts.fsync", True)),
            )


def _load_config(config_path: str | Path | None) -> Any:
    from .config import HarborIntegrationConfig, load_config

    if config_path is not None:
        return load_config(config_path)
    env_path = os.environ.get("DRESSAGE_HARBOR_INTEGRATION_CONFIG")
    if env_path:
        return _load_config(env_path)
    return HarborIntegrationConfig()


def _load_compat() -> ModuleType:
    from . import compat

    return compat


def _pending_trial_configs(compat: Any, job: Any) -> list[Any]:
    for name in ("pending_trial_configs", "get_pending_trial_configs"):
        function = getattr(compat, name, None)
        if function is not None:
            return list(function(job))
    raise AttributeError("Harbor compat module must expose pending_trial_configs(job)")


def _default_gateway_runtime() -> Any:
    from .gateway import GatewayRuntime

    return GatewayRuntime.get()


def _default_route_spec_factory() -> Callable[..., Any]:
    from .gateway import RouteSpec

    return RouteSpec


def _inject_agent_routing(
    agent: Any,
    config: Any,
    env_name: str,
    advertise_url: str,
) -> None:
    env = dict(getattr(agent, "env", {}) or {})
    advertise_url = advertise_url.rstrip("/")
    token_template = "${" + env_name + "}"
    protocol = _agent_protocol(agent, config)
    if protocol in {"openai", "both"}:
        env.update({
            "OPENAI_API_KEY": token_template,
            "OPENAI_BASE_URL": f"{advertise_url}/v1",
        })
    if protocol in {"anthropic", "both"}:
        env.update({
            "ANTHROPIC_API_KEY": token_template,
            "ANTHROPIC_AUTH_TOKEN": token_template,
            "ANTHROPIC_BASE_URL": advertise_url,
        })
        model_override = _config_get(config, "training.model_override", None)
        if model_override:
            model = str(model_override)
            env.update(
                {
                    "ANTHROPIC_MODEL": model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                    "CLAUDE_CODE_SUBAGENT_MODEL": model,
                }
            )
    agent.env = env
    hostname = urlparse(advertise_url).hostname
    hosts = list(getattr(agent, "extra_allowed_hosts", None) or [])
    if hostname:
        if hostname not in hosts:
            hosts.append(hostname)
    for host in _config_get(config, "security.additional_agent_egress_hosts", ()) or ():
        normalized = str(host).strip()
        if normalized and normalized not in hosts:
            hosts.append(normalized)
    agent.extra_allowed_hosts = hosts


def _routing_is_enforced(config: Any) -> bool:
    return _config_get(config, "security.routing_guarantee", "configure_only") == "enforced"


def _existing_task_directory(task_reference: Any) -> Path | None:
    getter = getattr(task_reference, "get_local_path", None)
    if not callable(getter):
        return None
    try:
        path = Path(getter()).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return None
    return path if (path / "task.toml").is_file() else None


def _network_mode(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).strip().lower().replace("_", "-")


def _task_network_class(task_config: Any) -> tuple[str, str | None]:
    environment = getattr(task_config, "environment", None)
    baseline = _network_mode(getattr(environment, "network_mode", "public"))
    if baseline not in {"no-network", "allowlist"}:
        return (
            "public",
            f"[environment].network_mode={baseline!r}",
        )

    phases: list[tuple[str, Any]] = [("agent", getattr(task_config, "agent", None))]
    for step in getattr(task_config, "steps", None) or []:
        phases.append((f"steps.{getattr(step, 'name', '?')}.agent", getattr(step, "agent", None)))
    for label, phase in phases:
        explicit = _network_mode(getattr(phase, "network_mode", None))
        effective = explicit or baseline
        if effective not in {"no-network", "allowlist"}:
            return (
                "public",
                f"[{label}].network_mode={effective!r}",
            )
    return "restricted", None


def _validate_task_network_config(task_config: Any, *, source: Path) -> None:
    network_class, reason = _task_network_class(task_config)
    if network_class == "public":
        raise RoutingPolicyError(
            f"{source}: {reason} cannot satisfy "
            "security.routing_guarantee='enforced'"
        )


def _audit_task_network_file(task_dir: Path) -> tuple[str, str]:
    config_path = task_dir / "task.toml"
    try:
        from harbor.models.task.config import TaskConfig

        task_config = TaskConfig.model_validate_toml(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RoutingPolicyError(
            f"cannot validate Harbor task network policy at {config_path}: {exc}"
        ) from exc
    payload = task_config.model_dump(mode="json", exclude_none=False)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    network_class, _ = _task_network_class(task_config)
    return digest, network_class


def _agent_protocol(agent: Any, config: Any) -> str:
    name = str(getattr(agent, "name", None) or "")
    import_path = str(getattr(agent, "import_path", None) or "")
    overrides = _config_get(config, "agent_protocol_overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise TypeError("agent_protocol_overrides must be a mapping")
    override = overrides.get(name) or overrides.get(import_path)
    if override is not None:
        protocol = str(override).lower()
        if protocol not in {"openai", "anthropic", "both"}:
            raise ValueError(
                f"invalid protocol override {override!r} for Agent {name or import_path!r}"
            )
        return protocol
    if name == "claude-code":
        return "anthropic"
    if name in {"codex", "qwen-coder"}:
        return "openai"
    raise ValueError(
        f"unsupported Harbor Agent {name or import_path!r}; configure "
        "agent_protocol_overrides as openai, anthropic, or both"
    )


_ANTHROPIC_ALTERNATIVE_AUTH_ENV = {
    "CLAUDE_FORCE_OAUTH",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
    "AWS_BEDROCK_RUNTIME_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "GOOGLE_APPLICATION_CREDENTIALS",
}

_OPENAI_ALTERNATIVE_AUTH_ENV = {
    "CODEX_FORCE_AUTH_JSON",
    "CODEX_AUTH_JSON_PATH",
}


def _reject_alternative_auth(agent: Any, protocol: str) -> None:
    env = dict(getattr(agent, "env", {}) or {})
    names: set[str] = set()
    if protocol in {"anthropic", "both"}:
        names.update(_ANTHROPIC_ALTERNATIVE_AUTH_ENV)
    if protocol in {"openai", "both"}:
        names.update(_OPENAI_ALTERNATIVE_AUTH_ENV)
    active = sorted(
        name
        for name in names
        if str(env.get(name) or os.environ.get(name) or "").strip()
    )
    if active:
        raise ValueError(
            "alternative Agent authentication would bypass the Dressage Gateway: "
            + ", ".join(active)
        )


def _logical_instance_id(job_id: str, task: str, agent: str) -> str:
    digest = hashlib.sha256(f"{job_id}\0{task}\0{agent}".encode()).hexdigest()[:32]
    return f"harbor-{digest}"


def _stable_dump(value: Any) -> str:
    if value is None:
        return "null"
    dump = getattr(value, "model_dump", None)
    if dump is not None:
        try:
            value = dump(mode="json", exclude_none=True)
        except TypeError:
            value = dump()
    elif hasattr(value, "__dict__"):
        value = vars(value)
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _deep_copy_model(value: Any) -> Any:
    model_copy = getattr(value, "model_copy", None)
    return model_copy(deep=True) if model_copy is not None else copy.deepcopy(value)


def _event_key(event: Any) -> tuple[str, str]:
    return str(getattr(event, "trial_name")), str(getattr(event, "trial_id"))


def _trial_dir(event: Any) -> Path | None:
    config = getattr(event, "config", None)
    trials_dir = getattr(config, "trials_dir", None)
    if trials_dir is None:
        return None
    return Path(trials_dir) / str(getattr(event, "trial_name"))


def _backend_url(config: Any) -> str:
    value = _config_first(
        config,
        (
            "backend.proxy_url",
            "backend.dressage_proxy_url",
            "gateway.backend.dressage_proxy_url",
            "router_base_url",
        ),
        None,
    )
    if not value:
        raise ValueError("Dressage backend proxy URL must be configured")
    return str(value).rstrip("/")


def _backend_headers(config: Any) -> dict[str, str]:
    backend = _config_get(config, "backend", None)
    service_headers = getattr(backend, "service_headers", None)
    if service_headers is not None:
        return dict(
            service_headers(
                os.environ,
                required=_config_get(config, "execution_mode", "rollout")
                == "training",
            )
        )
    env_name = _config_first(
        config,
        ("backend.service_api_key_env", "gateway.backend.service_api_key_env"),
        None,
    )
    if not env_name:
        if _config_get(config, "execution_mode", "rollout") == "training":
            raise ValueError("backend.service_api_key_env is required for training")
        return {}
    secret = os.environ.get(str(env_name))
    if not secret:
        if _config_get(config, "execution_mode", "rollout") == "training":
            raise ValueError(f"backend service credential env {env_name!r} is not set")
        return {}
    header = str(_config_get(config, "backend.service_api_key_header", "Authorization"))
    scheme_value = _config_get(config, "backend.service_api_key_scheme", "Bearer")
    scheme = "" if scheme_value is None else str(scheme_value)
    return {header: f"{scheme} {secret}".strip()}


def _gateway_public_url(config: Any, lease: Any) -> str:
    configured = _config_get(config, "gateway.advertise_url", None)
    if configured is not None:
        return str(configured).rstrip("/")
    for attribute in ("public_url", "advertise_url", "url"):
        value = getattr(lease, attribute, None)
        if value:
            return str(value).rstrip("/")
    raise ValueError(
        "Gateway lease did not expose a public URL for an ephemeral listener"
    )


def _config_first(
    config: Any,
    paths: tuple[str, ...],
    default: Any,
) -> Any:
    sentinel = object()
    for path in paths:
        value = _config_get(config, path, sentinel)
        if value is not sentinel and value is not None:
            return value
    return default


def _config_get(config: Any, path: str, default: Any) -> Any:
    value = config
    for part in path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return default
            value = value[part]
        else:
            if not hasattr(value, part):
                return default
            value = getattr(value, part)
    return value


def _sanitize_details(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        sensitive = {"authorization", "api-key", "x-api-key", "cookie", "token"}
        return {
            str(key): "<redacted>"
            if str(key).lower() in sensitive
            else _sanitize_details(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_details(item) for item in value]
    return str(value)


def _parse_mode(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text, 8)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


__all__ = [
    "AttemptPhase",
    "AttemptRecord",
    "DressageHarborPlugin",
    "RoutingPolicyError",
    "SlotRecord",
    "TrialBinding",
]
