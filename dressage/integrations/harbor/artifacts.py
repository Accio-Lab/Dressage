"""Durable Harbor trajectory bundles.

This module deliberately has no Harbor import.  Harbor is an optional Python
3.12-only integration while the core Dressage package remains importable on
Python 3.10.  The small amount of Harbor result introspection below therefore
uses the public object attributes rather than concrete Harbor model classes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence
import uuid

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "dressage.harbor.trajectory/v2"


logger = logging.getLogger(__name__)


class ProxyClientProtocol(Protocol):
    """Subset of :class:`ProxyClient` needed to commit a trajectory."""

    async def finalize_session(
        self,
        session_id: str,
        *,
        instance_id: str | None = None,
        label: Any | None = None,
    ) -> dict[str, Any]: ...

    async def read_trajectory(
        self,
        *,
        trajectory_id: str | None = None,
        session_id: str | None = None,
        instance_id: str | None = None,
        max_groups: int | None = None,
        segment_view: str | None = None,
        drain: bool = False,
    ) -> dict[str, Any]: ...


class SessionPayloadWriterProtocol(Protocol):
    """Legacy trajectory-payload writer used by the core rollout paths."""

    async def write_session_payload(
        self,
        trajectory_payload: dict[str, Any],
        *,
        session_id: str,
        instance_id: str,
    ) -> Path | None: ...

    async def drain(self) -> None: ...


class FailureStage(str, Enum):
    START = "start"
    AGENT = "agent"
    VERIFICATION = "verification"
    FINALIZE = "finalize"
    READ = "read"
    VALIDATE = "validate"
    WRITE = "write"
    CLEANUP = "cleanup"
    LIFECYCLE = "lifecycle"


class FinalizationCheckpoint(str, Enum):
    NOT_STARTED = "not_started"
    SESSION_FINALIZED = "session_finalized"
    TRAJECTORY_SNAPSHOTTED = "trajectory_snapshotted"
    ARTIFACT_COMMITTED = "artifact_committed"
    STORE_DRAINED = "store_drained"


class AttemptFailure(BaseModel):
    """A sanitized, machine-actionable attempt failure."""

    model_config = ConfigDict(extra="forbid")

    code: str
    stage: FailureStage | str
    message: str
    fatal: bool = True
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_exception(
        cls,
        code: str,
        stage: FailureStage | str,
        exc: BaseException,
        *,
        fatal: bool = True,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> "AttemptFailure":
        return cls(
            code=code,
            stage=stage,
            message=_one_line(str(exc)) or type(exc).__name__,
            fatal=fatal,
            retryable=retryable,
            details={"exception_type": type(exc).__name__, **dict(details or {})},
        )


class HarborTrajectoryBundle(BaseModel):
    """Versioned output of one *physical* Harbor Trial attempt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dressage.harbor.trajectory/v2"] = SCHEMA_VERSION
    run_id: str
    job_id: str
    trial_name: str
    trial_id: str
    session_id: str
    instance_id: str
    attempt_ordinal: int = 0

    task_name: str | None = None
    task_checksum: str | None = None
    agent_name: str | None = None
    routing_guarantee: Literal["configure_only", "enforced"]
    task_network_class: Literal["public", "restricted"]
    cancelled: bool = False
    # A physical attempt is not a final training artifact until job/group
    # reconciliation selects it.  Crash-before-reconciliation is therefore
    # fail-closed even when the token/reward payload itself is valid.
    superseded: bool = True
    reconciliation_generation: str | None = None

    rewards: dict[str, int | float] = Field(default_factory=dict)
    reward_key: str
    reward: int | float | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    trainable_token_count: int = 0

    expected_weight_version: str | None = None
    observed_weight_versions: list[str] = Field(default_factory=list)
    failures: list[AttemptFailure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trainable: bool = False
    finalization_checkpoint: FinalizationCheckpoint = (
        FinalizationCheckpoint.NOT_STARTED
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactCommitError(RuntimeError):
    """The trajectory could not be durably committed."""


class HarborArtifactStore:
    """Exactly-once, process-local trajectory commit coordinator.

    The store intentionally snapshots with ``drain=False`` and only drains the
    Dressage trajectory store after the bundle has been durably written.  A
    repeated END event awaits the original commit task instead of issuing a
    second non-idempotent ``finalize_session`` request.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str | None = None,
        reward_key: str = "reward",
        mode: str = "both",
        require_token_versions: bool = False,
        require_trainable_tokens: bool = True,
        file_mode: int = 0o600,
        dir_mode: int = 0o700,
        fsync: bool = True,
        session_payload_writer: SessionPayloadWriterProtocol | None = None,
    ) -> None:
        if mode not in {"disk", "memory", "both"}:
            raise ValueError("artifact mode must be 'disk', 'memory', or 'both'")
        if not reward_key:
            raise ValueError("reward_key must not be empty")
        if session_payload_writer is None:
            # Keep the core rollout writer import lazy so Harbor's optional
            # integration module remains inexpensive to import.
            from dressage.rollout.artifacts.writer import RolloutArtifactWriter

            session_payload_writer = RolloutArtifactWriter()
        self.root = Path(root).expanduser().resolve()
        self.run_id = run_id or uuid.uuid4().hex
        self.reward_key = reward_key
        self.mode = mode
        self.require_token_versions = require_token_versions
        self.require_trainable_tokens = require_trainable_tokens
        self.file_mode = file_mode
        self.dir_mode = dir_mode
        self.fsync = fsync
        self._session_payload_writer = session_payload_writer
        self._commit_tasks: dict[
            tuple[str, str], asyncio.Task[HarborTrajectoryBundle]
        ] = {}
        self._bundles: dict[tuple[str, str], HarborTrajectoryBundle] = {}

    @property
    def bundles(self) -> Mapping[tuple[str, str], HarborTrajectoryBundle]:
        return dict(self._bundles)

    async def cancel_pending_commits(self) -> None:
        """Cancel unfinished commit coordinators during bounded shutdown.

        Atomic file replacement may still finish in a worker thread, but every
        pre-reconciliation bundle is superseded and therefore fail-closed.
        """

        tasks = [task for task in self._commit_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def invalidate_bundle(
        self,
        bundle: HarborTrajectoryBundle,
        failure: AttemptFailure,
    ) -> None:
        """Durably make a previously committed bundle non-trainable."""

        bundle.failures = _deduplicate_failures([*bundle.failures, failure])
        bundle.superseded = True
        bundle.trainable = False
        self._bundles[(bundle.trial_name, bundle.trial_id)] = bundle
        if self.mode in {"disk", "both"}:
            await asyncio.to_thread(
                self._write_model_atomic,
                self.bundle_path(bundle),
                bundle,
            )

    async def commit_attempt(
        self,
        *,
        proxy_client: ProxyClientProtocol,
        job_id: str,
        trial_name: str,
        trial_id: str,
        session_id: str,
        instance_id: str,
        attempt_ordinal: int,
        trial_result: Any,
        failures: Sequence[AttemptFailure] = (),
        cancelled: bool = False,
        expected_weight_version: str | None = None,
        routing_guarantee: Literal["configure_only", "enforced"],
        task_network_class: Literal["public", "restricted"],
        label: Any | None = None,
        trial_dir: str | Path | None = None,
    ) -> HarborTrajectoryBundle:
        """Finalize, snapshot, validate, persist, then drain one attempt.

        The key includes the actual Harbor ``trial_id`` because retries reuse a
        logical TrialConfig/trial name while creating a new physical Trial.
        """

        key = (str(trial_name), str(trial_id))
        task = self._commit_tasks.get(key)
        if task is None:
            task = asyncio.create_task(
                self._commit_attempt_once(
                    proxy_client=proxy_client,
                    job_id=str(job_id),
                    trial_name=str(trial_name),
                    trial_id=str(trial_id),
                    session_id=str(session_id),
                    instance_id=str(instance_id),
                    attempt_ordinal=attempt_ordinal,
                    trial_result=trial_result,
                    failures=list(failures),
                    cancelled=cancelled,
                    expected_weight_version=expected_weight_version,
                    routing_guarantee=routing_guarantee,
                    task_network_class=task_network_class,
                    label=label,
                    trial_dir=Path(trial_dir).resolve() if trial_dir else None,
                ),
                name=f"harbor-artifact-{trial_id}",
            )
            self._commit_tasks[key] = task
        return await asyncio.shield(task)

    async def _commit_attempt_once(
        self,
        *,
        proxy_client: ProxyClientProtocol,
        job_id: str,
        trial_name: str,
        trial_id: str,
        session_id: str,
        instance_id: str,
        attempt_ordinal: int,
        trial_result: Any,
        failures: list[AttemptFailure],
        cancelled: bool,
        expected_weight_version: str | None,
        routing_guarantee: Literal["configure_only", "enforced"],
        task_network_class: Literal["public", "restricted"],
        label: Any | None,
        trial_dir: Path | None,
    ) -> HarborTrajectoryBundle:
        checkpoint = FinalizationCheckpoint.NOT_STARTED
        trajectory_payload: dict[str, Any] | None = None

        try:
            await proxy_client.finalize_session(
                session_id,
                instance_id=instance_id,
                label=label,
            )
            checkpoint = FinalizationCheckpoint.SESSION_FINALIZED
        except Exception as exc:
            # The endpoint has exactly-once effect and caches its response, so
            # one retry handles a timeout that happened before the first
            # request reached the server.  If both responses are lost, a
            # complete-marker read proves whether the server committed.
            try:
                await proxy_client.finalize_session(
                    session_id,
                    instance_id=instance_id,
                    label=label,
                )
                checkpoint = FinalizationCheckpoint.SESSION_FINALIZED
            except Exception as retry_exc:
                try:
                    recovery = await proxy_client.read_trajectory(
                        trajectory_id=session_id,
                        instance_id=instance_id,
                        drain=False,
                    )
                except Exception as read_exc:
                    failures.append(
                        AttemptFailure.from_exception(
                            "FINALIZE_FAILED",
                            FailureStage.FINALIZE,
                            exc,
                            retryable=True,
                            details={
                                "retry_exception_type": type(retry_exc).__name__,
                                "recovery_read_exception_type": type(read_exc).__name__,
                            },
                        )
                    )
                else:
                    if _payload_is_complete(
                        recovery,
                        expected_trajectory_id=session_id,
                        expected_instance_id=instance_id,
                    ):
                        trajectory_payload = recovery
                        checkpoint = FinalizationCheckpoint.SESSION_FINALIZED
                    else:
                        failures.append(
                            AttemptFailure.from_exception(
                                "FINALIZE_FAILED",
                                FailureStage.FINALIZE,
                                exc,
                                retryable=True,
                                details={
                                    "retry_exception_type": type(retry_exc).__name__,
                                },
                            )
                        )

        if trajectory_payload is None and checkpoint is FinalizationCheckpoint.SESSION_FINALIZED:
            try:
                trajectory_payload = await proxy_client.read_trajectory(
                    trajectory_id=session_id,
                    instance_id=instance_id,
                    drain=False,
                )
            except Exception as exc:
                failures.append(
                    AttemptFailure.from_exception(
                        "TRAJECTORY_READ_FAILED",
                        FailureStage.READ,
                        exc,
                        retryable=True,
                    )
                )

        segments = _payload_segments(trajectory_payload)
        if trajectory_payload is not None:
            checkpoint = FinalizationCheckpoint.TRAJECTORY_SNAPSHOTTED
            try:
                await self._session_payload_writer.write_session_payload(
                    trajectory_payload,
                    session_id=session_id,
                    instance_id=instance_id,
                )
                # The writer defaults to background I/O.  Harbor must wait for
                # the compatibility copy before destructively draining the
                # authoritative Proxy trajectory store.
                await self._session_payload_writer.drain()
            except Exception:
                # This mirror is diagnostic compatibility output.  Its
                # failure must not change reward validation, trainability, or
                # the durable bundle/manifest commit path.
                logger.warning(
                    "failed to write legacy Harbor session payload for "
                    "session_id=%s instance_id=%s",
                    session_id,
                    instance_id,
                    exc_info=True,
                )

        rewards = _extract_rewards(trial_result)
        failures.extend(_harbor_result_failures(trial_result))
        validation_failures, warnings, token_count, versions = validate_attempt(
            rewards=rewards,
            reward_key=self.reward_key,
            segments=segments,
            require_token_versions=self.require_token_versions,
            expected_weight_version=expected_weight_version,
            require_trainable_tokens=self.require_trainable_tokens,
            expected_trajectory_id=session_id,
            expected_instance_id=instance_id,
        )
        failures.extend(validation_failures)

        bundle = HarborTrajectoryBundle(
            run_id=self.run_id,
            job_id=job_id,
            trial_name=trial_name,
            trial_id=trial_id,
            session_id=session_id,
            instance_id=instance_id,
            attempt_ordinal=attempt_ordinal,
            task_name=_string_attr(trial_result, "task_name"),
            task_checksum=_string_attr(trial_result, "task_checksum"),
            agent_name=_nested_string_attr(trial_result, "agent_info", "name"),
            routing_guarantee=routing_guarantee,
            task_network_class=task_network_class,
            cancelled=cancelled,
            rewards=rewards,
            reward_key=self.reward_key,
            reward=rewards.get(self.reward_key),
            segments=segments,
            trainable_token_count=token_count,
            expected_weight_version=expected_weight_version,
            observed_weight_versions=versions,
            failures=_deduplicate_failures(failures),
            warnings=warnings,
            trainable=False,
            finalization_checkpoint=checkpoint,
            started_at=_datetime_attr(trial_result, "started_at"),
            finished_at=_datetime_attr(trial_result, "finished_at"),
        )
        bundle.trainable = (
            not cancelled
            and checkpoint is FinalizationCheckpoint.TRAJECTORY_SNAPSHOTTED
            and not any(failure.fatal for failure in bundle.failures)
            and token_count > 0
        )

        if self.mode in {"disk", "both"}:
            output_path = self.bundle_path(bundle)
            _ensure_outside_trial_dir(output_path, trial_dir)
            try:
                await asyncio.to_thread(self._write_model_atomic, output_path, bundle)
            except Exception as exc:
                raise ArtifactCommitError(
                    f"failed to persist Harbor bundle for {trial_name}/{trial_id}: "
                    f"{_one_line(str(exc))}"
                ) from exc
            checkpoint = FinalizationCheckpoint.ARTIFACT_COMMITTED
            bundle.finalization_checkpoint = checkpoint
            # Persist the checkpoint itself before the destructive read.
            await asyncio.to_thread(self._write_model_atomic, output_path, bundle)
        else:
            checkpoint = FinalizationCheckpoint.ARTIFACT_COMMITTED
            bundle.finalization_checkpoint = checkpoint

        if trajectory_payload is not None:
            try:
                await proxy_client.read_trajectory(
                    trajectory_id=session_id,
                    instance_id=instance_id,
                    drain=True,
                )
            except Exception as exc:
                bundle.failures.append(
                    AttemptFailure.from_exception(
                        "TRAJECTORY_DRAIN_FAILED",
                        FailureStage.CLEANUP,
                        exc,
                        fatal=False,
                        retryable=True,
                    )
                )
            else:
                checkpoint = FinalizationCheckpoint.STORE_DRAINED
                bundle.finalization_checkpoint = checkpoint

        # A failed cleanup does not invalidate already durable training data.
        if self.mode in {"disk", "both"}:
            await asyncio.to_thread(self._write_model_atomic, self.bundle_path(bundle), bundle)
        self._bundles[(trial_name, trial_id)] = bundle
        return bundle

    def bundle_path(self, bundle: HarborTrajectoryBundle) -> Path:
        return (
            self.root
            / _safe_part(bundle.run_id)
            / _safe_part(bundle.job_id)
            / "trials"
            / _safe_part(bundle.trial_name)
            / "attempts"
            / f"{bundle.attempt_ordinal:04d}-{_safe_part(bundle.trial_id)}"
            / "bundle.json"
        )

    async def write_job_manifest(
        self,
        job_result: Any,
        *,
        final_keys: Iterable[tuple[str, str]],
        routing_guarantee: Literal["configure_only", "enforced"],
        public_network_tasks: int,
        restricted_network_tasks: int,
        state: str = "completed",
    ) -> Path | None:
        """Write final-attempt pointers and one job-level reconciliation index."""

        if state not in {"completed", "partial", "aborted"}:
            raise ValueError("job manifest state must be completed, partial, or aborted")

        keys = [(str(name), str(trial_id)) for name, trial_id in final_keys]
        job_id = _string_attr(job_result, "id") or _job_id_from_keys(self._bundles, keys)
        if job_id is None:
            return None

        finals: list[dict[str, Any]] = []
        pointer_writes: list[tuple[Path, dict[str, Any]]] = []
        final_key_set = set(keys)
        generation = uuid.uuid4().hex
        job_bundles = {
            key: bundle
            for key, bundle in self._bundles.items()
            if bundle.job_id == job_id
        }
        for key, bundle in job_bundles.items():
            selected = key in final_key_set
            bundle.superseded = not selected
            bundle.reconciliation_generation = generation
            if not selected:
                code = "JOB_ABORTED" if state == "aborted" else "ATTEMPT_NOT_SELECTED"
                message = (
                    "Harbor Job aborted before artifact selection"
                    if state == "aborted"
                    else "physical attempt was not selected by final reconciliation"
                )
                bundle.failures = _deduplicate_failures(
                    [
                        *bundle.failures,
                        AttemptFailure(
                            code=code,
                            stage=FailureStage.LIFECYCLE,
                            message=message,
                            fatal=True,
                            retryable=False,
                        ),
                    ]
                )
                bundle.trainable = False

        for key in keys:
            bundle = self._bundles.get(key)
            if bundle is None:
                finals.append(
                    {
                        "trial_name": key[0],
                        "trial_id": key[1],
                        "missing": True,
                        "generation": generation,
                    }
                )
                continue
            record = {
                "trial_name": bundle.trial_name,
                "trial_id": bundle.trial_id,
                "bundle": str(self.bundle_path(bundle)),
                "trainable": bundle.trainable,
                "generation": generation,
            }
            finals.append(record)
            if self.mode in {"disk", "both"}:
                final_path = self.bundle_path(bundle).parents[2] / "final.json"
                pointer_writes.append((final_path, record))

        selected_trial_names = {name for name, _ in keys}
        for trial_name in sorted(
            {bundle.trial_name for bundle in job_bundles.values()}
            - selected_trial_names
        ):
            marker = {
                "trial_name": trial_name,
                "missing": True,
                "state": state,
                "trainable": False,
                "generation": generation,
            }
            finals.append(marker)
            if self.mode in {"disk", "both"}:
                representative = next(
                    bundle
                    for bundle in job_bundles.values()
                    if bundle.trial_name == trial_name
                )
                final_path = self.bundle_path(representative).parents[2] / "final.json"
                pointer_writes.append((final_path, marker))

        manifest = {
            "schema_version": "dressage.harbor.job/v2",
            "run_id": self.run_id,
            "job_id": job_id,
            "state": state,
            "routing_guarantee": routing_guarantee,
            "public_network_tasks": public_network_tasks,
            "restricted_network_tasks": restricted_network_tasks,
            "generation": generation,
            "committed": False,
            "final_trials": finals,
            "attempts": [
                {
                    "trial_name": bundle.trial_name,
                    "trial_id": bundle.trial_id,
                    "attempt_ordinal": bundle.attempt_ordinal,
                    "trainable": bundle.trainable,
                    "superseded": bundle.superseded,
                    "generation": bundle.reconciliation_generation,
                    "bundle": str(self.bundle_path(bundle)),
                }
                for bundle in job_bundles.values()
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.mode not in {"disk", "both"}:
            return None

        path = self.root / _safe_part(self.run_id) / _safe_part(job_id) / "manifest.json"
        # The manifest is the commit record.  Publish an uncommitted
        # generation before changing bundle convenience fields or pointers,
        # so any partial multi-file write remains fail-closed to readers.
        await asyncio.to_thread(self._write_json_atomic, path, manifest)
        for bundle in job_bundles.values():
            await asyncio.to_thread(
                self._write_model_atomic,
                self.bundle_path(bundle),
                bundle,
            )
        for final_path, record in pointer_writes:
            await asyncio.to_thread(self._write_json_atomic, final_path, record)
        manifest["committed"] = True
        manifest["committed_at"] = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(self._write_json_atomic, path, manifest)
        return path

    def _write_model_atomic(self, path: Path, model: BaseModel) -> None:
        self._write_json_atomic(path, model.model_dump(mode="json"))

    def _write_json_atomic(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=self.dir_mode)
        current = path.parent
        while current == self.root or self.root in current.parents:
            os.chmod(current, self.dir_mode)
            if current == self.root:
                break
            current = current.parent
        temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                self.file_mode,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, self.file_mode)
            if self.fsync:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _trajectory_finalization_failures(
    segments: Sequence[Mapping[str, Any]],
    *,
    expected_trajectory_id: str | None = None,
    expected_instance_id: str | None = None,
) -> list[AttemptFailure]:
    """Prove the snapshot is one complete, atomically finalized view."""

    failures: list[AttemptFailure] = []
    indices: list[int] = []
    counts: list[int] = []
    trajectory_ids: set[str] = set()
    instance_ids: set[str] = set()
    views: set[str] = set()
    finalization_ids: set[str] = set()
    marker_missing = False
    identity_field_missing = False

    for ordinal, segment in enumerate(segments):
        index = segment.get("segment_index")
        count = segment.get("segment_count")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            failures.append(
                _failure(
                    "SEGMENT_INDEX_INVALID",
                    f"segment {ordinal} has an invalid segment_index",
                    details={"segment_ordinal": ordinal},
                )
            )
        else:
            indices.append(index)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            failures.append(
                _failure(
                    "SEGMENT_COUNT_INVALID",
                    f"segment {ordinal} has an invalid segment_count",
                    details={"segment_ordinal": ordinal},
                )
            )
        else:
            counts.append(count)

        trajectory_id = segment.get("trajectory_id") or segment.get("session_id")
        instance_id = segment.get("instance_id")
        if isinstance(trajectory_id, str) and trajectory_id:
            trajectory_ids.add(trajectory_id)
        else:
            identity_field_missing = True
        if isinstance(instance_id, str) and instance_id:
            instance_ids.add(instance_id)
        else:
            identity_field_missing = True

        extra_info = segment.get("extra_info")
        if not isinstance(extra_info, Mapping):
            marker_missing = True
            continue
        view = extra_info.get("segment_view")
        if isinstance(view, str) and view:
            views.add(view)
        else:
            identity_field_missing = True
        finalization_id = extra_info.get("finalization_id")
        if (
            extra_info.get("finalization_complete") is not True
            or not isinstance(finalization_id, str)
            or not finalization_id
        ):
            marker_missing = True
        else:
            finalization_ids.add(finalization_id)

    if counts:
        expected_counts = set(counts)
        expected_count = next(iter(expected_counts))
        if (
            len(expected_counts) != 1
            or expected_count != len(segments)
            or len(indices) != len(segments)
            or set(indices) != set(range(expected_count))
            or len(indices) != len(set(indices))
        ):
            failures.append(
                _failure(
                    "TRAJECTORY_INCOMPLETE",
                    "trajectory segments are not the complete 0..N-1 set",
                    details={
                        "observed_indices": sorted(set(indices)),
                        "observed_counts": sorted(expected_counts),
                        "observed_segments": len(segments),
                    },
                )
            )

    identity_mismatch = (
        expected_trajectory_id is not None
        and trajectory_ids != {expected_trajectory_id}
    ) or (
        expected_instance_id is not None
        and instance_ids != {expected_instance_id}
    )
    if (
        len(trajectory_ids) != 1
        or len(instance_ids) != 1
        or len(views) != 1
        or identity_field_missing
        or identity_mismatch
    ):
        failures.append(
            _failure(
                "TRAJECTORY_IDENTITY_INVALID",
                "trajectory segments do not share one trajectory, instance, and view",
            )
        )
    if marker_missing or len(finalization_ids) != 1:
        failures.append(
            _failure(
                "FINALIZATION_MARKER_MISSING",
                "trajectory does not carry one complete finalization marker",
            )
        )
    return failures


def validate_attempt(
    *,
    rewards: Mapping[str, Any],
    reward_key: str,
    segments: Sequence[Mapping[str, Any]],
    require_token_versions: bool,
    expected_weight_version: str | None,
    require_trainable_tokens: bool = True,
    expected_trajectory_id: str | None = None,
    expected_instance_id: str | None = None,
) -> tuple[list[AttemptFailure], list[str], int, list[str]]:
    """Validate reward and raw token arrays without mutating the segments."""

    failures: list[AttemptFailure] = []
    warnings: list[str] = []
    reward = rewards.get(reward_key)
    if reward_key not in rewards:
        failures.append(
            _failure("REWARD_MISSING", f"reward key {reward_key!r} is missing")
        )
    elif isinstance(reward, bool) or not isinstance(reward, (int, float)):
        failures.append(_failure("REWARD_INVALID", "reward must be an int or float"))
    elif not math.isfinite(float(reward)):
        failures.append(_failure("REWARD_NON_FINITE", "reward must be finite"))

    if not segments:
        failures.append(_failure("TRAJECTORY_EMPTY", "trajectory has no segments"))
        return failures, warnings, 0, []

    failures.extend(
        _trajectory_finalization_failures(
            segments,
            expected_trajectory_id=expected_trajectory_id,
            expected_instance_id=expected_instance_id,
        )
    )

    trainable_count = 0
    observed_versions: list[str] = []
    seen_versions: set[str] = set()
    for segment_index, segment in enumerate(segments):
        arrays: dict[str, list[Any]] = {}
        for field in ("tokens", "full_loss_mask", "full_logprobs"):
            value = segment.get(field)
            if not isinstance(value, list):
                failures.append(
                    _failure(
                        "SEGMENT_FIELD_INVALID",
                        f"segment {segment_index} field {field!r} must be a list",
                        details={"segment_index": segment_index, "field": field},
                    )
                )
                value = []
            arrays[field] = value

        token_count = len(arrays["tokens"])
        for field in ("full_loss_mask", "full_logprobs"):
            if len(arrays[field]) != token_count:
                failures.append(
                    _failure(
                        "SEGMENT_LENGTH_MISMATCH",
                        f"segment {segment_index}: tokens={token_count}, "
                        f"{field}={len(arrays[field])}",
                        details={"segment_index": segment_index, "field": field},
                    )
                )

        if any(isinstance(token, bool) or not isinstance(token, int) for token in arrays["tokens"]):
            failures.append(
                _failure(
                    "TOKENS_INVALID",
                    f"segment {segment_index} tokens must contain only integers",
                    details={"segment_index": segment_index},
                )
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in arrays["full_logprobs"]
        ):
            failures.append(
                _failure(
                    "LOGPROBS_INVALID",
                    f"segment {segment_index} logprobs must be finite numbers",
                    details={"segment_index": segment_index},
                )
            )

        loss_mask = arrays["full_loss_mask"]
        invalid_mask = [value for value in loss_mask if value not in (0, 1, False, True)]
        if invalid_mask:
            failures.append(
                _failure(
                    "LOSS_MASK_INVALID",
                    f"segment {segment_index} loss mask must contain only 0/1",
                    details={"segment_index": segment_index},
                )
            )

        versions_raw = segment.get("full_versions")
        needs_versions = require_token_versions or expected_weight_version is not None
        if needs_versions and not isinstance(versions_raw, list):
            failures.append(
                _failure(
                    "TOKEN_VERSIONS_MISSING",
                    f"segment {segment_index} has no full_versions",
                    details={"segment_index": segment_index},
                )
            )
            versions_raw = []
        elif versions_raw is None:
            versions_raw = []
        elif not isinstance(versions_raw, list):
            failures.append(
                _failure(
                    "TOKEN_VERSIONS_INVALID",
                    f"segment {segment_index} full_versions must be a list",
                    details={"segment_index": segment_index},
                )
            )
            versions_raw = []

        if versions_raw and len(versions_raw) != token_count:
            failures.append(
                _failure(
                    "SEGMENT_LENGTH_MISMATCH",
                    f"segment {segment_index}: tokens={token_count}, "
                    f"full_versions={len(versions_raw)}",
                    details={"segment_index": segment_index, "field": "full_versions"},
                )
            )

        for index, mask in enumerate(loss_mask[:token_count]):
            if mask not in (1, True):
                continue
            trainable_count += 1
            version = versions_raw[index] if index < len(versions_raw) else None
            valid_version = (
                isinstance(version, str)
                and bool(version.strip())
                and version.strip().lower() != "unknown"
            )
            if needs_versions and not valid_version:
                failures.append(
                    _failure(
                        "TRAINABLE_TOKEN_VERSION_UNKNOWN",
                        f"segment {segment_index} trainable token {index} has no version",
                        details={
                            "segment_index": segment_index,
                            "token_index": index,
                        },
                    )
                )
                continue
            if valid_version:
                assert isinstance(version, str)
                if version not in seen_versions:
                    seen_versions.add(version)
                    observed_versions.append(version)

    if require_trainable_tokens and trainable_count == 0:
        failures.append(
            _failure("NO_TRAINABLE_TOKENS", "trajectory has no trainable tokens")
        )
    if require_token_versions and len(seen_versions) > 1:
        failures.append(
            _failure(
                "MULTIPLE_WEIGHT_VERSIONS",
                "trainable tokens span multiple weight versions",
                details={"versions": observed_versions},
            )
        )
    if expected_weight_version is not None and seen_versions != {expected_weight_version}:
        failures.append(
            _failure(
                "WEIGHT_VERSION_MISMATCH",
                "observed trainable token versions do not match the expected version",
                details={
                    "expected": expected_weight_version,
                    "observed": observed_versions,
                },
            )
        )
    return failures, warnings, trainable_count, observed_versions


def _payload_segments(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [dict(item) for item in data if isinstance(item, Mapping)]


def _payload_is_complete(
    payload: Mapping[str, Any] | None,
    *,
    expected_trajectory_id: str | None = None,
    expected_instance_id: str | None = None,
) -> bool:
    segments = _payload_segments(payload)
    return bool(segments) and not _trajectory_finalization_failures(
        segments,
        expected_trajectory_id=expected_trajectory_id,
        expected_instance_id=expected_instance_id,
    )


def _extract_rewards(result: Any) -> dict[str, int | float]:
    verifier_result = getattr(result, "verifier_result", None)
    rewards = getattr(verifier_result, "rewards", None)
    if rewards is None:
        for step in reversed(getattr(result, "step_results", None) or []):
            step_verifier = getattr(step, "verifier_result", None)
            rewards = getattr(step_verifier, "rewards", None)
            if rewards is not None:
                break
    if not isinstance(rewards, Mapping):
        return {}
    # Preserve malformed values long enough for validate_attempt to reject
    # them.  Pydantic validation happens only after validation has completed.
    return dict(rewards)  # type: ignore[return-value]


def _harbor_result_failures(result: Any) -> list[AttemptFailure]:
    failures: list[AttemptFailure] = []
    top = getattr(result, "exception_info", None)
    if top is not None:
        failures.append(_exception_info_failure(top, step_name=None))
    for step in getattr(result, "step_results", None) or []:
        exception_info = getattr(step, "exception_info", None)
        if exception_info is not None:
            failures.append(
                _exception_info_failure(
                    exception_info,
                    step_name=_string_attr(step, "step_name"),
                )
            )
    return failures


def _exception_info_failure(info: Any, *, step_name: str | None) -> AttemptFailure:
    exception_type = _string_attr(info, "exception_type") or "HarborException"
    message = _string_attr(info, "exception_message") or exception_type
    details = {"exception_type": exception_type}
    if step_name is not None:
        details["step_name"] = step_name
    return AttemptFailure(
        code="HARBOR_TRIAL_EXCEPTION",
        stage=FailureStage.AGENT,
        message=_one_line(message),
        fatal=True,
        retryable=exception_type not in {"CancelledError", "AgentTimeoutError"},
        details=details,
    )


def _failure(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> AttemptFailure:
    return AttemptFailure(
        code=code,
        stage=FailureStage.VALIDATE,
        message=message,
        fatal=True,
        details=dict(details or {}),
    )


def _deduplicate_failures(
    failures: Iterable[AttemptFailure],
) -> list[AttemptFailure]:
    result: list[AttemptFailure] = []
    seen: set[tuple[str, str, str]] = set()
    for failure in failures:
        key = (failure.code, str(failure.stage), failure.message)
        if key not in seen:
            seen.add(key)
            result.append(failure)
    return result


def _ensure_outside_trial_dir(path: Path, trial_dir: Path | None) -> None:
    if trial_dir is None:
        return
    try:
        path.relative_to(trial_dir)
    except ValueError:
        return
    raise ArtifactCommitError(
        f"artifact path {path} must be outside Harbor trial directory {trial_dir}"
    )


def _safe_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "unknown"


def _one_line(value: str) -> str:
    return " ".join(value.splitlines()).strip()


def _string_attr(value: Any, name: str) -> str | None:
    item = getattr(value, name, None)
    return None if item is None else str(item)


def _nested_string_attr(value: Any, parent: str, name: str) -> str | None:
    return _string_attr(getattr(value, parent, None), name)


def _datetime_attr(value: Any, name: str) -> datetime | None:
    item = getattr(value, name, None)
    return item if isinstance(item, datetime) else None


def _job_id_from_keys(
    bundles: Mapping[tuple[str, str], HarborTrajectoryBundle],
    keys: Sequence[tuple[str, str]],
) -> str | None:
    for key in keys:
        bundle = bundles.get(key)
        if bundle is not None:
            return bundle.job_id
    return None


__all__ = [
    "ArtifactCommitError",
    "AttemptFailure",
    "FailureStage",
    "FinalizationCheckpoint",
    "HarborArtifactStore",
    "HarborTrajectoryBundle",
    "ProxyClientProtocol",
    "SCHEMA_VERSION",
    "validate_attempt",
]
