from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import asyncpg
import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ACTIVE_STATES = (
    "ACCEPTED",
    "CANCELLING",
    "PREPARING",
    "WAITING_ROUTE",
    "AGENT_RUNNING",
    "VERIFYING",
)
HOLD_STATES = (*ACTIVE_STATES, "FAILED")
TERMINAL_STATES = ("SUCCEEDED", "FAILED", "CANCELLED")


def failure_reason(row: dict[str, Any]) -> str:
    """Display evidence-based reasons, not the broad retry-policy class."""
    if row.get("state") == "CANCELLED":
        return "主动取消"
    if row.get("agent_sigkill"):
        return "Agent退出137/SIGKILL（OOM等原因待核验）"
    if row.get("is_rapid_refill"):
        return "压缩快速回填止损"
    code = row.get("error_code") or row.get("failure_class")
    labels = {
        "ClaudeMaxTurnsError": "Agent轮数耗尽",
        "ContextWindowExceededError": "模型上下文超窗",
        "AgentTimeoutError": "Agent总时限耗尽",
        "AgentExecutionTimeoutError": "Agent总时限耗尽",
        "ImageDiskPressureError": "镜像磁盘水位不足",
        "ImagePullTransientError": "镜像拉取失败",
        "DockerAPIError": "Docker执行异常",
        "SANDBOX_RESOURCE_GUARD": "Sandbox资源保护清退",
        "EXECUTION_UNCERTAIN": "执行不确定/清理隔离",
        "ConnectError": "连接建立失败",
        "ConnectionError": "连接失败",
        "ReadTimeout": "传输读取超时",
        "ReadError": "传输读取失败",
        "ApiConnectionClosedError": "API连接中断",
        "APIConnectionError": "API连接失败",
        "NetworkConnectionError": "连接错误（原始分类，需核验退出码）",
        "VerifierTimeoutError": "验证超时",
        "AgentSetupTimeoutError": "Agent准备超时",
        "ApiRateLimitError": "推理API限流",
        "ApiOverloadedError": "推理API过载",
        "ApiInternalServerError": "推理API内部错误",
        "NonZeroAgentExitCodeError": "Agent非零退出（主因待核验）",
        "TASK_SETUP": "任务/环境准备失败",
        "RETRYABLE_INFRA": "基础设施异常（主因待核验）",
        "AGENT_RUNTIME": "Agent/验证执行异常（主因待核验）",
    }
    if row.get("is_seal_conflict"):
        return "路由收尾409（主因待核验，非网络定论）"
    if code == "HTTPStatusError":
        return "HTTP服务拒绝（非网络定论）"
    return labels.get(code, code or "未知失败（待核验）")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


class DashboardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkerSource(DashboardModel):
    worker_id: str
    observer_url: str

    @field_validator("observer_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value.rstrip("/"))
        if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
            raise ValueError("observer_url must be http://host:port")
        return value.rstrip("/")


class AlertSettings(DashboardModel):
    queue_oldest_sec: int = Field(default=120, ge=1)
    cpu_used_percent: float = Field(default=90.0, gt=0, le=100)
    memory_available_percent: float = Field(default=10.0, ge=0, lt=100)
    disk_free_bytes: int = Field(default=50 * 1024**3, ge=0)
    stale_warning_sec: int = Field(default=10, ge=1)
    stale_critical_sec: int = Field(default=30, ge=2)
    ownership_mismatch_sec: int = Field(default=60, ge=10)

    @model_validator(mode="after")
    def validate_stale(self) -> AlertSettings:
        if self.stale_critical_sec <= self.stale_warning_sec:
            raise ValueError("stale critical threshold must exceed warning threshold")
        return self


class DashboardConfig(DashboardModel):
    schema_version: str = "rollout-dashboard/v1"
    listen_host: str = Field(min_length=1)
    listen_port: int = Field(default=19600, ge=1, le=65535)
    refresh_interval_sec: float = Field(default=3.0, gt=0)
    history_interval_sec: float = Field(default=5.0, gt=0)
    history_retention_sec: int = Field(default=3600, ge=60)
    control_url: str
    database_url: str
    database_pool_size: int = Field(default=2, ge=1, le=4)
    database_statement_timeout_ms: int = Field(default=1000, ge=100, le=10000)
    node_timeout_sec: float = Field(default=2.0, gt=0, le=30)
    workers: list[WorkerSource] = Field(min_length=1)
    alerts: AlertSettings = Field(default_factory=AlertSettings)

    @model_validator(mode="after")
    def validate_config(self) -> DashboardConfig:
        if self.schema_version != "rollout-dashboard/v1":
            raise ValueError("unsupported dashboard schema")
        if self.history_interval_sec < self.refresh_interval_sec:
            raise ValueError(
                "history interval must not be shorter than refresh interval"
            )
        ids = [worker.worker_id for worker in self.workers]
        if len(ids) != len(set(ids)):
            raise ValueError("dashboard worker IDs must be unique")
        parsed = urlsplit(self.control_url.rstrip("/"))
        if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
            raise ValueError("control_url must be http://host:port")
        self.control_url = self.control_url.rstrip("/")
        return self


def load_dashboard_config(path: str | Path) -> DashboardConfig:
    payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("dashboard configuration must be a YAML object")
    return DashboardConfig.model_validate(payload)


class DatabaseReader:
    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self.pool: asyncpg.Pool | None = None

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def _pool(self) -> asyncpg.Pool:
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=1,
                max_size=self.config.database_pool_size,
                server_settings={
                    "application_name": "dressage-rollout-dashboard",
                    "default_transaction_read_only": "on",
                    "statement_timeout": str(self.config.database_statement_timeout_ms),
                    "lock_timeout": "100",
                },
            )
        return self.pool

    async def read(self) -> dict[str, Any]:
        pool = await self._pool()
        try:
            async with (
                pool.acquire() as connection,
                connection.transaction(readonly=True),
            ):
                worker_rows = await connection.fetch("""
                    SELECT worker_id,registration->>'incarnation_id' incarnation_id,state,
                           active_execution_ids AS reported_active_attempt_ids,last_heartbeat
                    FROM rollout_v4.workers ORDER BY worker_id""")
                active_rows = await connection.fetch("""
                    SELECT execution_id AS attempt_id,client_scope,worker_id,state,
                        body->'spec'->>'task_ref' task_ref,
                        (body->>'created_at')::timestamptz assigned_at,
                        (body->>'updated_at')::timestamptz updated_at
                    FROM rollout_v4.executions WHERE NOT cleanup_verified
                    ORDER BY sequence_no DESC LIMIT 200""")
                active_counts = await connection.fetch("""
                    SELECT worker_id,state,count(*) AS count FROM rollout_v4.executions
                    WHERE NOT cleanup_verified GROUP BY worker_id,state""")
                recent_counts = await connection.fetch(
                    """
                    SELECT state,count(*) AS count FROM rollout_v4.executions
                    WHERE state=ANY($1::text[]) AND
                        (body->>'updated_at')::timestamptz >= clock_timestamp()-interval '15 minutes'
                    GROUP BY state""",
                    list(TERMINAL_STATES),
                )
                recent_failures = await connection.fetch(
                    """
                    SELECT execution_id AS attempt_id,client_scope,worker_id,state,
                        body->'spec'->>'task_ref' task_ref,
                        COALESCE(body->'result'->>'failure_class',body->>'failure_code') failure_class,
                        body->'result'->>'failure_stage' failure_stage,
                        body->'result'->>'error_code' error_code,
                        COALESCE(body->'result'->>'error_message','') LIKE '%Command failed (exit 137)%' AS agent_sigkill,
                        COALESCE(body->'result'->>'error_message','') ILIKE '%rapid_refill_breaker%'
                            OR COALESCE(body->'result'->>'error_message','') ILIKE '%Autocompact is thrashing%' AS is_rapid_refill,
                        COALESCE(body->'result'->>'error_message','') ILIKE '%409%'
                            AND COALESCE(body->'result'->>'error_message','') ILIKE '%:seal%' AS is_seal_conflict,
                        (body->>'updated_at')::timestamptz updated_at
                    FROM rollout_v4.executions WHERE state=ANY($1::text[])
                    AND (body->>'updated_at')::timestamptz >= clock_timestamp()-interval '15 minutes'
                    ORDER BY sequence_no DESC LIMIT 50""",
                    ["FAILED", "CANCELLED"],
                )
                diagnostics = await connection.fetchrow("""
                    SELECT count(*) FILTER(WHERE state='ACCEPTED') AS queued_attempt_rows,
                        count(*) FILTER(WHERE state='SUCCEEDED'
                            AND body->'result'->>'agent_terminal_reason'='max_turns'
                            AND (body->>'updated_at')::timestamptz >= clock_timestamp()-interval '15 minutes') AS budget_ended_succeeded_15m,
                        count(*) FILTER(WHERE state='RESERVED') AS reserved_executions,
                        count(*) FILTER(WHERE body->'result' IS DISTINCT FROM 'null'::jsonb
                            AND NOT (body->>'result_acknowledged')::boolean) AS unacknowledged_results
                    FROM rollout_v4.executions""")
        except (asyncpg.PostgresError, TimeoutError):
            await self.close()
            raise

        def row_dict(row: asyncpg.Record) -> dict[str, Any]:
            return {key: _iso(value) for key, value in dict(row).items()}

        workers = [row_dict(row) for row in worker_rows]
        for worker in workers:
            active_ids = worker.get("reported_active_attempt_ids")
            if isinstance(active_ids, str):
                decoded = json.loads(active_ids)
                if not isinstance(decoded, list):
                    raise TypeError("reported active attempt IDs must be a JSON array")
                worker["reported_active_attempt_ids"] = decoded

        failures = [row_dict(row) for row in recent_failures]
        for row in failures:
            row["failure_reason"] = failure_reason(row)
        return {
            "workers": workers,
            "active_attempts": [row_dict(row) for row in active_rows],
            "active_counts": [row_dict(row) for row in active_counts],
            "recent_counts": [row_dict(row) for row in recent_counts],
            "recent_failures": failures,
            "diagnostics": row_dict(diagnostics),
        }


@dataclass
class CachedSource:
    data: Any = None
    last_success_at: datetime | None = None
    error: str | None = None

    def update(self, data: Any) -> None:
        self.data = data
        self.last_success_at = _utc_now()
        self.error = None

    def fail(self, exc: BaseException) -> None:
        self.error = f"{type(exc).__name__}: {str(exc)[:300]}"

    def view(self) -> dict[str, Any]:
        now = _utc_now()
        age = (
            None
            if self.last_success_at is None
            else max((now - self.last_success_at).total_seconds(), 0.0)
        )
        return {
            "last_success_at": _iso(self.last_success_at),
            "age_seconds": age,
            "error": self.error,
            "available": self.data is not None,
        }


class DashboardCollector:
    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.node_timeout_sec), trust_env=False
        )
        self.database = DatabaseReader(config)
        self.capacity = CachedSource()
        self.db = CachedSource()
        self.nodes = {worker.worker_id: CachedSource() for worker in config.workers}
        self.snapshot: dict[str, Any] | None = None
        points = max(1, int(config.history_retention_sec / config.history_interval_sec))
        self.history: deque[dict[str, Any]] = deque(maxlen=points)
        self._ownership_mismatch_since: dict[
            str, tuple[tuple[tuple[str, ...], ...], float]
        ] = {}
        self._orphan_since: dict[str, float] = {}
        self._last_history = 0.0
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        await self.refresh()
        self._task = asyncio.create_task(self._loop(), name="dashboard-collector")

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.client.aclose()
        await self.database.close()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.refresh_interval_sec
                )
            except TimeoutError:
                pass
            if not self._stop.is_set():
                await self.refresh()

    async def _capacity(self) -> None:
        try:
            response = await self.client.get(f"{self.config.control_url}/v4/capacity")
            response.raise_for_status()
            self.capacity.update(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            self.capacity.fail(exc)

    async def _database(self) -> None:
        try:
            self.db.update(await self.database.read())
        except (asyncpg.PostgresError, OSError, RuntimeError, TimeoutError) as exc:
            self.db.fail(exc)

    async def _node(self, worker: WorkerSource) -> None:
        source = self.nodes[worker.worker_id]
        try:
            response = await self.client.get(f"{worker.observer_url}/api/v1/snapshot")
            response.raise_for_status()
            data = response.json()
            if data.get("worker_id") != worker.worker_id:
                raise ValueError("node observer worker identity mismatch")
            source.update(data)
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            source.fail(exc)

    async def refresh(self) -> None:
        await asyncio.gather(
            self._capacity(),
            self._database(),
            *(self._node(worker) for worker in self.config.workers),
        )
        self.snapshot = self._build_snapshot()
        now = time.monotonic()
        if now - self._last_history >= self.config.history_interval_sec:
            self._last_history = now
            self.history.append(self._history_point(self.snapshot))

    def _build_snapshot(self) -> dict[str, Any]:
        sampled_monotonic = time.monotonic()
        capacity = self.capacity.data or {}
        database = self.db.data or {}
        db_workers = {row["worker_id"]: row for row in database.get("workers", [])}
        capacity_workers = {
            row["worker_id"]: row for row in capacity.get("workers", [])
        }
        counts: dict[str, Counter[str]] = {}
        for row in database.get("active_counts", []):
            counts.setdefault(str(row["worker_id"]), Counter())[str(row["state"])] = (
                int(row["count"])
            )
        active_ids = {
            str(row["attempt_id"])
            for row in database.get("active_attempts", [])
            if row.get("state") in ACTIVE_STATES
        }
        workers: list[dict[str, Any]] = []
        for configured in self.config.workers:
            worker_id = configured.worker_id
            node = self.nodes[worker_id].data or {}
            state_counts = counts.get(worker_id, Counter())
            worker_db = db_workers.get(worker_id, {})
            worker_runtime = node.get("worker") or {}
            db_reported = set(worker_db.get("reported_active_attempt_ids") or [])
            local_reported = set(worker_runtime.get("active_execution_ids") or [])
            worker_active_ids = {
                str(row["attempt_id"])
                for row in database.get("active_attempts", [])
                if row.get("worker_id") == worker_id
                and row.get("state") in ACTIVE_STATES
            }
            projects = (node.get("docker") or {}).get("projects") or []
            orphan_projects = [
                project
                for project in projects
                if project.get("attempt_id") not in active_ids
            ]
            raw_mismatch = not (worker_active_ids == db_reported == local_reported)
            mismatch_signature = (
                tuple(sorted(worker_active_ids)),
                tuple(sorted(db_reported)),
                tuple(sorted(local_reported)),
            )
            mismatch_since = self._track_signature(
                self._ownership_mismatch_since,
                worker_id,
                raw_mismatch,
                mismatch_signature,
                sampled_monotonic,
            )
            orphan_keys = {
                f"{worker_id}:{project.get('project')}" for project in orphan_projects
            }
            for key in [
                value
                for value in self._orphan_since
                if value.startswith(f"{worker_id}:") and value not in orphan_keys
            ]:
                self._orphan_since.pop(key, None)
            sustained_orphans = []
            for project in orphan_projects:
                key = f"{worker_id}:{project.get('project')}"
                since = self._track_condition(
                    self._orphan_since, key, True, sampled_monotonic
                )
                if sampled_monotonic - since >= self.config.alerts.stale_critical_sec:
                    sustained_orphans.append(project)
            workers.append(
                {
                    "worker_id": worker_id,
                    "control": capacity_workers.get(worker_id, {}),
                    "database": worker_db,
                    "active_attempts": sum(
                        state_counts.get(state, 0) for state in ACTIVE_STATES
                    ),
                    "cleanup_pending_attempts": state_counts.get("FAILED", 0),
                    "state_counts": dict(state_counts),
                    "ownership": {
                        "db_active_count": len(worker_active_ids),
                        "heartbeat_active_count": len(db_reported),
                        "local_active_count": len(local_reported),
                        "mismatch": raw_mismatch,
                        "mismatch_seconds": (
                            sampled_monotonic - mismatch_since if raw_mismatch else 0.0
                        ),
                        "mismatch_sustained": (
                            raw_mismatch
                            and sampled_monotonic - mismatch_since
                            >= self.config.alerts.ownership_mismatch_sec
                        ),
                    },
                    "node": node,
                    "orphan_candidate_projects": orphan_projects,
                    "orphan_sustained_projects": sustained_orphans,
                    "source": self.nodes[worker_id].view(),
                }
            )
        recent = {
            row["state"]: int(row["count"]) for row in database.get("recent_counts", [])
        }
        active_total = sum(worker["active_attempts"] for worker in workers)
        cleanup_pending_total = sum(
            worker["cleanup_pending_attempts"] for worker in workers
        )
        snapshot = {
            "schema_version": "rollout-dashboard-snapshot/v1",
            "sampled_at": _iso(_utc_now()),
            "cluster": {
                "reserved_executions": int(capacity.get("reserved_executions") or 0),
                "queued_attempts": int(capacity.get("queued_executions") or 0),
                "oldest_queued_at": capacity.get("oldest_queued_at"),
                "active_attempts": active_total,
                "cleanup_pending_attempts": cleanup_pending_total,
                "ready_workers": sum(
                    (worker.get("control") or {}).get("state") == "READY"
                    for worker in workers
                ),
                "total_workers": len(workers),
                "recent_15m": recent,
            },
            "workers": workers,
            "active_attempts": database.get("active_attempts", []),
            "recent_failures": database.get("recent_failures", []),
            "diagnostics": database.get("diagnostics", {}),
            "sources": {
                "capacity": self.capacity.view(),
                "database": self.db.view(),
                "nodes": {
                    worker_id: source.view() for worker_id, source in self.nodes.items()
                },
            },
        }
        snapshot["alerts"] = self._alerts(snapshot)
        return snapshot

    @staticmethod
    def _track_condition(
        tracker: dict[str, float],
        key: str,
        active: bool,
        sampled_monotonic: float,
    ) -> float:
        if not active:
            tracker.pop(key, None)
            return sampled_monotonic
        return tracker.setdefault(key, sampled_monotonic)

    @staticmethod
    def _track_signature(
        tracker: dict[str, tuple[tuple[tuple[str, ...], ...], float]],
        key: str,
        active: bool,
        signature: tuple[tuple[str, ...], ...],
        sampled_monotonic: float,
    ) -> float:
        if not active:
            tracker.pop(key, None)
            return sampled_monotonic
        current = tracker.get(key)
        if current is None or current[0] != signature:
            tracker[key] = (signature, sampled_monotonic)
            return sampled_monotonic
        return current[1]

    def _alerts(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []

        def add(
            level: str, code: str, message: str, worker_id: str | None = None
        ) -> None:
            alerts.append(
                {
                    "level": level,
                    "code": code,
                    "message": message,
                    "worker_id": worker_id,
                }
            )

        now = _utc_now()
        oldest = snapshot["cluster"].get("oldest_queued_at")
        if oldest:
            try:
                age = (now - datetime.fromisoformat(str(oldest))).total_seconds()
                snapshot["cluster"]["oldest_queue_age_seconds"] = max(age, 0.0)
                if age >= self.config.alerts.queue_oldest_sec:
                    add("warning", "QUEUE_OLD", f"oldest queue item is {age:.0f}s old")
            except ValueError:
                pass
        diagnostics = snapshot.get("diagnostics") or {}
        if int(diagnostics.get("unconsumed_outbox") or 0):
            add(
                "critical",
                "CONTROL_OUTBOX",
                "Control has unconsumed result outbox rows",
            )
        if int(diagnostics.get("queued_attempt_rows") or 0) != int(
            snapshot["cluster"].get("queued_attempts") or 0
        ):
            add("warning", "QUEUE_COUNTER_MISMATCH", "queue counter and rows disagree")
        for source_name in ("capacity", "database"):
            source = snapshot["sources"][source_name]
            age = source.get("age_seconds")
            if age is None or age >= self.config.alerts.stale_critical_sec:
                add(
                    "critical",
                    f"{source_name.upper()}_STALE",
                    f"{source_name} snapshot is stale",
                )
            elif age >= self.config.alerts.stale_warning_sec:
                add(
                    "warning",
                    f"{source_name.upper()}_STALE",
                    f"{source_name} snapshot is aging",
                )
        for worker in snapshot["workers"]:
            worker_id = worker["worker_id"]
            state = (worker.get("control") or {}).get("state", "NOT_READY")
            if state != "READY":
                add(
                    "critical",
                    "WORKER_NOT_READY",
                    f"Worker state is {state}",
                    worker_id,
                )
            if worker.get("cleanup_pending_attempts"):
                add(
                    "warning",
                    "CLEANUP_PENDING",
                    "Failed attempts are awaiting resource cleanup",
                    worker_id,
                )
            if worker["ownership"]["mismatch_sustained"]:
                add(
                    "warning",
                    "OWNERSHIP_MISMATCH",
                    "active attempt sources disagree",
                    worker_id,
                )
            if worker.get("orphan_sustained_projects"):
                add(
                    "critical",
                    "ORPHAN_CONTAINER",
                    "managed container has no active attempt",
                    worker_id,
                )
            node = worker.get("node") or {}
            node_errors = (node.get("sources") or {}).get("errors") or {}
            if node_errors:
                add(
                    "warning",
                    "NODE_SOURCE_ERROR",
                    "Node Observer has partial source errors",
                    worker_id,
                )
            for service_name, service_state in (node.get("services") or {}).items():
                if service_state == "active":
                    continue
                level = (
                    "warning"
                    if service_name == "dressage-distroll-image-cleaner.service"
                    else "critical"
                )
                add(
                    level,
                    "SERVICE_NOT_ACTIVE",
                    f"{service_name} is {service_state}",
                    worker_id,
                )
            cpu = node.get("cpu") or {}
            if (cpu.get("used_percent") or 0) >= self.config.alerts.cpu_used_percent:
                add(
                    "warning",
                    "CPU_HIGH",
                    f"physical CPU is {cpu['used_percent']:.1f}%",
                    worker_id,
                )
            memory = node.get("memory") or {}
            total = int(memory.get("total_bytes") or 0)
            available = int(memory.get("available_bytes") or 0)
            if (
                total
                and available / total * 100
                <= self.config.alerts.memory_available_percent
            ):
                add(
                    "warning",
                    "MEMORY_LOW",
                    "physical available memory is low",
                    worker_id,
                )
            for filesystem in node.get("filesystems") or []:
                if (
                    int(filesystem.get("free_bytes") or 0)
                    < self.config.alerts.disk_free_bytes
                ):
                    add(
                        "warning",
                        "DISK_LOW",
                        "filesystem free space is below threshold",
                        worker_id,
                    )
            images = node.get("images") or {}
            if int(images.get("failed") or 0):
                add(
                    "warning",
                    "IMAGE_PULL_FAILED",
                    "managed image pull failures exist",
                    worker_id,
                )
            if int(images.get("quarantined") or 0) or int(
                images.get("invalid_records") or 0
            ):
                add(
                    "critical",
                    "IMAGE_QUARANTINED",
                    "image cache has quarantined/invalid records",
                    worker_id,
                )
            outbox = node.get("outbox") or {}
            if int(outbox.get("pending") or 0):
                add(
                    "warning",
                    "WORKER_OUTBOX",
                    "Worker completion outbox is pending",
                    worker_id,
                )
            if int(outbox.get("rejected_recent_15m") or 0):
                add(
                    "critical",
                    "WORKER_OUTBOX_REJECTED",
                    "Worker has recently rejected completion records",
                    worker_id,
                )
            source = worker["source"]
            age = source.get("age_seconds")
            if age is None or age >= self.config.alerts.stale_critical_sec:
                add(
                    "critical",
                    "NODE_STALE",
                    "Node Observer snapshot is stale",
                    worker_id,
                )
            elif age >= self.config.alerts.stale_warning_sec:
                add(
                    "warning",
                    "NODE_STALE",
                    "Node Observer snapshot is aging",
                    worker_id,
                )
        return alerts

    @staticmethod
    def _history_point(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "sampled_at": snapshot["sampled_at"],
            "queued_attempts": snapshot["cluster"]["queued_attempts"],
            "active_attempts": snapshot["cluster"]["active_attempts"],
            "workers": {
                worker["worker_id"]: {
                    "active": worker["active_attempts"],
                    "cpu": (worker.get("node") or {})
                    .get("cpu", {})
                    .get("used_percent"),
                    "memory": (worker.get("node") or {})
                    .get("memory", {})
                    .get("used_percent"),
                }
                for worker in snapshot["workers"]
            },
        }


def create_dashboard_app(
    config: DashboardConfig, collector: DashboardCollector | None = None
) -> FastAPI:
    selected = collector or DashboardCollector(config)
    static_root = Path(str(files("rollout_service.observability").joinpath("static")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await selected.start()
        app.state.collector = selected
        try:
            yield
        finally:
            await selected.close()

    app = FastAPI(title="CPU Rollout Dashboard", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/dashboard.css", include_in_schema=False)
    async def css() -> FileResponse:
        return FileResponse(static_root / "dashboard.css", media_type="text/css")

    @app.get("/dashboard.js", include_in_schema=False)
    async def js() -> FileResponse:
        return FileResponse(
            static_root / "dashboard.js", media_type="application/javascript"
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = (
            selected.snapshot is not None
            and selected.db.data is not None
            and selected.capacity.data is not None
        )
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not-ready"},
        )

    @app.get("/api/v1/snapshot")
    async def snapshot() -> JSONResponse:
        if selected.snapshot is None:
            raise HTTPException(status_code=503, detail="snapshot not ready")
        return JSONResponse(selected.snapshot)

    @app.get("/api/v1/history")
    async def history(
        window: str = Query(default="30m", pattern=r"^(5|15|30|60)m$"),
    ) -> dict[str, Any]:
        seconds = int(window.removesuffix("m")) * 60
        cutoff = _utc_now().timestamp() - seconds
        points = [
            point
            for point in selected.history
            if datetime.fromisoformat(point["sampled_at"]).timestamp() >= cutoff
        ]
        return {"window": window, "points": points}

    return app


def main() -> None:
    config = load_dashboard_config(
        os.environ.get(
            "ROLLOUT_DASHBOARD_CONFIG", "/etc/dressage-distroll/dashboard.yaml"
        )
    )
    uvicorn.run(
        create_dashboard_app(config),
        host=config.listen_host,
        port=config.listen_port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
