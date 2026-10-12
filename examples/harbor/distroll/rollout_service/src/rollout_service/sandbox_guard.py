"""Worker-local, per-execution OOM stop-loss. No task blacklist or RL state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import SandboxGuardSettings
from .image_cache import run_command

logger = logging.getLogger(__name__)
current_guard: ContextVar[GuardAttempt | None] = ContextVar(
    "sandbox_guard", default=None
)


@dataclass(frozen=True)
class ResourceSample:
    at: float
    oom_kill: int
    full_usec: int
    memory_bytes: int
    pids: int
    pids_max_events: int


def read_sample(path: Path) -> ResourceSample:
    def counters(name):
        return dict(line.split() for line in (path / name).read_text().splitlines())

    pressure = (path / "memory.pressure").read_text().splitlines()
    full = next((line for line in pressure if line.startswith("full ")), None)
    if full is None:
        raise ValueError("cgroup memory full pressure counter is missing")
    return ResourceSample(
        at=time.monotonic(),
        oom_kill=int(counters("memory.events.local")["oom_kill"]),
        full_usec=int(dict(v.split("=") for v in full.split()[1:])["total"]),
        memory_bytes=int((path / "memory.current").read_text()),
        pids=int((path / "pids.current").read_text()),
        pids_max_events=int(counters("pids.events")["max"]),
    )


class GuardAttempt:
    def __init__(
        self, owner: SandboxGuard, execution_id: str, task_ref: str, client_scope: str
    ):
        self.owner = owner
        self.execution_id = execution_id
        self.task_ref = task_ref
        self.client_scope = client_scope
        self.triggered = asyncio.Event()
        self.reason: dict | None = None
        self.sealed = False
        self.creating = 0
        self.runner_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None
        self.containers: dict[str, tuple[Path, deque[ResourceSample]]] = {}
        self.events: list[dict] = []
        self._write_lock = asyncio.Lock()
        self._writes: set[asyncio.Task] = set()

    def begin_start(self):
        if self.sealed or self.triggered.is_set():
            raise RuntimeError("resource guard forbids late sandbox creation")
        self.creating += 1

    async def bind(self, container_id: str):
        inspected = await run_command(
            ["docker", "container", "inspect", container_id], timeout_sec=5
        )
        if inspected.returncode:
            raise RuntimeError("could not bind sandbox resource guard")
        info = json.loads(inspected.stdout)[0]
        labels = info["Config"]["Labels"] or {}
        expected = {
            "io.dressage.rollout.managed": "true",
            "io.dressage.rollout.worker_id": self.owner.worker_id,
            "io.dressage.rollout.attempt_id": self.execution_id,
        }
        if any(labels.get(k) != v for k, v in expected.items()):
            raise RuntimeError("sandbox resource guard identity mismatch")
        pid = int(info["State"]["Pid"])
        cgroup = next(
            line[3:]
            for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        root = Path("/sys/fs/cgroup").resolve()
        path = (root / cgroup.lstrip("/")).resolve()
        if path == root or not path.is_relative_to(root) or info["Id"] not in path.name:
            raise RuntimeError("invalid sandbox cgroup")
        async with asyncio.timeout(5):
            baseline = await asyncio.to_thread(read_sample, path)
        self.containers[info["Id"]] = (path, deque([baseline]))
        if self.sealed or self.triggered.is_set():
            # A start that raced cancellation remains owned; the fast cleanup
            # loop must see it, and Harbor must not proceed to agent execution.
            raise RuntimeError("resource guard interrupted sandbox creation")

    def update(self, container_id: str, sample: ResourceSample):
        _, samples = self.containers[container_id]
        samples.append(sample)
        cutoff = sample.at - self.owner.settings.oom_window_sec
        while len(samples) > 1 and samples[0].at < cutoff:
            samples.popleft()
        first = samples[0]
        elapsed = sample.at - first.at
        if elapsed <= 0 or self.reason is not None:
            return
        kills = sample.oom_kill - first.oom_kill
        pressure = (sample.full_usec - first.full_usec) / (elapsed * 1_000_000)
        if (
            kills < self.owner.settings.oom_kill_threshold
            or pressure < self.owner.settings.memory_full_ratio
        ):
            return
        self.reason = {
            "container_id": container_id,
            "rule": "repeated_oom_kill_with_memory_pressure",
            "oom_kill_delta": kills,
            "window_sec": elapsed,
            "memory_full_ratio": pressure,
            "memory_bytes": sample.memory_bytes,
            "pids": sample.pids,
            "pids_max_events": sample.pids_max_events,
        }
        if self.owner.settings.mode == "enforce":
            self.triggered.set()
            if self.owner.on_trigger is not None:
                self.owner.on_trigger(self)
        logger.warning(
            "sandbox_resource_guard execution=%s worker=%s mode=%s container=%s oom_kills=%s memory_full_ratio=%.3f",
            self.execution_id,
            self.owner.worker_id,
            self.owner.settings.mode,
            container_id,
            kills,
            pressure,
        )
        self.record(
            "triggered" if self.triggered.is_set() else "observed", **self.reason
        )

    def record(self, stage: str, **fields):
        self.events.append(
            {"at": datetime.now(UTC).isoformat(), "stage": stage, **fields}
        )
        if len(self.events) > 64:
            self.events = self.events[:1] + self.events[-63:]
        payload = {
            "execution_id": self.execution_id,
            "task_ref": self.task_ref,
            "client_scope": self.client_scope,
            "settings": self.owner.settings.model_dump(mode="json"),
            "worker_id": self.owner.worker_id,
            "worker_incarnation": self.owner.incarnation,
            "mode": self.owner.settings.mode,
            "events": list(self.events),
        }
        task = asyncio.create_task(self._persist(payload))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _persist(self, payload):
        try:
            async with self._write_lock:
                name = hashlib.sha256(self.execution_id.encode()).hexdigest() + ".json"
                await asyncio.to_thread(
                    self.owner.writer, self.owner.path / name, payload
                )
        except Exception:
            logger.exception(
                "resource guard audit write failed execution=%s", self.execution_id
            )

    async def flush(self):
        if self._writes:
            await asyncio.wait(list(self._writes), timeout=3)

    def seal(self):
        if self.creating:
            raise RuntimeError("sandbox creation still pending")
        self.sealed = True
        self.owner.attempts.pop(self.execution_id, None)


class SandboxGuard:
    def __init__(
        self, settings, state_dir, worker_id, incarnation, writer, on_trigger=None
    ):
        self.settings: SandboxGuardSettings = settings
        self.worker_id = worker_id
        self.incarnation = incarnation
        self.writer = writer
        self.on_trigger = on_trigger
        self.path = state_dir / "resource-guard"
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.attempts: dict[str, GuardAttempt] = {}
        self.healthy = True
        self._poll_lock = asyncio.Lock()

    def check_support(self):
        if self.settings.mode != "off":
            root = Path("/sys/fs/cgroup")
            controllers = (root / "cgroup.controllers").read_text().split()
            if not {"memory", "pids"}.issubset(controllers):
                raise RuntimeError(
                    "sandbox guard requires cgroup v2 memory and pids controllers"
                )
            # The hierarchy root need not expose per-cgroup memory files.
            relative = next(
                line[3:]
                for line in Path("/proc/self/cgroup").read_text().splitlines()
                if line.startswith("0::")
            )
            read_sample(root / relative.lstrip("/"))

    def register(self, execution_id, task_ref, client_scope):
        watch = GuardAttempt(self, execution_id, task_ref, client_scope)
        self.attempts[execution_id] = watch
        return watch

    async def poll(self):
        healthy = True
        for watch in list(self.attempts.values()):
            healthy = await self.sample(watch) and healthy
        self.healthy = healthy

    async def finish(self, watch):
        # Drain an already-started sample before sealing the result. The read
        # must not apply a new guard decision after completion is published.
        async with asyncio.timeout(3):
            async with self._poll_lock:
                watch.seal()

    async def sample(self, watch):
        healthy = True
        async with self._poll_lock:
            for container_id, (path, _) in list(watch.containers.items()):
                if watch.sealed:
                    break
                try:
                    async with asyncio.timeout(self.settings.sample_interval_sec):
                        sample = await asyncio.to_thread(read_sample, path)
                except FileNotFoundError:
                    # A vanished cgroup is not proof of Docker cleanup. Worker
                    # still verifies the entire execution before releasing charge.
                    watch.containers.pop(container_id, None)
                    continue
                except (OSError, ValueError, KeyError, StopIteration):
                    healthy = False
                    logger.exception(
                        "sandbox guard sampling failed execution=%s", watch.execution_id
                    )
                    continue
                if not watch.sealed and container_id in watch.containers:
                    watch.update(container_id, sample)
        if not healthy:
            self.healthy = False
        return healthy

    async def run(self):
        self.check_support()
        while True:
            if self.settings.mode != "off":
                await self.poll()
            await asyncio.sleep(self.settings.sample_interval_sec)
