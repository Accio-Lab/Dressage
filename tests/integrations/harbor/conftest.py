from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import SimpleNamespace

import pytest

from dressage.integrations.harbor.config import HarborIntegrationConfig


@pytest.fixture(autouse=True)
def _isolate_legacy_session_payloads(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "DRESSAGE_TRAJECTORY_PAYLOAD_LOG_DIR",
        str(tmp_path / "legacy-session-payloads"),
    )
    monkeypatch.setenv("DRESSAGE_LOG_WRITE_MODE", "await")


@pytest.fixture
def integration_config_factory(tmp_path):
    def create(*, execution_mode="rollout", **updates):
        payload = {
            "execution_mode": execution_mode,
            "artifacts": {"mode": "both", "root": str(tmp_path / "artifacts")},
        }
        if execution_mode == "training":
            payload.update(
                {
                    "security": {"routing_guarantee": "enforced"},
                    "training": {
                        "model_override": "train-model",
                        "group_max_retries": 1,
                        "min_live_group_ratio": 0.5,
                    },
                }
            )
        for section, value in updates.items():
            if isinstance(value, dict) and isinstance(payload.get(section), dict):
                payload[section] = {**payload[section], **value}
            else:
                payload[section] = value
        return HarborIntegrationConfig.model_validate(payload)

    return create


@pytest.fixture
def segment_factory():
    def create(
        trajectory_id="physical-1",
        instance_id="logical-a",
        *,
        version="v1",
        segment_index=0,
        segment_count=1,
    ):
        return {
            "trajectory_id": trajectory_id,
            "instance_id": instance_id,
            "segment_index": segment_index,
            "segment_count": segment_count,
            "tokens": [11, 12, 13],
            "full_loss_mask": [0, 1, 1],
            "full_logprobs": [0.0, -0.1, -0.2],
            "full_versions": ["input", version, version],
            "extra_info": {
                "segment_view": "timeline",
                "finalization_complete": True,
                "finalization_id": f"final-{trajectory_id}",
            },
        }

    return create


@pytest.fixture
def trial_result_factory():
    def create(*, reward=1.0, exception_info=None, trial_name="trial-a"):
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            id=f"physical-{trial_name}",
            trial_name=trial_name,
            task_name="task-a",
            task_checksum="checksum-a",
            agent_info=SimpleNamespace(name="codex"),
            verifier_result=SimpleNamespace(rewards={"reward": reward}),
            exception_info=exception_info,
            step_results=None,
            started_at=now,
            finished_at=now,
        )

    return create


class RecordingProxy:
    def __init__(self, segments=(), *, finalize_error=None, read_error=None):
        self.segments = list(segments)
        self.finalize_error = finalize_error
        self.read_error = read_error
        self.finalize_calls = []
        self.read_calls = []

    async def finalize_session(self, session_id, **kwargs):
        self.finalize_calls.append(session_id)
        if self.finalize_error:
            raise self.finalize_error
        return {"success": True}

    async def read_trajectory(self, *, trajectory_id=None, drain=False, **kwargs):
        self.read_calls.append((trajectory_id, drain))
        if self.read_error and not drain:
            raise self.read_error
        data = list(self.segments)
        if drain:
            self.segments.clear()
        return {"success": bool(data), "data": data, "drained": drain}


@pytest.fixture
def proxy_client_factory():
    return RecordingProxy


class RecordingHandle:
    def __init__(self, *, fail_open=False, fail_close=False):
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.turns = []
        self.quiesce_calls = 0
        self.close_calls = 0
        self.broken = []

    async def open_turn(self, turn_id, backend_session_id=None):
        if self.fail_open:
            raise RuntimeError("open failed")
        self.turns.append(turn_id)

    async def quiesce(self, timeout=None):
        self.quiesce_calls += 1
        return {}

    async def mark_broken(self, reason):
        self.broken.append(reason)

    async def close(self, *, tombstone=True):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("close failed")


class RecordingSecrets:
    def __init__(self):
        self.values = {}
        self.counter = 0
        self.rotate_calls = []
        self.closed_jobs = []

    def create(self, *, job_id, slot_id, env_name=None):
        self.counter += 1
        self.values[slot_id] = f"secret-{self.counter}"
        return self.values[slot_id]

    def current(self, slot_id):
        return self.values[slot_id]

    def fingerprint(self, slot_id):
        import hashlib

        return hashlib.sha256(self.values[slot_id].encode()).hexdigest()

    def rotate(self, slot_id):
        self.rotate_calls.append(slot_id)
        self.counter += 1
        self.values[slot_id] = f"secret-{self.counter}"
        return self.values[slot_id]

    def delete(self, slot_id):
        self.values.pop(slot_id, None)

    def close_job(self, job_id):
        self.closed_jobs.append(job_id)


class RecordingLease:
    def __init__(self, *, fail_open=False, fail_register=False, fail_close=False):
        self.public_url = "http://127.0.0.1:39123"
        self.fail_open = fail_open
        self.fail_register = fail_register
        self.fail_close = fail_close
        self.specs = []
        self.handles = []
        self.release_calls = 0

    async def register(self, spec):
        self.specs.append(spec)
        if self.fail_register:
            raise RuntimeError("register failed")
        handle = RecordingHandle(fail_open=self.fail_open, fail_close=self.fail_close)
        self.handles.append(handle)
        return handle

    async def release(self):
        self.release_calls += 1


class RecordingRuntime:
    def __init__(self, **failures):
        self.secret_slots = RecordingSecrets()
        self.lease = RecordingLease(**failures)

    async def acquire(self, config):
        return self.lease


@pytest.fixture
def runtime_factory():
    return RecordingRuntime


@pytest.fixture
def trial_factory(tmp_path):
    def create(name="trial-a", *, agent_name="codex"):
        task_dir = tmp_path / "resolved-tasks" / name
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.toml").write_text('schema_version = "1.3"\n')
        return SimpleNamespace(
            trial_name=name,
            agent=SimpleNamespace(
                name=agent_name,
                import_path=None,
                n_concurrent=2,
                concurrency_group=None,
                concurrency_key="agent:original",
                env={},
                extra_allowed_hosts=[],
            ),
            task=SimpleNamespace(get_local_path=lambda: task_dir),
            trials_dir=tmp_path / "harbor-trials",
        )

    return create


@pytest.fixture
def job_factory(trial_result_factory):
    class Job:
        def __init__(self, configs):
            self.id = "job-a"
            self.is_resuming = False
            self.configs = configs
            self.hooks = {}

        def on_trial_started(self, callback):
            self.hooks["start"] = callback

        def on_agent_started(self, callback):
            self.hooks["agent_start"] = callback

        def on_agent_ended(self, callback):
            self.hooks["agent_end"] = callback

        def on_verification_started(self, callback):
            self.hooks["verification"] = callback

        def on_trial_cancelled(self, callback):
            self.hooks["cancel"] = callback

        def on_trial_ended(self, callback):
            self.hooks["end"] = callback

        def event(self, trial, trial_id="physical-1"):
            return SimpleNamespace(
                trial_name=trial.trial_name,
                trial_id=trial_id,
                config=trial,
                result=trial_result_factory(trial_name=trial.trial_name),
            )

    return Job


@dataclass
class SampleLike:
    group_index: int
    index: int
    prompt: str = "harbor://task"
    metadata: dict = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    reward: float | dict | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    remove_sample: bool = False

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING


@pytest.fixture
def group_factory():
    def create(position):
        return [
            SampleLike(
                group_index=position,
                index=position * 2 + attempt,
                metadata={
                    "harbor_spec_id": f"spec-{position}",
                    "harbor_attempt_slot": attempt,
                    "harbor_instance_id": f"instance-{position}",
                },
            )
            for attempt in range(2)
        ]

    return create
