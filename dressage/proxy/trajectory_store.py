"""In-memory trajectory storage grouped by ``instance_id``."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectorySegment:
    """One finalized trajectory segment stored by the proxy."""

    uid: str
    trajectory_id: str
    turn_id: str
    instance_id: str
    segment_index: int
    segment_count: int
    messages: list[dict]
    tools: list[dict[str, Any]] | None
    tokens: list[int]
    full_logprobs: list[float]
    full_loss_mask: list[int]
    aligned_response_length: int
    full_versions: list[str] | None = None
    routed_experts: str | None = None
    routed_experts_chunks: list[dict[str, Any]] | None = None
    routed_experts_parts: list[dict[str, Any]] | None = None
    label: Any | None = None
    finish_reason: str = "stop"
    timestamp: float = field(default_factory=time.time)
    extra_info: dict = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return self.trajectory_id

    def to_dict(self) -> dict:
        data = {
            "uid": self.uid,
            "session_id": self.session_id,
            "trajectory_id": self.trajectory_id,
            "turn_id": self.turn_id,
            "instance_id": self.instance_id,
            "segment_index": self.segment_index,
            "segment_count": self.segment_count,
            "messages": self.messages,
            "tools": self.tools,
            "tokens": self.tokens,
            "full_logprobs": self.full_logprobs,
            "full_loss_mask": self.full_loss_mask,
            "aligned_response_length": self.aligned_response_length,
            "label": self.label,
            "finish_reason": self.finish_reason,
            "timestamp": self.timestamp,
            "extra_info": self.extra_info,
        }
        if self.full_versions is not None:
            data["full_versions"] = self.full_versions
        if self.routed_experts is not None:
            data["routed_experts"] = self.routed_experts
        if self.routed_experts_chunks is not None:
            data["routed_experts_chunks"] = self.routed_experts_chunks
        if self.routed_experts_parts is not None:
            data["routed_experts_parts"] = self.routed_experts_parts
        return data


TrajectoryItem = TrajectorySegment


class TrajectoryStore:
    """Thread-safe store supporting both exact reads and batch draining."""

    def __init__(self, min_group_size: int = 1, group_timeout: float = 300.0):
        self._lock = threading.Lock()
        self._by_instance: dict[str, list[TrajectorySegment]] = {}
        self._by_trajectory: dict[str, list[TrajectorySegment]] = {}
        self._instance_timestamps: dict[str, float] = {}
        self._min_group_size = min_group_size
        self._group_timeout = group_timeout

    @staticmethod
    def _trajectory_key(item: TrajectorySegment) -> str:
        return item.trajectory_id

    @staticmethod
    def _item_segment_view(item: TrajectorySegment) -> str | None:
        value = item.extra_info.get("segment_view")
        return str(value) if value is not None else None

    @staticmethod
    def _item_token_build_mode(item: TrajectorySegment) -> str | None:
        value = item.extra_info.get("token_build_mode")
        return str(value) if value is not None else None

    @classmethod
    def _default_segment_view(cls, items: list[TrajectorySegment]) -> str | None:
        token_build_modes = {cls._item_token_build_mode(item) for item in items}
        if "tito" in token_build_modes:
            return "lineage"
        if "snapshot" in token_build_modes:
            return "timeline"
        return None

    @classmethod
    def _filter_segment_view(
        cls,
        items: list[TrajectorySegment],
        segment_view: str | None,
    ) -> list[TrajectorySegment]:
        selected_view = segment_view or cls._default_segment_view(items)
        if selected_view is None:
            return list(items)
        return [
            item for item in items if cls._item_segment_view(item) == selected_view
        ]

    def write(self, item: TrajectorySegment) -> None:
        with self._lock:
            self._by_instance.setdefault(item.instance_id, []).append(item)
            self._by_trajectory.setdefault(item.trajectory_id, []).append(item)
            self._instance_timestamps[item.instance_id] = time.time()

    def write_dict(self, data: dict) -> TrajectorySegment:
        trajectory_id = data.get("trajectory_id") or data.get("session_id")
        if trajectory_id is None:
            trajectory_id = str(uuid.uuid4())
        for key in ("tokens", "full_logprobs", "full_loss_mask"):
            if key not in data:
                raise ValueError(f"trajectory segment missing required field: {key}")

        token_count = len(data["tokens"])
        for key in ("full_logprobs", "full_loss_mask"):
            if len(data[key]) != token_count:
                raise ValueError(
                    f"trajectory segment field length mismatch: {key} has "
                    f"{len(data[key])}, tokens has {token_count}"
                )
        if data.get("full_versions") is not None and len(data["full_versions"]) != token_count:
            raise ValueError(
                "trajectory segment field length mismatch: full_versions has "
                f"{len(data['full_versions'])}, tokens has {token_count}"
            )

        item = TrajectorySegment(
            uid=data.get("uid", str(uuid.uuid4())),
            trajectory_id=trajectory_id,
            turn_id=data["turn_id"],
            instance_id=data.get("instance_id", str(uuid.uuid4())),
            segment_index=data.get("segment_index", 0),
            segment_count=data.get("segment_count", 1),
            messages=data["messages"],
            tools=data.get("tools"),
            tokens=data["tokens"],
            full_logprobs=data["full_logprobs"],
            full_loss_mask=data["full_loss_mask"],
            aligned_response_length=data.get("aligned_response_length", 0),
            full_versions=(
                None
                if data.get("full_versions") is None
                else [str(value) for value in data["full_versions"]]
            ),
            routed_experts=data.get("routed_experts"),
            routed_experts_chunks=data.get("routed_experts_chunks"),
            routed_experts_parts=data.get("routed_experts_parts"),
            label=data.get("label"),
            finish_reason=data.get("finish_reason", "stop"),
            extra_info=data.get("extra_info", {}),
        )
        self.write(item)
        return item

    def read_trajectory(
        self,
        trajectory_id: str,
        instance_id: str | None = None,
        segment_view: str | None = None,
    ) -> list[dict]:
        with self._lock:
            items = self._by_trajectory.get(trajectory_id, [])
            if instance_id is not None:
                items = [item for item in items if item.instance_id == instance_id]
            items = self._filter_segment_view(items, segment_view)
            items = sorted(items, key=lambda item: (item.segment_index, item.timestamp))
            return [copy.deepcopy(item.to_dict()) for item in items]

    def pop_trajectory(
        self,
        trajectory_id: str,
        instance_id: str | None = None,
        segment_view: str | None = None,
    ) -> list[dict]:
        """Read and remove finalized segments for one trajectory.

        Exact trajectory reads are used by rollout workers immediately after
        finalization. Without removal, long-running fully async rollouts keep
        every completed segment in memory for the lifetime of the proxy.
        """
        with self._lock:
            existing_items = self._by_trajectory.get(trajectory_id, [])
            if instance_id is None:
                matched = list(existing_items)
                remaining_by_trajectory = []
            else:
                matched = [
                    item for item in existing_items if item.instance_id == instance_id
                ]
                remaining_by_trajectory = [
                    item for item in existing_items if item.instance_id != instance_id
                ]

            if not matched:
                return []

            returned = self._filter_segment_view(matched, segment_view)

            if remaining_by_trajectory:
                self._by_trajectory[trajectory_id] = remaining_by_trajectory
            else:
                self._by_trajectory.pop(trajectory_id, None)

            matched_uids = {item.uid for item in matched}
            affected_instances = {item.instance_id for item in matched}
            for affected_instance_id in affected_instances:
                remaining_by_instance = [
                    item
                    for item in self._by_instance.get(affected_instance_id, [])
                    if item.uid not in matched_uids
                ]
                if remaining_by_instance:
                    self._by_instance[affected_instance_id] = remaining_by_instance
                    self._instance_timestamps[affected_instance_id] = time.time()
                else:
                    self._by_instance.pop(affected_instance_id, None)
                    self._instance_timestamps.pop(affected_instance_id, None)

            returned = sorted(
                returned, key=lambda item: (item.segment_index, item.timestamp)
            )
            return [copy.deepcopy(item.to_dict()) for item in returned]

    def read_session(self, session_id: str, instance_id: str | None = None) -> list[dict]:
        return self.read_trajectory(session_id, instance_id=instance_id)

    def _ready_instances_locked(self) -> dict[str, list[TrajectorySegment]]:
        now = time.time()
        ready: dict[str, list[TrajectorySegment]] = {}
        for instance_id, items in list(self._by_instance.items()):
            trajectory_count = len({self._trajectory_key(item) for item in items})
            size_ok = trajectory_count >= self._min_group_size
            timed_out = (
                now - self._instance_timestamps.get(instance_id, now)
            ) > self._group_timeout
            if size_ok or timed_out:
                ready[instance_id] = items
        return ready

    def read_batch(
        self,
        max_groups: int | None = None,
    ) -> list[list[dict]]:
        with self._lock:
            ready = self._ready_instances_locked()
            if not ready:
                return []

            instance_ids = list(ready.keys())
            if max_groups is not None:
                instance_ids = instance_ids[:max_groups]

            groups: list[list[dict]] = []
            for instance_id in instance_ids:
                items = sorted(
                    self._by_instance.pop(instance_id),
                    key=lambda item: (item.trajectory_id, item.segment_index, item.timestamp),
                )
                for item in items:
                    trajectory_items = self._by_trajectory.get(item.trajectory_id, [])
                    trajectory_items = [
                        existing for existing in trajectory_items if existing.uid != item.uid
                    ]
                    if trajectory_items:
                        self._by_trajectory[item.trajectory_id] = trajectory_items
                else:
                    self._by_trajectory.pop(item.trajectory_id, None)
                self._instance_timestamps.pop(instance_id, None)
                by_trajectory: dict[str, list[TrajectorySegment]] = {}
                for item in items:
                    by_trajectory.setdefault(item.trajectory_id, []).append(item)
                default_items: list[TrajectorySegment] = []
                for trajectory_items in by_trajectory.values():
                    default_items.extend(self._filter_segment_view(trajectory_items, None))
                default_items = sorted(
                    default_items,
                    key=lambda item: (
                        item.trajectory_id,
                        item.segment_index,
                        item.timestamp,
                    ),
                )
                groups.append([copy.deepcopy(item.to_dict()) for item in default_items])
            return groups

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_items": sum(len(items) for items in self._by_instance.values()),
                "total_instances": len(self._by_instance),
                "ready_instances": len(self._ready_instances_locked()),
            }
