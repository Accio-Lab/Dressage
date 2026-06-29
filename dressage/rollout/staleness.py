"""Trajectory-level version staleness helpers for Dressage async rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_NON_REAL_VERSIONS = {"", "-1", "unknown", "none"}


@dataclass(frozen=True)
class StalenessConfig:
    keep_versions: int | None = None

    @property
    def enabled(self) -> bool:
        return self.keep_versions is not None


@dataclass(frozen=True)
class TrajectoryVersionInfo:
    key: str
    version: str


@dataclass(frozen=True)
class PendingGroup:
    group_id: int
    samples: list[Any]


def config_from_args(args: Any) -> StalenessConfig:
    raw = getattr(args, "dressage_staleness_keep_versions", None)
    if raw is None:
        return StalenessConfig()
    value = int(raw)
    if value < 0:
        raise ValueError("dressage_staleness_keep_versions must be an integer >= 0")
    if value == 0:
        return StalenessConfig()
    return StalenessConfig(keep_versions=value)


def real_version(value: Any) -> str | None:
    if value is None:
        return None
    version = str(value).strip()
    if version.lower() in _NON_REAL_VERSIONS:
        return None
    return version


def trajectory_key(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None)
    value = metadata.get("parent_traj_id") if isinstance(metadata, dict) else None
    return "" if value is None else str(value)


def _segment_index(sample: Any, order: int) -> tuple[int, int]:
    metadata = getattr(sample, "metadata", None)
    value = metadata.get("segment_index", 0) if isinstance(metadata, dict) else 0
    return int(value), order


def trajectory_version_infos(group: list[Any]) -> list[TrajectoryVersionInfo]:
    latest_by_key: dict[str, tuple[tuple[int, int], str]] = {}
    for order, sample in enumerate(group):
        metadata = getattr(sample, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        version = real_version(metadata.get("dressage_end_token_version"))
        if version is None:
            continue

        key = trajectory_key(sample)
        if not key:
            continue
        position = _segment_index(sample, order)
        if key not in latest_by_key or latest_by_key[key][0] <= position:
            latest_by_key[key] = (position, version)

    return [
        TrajectoryVersionInfo(key=key, version=version)
        for key, (_, version) in latest_by_key.items()
    ]


@dataclass
class StalenessTracker:
    config: StalenessConfig
    versions: list[str] = field(default_factory=list)

    @property
    def current_version_index(self) -> int | None:
        return len(self.versions) - 1 if self.versions else None

    @property
    def cutoff_version_index(self) -> int | None:
        if not self.config.enabled or not self.versions:
            return None
        return max(0, len(self.versions) - int(self.config.keep_versions))

    @property
    def current_version_label(self) -> str | None:
        return self.versions[-1] if self.versions else None

    @property
    def cutoff_version_label(self) -> str | None:
        cutoff = self.cutoff_version_index
        return None if cutoff is None else self.versions[cutoff]

    def observe_group(self, group: list[Any]) -> bool:
        if not self.config.enabled:
            return False

        previous_count = len(self.versions)
        for info in trajectory_version_infos(group):
            if info.version not in self.versions:
                self.versions.append(info.version)
        return len(self.versions) != previous_count

    def should_drop_group(self, group: list[Any]) -> bool:
        cutoff = self.cutoff_version_index
        if cutoff is None:
            return False

        return any(
            self.version_index(info.version) < cutoff
            for info in trajectory_version_infos(group)
        )

    def version_index(self, version: str) -> int:
        return self.versions.index(version)


@dataclass
class StalenessGroupFilter:
    tracker: StalenessTracker
    rollout_name: str
    dropped_groups: int = 0

    def observe_group(self, group: list[Any]) -> bool:
        return self.tracker.observe_group(group)

    def observe_completed(self, completed_groups: list[Any]) -> bool:
        advanced_version = False
        for completed in completed_groups:
            result = getattr(completed, "result", None)
            if result is not None and self.observe_group(result):
                advanced_version = True
        return advanced_version

    def keep_group(self, group_id: int, group: list[Any], logger: Any) -> bool:
        if not self.tracker.config.enabled:
            return True
        if not self.tracker.should_drop_group(group):
            return True
        self._drop_group(group_id, logger)
        return False

    def filter_pending(
        self,
        groups: list[PendingGroup],
        logger: Any,
    ) -> list[PendingGroup]:
        if not self.tracker.config.enabled:
            return groups
        return [
            group
            for group in groups
            if self.keep_group(group.group_id, group.samples, logger)
        ]

    def metrics_for_groups(self, groups: list[list[Any]]) -> dict[str, float]:
        if not self.tracker.config.enabled:
            return {}

        metrics: dict[str, float] = {
            "staleness/dropped_groups": float(self.dropped_groups),
        }
        current = self.tracker.current_version_index
        if current is None:
            return metrics

        metrics["staleness/current_version_index"] = float(current)
        cutoff = self.tracker.cutoff_version_index
        if cutoff is not None:
            metrics["staleness/cutoff_version_index"] = float(cutoff)

        gaps = [
            current - self.tracker.version_index(info.version)
            for group in groups
            for info in trajectory_version_infos(group)
        ]
        if gaps:
            metrics.update({
                "staleness/version_gap_min": float(min(gaps)),
                "staleness/version_gap_max": float(max(gaps)),
                "staleness/version_gap_mean": float(sum(gaps) / len(gaps)),
            })
        return metrics

    def _drop_group(self, group_id: int, logger: Any) -> None:
        self.dropped_groups += 1
        logger.info(
            "dropping stale Dressage %s async rollout group %s: cutoff_version_index=%s "
            "cutoff_version=%s current_version_index=%s current_version=%s "
            "dropped_stale_groups=%s",
            self.rollout_name,
            group_id,
            self.tracker.cutoff_version_index,
            self.tracker.cutoff_version_label,
            self.tracker.current_version_index,
            self.tracker.current_version_label,
            self.dropped_groups,
        )
