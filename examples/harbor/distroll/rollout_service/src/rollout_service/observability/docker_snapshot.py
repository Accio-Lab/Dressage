from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MANAGED_LABEL = "io.dressage.rollout.managed"
WORKER_LABEL = "io.dressage.rollout.worker_id"
ATTEMPT_LABEL = "io.dressage.rollout.attempt_id"
RUN_LABEL = "io.dressage.rollout.run_id"
GROUP_LABEL = "io.dressage.rollout.group_id"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


def parse_json_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError("Docker JSON line is not an object")
        rows.append(value)
    return rows


def parse_label_string(value: str | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for raw in (value or "").split(","):
        key, separator, item = raw.partition("=")
        if separator and key:
            labels[key] = item
    return labels


def aggregate_managed_containers(
    rows: Iterable[dict[str, Any]], *, worker_id: str
) -> dict[str, Any]:
    prepared: list[tuple[dict[str, Any], dict[str, str]]] = []
    anchors: dict[str, dict[str, str]] = {}
    for row in rows:
        labels = parse_label_string(str(row.get("Labels") or ""))
        prepared.append((row, labels))
        if (
            labels.get(MANAGED_LABEL) == "true"
            and labels.get(WORKER_LABEL) == worker_id
        ):
            project = labels.get(COMPOSE_PROJECT_LABEL)
            if project:
                anchors[project] = labels

    projects: list[dict[str, Any]] = []
    running_total = 0
    container_total = 0
    state_counts: Counter[str] = Counter()
    for project, identity in sorted(anchors.items()):
        members = [
            row
            for row, labels in prepared
            if labels.get(COMPOSE_PROJECT_LABEL) == project
        ]
        running = sum(
            str(row.get("State") or "").lower() == "running" for row in members
        )
        for row in members:
            state_counts[str(row.get("State") or "unknown").lower()] += 1
        running_total += running
        container_total += len(members)
        projects.append(
            {
                "project": project,
                "attempt_id": identity.get(ATTEMPT_LABEL),
                "run_id": identity.get(RUN_LABEL),
                "group_id": identity.get(GROUP_LABEL),
                "running_containers": running,
                "total_containers": len(members),
            }
        )
    return {
        "managed_running_containers": running_total,
        "managed_total_containers": container_total,
        "active_compose_projects": sum(
            project["running_containers"] > 0 for project in projects
        ),
        "state_counts": dict(state_counts),
        "projects": projects,
    }


def local_image_refs(rows: Iterable[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for row in rows:
        repository = str(row.get("Repository") or "")
        digest = str(row.get("Digest") or "")
        if repository and repository != "<none>" and digest.startswith("sha256:"):
            refs.add(f"{repository}@{digest}")
    return refs


def aggregate_image_cache(records_dir: Path, present_refs: set[str]) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    managed_records = 0
    managed_present = 0
    active = 0
    invalid_records = 0
    for path in sorted(records_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_records += 1
            continue
        if record.get("managed_origin") != "pulled_by_rollout_service":
            continue
        managed_records += 1
        state = str(record.get("state") or "UNKNOWN").upper()
        state_counts[state] += 1
        image_ref = str(record.get("image_ref") or "")
        if state == "READY" and image_ref in present_refs:
            managed_present += 1
        if record.get("active_attempt_id"):
            active += 1
    return {
        "managed_records": managed_records,
        "managed_present": managed_present,
        "active": active,
        "pulling": state_counts["PULLING"],
        "ready": state_counts["READY"],
        "failed": state_counts["FAILED"],
        "quarantined": state_counts["QUARANTINED"],
        "evicted": state_counts["EVICTED"] + state_counts["EVICTED_REF_ONLY"],
        "invalid_records": invalid_records,
        "state_counts": dict(state_counts),
    }
