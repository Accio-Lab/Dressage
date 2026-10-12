from __future__ import annotations

import os
from pathlib import Path


def capture_host_resource_snapshot() -> dict[str, object]:
    """Best-effort diagnostics only; callers must not use this for admission."""
    snapshot: dict[str, object] = {}
    try:
        snapshot["load_average"] = list(os.getloadavg())
    except OSError:
        pass
    try:
        values: dict[str, int] = {}
        selected = {"MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, _, raw = line.partition(" ")
            if name in selected:
                values[name.rstrip(":")] = int(raw.split()[0]) // 1024
        snapshot["memory_mb"] = values
    except (OSError, ValueError, IndexError):
        pass
    pressure: dict[str, str] = {}
    for resource in ("cpu", "memory", "io"):
        try:
            pressure[resource] = (
                (Path("/proc/pressure") / resource).read_text(encoding="utf-8").strip()
            )
        except OSError:
            continue
    if pressure:
        snapshot["pressure"] = pressure
    return snapshot
