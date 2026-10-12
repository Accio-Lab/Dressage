from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def parse_cpu_line(line: str) -> tuple[int, ...]:
    fields = line.split()
    if not fields or fields[0] != "cpu":
        raise ValueError("expected aggregate cpu line")
    values = tuple(int(value) for value in fields[1:])
    if len(values) < 4:
        raise ValueError("aggregate cpu line is incomplete")
    return values


def read_cpu_times(path: Path = Path("/proc/stat")) -> tuple[int, ...]:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return parse_cpu_line(first)


def cpu_used_percent(
    previous: tuple[int, ...], current: tuple[int, ...]
) -> float | None:
    width = min(len(previous), len(current))
    if width < 4:
        return None
    deltas = [current[index] - previous[index] for index in range(width)]
    if any(value < 0 for value in deltas):
        return None
    total = sum(deltas)
    if total <= 0:
        return None
    idle = deltas[3] + (deltas[4] if width > 4 else 0)
    return round(max(0.0, min(100.0, (1.0 - idle / total) * 100.0)), 3)


def read_memory(path: Path = Path("/proc/meminfo")) -> dict[str, int | float]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        multiplier = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
        values[name] = value * multiplier
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(total - available, 0)
    used_percent = (used / total * 100.0) if total else 0.0
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
        "used_percent": round(used_percent, 3),
        "swap_total_bytes": swap_total,
        "swap_free_bytes": swap_free,
    }


def parse_psi(text: str) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        values: dict[str, float | int] = {}
        for field in fields[1:]:
            name, separator, raw = field.partition("=")
            if not separator:
                continue
            try:
                values[name] = int(raw) if name == "total" else float(raw)
            except ValueError:
                continue
        result[fields[0]] = values
    return result


def read_pressure(root: Path = Path("/proc/pressure")) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for resource in ("cpu", "memory", "io"):
        try:
            result[resource] = parse_psi((root / resource).read_text(encoding="utf-8"))
        except OSError:
            continue
    return result


def read_load() -> dict[str, Any]:
    values = os.getloadavg()
    cores = os.cpu_count() or 1
    return {
        "logical_cores": cores,
        "load1": round(values[0], 3),
        "load5": round(values[1], 3),
        "load15": round(values[2], 3),
        "load1_per_core": round(values[0] / cores, 4),
    }
