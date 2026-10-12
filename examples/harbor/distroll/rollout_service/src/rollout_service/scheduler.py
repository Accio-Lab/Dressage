from __future__ import annotations

from agentic_rollout_contract.models import (
    AllocatedCharge,
    SchedulingCharge,
    VirtualCapacity,
)


def add_charge(allocated: AllocatedCharge, charge: SchedulingCharge) -> AllocatedCharge:
    return AllocatedCharge(
        cpu_units=allocated.cpu_units + charge.cpu_units,
        memory_mb=allocated.memory_mb + charge.memory_mb,
    )


def available_capacity(
    total: VirtualCapacity, used: AllocatedCharge
) -> AllocatedCharge:
    return AllocatedCharge(
        cpu_units=max(total.cpu_units - used.cpu_units, 0),
        memory_mb=max(total.memory_mb - used.memory_mb, 0),
    )


def dominant_load(total: VirtualCapacity, used: AllocatedCharge) -> float:
    return max(used.cpu_units / total.cpu_units, used.memory_mb / total.memory_mb)
