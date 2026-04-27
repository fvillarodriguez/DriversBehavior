from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkerResources:
    cpu_threads: int
    memory_limit_bytes: int
    dask_resources: dict[str, float]


def memory_limit(total_ram_bytes: int, fraction: float = 0.9) -> int:
    return int(total_ram_bytes * fraction)


def dask_resource_flags(resources: dict[str, float]) -> list[str]:
    if not resources:
        return []
    value = ",".join(f"{key}={amount}" for key, amount in sorted(resources.items()))
    return ["--resources", value]

