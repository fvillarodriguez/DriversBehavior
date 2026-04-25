"""
Helpers for connecting experiment pipelines to the configured Ray Cluster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from src import ray_cluster_manager


EXECUTION_BACKEND_LOCAL = "local"
EXECUTION_BACKEND_RAY_CLUSTER = "ray_cluster"
EXECUTION_BACKEND_VALUES = (
    EXECUTION_BACKEND_LOCAL,
    EXECUTION_BACKEND_RAY_CLUSTER,
)


@dataclass(frozen=True)
class RayClusterRuntime:
    config: ray_cluster_manager.RayClusterConfig
    status: ray_cluster_manager.CommandResult
    ray_module: Any
    nodes: List[Dict[str, object]]
    cluster_resources: Dict[str, object]
    total_cpus: int
    max_node_cpus: int
    active_nodes: int


def normalize_execution_backend(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", EXECUTION_BACKEND_LOCAL}:
        return EXECUTION_BACKEND_LOCAL
    if text == EXECUTION_BACKEND_RAY_CLUSTER:
        return EXECUTION_BACKEND_RAY_CLUSTER
    return EXECUTION_BACKEND_LOCAL


def _resource_cpu_value(value: object) -> int:
    try:
        numeric = float(value)
    except Exception:
        return 0
    if not math.isfinite(numeric):
        return 0
    return max(0, int(math.floor(numeric)))


def connect_ray_cluster(*, ray_module: Any | None = None) -> RayClusterRuntime:
    config = ray_cluster_manager.automatic_bridge_config(
        ray_cluster_manager.load_config()
    )
    blockers = ray_cluster_manager.blocking_checks(
        [
            *ray_cluster_manager.runtime_connection_health_checks(config),
            ray_cluster_manager.check_remote_repo_path(config),
        ]
    )
    if blockers:
        raise RuntimeError(
            "No se puede conectar al Ray Cluster hasta corregir estos checks:\n"
            + ray_cluster_manager.checks_to_text(blockers)
        )
    status = ray_cluster_manager.ray_status(config)
    if not status.ok:
        detail = status.combined_output or "Ray Cluster no responde."
        raise RuntimeError(detail)

    if ray_module is None:
        import ray as imported_ray  # type: ignore

        ray_module = imported_ray

    if not ray_module.is_initialized():
        import src as src_package  # type: ignore

        ray_module.init(
            address=config.ray_address,
            ignore_reinit_error=True,
            runtime_env={"py_modules": [src_package]},
        )

    raw_nodes = list(ray_module.nodes() or [])
    alive_nodes = [dict(node) for node in raw_nodes if bool(node.get("Alive"))]
    cluster_resources = dict(ray_module.cluster_resources() or {})
    total_cpus = _resource_cpu_value(cluster_resources.get("CPU"))
    max_node_cpus = max(
        [_resource_cpu_value((node.get("Resources") or {}).get("CPU")) for node in alive_nodes]
        or [0]
    )
    if total_cpus <= 0:
        total_cpus = max(1, max_node_cpus)
    if max_node_cpus <= 0:
        max_node_cpus = max(1, total_cpus)

    return RayClusterRuntime(
        config=config,
        status=status,
        ray_module=ray_module,
        nodes=alive_nodes,
        cluster_resources=cluster_resources,
        total_cpus=int(total_cpus),
        max_node_cpus=int(max_node_cpus),
        active_nodes=int(len(alive_nodes)),
    )
