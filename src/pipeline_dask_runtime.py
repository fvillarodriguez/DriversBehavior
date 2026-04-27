"""
Helpers for connecting experiment pipelines to the configured Dask Cluster.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DASK_APP_DIR = ROOT_DIR / "dask"
DASK_CONFIG_PATH = DASK_APP_DIR / "config.yaml"

EXECUTION_BACKEND_LOCAL = "local"
EXECUTION_BACKEND_DASK_CLUSTER = "dask_cluster"
EXECUTION_BACKEND_VALUES = (
    EXECUTION_BACKEND_LOCAL,
    EXECUTION_BACKEND_DASK_CLUSTER,
)


@dataclass(frozen=True)
class DaskClusterRuntime:
    config: Any
    cluster: Any
    client: Any
    scheduler_info: Dict[str, object]
    workers: Dict[str, object]
    total_cpus: int
    max_node_cpus: int
    active_nodes: int
    address: str
    dashboard_url: str | None
    has_cpu_resource: bool


def normalize_execution_backend(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", EXECUTION_BACKEND_LOCAL}:
        return EXECUTION_BACKEND_LOCAL
    if text == EXECUTION_BACKEND_DASK_CLUSTER:
        return EXECUTION_BACKEND_DASK_CLUSTER
    return EXECUTION_BACKEND_LOCAL


def connect_dask_cluster(*, cluster: Any | None = None) -> DaskClusterRuntime:
    if cluster is None:
        DaskCluster = _load_dask_cluster_class()
        cluster = DaskCluster(config_path=DASK_CONFIG_PATH)

    cluster.start()
    client = cluster.client()
    wait_for_workers = getattr(client, "wait_for_workers", None)
    if callable(wait_for_workers):
        try:
            wait_for_workers(n_workers=1, timeout=10)
        except Exception:
            pass
    scheduler_info = dict(client.scheduler_info() or {})
    workers = dict(scheduler_info.get("workers") or {})
    worker_cpus = [_worker_cpu_count(worker) for worker in workers.values()]
    total_cpus = sum(worker_cpus)
    max_node_cpus = max(worker_cpus or [0])
    if total_cpus <= 0:
        total_cpus = max(1, max_node_cpus)
    if max_node_cpus <= 0:
        max_node_cpus = max(1, total_cpus)

    status = {}
    try:
        status = dict(cluster.status() or {})
    except Exception:
        status = {}
    scheduler = status.get("scheduler") if isinstance(status.get("scheduler"), dict) else {}
    address = str(
        scheduler.get("address")
        or getattr(getattr(client, "scheduler", None), "address", "")
        or scheduler_info.get("address")
        or ""
    )
    dashboard_url = str(status.get("dashboard") or "") or None
    has_cpu_resource = any(_worker_resource_cpu(worker) > 0 for worker in workers.values())

    return DaskClusterRuntime(
        config=getattr(cluster, "config", None),
        cluster=cluster,
        client=client,
        scheduler_info=scheduler_info,
        workers=workers,
        total_cpus=int(total_cpus),
        max_node_cpus=int(max_node_cpus),
        active_nodes=int(len(workers)),
        address=address,
        dashboard_url=dashboard_url,
        has_cpu_resource=bool(has_cpu_resource),
    )


def _load_dask_cluster_class():
    try:
        from cluster_app.sdk import DaskCluster

        return DaskCluster
    except ModuleNotFoundError:
        if str(DASK_APP_DIR) not in sys.path:
            sys.path.insert(0, str(DASK_APP_DIR))
        try:
            from cluster_app.sdk import DaskCluster
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "No se puede importar Dask Cluster App. Instale el complemento con "
                "`pip install -e ./dask` en el entorno de Streamlit."
            ) from exc
        return DaskCluster


def _resource_cpu_value(value: object) -> int:
    try:
        numeric = float(value)
    except Exception:
        return 0
    if not math.isfinite(numeric):
        return 0
    return max(0, int(math.floor(numeric)))


def _worker_resource_cpu(worker: object) -> int:
    if not isinstance(worker, dict):
        return 0
    resources = worker.get("resources")
    if not isinstance(resources, dict):
        return 0
    return _resource_cpu_value(resources.get("CPU"))


def _worker_cpu_count(worker: object) -> int:
    if not isinstance(worker, dict):
        return 0
    resource_cpu = _worker_resource_cpu(worker)
    if resource_cpu > 0:
        return resource_cpu
    return _resource_cpu_value(worker.get("nthreads"))


def dask_submit_resources(
    runtime: DaskClusterRuntime,
    trial_cpus: int | None,
) -> dict[str, float] | None:
    if not runtime.has_cpu_resource:
        return None
    cpus = max(1, int(trial_cpus or 1))
    return {"CPU": float(cpus)}
