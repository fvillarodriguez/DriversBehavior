from __future__ import annotations

from typing import Any

from cluster_app.sdk import DaskCluster


def render_cluster_panel(cluster: DaskCluster | None = None, key: str = "dask_cluster") -> DaskCluster:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "streamlit is required to render the Dask cluster panel. "
            "Install streamlit in the ML app environment."
        ) from exc

    if cluster is None:
        if key not in st.session_state:
            st.session_state[key] = DaskCluster()
        cluster = st.session_state[key]

    status = cluster.status()
    scheduler = _as_mapping(status.get("scheduler"))
    worker = _as_mapping(status.get("worker"))
    nodes = status.get("nodes") or []

    st.subheader("Dask Cluster")
    scheduler_col, worker_col, nodes_col = st.columns(3)
    scheduler_col.metric("Scheduler", "running" if scheduler.get("running") else "stopped")
    worker_col.metric("Worker", "running" if worker.get("running") else "stopped")
    nodes_col.metric("Nodes", str(len(nodes)))

    scheduler_running = bool(scheduler.get("running"))
    scheduler_is_local = scheduler.get("local", True)
    worker_running = bool(worker.get("running"))

    if scheduler_col.button("Start", key=f"{key}_scheduler_start", disabled=scheduler_running):
        _run_action(st, cluster.start_scheduler, "No se pudo iniciar el scheduler.")
    if scheduler_col.button(
        "Stop",
        key=f"{key}_scheduler_stop",
        disabled=not scheduler_running or not scheduler_is_local,
    ):
        _run_action(st, cluster.stop_scheduler, "No se pudo detener el scheduler.")

    if worker_col.button(
        "Start",
        key=f"{key}_worker_start",
        disabled=worker_running or not scheduler_running,
    ):
        _run_action(st, cluster.start_worker, "No se pudo iniciar el worker.")
    if worker_col.button("Stop", key=f"{key}_worker_stop", disabled=not worker_running):
        _run_action(st, cluster.stop_worker, "No se pudo detener el worker.")

    if nodes_col.button("Refresh", key=f"{key}_refresh"):
        st.rerun()

    dashboard = status.get("dashboard")
    if dashboard:
        st.link_button("Open Dask Dashboard", str(dashboard))
    if scheduler.get("address"):
        st.caption(f"Scheduler: {scheduler['address']}")
    return cluster


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _run_action(st: Any, action: Any, error_prefix: str) -> None:
    try:
        action()
    except Exception as exc:
        st.error(f"{error_prefix} {exc}")
        return
    st.rerun()
