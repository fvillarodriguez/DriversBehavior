#!/usr/bin/env python3
"""
Streamlit page for managing the local two-node Ray cluster.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import pandas as pd
import streamlit as st

from src import ray_cluster_manager as ray_cluster


def _render_result(result: ray_cluster.CommandResult) -> None:
    if result.ok:
        st.success("Comando completado.")
    else:
        st.error(f"Comando fallo con codigo {result.returncode}.")
    st.code(f"$ {result.command}\n\n{result.combined_output or '(sin salida)'}", language="text")


def _render_results(results: Iterable[ray_cluster.CommandResult]) -> None:
    st.code(ray_cluster.command_outputs_to_text(results) or "(sin salida)", language="text")
    if all(result.ok for result in results):
        st.success("Operacion completada.")
    else:
        st.error("Una o mas operaciones fallaron.")


def _config_from_widgets(current: ray_cluster.RayClusterConfig) -> ray_cluster.RayClusterConfig:
    col_head, col_worker = st.columns(2)
    with col_head:
        head_ip = st.text_input("IP head", value=current.head_ip, key="ray_head_ip")
        head_cpus = st.number_input(
            "CPUs Ray en head",
            min_value=1,
            max_value=128,
            value=max(1, int(current.head_cpus)),
            step=1,
            key="ray_head_cpus",
        )
    with col_worker:
        worker_ip = st.text_input("IP worker", value=current.worker_ip, key="ray_worker_ip")
        worker_reserved = st.number_input(
            "CPUs reservadas en worker",
            min_value=0,
            max_value=32,
            value=max(0, int(current.worker_reserved_cpus)),
            step=1,
            key="ray_worker_reserved_cpus",
        )

    st.divider()
    col_ssh, col_repo = st.columns(2)
    with col_ssh:
        ssh_user = st.text_input("Usuario SSH worker", value=current.ssh_user, key="ray_ssh_user")
        ssh_key_path = st.text_input("Llave SSH privada", value=current.ssh_key_path, key="ray_ssh_key")
    with col_repo:
        remote_repo_path = st.text_input(
            "Ruta repo en worker",
            value=current.remote_repo_path,
            key="ray_remote_repo_path",
        )
        command_timeout = st.number_input(
            "Timeout comandos (s)",
            min_value=5,
            max_value=300,
            value=max(5, int(current.command_timeout_s)),
            step=5,
            key="ray_timeout",
        )

    with st.expander("Puertos Ray fijos", expanded=False):
        st.caption("Estos puertos se mantienen fijos para facilitar firewall y diagnostico.")
        cols = st.columns(4)
        cols[0].metric("Head", current.head_port)
        cols[1].metric("Dashboard", current.dashboard_port)
        cols[2].metric("Object manager", current.object_manager_port)
        cols[3].metric("Ray Client", current.ray_client_port)
        st.caption(f"Workers: {current.worker_port_min}-{current.worker_port_max}")

    return replace(
        current,
        head_ip=head_ip.strip(),
        worker_ip=worker_ip.strip(),
        ssh_user=ssh_user.strip(),
        ssh_key_path=ssh_key_path.strip(),
        remote_repo_path=remote_repo_path.strip(),
        head_cpus=int(head_cpus),
        worker_reserved_cpus=int(worker_reserved),
        command_timeout_s=int(command_timeout),
    )


def _render_config_tab(config: ray_cluster.RayClusterConfig) -> ray_cluster.RayClusterConfig:
    st.subheader("Configuracion del cluster")
    updated = _config_from_widgets(config)

    warnings = ray_cluster.check_config_warnings(updated)
    if warnings:
        for warning in warnings:
            st.warning(warning)

    col_save, col_reload = st.columns([1, 5])
    with col_save:
        if st.button("Guardar configuracion", type="primary", width="stretch"):
            ray_cluster.save_config(updated)
            st.success("Configuracion guardada.")
            st.rerun()
    with col_reload:
        st.caption(f"Archivo: {ray_cluster.CONFIG_FILE}")

    st.divider()
    st.subheader("Comandos para configurar Thunderbolt Bridge")
    st.caption("Ejecutelos manualmente si macOS pide permisos de administrador.")
    st.code(
        "\n".join(
            [
                f"# Head ({ray_cluster.local_hostname()})",
                ray_cluster.bridge_manual_command(updated.head_ip, updated.netmask),
                "",
                "# Worker",
                ray_cluster.bridge_manual_command(updated.worker_ip, updated.netmask),
            ]
        ),
        language="bash",
    )
    return updated


def _checks_to_dataframe(checks: list[ray_cluster.CheckResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": check.name,
                "estado": "OK" if check.ok else "Pendiente",
                "detalle": check.detail,
                "comando": check.command,
            }
            for check in checks
        ]
    )


def _render_preflight_tab(config: ray_cluster.RayClusterConfig) -> None:
    st.subheader("Preflight")
    st.caption("Valida red Thunderbolt, SSH, Python, Ray y deja procesos Ray detenidos antes de iniciar.")
    if st.button("Ejecutar preflight", type="primary"):
        with st.spinner("Ejecutando validaciones..."):
            st.session_state["ray_preflight_checks"] = ray_cluster.run_preflight(config)

    checks = st.session_state.get("ray_preflight_checks")
    if checks:
        df = _checks_to_dataframe(checks)
        st.dataframe(df, width="stretch", hide_index=True)
        ok_count = sum(1 for check in checks if check.ok)
        st.metric("Checks OK", f"{ok_count}/{len(checks)}")
        if ok_count == len(checks):
            st.success("Preflight listo para iniciar cluster.")
        else:
            st.warning("Resuelva los checks pendientes antes de usar el cluster para trabajos reales.")
    else:
        st.info("Ejecute preflight antes de iniciar el cluster por primera vez.")


def _render_control_tab(config: ray_cluster.RayClusterConfig) -> None:
    st.subheader("Control del cluster")
    st.caption("La app corre en el head y administra el worker por SSH.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Iniciar head", type="primary", width="stretch"):
            with st.spinner("Iniciando head..."):
                _render_result(ray_cluster.start_head(config))
        if st.button("Detener head", width="stretch"):
            with st.spinner("Deteniendo head..."):
                _render_result(ray_cluster.stop_head(config))
    with col2:
        if st.button("Iniciar worker", type="primary", width="stretch"):
            with st.spinner("Iniciando worker por SSH..."):
                _render_result(ray_cluster.start_worker(config))
        if st.button("Detener worker", width="stretch"):
            with st.spinner("Deteniendo worker por SSH..."):
                _render_result(ray_cluster.stop_worker(config))
    with col3:
        if st.button("Iniciar cluster", type="primary", width="stretch"):
            with st.spinner("Iniciando cluster completo..."):
                _render_results(ray_cluster.start_cluster(config))
        if st.button("Detener cluster", width="stretch"):
            with st.spinner("Deteniendo cluster completo..."):
                _render_results(ray_cluster.stop_cluster(config))
        if st.button("Reiniciar cluster", width="stretch"):
            with st.spinner("Reiniciando cluster completo..."):
                _render_results(ray_cluster.restart_cluster(config))

    st.divider()
    st.subheader("Comandos equivalentes")
    st.caption("Referencia de auditoria; la UI ejecuta estos comandos con parametros fijos.")
    st.code(" ".join(ray_cluster.build_head_start_args(config)), language="bash")
    st.code(ray_cluster.build_worker_start_script(config), language="bash")


def _render_monitor_tab(config: ray_cluster.RayClusterConfig) -> None:
    st.subheader("Monitor")
    col_status, col_dashboard = st.columns([1, 1])
    with col_status:
        if st.button("Actualizar ray status", type="primary"):
            with st.spinner("Consultando Ray..."):
                st.session_state["ray_status_result"] = ray_cluster.ray_status(config)
    with col_dashboard:
        st.link_button("Abrir dashboard Ray", config.dashboard_url, width="stretch")
        st.caption(config.dashboard_url)

    status_result = st.session_state.get("ray_status_result")
    if status_result:
        if status_result.ok:
            st.success("Ray responde.")
            summary = ray_cluster.parse_ray_status_summary(status_result.stdout)
            col_nodes, col_cpu = st.columns(2)
            col_nodes.metric("Nodos activos", summary.get("active_nodes", 0))
            col_cpu.metric("CPU", summary.get("usage", {}).get("CPU", "-"))
        else:
            st.error("Ray no responde o el cluster esta detenido.")
        st.code(status_result.combined_output or "(sin salida)", language="text")
    else:
        st.info("Actualice el estado para ver recursos y nodos.")

    st.divider()
    st.subheader("Logs")
    col_local, col_remote = st.columns(2)
    with col_local:
        if st.button("Ver logs head", width="stretch"):
            with st.spinner("Leyendo logs locales..."):
                st.session_state["ray_head_logs"] = ray_cluster.tail_logs(config, remote=False)
        logs = st.session_state.get("ray_head_logs")
        if logs:
            st.code(logs.combined_output or "(sin logs)", language="text")
    with col_remote:
        if st.button("Ver logs worker", width="stretch"):
            with st.spinner("Leyendo logs remotos..."):
                st.session_state["ray_worker_logs"] = ray_cluster.tail_logs(config, remote=True)
        logs = st.session_state.get("ray_worker_logs")
        if logs:
            st.code(logs.combined_output or "(sin logs)", language="text")


def _render_benchmark_tab(config: ray_cluster.RayClusterConfig) -> None:
    st.subheader("Prueba distribuida")
    tasks = st.number_input(
        "Numero de tareas Ray",
        min_value=1,
        max_value=10000,
        value=80,
        step=10,
        key="ray_benchmark_tasks",
    )
    if st.button("Ejecutar prueba distribuida", type="primary"):
        with st.spinner("Ejecutando benchmark Ray..."):
            result, payload = ray_cluster.run_distributed_benchmark(config, tasks=int(tasks))
            st.session_state["ray_benchmark_result"] = result
            st.session_state["ray_benchmark_payload"] = payload

    result = st.session_state.get("ray_benchmark_result")
    payload = st.session_state.get("ray_benchmark_payload")
    if result:
        if result.ok and payload:
            st.success("Benchmark completado.")
            host_counts = payload.get("tasks_by_host", {})
            if isinstance(host_counts, dict) and host_counts:
                st.bar_chart(pd.Series(host_counts, name="tareas"))
            col_tasks, col_nodes = st.columns(2)
            col_tasks.metric("Tareas", payload.get("tasks", "-"))
            col_nodes.metric("Nodos vivos", sum(1 for node in payload.get("nodes", []) if node.get("alive")))
            with st.expander("Payload Ray", expanded=False):
                st.json(payload)
        else:
            st.error("Benchmark fallo.")
            st.code(result.combined_output or "(sin salida)", language="text")
    else:
        st.info("Inicie el cluster y ejecute la prueba para confirmar tareas en ambos Macs.")


def main(set_page_config: bool = False, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="Ray Cluster", layout="wide")

    st.title("Ray Cluster")
    st.markdown(
        "Administra el cluster Ray por Thunderbolt Bridge desde Streamlit. "
        "La app corre en el head y controla el worker por SSH con llave."
    )

    config = ray_cluster.load_config()
    tabs = st.tabs(["Configuracion", "Preflight", "Control", "Monitor", "Prueba distribuida"])
    with tabs[0]:
        config = _render_config_tab(config)
    with tabs[1]:
        _render_preflight_tab(config)
    with tabs[2]:
        _render_control_tab(config)
    with tabs[3]:
        _render_monitor_tab(config)
    with tabs[4]:
        _render_benchmark_tab(config)

    if show_exit_button and st.button("Cerrar"):
        raise SystemExit(0)


if __name__ == "__main__":
    main(set_page_config=True)
