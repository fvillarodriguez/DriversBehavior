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
    automatic = ray_cluster.automatic_bridge_config(current)
    st.caption(
        "Head y worker usan un perfil automatico de Thunderbolt Bridge. "
        "Solo debes ajustar el usuario SSH, la ruta del repo y los recursos; la red queda fija y simple."
    )

    col_bridge_head, col_bridge_worker, col_bridge_mask = st.columns(3)
    col_bridge_head.metric("Head (este Mac)", automatic.head_ip)
    col_bridge_worker.metric("Worker (Mac remoto)", automatic.worker_ip)
    col_bridge_mask.metric("Mascara", automatic.netmask)

    col_head, col_worker = st.columns(2)
    with col_head:
        head_cpus = st.number_input(
            "CPUs Ray en head",
            min_value=1,
            max_value=128,
            value=max(1, int(automatic.head_cpus)),
            step=1,
            key="ray_head_cpus",
        )
    with col_worker:
        worker_reserved = st.number_input(
            "CPUs reservadas en worker",
            min_value=0,
            max_value=32,
            value=max(0, int(automatic.worker_reserved_cpus)),
            step=1,
            key="ray_worker_reserved_cpus",
        )

    st.divider()
    col_ssh, col_repo = st.columns(2)
    with col_ssh:
        ssh_user = st.text_input("Usuario SSH worker", value=automatic.ssh_user, key="ray_ssh_user")
        st.caption("La llave SSH se detecta o genera automaticamente en este Mac.")
    with col_repo:
        remote_repo_path = st.text_input(
            "Ruta repo en worker",
            value=automatic.remote_repo_path,
            key="ray_remote_repo_path",
        )
        command_timeout = st.number_input(
            "Timeout comandos (s)",
            min_value=5,
            max_value=300,
            value=max(5, int(automatic.command_timeout_s)),
            step=5,
            key="ray_timeout",
        )

    with st.expander("Puertos Ray fijos", expanded=False):
        st.caption("Estos puertos se mantienen fijos para facilitar firewall y diagnostico.")
        cols = st.columns(4)
        cols[0].metric("Head", automatic.head_port)
        cols[1].metric("Dashboard", automatic.dashboard_port)
        cols[2].metric("Object manager", automatic.object_manager_port)
        cols[3].metric("Ray Client", automatic.ray_client_port)
        st.caption(f"Workers: {automatic.worker_port_min}-{automatic.worker_port_max}")

    return ray_cluster.automatic_bridge_config(
        replace(
            automatic,
            ssh_user=ssh_user.strip(),
            remote_repo_path=remote_repo_path.strip(),
            head_cpus=int(head_cpus),
            worker_reserved_cpus=int(worker_reserved),
            command_timeout_s=int(command_timeout),
        )
    )


def _render_ssh_auto_tools(config: ray_cluster.RayClusterConfig) -> None:
    st.divider()
    st.subheader("SSH automatica")
    st.caption(
        "La app detecta o genera una llave local y puede autorizarla automaticamente en el worker. "
        "La password solo se usa en el primer enlace."
    )

    col_password, col_button = st.columns([2, 1])
    with col_password:
        password = st.text_input(
            "Password del worker (solo si es la primera vez)",
            type="password",
            key="ray_worker_ssh_password",
        )
    with col_button:
        st.write("")
        if st.button("Preparar SSH", type="primary", key="ray_prepare_ssh", width="stretch"):
            with st.spinner("Preparando SSH automatica..."):
                st.session_state["ray_prepare_ssh_results"] = ray_cluster.prepare_ssh_access(config, password=password)

    prepare_results = st.session_state.get("ray_prepare_ssh_results")
    if prepare_results:
        _render_results(prepare_results)


def _render_config_tab(config: ray_cluster.RayClusterConfig) -> ray_cluster.RayClusterConfig:
    st.subheader("Configuracion del cluster")
    updated = _config_from_widgets(config)

    warnings = ray_cluster.check_config_warnings(updated)
    if warnings:
        for warning in warnings:
            st.warning(warning)

    _render_ssh_auto_tools(updated)

    col_auto, col_save, col_reload = st.columns([1, 1, 4])
    with col_auto:
        if st.button("Aplicar conexion automatica", type="primary", width="stretch"):
            with st.spinner("Configurando Thunderbolt Bridge en head y worker..."):
                st.session_state["ray_bridge_apply_results"] = ray_cluster.apply_automatic_bridge(updated)
            results = st.session_state.get("ray_bridge_apply_results") or []
            if results and all(result.ok for result in results):
                ray_cluster.save_config(updated)
                st.success("Thunderbolt Bridge configurado y perfil guardado.")
            else:
                st.warning("La configuracion automatica quedo incompleta. Revise la salida y, si hace falta, use el diagnostico manual.")
    with col_save:
        if st.button("Guardar configuracion", width="stretch"):
            ray_cluster.save_config(updated)
            st.success("Configuracion guardada.")
    with col_reload:
        st.caption(f"Archivo: {ray_cluster.CONFIG_FILE}")

    bridge_results = st.session_state.get("ray_bridge_apply_results")
    if bridge_results:
        st.divider()
        st.subheader("Resultado de conexion automatica")
        _render_results(bridge_results)

    st.divider()
    st.subheader("Sincronizacion head/worker")
    st.caption(
        "La conexion usa siempre el mismo perfil Thunderbolt Bridge. "
        "Primero puedes preparar SSH automaticamente y luego aplicar la conexion Thunderbolt para dejar ambos Macs sincronizados."
    )
    with st.expander("Diagnostico manual", expanded=False):
        st.caption("Usa estos comandos solo si macOS o el worker no permiten la configuracion automatica.")
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
    st.caption("Valida el perfil automatico de Thunderbolt Bridge, SSH, Python, Ray y deja procesos Ray detenidos antes de iniciar.")
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
    st.caption("Este Mac opera como head y administra el worker por SSH sobre Thunderbolt Bridge.")

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
        "Administra el cluster Ray con una conexion head/worker simple y fija sobre Thunderbolt Bridge. "
        "La red se maneja con un perfil automatico y el head controla el worker por SSH con llave."
    )

    config = ray_cluster.automatic_bridge_config(ray_cluster.load_config())
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
