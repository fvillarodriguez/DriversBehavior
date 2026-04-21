#!/usr/bin/env python3
"""
Streamlit page for managing the local two-node Ray cluster.
"""
from __future__ import annotations

import json
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


def _check_state(check: ray_cluster.CheckResult) -> str:
    if check.ok:
        return "OK"
    if check.blocking:
        return "Bloqueante"
    return "Advertencia"


def _config_signature(config: ray_cluster.RayClusterConfig) -> str:
    return json.dumps(ray_cluster.config_to_json_dict(config), sort_keys=True)


def _action_blockers(
    snapshot: ray_cluster.RayHealthSnapshot | None,
    *,
    stale: bool,
    action: str,
) -> list[ray_cluster.CheckResult]:
    if stale or snapshot is None:
        return [
            ray_cluster.CheckResult(
                name="Salud Ray",
                ok=False,
                detail="Actualice la salud del cluster para validar esta accion con la configuracion actual.",
                blocking=True,
            )
        ]
    action_checks = {
        "head": snapshot.head_start_checks,
        "worker": snapshot.worker_start_checks,
        "cluster": snapshot.worker_start_checks,
        "benchmark": snapshot.benchmark_checks,
    }
    return ray_cluster.blocking_checks(action_checks[action])


def _render_blocker_notice(title: str, checks: list[ray_cluster.CheckResult]) -> None:
    if not checks:
        return
    st.error(f"{title}\n{ray_cluster.checks_to_text(checks)}")


def _render_health_console(config: ray_cluster.RayClusterConfig) -> tuple[ray_cluster.RayHealthSnapshot | None, bool]:
    st.subheader("Salud del cluster")
    st.caption(
        "Resume entorno local, espacio en /tmp, GCS, dashboard y validacion de la ruta remota del worker."
    )

    signature = _config_signature(config)
    refresh = st.button("Actualizar salud Ray", type="primary", key="ray_health_refresh")
    if refresh or "ray_health_snapshot" not in st.session_state:
        with st.spinner("Auditando salud de Ray..."):
            st.session_state["ray_health_snapshot"] = ray_cluster.collect_health_snapshot(config)
            st.session_state["ray_health_signature"] = signature

    snapshot = st.session_state.get("ray_health_snapshot")
    stale = st.session_state.get("ray_health_signature") != signature
    if stale:
        st.warning(
            "La configuracion actual cambio desde el ultimo audit. Actualice la salud antes de iniciar, reiniciar o ejecutar benchmark."
        )

    if snapshot is None:
        st.info("Actualice la salud del cluster para ver el estado actual.")
        return None, True

    cols = st.columns(len(snapshot.summary_checks))
    for col, check in zip(cols, snapshot.summary_checks):
        with col:
            st.metric(check.name, _check_state(check))
            st.caption(check.detail)

    with st.expander("Checks detallados del entorno local", expanded=False):
        df = _checks_to_dataframe(list(snapshot.local_environment_checks))
        st.dataframe(df, width="stretch", hide_index=True)

    return snapshot, stale


def _config_from_widgets(current: ray_cluster.RayClusterConfig) -> ray_cluster.RayClusterConfig:
    automatic = ray_cluster.automatic_bridge_config(current)
    st.caption(
        "Head y worker usan un perfil automatico de Thunderbolt Bridge. "
        "Solo debes ajustar el usuario SSH, la ruta del repo y los recursos; la red queda fija y simple."
    )

    col_bridge_head, col_bridge_worker, col_bridge_mask = st.columns(3)
    col_bridge_head.metric("Head", automatic.head_ip)
    col_bridge_worker.metric("Worker", automatic.worker_ip)
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

    col_auto, col_worker, col_save, col_reload = st.columns([1, 1, 1, 3])
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
    with col_worker:
        if st.button("Arrancar este Mac como Worker", width="stretch"):
            with st.spinner("Iniciando worker local..."):
                st.session_state["ray_config_worker_start_result"] = ray_cluster.start_local_worker(updated)
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

    worker_start_result = st.session_state.get("ray_config_worker_start_result")
    if worker_start_result:
        st.divider()
        st.subheader("Resultado de arranque Worker")
        _render_result(worker_start_result)

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
                "estado": _check_state(check),
                "detalle": check.detail,
                "comando": check.command,
            }
            for check in checks
        ]
    )


def _render_preflight_tab(
    config: ray_cluster.RayClusterConfig,
    snapshot: ray_cluster.RayHealthSnapshot | None,
    stale: bool,
) -> None:
    st.subheader("Preflight")
    st.caption("Usa el preflight que corresponde al rol de este Mac. En el worker local no se requiere SSH.")
    if stale:
        st.warning("El snapshot de salud esta desactualizado; el preflight ejecutara checks frescos con la configuracion actual.")

    preflight_head, preflight_worker = st.tabs(["Head / controlador", "Worker local"])
    with preflight_head:
        st.caption(
            "Para ejecutar en el Mac head: valida Thunderbolt en 10.10.10.1, SSH hacia el worker, "
            "Python, Ray y puertos del cluster."
        )
        if st.button("Ejecutar preflight head", type="primary"):
            with st.spinner("Ejecutando validaciones de head..."):
                st.session_state["ray_preflight_head_checks"] = ray_cluster.run_preflight(config)

        checks = st.session_state.get("ray_preflight_head_checks")
        if checks:
            _render_checks(checks, success_text="Preflight head listo.")
        else:
            st.info("Ejecute este preflight solo desde el Mac head.")

    with preflight_worker:
        st.caption(
            "Para ejecutar en este Mac cuando es worker: valida Thunderbolt en 10.10.10.2, "
            "conectividad al head 10.10.10.1, Python, Ray y puertos locales del worker."
        )
        if st.button("Ejecutar preflight worker", type="primary"):
            with st.spinner("Ejecutando validaciones de worker..."):
                st.session_state["ray_preflight_worker_checks"] = ray_cluster.run_worker_preflight(config)

        checks = st.session_state.get("ray_preflight_worker_checks")
        if checks:
            _render_checks(checks, success_text="Preflight worker listo.")
        else:
            st.info("Ejecute este preflight desde el Mac con IP Thunderbolt 10.10.10.2.")


def _render_checks(checks: list[ray_cluster.CheckResult], *, success_text: str) -> None:
    df = _checks_to_dataframe(checks)
    st.dataframe(df, width="stretch", hide_index=True)
    ok_count = sum(1 for check in checks if check.ok)
    warning_count = sum(1 for check in checks if (not check.ok and not check.blocking))
    blocking_count = sum(1 for check in checks if (not check.ok and check.blocking))
    col_ok, col_warn, col_block = st.columns(3)
    col_ok.metric("Checks OK", ok_count)
    col_warn.metric("Advertencias", warning_count)
    col_block.metric("Bloqueantes", blocking_count)
    if warning_count == 0 and blocking_count == 0:
        st.success(success_text)
    else:
        if blocking_count:
            st.error("Resuelva los checks bloqueantes antes de usar el cluster para trabajos reales.")
        if warning_count:
            st.warning("Hay advertencias que conviene atender para evitar degradacion de Ray.")


def _render_control_tab(
    config: ray_cluster.RayClusterConfig,
    snapshot: ray_cluster.RayHealthSnapshot | None,
    stale: bool,
) -> None:
    st.subheader("Control del cluster")
    st.caption("El head se administra en el Mac 10.10.10.1. El worker local se administra en el Mac 10.10.10.2, sin SSH.")
    head_blockers = _action_blockers(snapshot, stale=stale, action="head")
    worker_blockers = _action_blockers(snapshot, stale=stale, action="worker")
    cluster_blockers = _action_blockers(snapshot, stale=stale, action="cluster")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Iniciar head", type="primary", width="stretch", disabled=bool(head_blockers)):
            with st.spinner("Iniciando head..."):
                _render_result(ray_cluster.start_head(config))
        if st.button("Detener head", width="stretch"):
            with st.spinner("Deteniendo head..."):
                _render_result(ray_cluster.stop_head(config))
    with col2:
        if st.button("Iniciar worker remoto", width="stretch", disabled=bool(worker_blockers)):
            with st.spinner("Iniciando worker por SSH..."):
                _render_result(ray_cluster.start_worker(config))
        if st.button("Detener worker remoto", width="stretch"):
            with st.spinner("Deteniendo worker por SSH..."):
                _render_result(ray_cluster.stop_worker(config))
    with col3:
        if st.button("Iniciar worker local", type="primary", width="stretch", disabled=bool(head_blockers)):
            with st.spinner("Iniciando worker local..."):
                _render_result(ray_cluster.start_local_worker(config))
        if st.button("Detener worker local", width="stretch"):
            with st.spinner("Deteniendo worker local..."):
                _render_result(ray_cluster.stop_local_worker(config))
    with col4:
        if st.button("Iniciar cluster", type="primary", width="stretch", disabled=bool(cluster_blockers)):
            with st.spinner("Iniciando cluster completo..."):
                _render_results(ray_cluster.start_cluster(config))
        if st.button("Detener cluster", width="stretch"):
            with st.spinner("Deteniendo cluster completo..."):
                _render_results(ray_cluster.stop_cluster(config))
        if st.button("Reiniciar cluster", width="stretch", disabled=bool(cluster_blockers)):
            with st.spinner("Reiniciando cluster completo..."):
                _render_results(ray_cluster.restart_cluster(config))

    _render_blocker_notice("Head local bloqueado:", head_blockers)
    _render_blocker_notice("Worker remoto / cluster bloqueado:", worker_blockers)

    st.divider()
    st.subheader("Comandos equivalentes")
    st.caption("Referencia de auditoria; la UI ejecuta estos comandos con parametros fijos.")
    st.code(" ".join(ray_cluster.build_head_start_args(config)), language="bash")
    st.code(" ".join(ray_cluster.build_worker_start_args(config, block=True)), language="bash")
    st.code(ray_cluster.build_worker_start_script(config), language="bash")


def _render_monitor_tab(
    config: ray_cluster.RayClusterConfig,
    snapshot: ray_cluster.RayHealthSnapshot | None,
    stale: bool,
) -> None:
    st.subheader("Monitor")
    col_status, col_dashboard = st.columns([1, 1])
    with col_status:
        if st.button("Actualizar ray status", type="primary"):
            with st.spinner("Consultando Ray..."):
                st.session_state["ray_status_result"] = ray_cluster.ray_status(config)
    with col_dashboard:
        dashboard_ok = bool(snapshot and not stale and snapshot.dashboard_check.ok)
        if dashboard_ok:
            st.link_button("Abrir dashboard Ray", config.dashboard_url, width="stretch")
            st.caption(config.dashboard_url)
        else:
            st.button("Dashboard no disponible", width="stretch", disabled=True, key="ray_dashboard_disabled")
            if snapshot is not None:
                st.caption(snapshot.dashboard_check.detail)
            else:
                st.caption(config.dashboard_url)

    if stale:
        st.warning("El snapshot de salud esta desactualizado; actualicelo para validar dashboard y GCS.")
    elif snapshot is not None:
        if not snapshot.gcs_check.ok:
            st.error(snapshot.gcs_check.detail)
        elif not snapshot.dashboard_check.ok:
            st.warning(snapshot.dashboard_check.detail)

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


def _render_benchmark_tab(
    config: ray_cluster.RayClusterConfig,
    snapshot: ray_cluster.RayHealthSnapshot | None,
    stale: bool,
) -> None:
    st.subheader("Prueba distribuida")
    benchmark_blockers = _action_blockers(snapshot, stale=stale, action="benchmark")
    tasks = st.number_input(
        "Numero de tareas Ray",
        min_value=1,
        max_value=10000,
        value=80,
        step=10,
        key="ray_benchmark_tasks",
    )
    if st.button("Ejecutar prueba distribuida", type="primary", disabled=bool(benchmark_blockers)):
        with st.spinner("Ejecutando benchmark Ray..."):
            result, payload = ray_cluster.run_distributed_benchmark(config, tasks=int(tasks))
            st.session_state["ray_benchmark_result"] = result
            st.session_state["ray_benchmark_payload"] = payload

    _render_blocker_notice("Benchmark bloqueado:", benchmark_blockers)

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
        "El head y el worker tienen preflights separados porque validan responsabilidades distintas."
    )

    config = ray_cluster.automatic_bridge_config(ray_cluster.load_config())
    health_placeholder = st.container()
    tabs = st.tabs(["Configuracion", "Preflight", "Control", "Monitor", "Prueba distribuida"])
    with tabs[0]:
        config = _render_config_tab(config)
    with health_placeholder:
        snapshot, stale = _render_health_console(config)
    with tabs[1]:
        _render_preflight_tab(config, snapshot, stale)
    with tabs[2]:
        _render_control_tab(config, snapshot, stale)
    with tabs[3]:
        _render_monitor_tab(config, snapshot, stale)
    with tabs[4]:
        _render_benchmark_tab(config, snapshot, stale)

    if show_exit_button and st.button("Cerrar"):
        raise SystemExit(0)


if __name__ == "__main__":
    main(set_page_config=True)
