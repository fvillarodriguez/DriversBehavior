#!/usr/bin/env python3
"""Streamlit page to manage the JHU Rockfish HPC cluster from the Tesis app.

Six tabs:
    1. Configuración    — JHED, PI account, paths, módulos
    2. Salud & Conexión — audita SSH/módulos/quotas/GPU access/venv
    3. Sincronización   — rsync push + link Globus + tamaño remoto
    4. Launcher         — toggle Local/Rockfish + wizard run_gat_training
    5. Jobs             — squeue + cancelar
    6. Logs & Métricas  — tail stdout/stderr + jobstats + rsync pull
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from src import rockfish_manager as rf
from src import notification_system

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTADOS_DIR = ROOT_DIR / "Resultados"

SESSION_KEYS = {
    "config": "rockfish_config",
    "snapshot": "rockfish_health_snapshot",
    "signature": "rockfish_health_signature",
    "submit_log": "rockfish_submit_log",
    "last_jobid": "rockfish_last_jobid",
    "auto_refresh_jobs": "rockfish_auto_refresh_jobs",
    "auto_refresh_logs": "rockfish_auto_refresh_logs",
    "local_proc": "rockfish_local_proc_pid",
    "notified_jobs": "rockfish_notified_jobs",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_config() -> rf.RockfishConfig:
    if SESSION_KEYS["config"] not in st.session_state:
        st.session_state[SESSION_KEYS["config"]] = rf.load_config()
    return st.session_state[SESSION_KEYS["config"]]


def _save_config(cfg: rf.RockfishConfig) -> None:
    rf.save_config(cfg)
    st.session_state[SESSION_KEYS["config"]] = cfg


def _config_signature(cfg: rf.RockfishConfig) -> str:
    return json.dumps(rf.config_to_json_dict(cfg), sort_keys=True)


def _render_command_result(label: str, result: rf.CommandResult, *, expanded: bool = False) -> None:
    icon = "✅" if result.ok else "❌"
    with st.expander(f"{icon} {label} (rc={result.returncode})", expanded=expanded):
        st.code(f"$ {result.command}\n\n{result.combined_output or '(sin salida)'}", language="text")


def _check_badge(check: rf.CheckResult) -> tuple[str, str]:
    if check.ok:
        return "OK", "✅"
    if check.blocking:
        return "Bloqueante", "🛑"
    return "Aviso", "⚠️"


def _checks_to_df(checks: Iterable[rf.CheckResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Check": c.name, "Estado": _check_badge(c)[0], "Detalle": c.detail}
            for c in checks
        ]
    )


def _notify_job_finished(cfg: rf.RockfishConfig, job_id: str, state: str) -> None:
    if not cfg.notify_via_app:
        return
    seen: set[str] = st.session_state.setdefault(SESSION_KEYS["notified_jobs"], set())
    key = f"{job_id}:{state}"
    if key in seen:
        return
    seen.add(key)
    try:
        notification_system.send_notification_email(
            subject=f"Rockfish job {job_id} {state}",
            history_entry={
                "job_id": job_id,
                "state": state,
                "cluster": "rockfish",
                "host": cfg.ssh_host,
                "account": cfg.pi_account,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tab 1 — Configuración
# ---------------------------------------------------------------------------


def _tab_config(cfg: rf.RockfishConfig) -> rf.RockfishConfig:
    st.subheader("Identidad y cuenta")
    col1, col2, col3 = st.columns(3)
    jhed = col1.text_input("JHED ID", value=cfg.jhed, key="rf_cfg_jhed",
                            help="Tu usuario JHU (login SSH).")
    pi_account = col2.text_input(
        "PI account (<PI>_gpu)",
        value=cfg.pi_account,
        key="rf_cfg_pi_account",
        help="Account SLURM con acceso a GPU. Te lo da tu PI o `sacctmgr show assoc`.",
    )
    group = col3.text_input(
        "Grupo Unix",
        value=cfg.group,
        key="rf_cfg_group",
        help="Grupo para /data/ y /scratch16/. Lo ves con `id -gn` en Rockfish.",
    )

    st.subheader("Hosts y SSH")
    col1, col2 = st.columns(2)
    ssh_host = col1.text_input("Login host", value=cfg.ssh_host, key="rf_cfg_ssh_host")
    dtn_host = col2.text_input("DTN host (rsync)", value=cfg.dtn_host, key="rf_cfg_dtn_host")
    col1, col2 = st.columns(2)
    ssh_key_path = col1.text_input("Ruta SSH key", value=cfg.ssh_key_path, key="rf_cfg_ssh_key")
    control_path = col2.text_input("ControlPath", value=cfg.control_path, key="rf_cfg_control_path",
                                    help="Socket para ControlMaster. Debe coincidir con el de tu sesión manual.")

    st.subheader("Paths remotos")
    col1, col2 = st.columns(2)
    remote_repo = col1.text_input("Remote repo dir", value=cfg.remote_repo_dir, key="rf_cfg_remote_repo",
                                   placeholder=f"/home/{jhed}/Tesis")
    remote_data = col2.text_input("Remote data dir", value=cfg.remote_data_dir, key="rf_cfg_remote_data",
                                   placeholder=f"/data/{group}/{jhed}/Tesis")
    col1, col2 = st.columns(2)
    remote_scratch = col1.text_input("Remote scratch dir", value=cfg.remote_scratch_dir,
                                     key="rf_cfg_remote_scratch",
                                     placeholder=f"/scratch16/{group}/{jhed}/Tesis")
    remote_venv = col2.text_input("Remote venv path", value=cfg.remote_venv_path,
                                   key="rf_cfg_remote_venv",
                                   placeholder=f"/home/{jhed}/Tesis/venv_gpu")

    st.subheader("Módulos Lmod")
    col1, col2 = st.columns(2)
    python_module = col1.text_input("Python module", value=cfg.python_module, key="rf_cfg_python_module")
    cuda_module = col2.text_input("CUDA module", value=cfg.cuda_module, key="rf_cfg_cuda_module")

    st.subheader("Defaults para sbatch")
    col1, col2, col3 = st.columns(3)
    default_partition = col1.selectbox(
        "Partition", list(rf.GPU_PARTITIONS.keys()),
        index=list(rf.GPU_PARTITIONS.keys()).index(cfg.default_partition)
            if cfg.default_partition in rf.GPU_PARTITIONS else 1,
        key="rf_cfg_partition",
    )
    default_qos = col2.text_input("QoS", value=cfg.default_qos, key="rf_cfg_qos")
    default_time = col3.text_input("Walltime (HH:MM:SS)", value=cfg.default_time, key="rf_cfg_time")
    col1, col2, col3 = st.columns(3)
    default_gpus = col1.number_input("GPUs default", min_value=1, max_value=10,
                                      value=int(cfg.default_gpus), key="rf_cfg_gpus")
    default_cpus = col2.number_input("CPUs/task default", min_value=1, max_value=64,
                                      value=int(cfg.default_cpus), key="rf_cfg_cpus")
    default_mem = col3.number_input("RAM (GB) default", min_value=4, max_value=512,
                                     value=int(cfg.default_mem_gb), key="rf_cfg_mem")

    st.subheader("Notificaciones")
    col1, col2 = st.columns(2)
    notification_email = col1.text_input(
        "Email --mail-user",
        value=cfg.notification_email,
        key="rf_cfg_mail",
        placeholder=f"{jhed}@jh.edu",
    )
    notify_via_app = col2.checkbox(
        "También usar notification_system (SMTP propio)",
        value=cfg.notify_via_app,
        key="rf_cfg_notify_via_app",
        help="Si está activo, la pestaña Jobs te avisa al terminar usando email_config.json.",
    )

    new_cfg = replace(
        cfg,
        jhed=jhed.strip(),
        pi_account=pi_account.strip(),
        group=group.strip(),
        ssh_host=ssh_host.strip(),
        dtn_host=dtn_host.strip(),
        ssh_key_path=ssh_key_path.strip(),
        control_path=control_path.strip(),
        remote_repo_dir=remote_repo.strip(),
        remote_data_dir=remote_data.strip(),
        remote_scratch_dir=remote_scratch.strip(),
        remote_venv_path=remote_venv.strip(),
        python_module=python_module.strip(),
        cuda_module=cuda_module.strip(),
        default_partition=default_partition,
        default_qos=default_qos.strip(),
        default_time=default_time.strip(),
        default_gpus=int(default_gpus),
        default_cpus=int(default_cpus),
        default_mem_gb=int(default_mem),
        notification_email=notification_email.strip(),
        notify_via_app=bool(notify_via_app),
    )

    col_save, col_auto, col_reset = st.columns(3)
    if col_save.button("💾 Guardar config", type="primary", width="stretch", key="rf_cfg_save"):
        _save_config(new_cfg)
        st.success(f"Config guardada en {rf.CONFIG_FILE}")
    if col_auto.button("🔮 Autodetectar paths", width="stretch", key="rf_cfg_autodetect"):
        autodetected = new_cfg.autodetect_paths()
        _save_config(autodetected)
        st.success("Paths remotos completados con valores por defecto.")
        st.rerun()
    if col_reset.button("↺ Reset", width="stretch", key="rf_cfg_reset"):
        _save_config(rf.default_config())
        st.warning("Config reseteada a valores por defecto.")
        st.rerun()

    return new_cfg


# ---------------------------------------------------------------------------
# Tab 2 — Salud
# ---------------------------------------------------------------------------


def _tab_health(cfg: rf.RockfishConfig) -> None:
    st.subheader("Auditoría del entorno Rockfish")
    st.caption(
        "Valida SSH ControlMaster, módulos Lmod, quotas, acceso a GPU partitions y venv CUDA. "
        "El SSH debe estar inicializado previamente en una terminal:"
    )
    if cfg.jhed and cfg.control_path:
        st.code(
            f"ssh -fNM -o ControlMaster=auto -o ControlPath={cfg.control_path} -o ControlPersist=4h "
            f"{cfg.jhed}@{cfg.ssh_host}",
            language="bash",
        )
    else:
        st.warning("Configura JHED y ControlPath en la pestaña Configuración antes de auditar.")

    signature = _config_signature(cfg)
    refresh = st.button("🔍 Auditar Rockfish", type="primary", key="rf_audit")
    if refresh or SESSION_KEYS["snapshot"] not in st.session_state:
        with st.spinner("Ejecutando checks contra Rockfish..."):
            st.session_state[SESSION_KEYS["snapshot"]] = rf.collect_health_snapshot(cfg)
            st.session_state[SESSION_KEYS["signature"]] = signature

    snapshot: rf.RockfishHealthSnapshot | None = st.session_state.get(SESSION_KEYS["snapshot"])
    stale = st.session_state.get(SESSION_KEYS["signature"]) != signature
    if stale:
        st.warning("La config cambió desde la última auditoría. Vuelve a auditar antes de operar.")

    if snapshot is None:
        st.info("Aún no hay auditoría. Pulsa el botón para empezar.")
        return

    cols = st.columns(len(snapshot.summary))
    for col, check in zip(cols, snapshot.summary):
        with col:
            label, icon = _check_badge(check)
            st.metric(check.name, f"{icon} {label}")
            st.caption(check.detail[:120])

    blockers = rf.blocking_checks(snapshot.summary)
    if blockers:
        st.error("Hay bloqueantes:\n" + rf.checks_to_text(blockers))

    with st.expander("Módulos Lmod", expanded=False):
        if snapshot.modules:
            st.dataframe(_checks_to_df(snapshot.modules), width="stretch", hide_index=True)
        else:
            st.caption("Sin datos (SSH no disponible).")

    with st.expander("Filesystems / quotas", expanded=False):
        if snapshot.quotas:
            st.dataframe(_checks_to_df(snapshot.quotas), width="stretch", hide_index=True)
        else:
            st.caption("Sin datos (SSH no disponible).")

    st.caption(f"Última auditoría: {time.strftime('%H:%M:%S', time.localtime(snapshot.timestamp))}")


# ---------------------------------------------------------------------------
# Tab 3 — Sincronización
# ---------------------------------------------------------------------------


def _tab_sync(cfg: rf.RockfishConfig) -> None:
    st.subheader("Sincronización de código y datos")

    if not cfg.remote_repo_dir:
        st.warning("Configura `remote_repo_dir` en la pestaña Configuración.")
        return

    st.markdown("**1. Código (rsync push al DTN)**")
    st.caption(
        f"Local: `{ROOT_DIR}` → Remoto: `{cfg.dtn_target}:{cfg.remote_repo_dir}`. "
        "Excluye Datos/, Resultados/, cache/, venv*, .git/, docs/, papers/, simulación/, NLP/."
    )
    col1, col2 = st.columns(2)
    dry_run = col1.checkbox("Dry-run (solo listar)", value=True, key="rf_sync_dry")
    delete_remote = col2.checkbox("--delete (borrar archivos remotos faltantes)", value=False,
                                   key="rf_sync_delete")

    if st.button("🚀 Sincronizar código ahora", type="primary", key="rf_sync_push"):
        extras = ("--delete",) if delete_remote else ()
        with st.spinner("Ejecutando rsync..."):
            result = rf.rsync_push(cfg, ROOT_DIR, cfg.remote_repo_dir, dry_run=dry_run, extra_args=extras)
        _render_command_result("rsync push código", result, expanded=True)

    st.markdown("---")
    st.markdown("**2. Datos (~40 GB) — usar Globus**")
    st.caption("scp/rsync de 40 GB sobre login no es realista (timeout + caída). Globus reanuda.")
    st.markdown(
        "- Abre https://app.globus.org → SSO de JHU\n"
        f"- Colección remota: **Rockfish User Data** → carpeta `/data/{cfg.group or '<group>'}/{cfg.jhed or '<jhed>'}/Tesis/Datos/`\n"
        f"- Origen local: tu máquina → `{ROOT_DIR / 'Datos'}`\n"
        "- Pulsa **Start** y deja que termine en background"
    )

    st.markdown("---")
    st.markdown("**3. Bootstrap del venv CUDA (una vez)**")
    st.caption("Sube scripts/slurm/setup_rockfish_env.sh al repo remoto. Luego corre `bash scripts/slurm/setup_rockfish_env.sh` "
                "DENTRO de un job interactivo (no en login node).")
    if st.button("📤 Subir setup_rockfish_env.sh", key="rf_sync_push_setup"):
        with st.spinner("Subiendo..."):
            res = rf.push_setup_script(cfg)
        _render_command_result("push setup script", res, expanded=True)

    st.markdown("---")
    st.markdown("**4. Inspección remota**")
    if st.button("📊 Ver uso de disco remoto", key="rf_sync_du"):
        with st.spinner("Consultando..."):
            res = rf.ssh_run(
                cfg,
                f"du -sh {shlex.quote(cfg.remote_repo_dir)} {shlex.quote(cfg.remote_data_dir or '/dev/null')} "
                f"{shlex.quote(cfg.remote_scratch_dir or '/dev/null')} 2>/dev/null",
                timeout=30,
            )
        _render_command_result("du -sh", res, expanded=True)


# ---------------------------------------------------------------------------
# Tab 4 — Launcher
# ---------------------------------------------------------------------------


def _list_local_graphs() -> list[str]:
    if not RESULTADOS_DIR.exists():
        return []
    return sorted(str(p.relative_to(ROOT_DIR)) for p in RESULTADOS_DIR.glob("highway_graph_*.pt"))


def _list_local_hparams() -> list[str]:
    if not RESULTADOS_DIR.exists():
        return []
    return sorted(str(p.relative_to(ROOT_DIR)) for p in RESULTADOS_DIR.glob("optuna_hyperparams_*.csv"))


def _build_jobspec_form(cfg: rf.RockfishConfig) -> rf.JobSpec:
    st.markdown("**Parámetros de `run_gat_training`**")
    col1, col2 = st.columns(2)
    graphs = _list_local_graphs()
    hparams = _list_local_hparams()
    graph_choice = col1.selectbox("Grafo (.pt)", options=["<custom>"] + graphs, index=1 if graphs else 0,
                                   key="rf_graph_pick")
    if graph_choice == "<custom>":
        graph_path = col1.text_input("Ruta grafo", value=st.session_state.get("rf_graph_custom", ""), key="rf_graph_custom")
    else:
        graph_path = graph_choice
    hparams_choice = col2.selectbox("Hparams (.csv)", options=["<ninguno>"] + hparams, index=0, key="rf_hp_pick")
    hparams_path = "" if hparams_choice == "<ninguno>" else hparams_choice
    hparams_index = col2.number_input("Hparams index", min_value=0, value=0, step=1, key="rf_js_hp_index")

    col1, col2, col3 = st.columns(3)
    purpose = col1.text_input("Purpose tag", value="rockfish_run", key="rf_js_purpose")
    max_epochs = col2.number_input("max_epochs", min_value=1, value=50, step=1, key="rf_js_max_epochs")
    seed = col3.number_input("Seed", value=19091985, step=1, key="rf_js_seed")

    col1, col2, col3 = st.columns(3)
    early_stop = col1.checkbox("Early stop", value=True, key="rf_js_early_stop")
    early_stop_patience = col2.number_input("ES patience", min_value=1, value=8, key="rf_js_es_patience")
    early_stop_min_delta = col3.number_input("ES min delta", value=1e-6, format="%.1e",
                                              key="rf_js_es_min_delta")

    col1, col2, col3 = st.columns(3)
    accumulation_steps = col1.number_input("Grad accumulation steps", min_value=1, value=1,
                                            key="rf_js_accum")
    sampler_modes = ["neighbor", "cluster", "saint_node", "saint_edge", "saint_rw"]
    train_sampler_mode = col2.selectbox("Train sampler", sampler_modes, index=0, key="rf_js_sampler")
    smote_choice = col3.selectbox("GraphSMOTE", ["(auto)", "Forzar", "Deshabilitar"], index=0,
                                   key="rf_js_smote")
    force_smote: Optional[bool] = {"(auto)": None, "Forzar": True, "Deshabilitar": False}[smote_choice]

    extra = st.text_input("CLI extra (avanzado)", value="", key="rf_js_extra",
                          help="Flags adicionales pasados crudos a `python -m src.gnn_main`.")

    st.markdown("**Recursos SLURM**")
    col1, col2, col3, col4 = st.columns(4)
    partition = col1.selectbox("Partition", list(rf.GPU_PARTITIONS.keys()),
                               index=list(rf.GPU_PARTITIONS.keys()).index(cfg.default_partition)
                                     if cfg.default_partition in rf.GPU_PARTITIONS else 1,
                               key="rf_js_partition")
    gpus = col2.number_input("GPUs", min_value=1, max_value=10, value=int(cfg.default_gpus),
                              key="rf_js_gpus")
    cpus = col3.number_input("CPUs/task", min_value=1, max_value=64, value=int(cfg.default_cpus),
                              key="rf_js_cpus")
    mem_gb = col4.number_input("RAM (GB)", min_value=4, max_value=512, value=int(cfg.default_mem_gb),
                                key="rf_js_mem")
    col1, col2 = st.columns(2)
    time_limit = col1.text_input("Walltime (HH:MM:SS o D-HH:MM:SS)", value=cfg.default_time,
                                  key="rf_js_time")
    job_name = col2.text_input("Job name", value="gnn_exp", key="rf_js_name")

    return rf.JobSpec(
        job_name=job_name,
        partition=partition,
        qos=cfg.default_qos,
        gpus=int(gpus),
        cpus_per_task=int(cpus),
        mem_gb=int(mem_gb),
        time_limit=time_limit,
        graph_path=graph_path,
        hparams_path=hparams_path,
        hparams_index=int(hparams_index),
        purpose=purpose,
        max_epochs=int(max_epochs),
        early_stop=bool(early_stop),
        early_stop_patience=int(early_stop_patience),
        early_stop_min_delta=float(early_stop_min_delta),
        accumulation_steps=int(accumulation_steps),
        train_sampler_mode=train_sampler_mode,
        force_use_graphsmote=force_smote,
        seed=int(seed),
        extra_cli_args=extra,
    )


def _resolve_remote_graph_path(cfg: rf.RockfishConfig, local_relative: str) -> str:
    """Map a local path under Resultados/ to its expected remote counterpart on /data/."""
    p = Path(local_relative)
    try:
        rel = p.relative_to("Resultados") if p.is_absolute() is False and str(p).startswith("Resultados") else p
    except Exception:
        rel = p
    return f"{cfg.remote_data_dir.rstrip('/')}/{str(rel)}" if cfg.remote_data_dir else str(p)


def _spec_for_remote(cfg: rf.RockfishConfig, spec: rf.JobSpec) -> rf.JobSpec:
    """Rewrite local paths to remote paths in the JobSpec."""
    remote_graph = _resolve_remote_graph_path(cfg, spec.graph_path) if spec.graph_path else ""
    remote_hp = _resolve_remote_graph_path(cfg, spec.hparams_path) if spec.hparams_path else ""
    return replace(spec, graph_path=remote_graph, hparams_path=remote_hp)


def _tab_launcher(cfg: rf.RockfishConfig) -> None:
    st.subheader("Lanzador de entrenamiento GNN")
    backend = st.radio(
        "Backend de ejecución",
        options=["Rockfish (sbatch)", "Local (MPS/CUDA)"],
        index=0,
        horizontal=True,
        key="rf_launcher_backend",
    )
    spec = _build_jobspec_form(cfg)
    submit_log: list[str] = st.session_state.setdefault(SESSION_KEYS["submit_log"], [])

    if backend.startswith("Rockfish"):
        if not cfg.pi_account or not cfg.remote_repo_dir or not cfg.remote_venv_path:
            st.error("Faltan campos en Configuración: pi_account / remote_repo_dir / remote_venv_path.")
            return
        remote_spec = _spec_for_remote(cfg, spec)
        try:
            preview = rf.render_sbatch(remote_spec, cfg)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        with st.expander("Preview del sbatch generado", expanded=False):
            st.code(preview, language="bash")
        if st.button("📨 Enviar a Rockfish (sbatch)", type="primary", key="rf_launch_submit"):
            with st.spinner("Subiendo y enviando..."):
                upload = rf.upload_sbatch(cfg, preview)
                if not upload.ok:
                    _render_command_result("upload sbatch", upload, expanded=True)
                    return
                remote_path = upload.stdout.strip()
                submit = rf.submit_sbatch(cfg, remote_path)
            _render_command_result("submit sbatch", submit, expanded=True)
            jobid = rf.parse_submit_output(submit.stdout) if submit.ok else None
            if jobid:
                st.success(f"Job enviado con ID **{jobid}** ({remote_spec.job_name}).")
                st.session_state[SESSION_KEYS["last_jobid"]] = jobid
                submit_log.append(f"{time.strftime('%H:%M:%S')}  {jobid}  {remote_spec.job_name}")
            else:
                st.warning("No se pudo extraer JobID de la salida. Revisa el log.")
    else:
        # Local backend — fork python -m src.gnn_main against the local repo
        st.caption(f"Ejecuta `python -m src.gnn_main ...` en este Mac usando el venv actual. Útil para humo local con MPS o CUDA local.")
        if st.button("▶️ Correr en local", type="primary", key="rf_launch_local"):
            argv = [sys.executable, "-u"] + spec.to_python_cli()
            log_path = ROOT_DIR / "Resultados" / "rockfish_cluster" / f"local_run_{int(time.time())}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(ROOT_DIR),
                    stdout=open(log_path, "w", buffering=1),
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": str(ROOT_DIR)},
                )
                st.session_state[SESSION_KEYS["local_proc"]] = proc.pid
                st.success(f"Subproceso lanzado (PID {proc.pid}). Log: {log_path}")
            except Exception as exc:
                st.error(f"No se pudo lanzar: {exc}")
        local_pid = st.session_state.get(SESSION_KEYS["local_proc"])
        if local_pid:
            st.caption(f"PID activo: {local_pid}")

    if submit_log:
        st.markdown("**Historial de envíos (sesión)**")
        st.code("\n".join(submit_log[-15:]), language="text")


# ---------------------------------------------------------------------------
# Tab 5 — Jobs
# ---------------------------------------------------------------------------


def _tab_jobs(cfg: rf.RockfishConfig) -> None:
    st.subheader("Jobs en cola / corriendo")
    if not cfg.jhed:
        st.warning("Configura JHED primero.")
        return
    col1, col2 = st.columns([1, 1])
    refresh = col1.button("🔄 Refrescar squeue", type="primary", key="rf_jobs_refresh")
    auto = col2.checkbox("Auto-refresh 10s", value=False, key=SESSION_KEYS["auto_refresh_jobs"])
    if refresh or auto:
        with st.spinner("Consultando squeue..."):
            res, df = rf.list_jobs(cfg)
        if not res.ok:
            _render_command_result("squeue", res, expanded=True)
        elif df.empty:
            st.info("No hay jobs propios en cola.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)
            terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"}
            for _, row in df.iterrows():
                if row["State"] in terminal_states:
                    _notify_job_finished(cfg, row["JobID"], row["State"])
        st.caption(f"Última consulta: {time.strftime('%H:%M:%S')}")

    st.markdown("---")
    st.markdown("**Cancelar job**")
    col1, col2 = st.columns([2, 1])
    job_to_cancel = col1.text_input("JobID a cancelar",
                                     value=st.session_state.get(SESSION_KEYS["last_jobid"], ""),
                                     key="rf_jobs_cancel_id")
    if col2.button("❌ scancel", key="rf_jobs_cancel_btn"):
        if not job_to_cancel.strip():
            st.warning("Ingresa un JobID.")
        else:
            with st.spinner("Cancelando..."):
                res = rf.cancel_job(cfg, job_to_cancel.strip())
            _render_command_result("scancel", res, expanded=True)

    if auto:
        time.sleep(10)
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 6 — Logs & métricas
# ---------------------------------------------------------------------------


def _tab_logs(cfg: rf.RockfishConfig) -> None:
    st.subheader("Logs y métricas")
    col1, col2, col3 = st.columns([2, 1, 1])
    job_id = col1.text_input("JobID",
                              value=st.session_state.get(SESSION_KEYS["last_jobid"], ""),
                              key="rf_logs_jobid")
    kind = col2.selectbox("Stream", ["stdout", "stderr"], index=0, key="rf_logs_kind")
    lines = col3.number_input("Líneas", min_value=50, max_value=5000, value=500, step=50,
                               key="rf_logs_lines")
    auto = st.checkbox("Auto-refresh 10s", value=False, key=SESSION_KEYS["auto_refresh_logs"])

    col_log, col_stats, col_pull = st.columns(3)
    show_log = col_log.button("📜 Tail log", type="primary", key="rf_logs_tail")
    show_stats = col_stats.button("📈 jobstats", key="rf_logs_stats")
    pull_results = col_pull.button("⬇️ rsync pull Resultados/", key="rf_logs_pull")

    if show_log or auto:
        if not job_id.strip():
            st.warning("Ingresa un JobID.")
        else:
            with st.spinner("Leyendo log..."):
                res = rf.tail_log(cfg, job_id.strip(), kind="err" if kind == "stderr" else "out", lines=int(lines))
            if res.ok:
                st.code(res.stdout or "(log vacío o archivo no encontrado)", language="text")
            else:
                _render_command_result("tail log", res, expanded=True)

    if show_stats:
        if not job_id.strip():
            st.warning("Ingresa un JobID.")
        else:
            with st.spinner("Calculando jobstats..."):
                res = rf.job_stats(cfg, job_id.strip())
            _render_command_result("jobstats", res, expanded=True)

    if pull_results:
        if not cfg.remote_data_dir:
            st.warning("Configura remote_data_dir primero.")
        else:
            local_dest = ROOT_DIR / "Resultados" / "rockfish_pull"
            with st.spinner("Bajando Resultados/ desde Rockfish..."):
                res = rf.rsync_pull(cfg, f"{cfg.remote_data_dir.rstrip('/')}/Resultados",
                                     local_dest, timeout=1800)
            _render_command_result("rsync pull", res, expanded=True)
            if res.ok:
                st.success(f"Resultados sincronizados en {local_dest}")

    if auto:
        time.sleep(10)
        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(set_page_config: bool = False, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="Rockfish Cluster", layout="wide")
    st.title("Rockfish Cluster")
    st.caption("Administra el cluster JHU Rockfish (SSH + sbatch) desde la app de tesis.")

    cfg = _get_config()
    tabs = st.tabs([
        "⚙️ Configuración",
        "🩺 Salud & Conexión",
        "🔄 Sincronización",
        "🚀 Launcher",
        "🧾 Jobs",
        "📋 Logs & Métricas",
    ])
    with tabs[0]:
        new_cfg = _tab_config(cfg)
        if _config_signature(new_cfg) != _config_signature(cfg):
            st.session_state[SESSION_KEYS["config"]] = new_cfg
    with tabs[1]:
        _tab_health(_get_config())
    with tabs[2]:
        _tab_sync(_get_config())
    with tabs[3]:
        _tab_launcher(_get_config())
    with tabs[4]:
        _tab_jobs(_get_config())
    with tabs[5]:
        _tab_logs(_get_config())

    if show_exit_button and st.button("Cerrar", key="rf_exit"):
        raise SystemExit(0)


if __name__ == "__main__":
    main(set_page_config=True)
