#!/usr/bin/env python3
"""
Streamlit page — TRC umbrella paper pipeline manager.

Surfaces every step of `src/trc_paper/` from the main app so the user can
launch, monitor, and inspect runs without leaving the UI. Background runs are
controlled via subprocess.Popen plus PID files persisted under
Resultados/trc_paper/logs/.

Manuscript material lives in papers/dynamic_clusters_trc/ and is not edited
from this page.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT_DIR / "src" / "trc_paper"
CONFIG_DEFAULT = PACKAGE_DIR / "config" / "default.yaml"
CONFIG_K8 = PACKAGE_DIR / "config" / "k8_sensitivity.yaml"
RESULTS_ROOT = ROOT_DIR / "Resultados" / "trc_paper"
RESULTS_ROOT_K8 = ROOT_DIR / "Resultados" / "trc_paper_k8"
LOGS_ROOT = RESULTS_ROOT / "logs"
PID_DIR = LOGS_ROOT / "pids"
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
PYTHON_BIN = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
MANUSCRIPT_DIR = ROOT_DIR / "papers" / "dynamic_clusters_trc"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def _run_tag(cfg: Dict[str, Any]) -> str:
    gmm = cfg.get("dynamic_gmm", {})
    return f"k{gmm.get('k', 5)}_{gmm.get('date_start', '????')}_{gmm.get('date_end', '????')}"


def _results_root_for(cfg: Dict[str, Any]) -> Path:
    rel = cfg.get("paths", {}).get("results_root", "Resultados/trc_paper")
    return (ROOT_DIR / rel).resolve()


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


@dataclass
class JobStatus:
    name: str
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    log_path: Optional[Path] = None
    cmd: Optional[List[str]] = None

    @property
    def running(self) -> bool:
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def _pid_file(job_name: str) -> Path:
    PID_DIR.mkdir(parents=True, exist_ok=True)
    return PID_DIR / f"{job_name}.json"


def _persist_job(status: JobStatus) -> None:
    payload = {
        "name": status.name,
        "pid": status.pid,
        "started_at": status.started_at.isoformat() if status.started_at else None,
        "finished_at": status.finished_at.isoformat() if status.finished_at else None,
        "exit_code": status.exit_code,
        "log_path": str(status.log_path) if status.log_path else None,
        "cmd": status.cmd,
    }
    _pid_file(status.name).write_text(json.dumps(payload, indent=2))


def _load_job(job_name: str) -> Optional[JobStatus]:
    p = _pid_file(job_name)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    s = JobStatus(
        name=d.get("name", job_name),
        pid=d.get("pid"),
        started_at=datetime.fromisoformat(d["started_at"]) if d.get("started_at") else None,
        finished_at=datetime.fromisoformat(d["finished_at"]) if d.get("finished_at") else None,
        exit_code=d.get("exit_code"),
        log_path=Path(d["log_path"]) if d.get("log_path") else None,
        cmd=d.get("cmd"),
    )
    if not s.running and s.exit_code is None and s.log_path and s.log_path.exists():
        # Process ended without us seeing it — surface as unknown exit
        s.exit_code = -1
        s.finished_at = datetime.fromtimestamp(s.log_path.stat().st_mtime)
        _persist_job(s)
    return s


def _launch_job(name: str, cmd: List[str], log_path: Path) -> JobStatus:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        start_new_session=True,
    )
    status = JobStatus(
        name=name,
        pid=proc.pid,
        started_at=datetime.now(),
        log_path=log_path,
        cmd=cmd,
    )
    _persist_job(status)
    return status


def _cancel_job(status: JobStatus) -> None:
    if status.pid is None:
        return
    try:
        os.killpg(status.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        st.warning(f"No tengo permisos para detener PID {status.pid}.")
    time.sleep(1)
    if status.running:
        try:
            os.killpg(status.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# Pipeline definition (single source of truth)
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    key: str
    title: str
    description: str
    script: str           # filename inside src/trc_paper/
    output_template: str  # path relative to results_root, with {run_tag}
    inputs: List[str] = field(default_factory=list)
    long_running: bool = False
    builder: Optional[Callable[[Dict[str, Any], Dict[str, Path], Path, str], List[str]]] = None


def _abs_input(cfg: Dict[str, Any], key: str) -> Path:
    return ROOT_DIR / cfg["paths"][key]


def _build_validate_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    gmm = cfg["dynamic_gmm"]
    results = _results_root_for(cfg)
    out = results / "validation" / f"{run_tag}_validation.json"
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "validate_data.py"),
        "--flow-db", str(_abs_input(cfg, "flow_db")),
        "--porticos-csv", str(_abs_input(cfg, "porticos_csv")),
        "--events-db", str(_abs_input(cfg, "events_db")),
        "--date-start", gmm["date_start"],
        "--date-end", gmm["date_end"],
        "--output", str(out),
    ]


def _build_dynamic_gmm_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    out_db = results / "dynamic_gmm" / f"{run_tag}_assignments.duckdb"
    out_model = results / "dynamic_gmm" / f"{run_tag}_model.joblib"
    out_meta = results / "dynamic_gmm" / f"{run_tag}_run.json"
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "run_dynamic_gmm.py"),
        "--config", str(CONFIG_DEFAULT if cfg["dynamic_gmm"]["k"] == 5 else CONFIG_K8),
        "--k", str(cfg["dynamic_gmm"]["k"]),
        "--output-db", str(out_db),
        "--output-model", str(out_model),
        "--output-metadata", str(out_meta),
        "--parallel-jobs", str(cfg["dynamic_gmm"].get("parallel_jobs", 4)),
    ]


def _build_entropy_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    assignments = results / "dynamic_gmm" / f"{run_tag}_assignments.duckdb"
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "compute_entropy.py"),
        "--assignments-db", str(assignments),
        "--flow-db", str(_abs_input(cfg, "flow_db")),
        "--output-15min", str(results / "entropy" / f"{run_tag}_H_portico_15min.parquet"),
        "--output-5min", str(results / "entropy" / f"{run_tag}_H_portico_5min.parquet"),
        "--output-60min", str(results / "entropy" / f"{run_tag}_H_portico_60min.parquet"),
        "--output-summary", str(results / "entropy" / f"{run_tag}_H_summary.json"),
    ]


def _build_markov_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "markov_matrix.py"),
        "--assignments-db", str(results / "dynamic_gmm" / f"{run_tag}_assignments.duckdb"),
        "--step", cfg["markov"]["step"],
        "--subpopulation", cfg["markov"]["subpopulation"],
        "--bootstrap-replicas", str(cfg["markov"]["bootstrap"]["n_replicas"]),
        "--output-global", str(results / "markov" / f"{run_tag}_P_global.parquet"),
        "--output-bootstrap", str(results / "markov" / f"{run_tag}_P_bootstrap.parquet"),
        "--output-summary", str(results / "markov" / f"{run_tag}_P_summary.json"),
    ]


def _build_homogeneity_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "homogeneity_test.py"),
        "--assignments-db", str(results / "dynamic_gmm" / f"{run_tag}_assignments.duckdb"),
        "--p-global", str(results / "markov" / f"{run_tag}_P_global.parquet"),
        "--output-result", str(results / "markov" / f"{run_tag}_homogeneity.json"),
        "--output-p-per-split", str(results / "markov" / f"{run_tag}_P_per_split.parquet"),
        "--config", str(CONFIG_DEFAULT),
    ]


def _build_stationary_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "stationary_asymmetry.py"),
        "--p-global", str(results / "markov" / f"{run_tag}_P_global.parquet"),
        "--output-result", str(results / "markov" / f"{run_tag}_stationary.json"),
        "--output-pairs", str(results / "markov" / f"{run_tag}_asymmetry_pairs.parquet"),
    ]


def _build_covid_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "covid_decomposition.py"),
        "--p-per-split", str(results / "markov" / f"{run_tag}_P_per_split.parquet"),
        "--h-15min", str(results / "entropy" / f"{run_tag}_H_portico_15min.parquet"),
        "--output-result", str(results / "covid" / f"{run_tag}_decomposition.json"),
        "--output-timeline", str(results / "covid" / f"{run_tag}_share_timeline.parquet"),
        "--config", str(CONFIG_DEFAULT),
    ]


def _build_event_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "event_matching.py"),
        "--h-15min", str(results / "entropy" / f"{run_tag}_H_portico_15min.parquet"),
        "--events-db", str(_abs_input(cfg, "events_db")),
        "--porticos-csv", str(_abs_input(cfg, "porticos_csv")),
        "--output-matched", str(results / "events" / f"{run_tag}_matched_pairs.parquet"),
        "--output-summary", str(results / "events" / f"{run_tag}_event_summary.json"),
        "--config", str(CONFIG_DEFAULT),
    ]


def _build_integration_cmd(cfg, _outputs, log, run_tag) -> List[str]:
    results = _results_root_for(cfg)
    return [
        PYTHON_BIN,
        str(PACKAGE_DIR / "integration_h_bound.py"),
        "--stationary", str(results / "markov" / f"{run_tag}_stationary.json"),
        "--h-15min", str(results / "entropy" / f"{run_tag}_H_portico_15min.parquet"),
        "--homogeneity", str(results / "markov" / f"{run_tag}_homogeneity.json"),
        "--output-result", str(results / "integration" / f"{run_tag}_h_bound.json"),
        "--output-crosstab", str(results / "integration" / f"{run_tag}_crosstab.parquet"),
    ]


PIPELINE: List[PipelineStep] = [
    PipelineStep(
        key="validate",
        title="0 · Validate data",
        description=(
            "Verifica que flujos.duckdb, Porticos.csv y eventos.duckdb estén "
            "presentes, cubran 2018-2024 y mapeen consistentemente. "
            "Salida: JSON con `ready_for_phase_1=true/false`."
        ),
        script="validate_data.py",
        output_template="validation/{run_tag}_validation.json",
        inputs=["flow_db", "porticos_csv", "events_db"],
        builder=_build_validate_cmd,
    ),
    PipelineStep(
        key="dynamic_gmm",
        title="1 · Dynamic GMM",
        description=(
            "Regenera dynamic_assignments y dynamic_window_summary con la GMM "
            "deslizante de 7 días sobre todo el rango configurado. ESTA ES LA "
            "ETAPA CARA (~30-50h en CPU). Checkpoint automático."
        ),
        script="run_dynamic_gmm.py",
        output_template="dynamic_gmm/{run_tag}_assignments.duckdb",
        inputs=["validate"],
        long_running=True,
        builder=_build_dynamic_gmm_cmd,
    ),
    PipelineStep(
        key="entropy",
        title="2 · Macroscopic entropy H_{p,τ}",
        description=(
            "Calcula H_{p,τ} sobre soft membership por pórtico × bucket "
            "(5/15/60 min). Une asignaciones con detecciones vía ASOF JOIN."
        ),
        script="compute_entropy.py",
        output_template="entropy/{run_tag}_H_portico_15min.parquet",
        inputs=["dynamic_gmm"],
        builder=_build_entropy_cmd,
    ),
    PipelineStep(
        key="markov",
        title="3 · Markov matrix P_{ij}",
        description=(
            "Estima la matriz de transición soft P_{ij} a paso 1 semana, con "
            "bootstrap por patente (1000 réplicas por defecto)."
        ),
        script="markov_matrix.py",
        output_template="markov/{run_tag}_P_global.parquet",
        inputs=["dynamic_gmm"],
        long_running=True,
        builder=_build_markov_cmd,
    ),
    PipelineStep(
        key="homogeneity",
        title="4 · Homogeneity test",
        description=(
            "Test χ² robusto (Leskelä 2026) y distancias TV de la matriz "
            "Markov por split temporal (años + fases COVID)."
        ),
        script="homogeneity_test.py",
        output_template="markov/{run_tag}_homogeneity.json",
        inputs=["dynamic_gmm", "markov"],
        builder=_build_homogeneity_cmd,
    ),
    PipelineStep(
        key="stationary",
        title="5 · Stationary π & Kolmogorov asymmetry",
        description=(
            "Distribución estacionaria π, tiempo de mezcla, y test de "
            "balance detallado (Kolmogorov) para detectar irreversibilidad."
        ),
        script="stationary_asymmetry.py",
        output_template="markov/{run_tag}_stationary.json",
        inputs=["markov"],
        builder=_build_stationary_cmd,
    ),
    PipelineStep(
        key="covid",
        title="6 · COVID decomposition",
        description=(
            "Decomposición pre/lockdown/paso a paso/post-COVID. π por fase, "
            "TV y KL vs baseline pre-COVID. Timeline semanal de shares."
        ),
        script="covid_decomposition.py",
        output_template="covid/{run_tag}_decomposition.json",
        inputs=["homogeneity", "entropy"],
        builder=_build_covid_cmd,
    ),
    PipelineStep(
        key="events",
        title="7 · Event matching",
        description=(
            "Matching de H_{p,τ} contra accidentes + averías + objetos en "
            "calzada (4.242 eventos). Wilcoxon pre/post, Cohen's d."
        ),
        script="event_matching.py",
        output_template="events/{run_tag}_matched_pairs.parquet",
        inputs=["entropy"],
        builder=_build_event_cmd,
    ),
    PipelineStep(
        key="integration",
        title="8 · H-bound integration",
        description=(
            "Verificación empírica del teorema H-bound H(π) ≤ Ē[H] + R. "
            "Tabla cruzada espacial."
        ),
        script="integration_h_bound.py",
        output_template="integration/{run_tag}_h_bound.json",
        inputs=["stationary", "entropy", "homogeneity"],
        builder=_build_integration_cmd,
    ),
]


# ---------------------------------------------------------------------------
# UI: helpers
# ---------------------------------------------------------------------------


def _format_age(p: Path) -> str:
    if not p.exists():
        return "—"
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    delta = datetime.now() - mtime
    if delta.days >= 1:
        return f"hace {delta.days} d"
    hours = delta.seconds // 3600
    if hours >= 1:
        return f"hace {hours} h"
    minutes = delta.seconds // 60
    return f"hace {minutes} min" if minutes else "ahora"


def _status_icon(running: bool, output_exists: bool, exit_code: Optional[int]) -> str:
    if running:
        return ":material/hourglass:"
    if exit_code is not None and exit_code != 0:
        return ":material/error:"
    if output_exists:
        return ":material/check_circle:"
    return ":material/pending:"


def _tail_log(path: Path, n_lines: int = 200) -> str:
    if not path.exists():
        return ""
    try:
        # Read just last ~64KB to avoid loading huge logs
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > 65536:
                fh.seek(-65536, os.SEEK_END)
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"(error reading log: {exc})"
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def _output_path(cfg: Dict[str, Any], step: PipelineStep) -> Path:
    results = _results_root_for(cfg)
    return results / step.output_template.format(run_tag=_run_tag(cfg))


def _render_step_card(step: PipelineStep, cfg: Dict[str, Any]) -> None:
    run_tag = _run_tag(cfg)
    job_name = f"{step.key}_{run_tag}"
    job = _load_job(job_name)
    output = _output_path(cfg, step)
    output_exists = output.exists()
    running = bool(job and job.running)
    exit_code = job.exit_code if job else None

    icon = _status_icon(running, output_exists, exit_code)
    header_cols = st.columns([0.07, 0.55, 0.38])
    header_cols[0].markdown(f"### {icon}")
    header_cols[1].markdown(f"**{step.title}**")
    header_cols[2].caption(
        f"Output: {output.relative_to(ROOT_DIR) if output.exists() else '(no generado)'}"
        f"  ·  Edad: {_format_age(output)}"
    )

    with st.expander("Detalle", expanded=False):
        st.markdown(step.description)

        # Action row
        act_cols = st.columns([0.25, 0.25, 0.5])
        launch_label = "Re-lanzar" if running else "Lanzar"
        if running:
            launch_label = "Corriendo…"
        launch_disabled = running or step.builder is None
        if act_cols[0].button(launch_label, key=f"launch_{job_name}", disabled=launch_disabled):
            _start_step(step, cfg, job_name)
            st.rerun()
        if running and act_cols[1].button("Cancelar", key=f"cancel_{job_name}"):
            _cancel_job(job)
            st.warning(f"Señal SIGTERM enviada a PID {job.pid}.")
            st.rerun()

        if job:
            meta_cols = st.columns(4)
            meta_cols[0].caption(f"PID: {job.pid or '—'}")
            meta_cols[1].caption(
                f"Inicio: {job.started_at.strftime('%H:%M:%S') if job.started_at else '—'}"
            )
            meta_cols[2].caption(
                f"Fin: {job.finished_at.strftime('%H:%M:%S') if job.finished_at else '—'}"
            )
            meta_cols[3].caption(f"Exit: {exit_code if exit_code is not None else '—'}")

            log_path = job.log_path
            if log_path and log_path.exists():
                st.caption(f"Log: `{log_path.relative_to(ROOT_DIR)}`")
                log_text = _tail_log(log_path, n_lines=80)
                if log_text:
                    st.code(log_text, language=None)

        # Output preview
        if output_exists:
            _preview_output(output)


def _start_step(step: PipelineStep, cfg: Dict[str, Any], job_name: str) -> None:
    if step.builder is None:
        st.error("Este paso no tiene builder definido.")
        return
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_ROOT / f"{job_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    cmd = step.builder(cfg, {}, log_path, _run_tag(cfg))
    try:
        _launch_job(job_name, cmd, log_path)
        st.success(f"Lanzado: `{' '.join(cmd[:3])}…`  →  log: {log_path.name}")
    except FileNotFoundError as exc:
        st.error(f"No se pudo lanzar: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error lanzando proceso: {exc}")


def _preview_output(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            st.error(f"JSON inválido: {exc}")
            return
        st.json(payload, expanded=False)
    elif suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Parquet inválido: {exc}")
            return
        st.caption(f"Filas: {len(df):,} · Columnas: {len(df.columns)}")
        st.dataframe(df.head(50), use_container_width=True, height=240)
    elif suffix == ".duckdb":
        st.caption(
            f"DuckDB de {path.stat().st_size / 1e6:.1f} MB · usar Files o "
            "scripts dedicados para inspeccionar."
        )
    else:
        st.caption(f"({suffix or 'sin extensión'} — sin preview específico)")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _tab_overview(cfg: Dict[str, Any]) -> None:
    st.subheader("Vista general del pipeline TRC")
    st.caption(
        "Manuscript en `papers/dynamic_clusters_trc/`. Resultados en "
        f"`Resultados/trc_paper{'/' if cfg['dynamic_gmm']['k']==5 else '_k8/'}`."
    )

    rows = []
    for step in PIPELINE:
        out = _output_path(cfg, step)
        job = _load_job(f"{step.key}_{_run_tag(cfg)}")
        rows.append({
            "Paso": step.title,
            "Estado": (
                "corriendo" if (job and job.running)
                else "ok" if out.exists()
                else "pendiente"
            ),
            "Output": str(out.relative_to(ROOT_DIR)) if out.exists() else "—",
            "Edad output": _format_age(out),
            "Tamaño (MB)": f"{out.stat().st_size / 1e6:.1f}" if out.exists() else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Cómo usar esta página:**\n"
        "1. Verificá la pestaña **Config** y ajustá K (5 o 8), rango y otros parámetros.\n"
        "2. Lanzá `Validate data` y esperá `ready_for_phase_1=true`.\n"
        "3. Lanzá `Dynamic GMM` (etapa cara, ~30-50h). Se ejecuta en background con checkpoint.\n"
        "4. El resto del pipeline corre en minutos-horas. La página detecta sus outputs automáticamente."
    )


def _tab_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    st.subheader("Configuración del pipeline")
    st.caption(
        "Los cambios se persisten en `src/trc_paper/config/default.yaml`. "
        "K=8 sensibilidad usa `k8_sensitivity.yaml` con results_root separado."
    )

    gmm_cfg = cfg.setdefault("dynamic_gmm", {})
    paths_cfg = cfg.setdefault("paths", {})

    c1, c2, c3 = st.columns(3)
    new_k = c1.selectbox(
        "K (clusters)", options=[5, 8],
        index=[5, 8].index(int(gmm_cfg.get("k", 5))),
        help="K=5 es la configuración principal del paper. K=8 = análisis de sensibilidad.",
    )
    new_start = c2.text_input("date_start", value=gmm_cfg.get("date_start", "2018-01-01"))
    new_end = c3.text_input("date_end", value=gmm_cfg.get("date_end", "2024-09-30"))

    c4, c5, c6 = st.columns(3)
    new_window = c4.number_input(
        "window_days", min_value=1, max_value=30,
        value=int(gmm_cfg.get("window_days", 7)),
    )
    new_parallel = c5.number_input(
        "parallel_jobs", min_value=1, max_value=16,
        value=int(gmm_cfg.get("parallel_jobs", 4)),
    )
    new_step = c6.selectbox(
        "Markov step", options=["1D", "1W", "1M"],
        index=["1D", "1W", "1M"].index(cfg.get("markov", {}).get("step", "1W")),
    )

    if st.button("Guardar configuración", type="primary"):
        gmm_cfg["k"] = int(new_k)
        gmm_cfg["date_start"] = str(new_start)
        gmm_cfg["date_end"] = str(new_end)
        gmm_cfg["window_days"] = int(new_window)
        gmm_cfg["parallel_jobs"] = int(new_parallel)
        cfg.setdefault("markov", {})["step"] = new_step

        target = CONFIG_DEFAULT if int(new_k) == 5 else CONFIG_K8
        _save_yaml(target, cfg)
        st.success(f"Guardado en {target.relative_to(ROOT_DIR)}.")

    st.markdown("**Paths:**")
    cols = st.columns(3)
    cols[0].caption(f"flow_db: `{paths_cfg.get('flow_db')}`")
    cols[1].caption(f"porticos_csv: `{paths_cfg.get('porticos_csv')}`")
    cols[2].caption(f"events_db: `{paths_cfg.get('events_db')}`")
    cols2 = st.columns(2)
    cols2[0].caption(f"results_root: `{paths_cfg.get('results_root')}`")
    cols2[1].caption(f"logs_root: `{paths_cfg.get('logs_root')}`")

    return cfg


def _tab_run(cfg: Dict[str, Any]) -> None:
    st.subheader("Ejecución por etapa")
    st.caption(
        "Cada etapa puede lanzarse independientemente. Las dependencias se "
        "muestran en el `Detalle` y se verifican al lanzar (output del paso "
        "previo debe existir)."
    )
    for step in PIPELINE:
        _render_step_card(step, cfg)


def _tab_results(cfg: Dict[str, Any]) -> None:
    st.subheader("Artefactos generados")
    results = _results_root_for(cfg)
    if not results.exists():
        st.info(f"Aún no se generaron artefactos en `{results.relative_to(ROOT_DIR)}`.")
        return
    rows = []
    for p in sorted(results.rglob("*")):
        if p.is_dir():
            continue
        if "pids" in p.parts or p.name.startswith(".") or p.suffix == ".tmp":
            continue
        rows.append({
            "Path": str(p.relative_to(ROOT_DIR)),
            "Tipo": p.suffix.lstrip(".") or "(sin ext)",
            "Tamaño (MB)": round(p.stat().st_size / 1e6, 3),
            "Modificado": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    if not rows:
        st.info("Sin artefactos todavía.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=480)


def _tab_logs(cfg: Dict[str, Any]) -> None:
    st.subheader("Logs de ejecución")
    if not LOGS_ROOT.exists():
        st.info(f"No hay logs aún ({LOGS_ROOT.relative_to(ROOT_DIR)}).")
        return
    logs = sorted(LOGS_ROOT.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        st.info("Sin archivos `.log`.")
        return
    options = {f"{p.name}  ({_format_age(p)})": p for p in logs}
    selection = st.selectbox("Archivo de log", list(options.keys()))
    if selection:
        path = options[selection]
        st.caption(f"Path: `{path.relative_to(ROOT_DIR)}` · {path.stat().st_size/1024:.1f} KB")
        n_lines = st.slider("Líneas a mostrar", 50, 1000, 200, step=50)
        text = _tail_log(path, n_lines=n_lines)
        st.code(text or "(log vacío)", language=None)
        if st.button("Auto-refresh cada 5s"):
            st.session_state["trc_logs_autorefresh"] = True
        if st.session_state.get("trc_logs_autorefresh"):
            time.sleep(5)
            st.rerun()


def _tab_manuscript() -> None:
    st.subheader("Manuscrito (no se edita desde acá)")
    st.caption("Material del paper bajo `papers/dynamic_clusters_trc/`.")
    if not MANUSCRIPT_DIR.exists():
        st.error(f"No existe la carpeta del manuscrito: {MANUSCRIPT_DIR.relative_to(ROOT_DIR)}.")
        return
    rows = []
    for p in sorted(MANUSCRIPT_DIR.rglob("*")):
        if p.is_dir():
            continue
        rows.append({
            "Path": str(p.relative_to(ROOT_DIR)),
            "Tipo": p.suffix.lstrip(".") or "(sin ext)",
            "Tamaño (KB)": round(p.stat().st_size / 1024, 1),
            "Modificado": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)
    st.markdown(
        "Para editar el manuscrito usá la pestaña **LaTeX** del menú principal "
        "(o un editor externo). El roadmap del paper está en "
        f"`{(MANUSCRIPT_DIR / 'manuscript' / 'timeline.md').relative_to(ROOT_DIR)}`."
    )


# ---------------------------------------------------------------------------
# Streamlit entrypoint
# ---------------------------------------------------------------------------


def main(set_page_config: bool = True, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="TRC Paper Pipeline", layout="wide", page_icon=":material/science:")
    st.title("TRC Paper · Dynamic Latent-Class Pipeline")
    st.caption(
        "Pipeline para el manuscript umbrella en *Transportation Research Part C*. "
        "Implementación en `src/trc_paper/`, resultados en `Resultados/trc_paper/`."
    )

    cfg = _load_yaml(CONFIG_DEFAULT)
    if not cfg:
        st.error(f"No pude leer `{CONFIG_DEFAULT.relative_to(ROOT_DIR)}`.")
        return

    # Variant switcher (K=5 default, K=8 sensitivity)
    variant = st.sidebar.radio(
        "Variante",
        options=["K=5 (principal)", "K=8 (sensibilidad)"],
        horizontal=False,
        key="trc_variant",
    )
    if variant.startswith("K=8"):
        cfg = _load_yaml(CONFIG_K8) or cfg
        # Force results root to k8 even if K8 yaml only overrides selected fields
        cfg.setdefault("dynamic_gmm", {})["k"] = 8
        cfg.setdefault("paths", {})["results_root"] = "Resultados/trc_paper_k8"
        cfg["paths"]["logs_root"] = "Resultados/trc_paper_k8/logs"
    else:
        cfg.setdefault("dynamic_gmm", {})["k"] = 5

    tabs = st.tabs([
        "Overview", "Config", "Run", "Results", "Logs", "Manuscript",
    ])
    with tabs[0]:
        _tab_overview(cfg)
    with tabs[1]:
        cfg = _tab_config(cfg)
    with tabs[2]:
        _tab_run(cfg)
    with tabs[3]:
        _tab_results(cfg)
    with tabs[4]:
        _tab_logs(cfg)
    with tabs[5]:
        _tab_manuscript()

    if show_exit_button:
        if st.sidebar.button("Cerrar app"):
            os._exit(0)


if __name__ == "__main__":
    main(set_page_config=True, show_exit_button=True)
