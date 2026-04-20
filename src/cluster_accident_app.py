#!/usr/bin/env python3
"""
Streamlit app to evaluate accident prediction with and without cluster variables.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import (
    DEFAULT_INTERVAL_MINUTES,
    FlowSampleSelection,
    add_accident_target,
    compute_cluster_features,
    compute_flow_features,
    find_candidate_porticos,
    get_flow_db_summary,
    load_flujos,
    load_flujos_range,
    load_porticos,
    process_accidentes_df,
    get_portico_segments,
    _slugify,
    read_csv_with_progress,
)
from src.model_training import (
    BALANCE_STRATEGY_LABELS,
    CALIBRATION_METHOD_LABELS,
    OPTUNA_OBJECTIVE_LABELS,
    THRESHOLD_OBJECTIVE_LABELS,
    THRESHOLD_PROTOCOL_LABELS,
    build_model as _build_model,
    fit_score_calibrator as _fit_score_calibrator,
    get_model_scores as _get_model_scores,
    normalize_calibration_method as _normalize_calibration_method,
    normalize_optuna_objective_metric as _normalize_optuna_objective_metric,
    normalize_threshold_objective,
    optuna_objective_direction as _optuna_objective_direction,
    score_optuna_objective as _score_optuna_objective,
    temporal_train_test_split as _temporal_train_test_split,
    split_train_val_for_threshold as _split_train_val_for_threshold,
    train_model as _train_model,
    train_model_on_split as _train_model_on_split,
)
from src.model_xai import compute_xai_report, save_xai_bundle
from src.experiments_logic import (
    CALIBRATION_SWEEP_BALANCE_MODES,
    CALIBRATION_SWEEP_MULTIOBJECTIVE_DIRECTIONS,
    CALIBRATION_SWEEP_MULTIOBJECTIVE_KEY,
    CALIBRATION_SWEEP_MULTIOBJECTIVE_LABEL,
    CALIBRATION_SWEEP_MULTIOBJECTIVE_METRICS,
    CALIBRATION_SWEEP_MULTIOBJECTIVE_PROTOCOL_VERSION,
    CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
    CALIBRATION_SWEEP_OBJECTIVE_MODE_SCALAR,
    CALIBRATION_SWEEP_PROTOCOL_FAMILY,
    CALIBRATION_SWEEP_PROTOCOL_VERSION,
    CONTROLLED_COMPARISON_MODELS,
    FROZEN_TUNING_ABLATION_CONFIG,
    FROZEN_TUNING_ABLATION_FEATURE_SETS,
    FROZEN_TUNING_ABLATION_PROTOCOL_FAMILY,
    ExperimentsRunner,
    _calibration_multiobjective_far_gate,
    _calibration_multiobjective_pruning_proxy_from_metrics,
    _calibration_multiobjective_trials_dataframe,
    _calibration_multiobjective_values_from_metrics,
    _select_calibration_multiobjective_trial,
    _controlled_comparison_paths,
    _k_grid_values,
    build_controlled_comparison_context,
    estimate_controlled_comparison_parallelism,
    preview_controlled_comparison_checkpoint,
)

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import psutil
except ImportError:
    psutil = None

RESULTS_DIR = ROOT_DIR / "Resultados"
DATA_DIR = ROOT_DIR / "Datos"
HISTORY_PATH = RESULTS_DIR / "experiment_history.jsonl"
MODELS_DIR = RESULTS_DIR / "model_history"
CLUSTER_LABEL_PATTERN = re.compile(
    r"^cluster_(?P<method>kmeans|gmm|hdbscan)(?:_k(?P<k>\d+))?(?:.*)?\.csv$"
)
XAI_GROUP_COLOR_DOMAIN = ["Base", "Cluster"]
XAI_GROUP_COLOR_RANGE = ["#2f6c7a", "#c66a10"]
XAI_FEATURE_VALUE_RANGE = ["#2f6c7a", "#f4f1de", "#c66a10"]
OPTUNA_BALANCE_MODE_ORDER = ("none", "smote")
OPTUNA_BALANCE_MODE_LABELS = {
    "none": "Sin SMOTE",
    "smote": "Con SMOTE",
}
CALIBRATION_METHOD_ORDER = ("sigmoid", "isotonic", "none")
EXPERIMENT_RESULT_TIMESTAMP_RE = re.compile(
    r"(?:experiments_results|find_samples_sizes_results|best_highway_section_results|"
    r"best_highway_section_k_results|best_highway_section_controlled_summary|"
    r"controlled_comparison_summary)_(\d{8}_\d{6})\.csv$"
)
CALIBRATION_SWEEP_RUN_RE = re.compile(r"^calibration_sweep_(\d{8}_\d{6})_.+")
CALIBRATION_SWEEP_HISTORY_FILENAMES = (
    "best_summary.csv",
    "leaderboard.csv",
    "grid_results.csv",
)


class _StreamlitProgress:
    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.value = 0
        self.text = st.empty()
        self.bar = st.progress(0)

    def set_description(self, label: str) -> None:
        self.text.text(label)

    def update(self, step: int = 1) -> None:
        self.value = min(self.total, self.value + int(step))
        percent = int((self.value / self.total) * 100)
        self.bar.progress(percent)

    def close(self) -> None:
        self.text.empty()


def _format_duration_compact(seconds: Optional[float]) -> str:
    if seconds is None or not np.isfinite(float(seconds)):
        return "-"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _init_state() -> None:
    st.session_state.setdefault("accidents_df", None)
    st.session_state.setdefault("accident_files", [])
    st.session_state.setdefault("porticos_source", None)
    st.session_state.setdefault("flow_df", None)
    st.session_state.setdefault("flow_batch_paths", [])
    st.session_state.setdefault("flow_rows_loaded", 0)
    st.session_state.setdefault("flow_features_df", None)
    st.session_state.setdefault("flow_features_path", None)
    st.session_state.setdefault("flow_features_source", None)
    st.session_state.setdefault("flow_features_tramo", None)
    st.session_state.setdefault("flow_features_tramo_label", None)
    st.session_state.setdefault("cluster_features_df", None)
    st.session_state.setdefault("cluster_features_path", None)
    st.session_state.setdefault("cluster_features_source", None)
    st.session_state.setdefault("selected_features", None)
    st.session_state.setdefault("feature_importances_df", None)
    st.session_state.setdefault("feature_selection_store", {})
    st.session_state.setdefault("feature_selection_active_key", None)
    st.session_state.setdefault("balanced_base_df", None)
    st.session_state.setdefault("balanced_cluster_only_df", None)
    st.session_state.setdefault("balanced_cluster_df", None)
    st.session_state.setdefault("use_balanced_base", False)
    st.session_state.setdefault("use_balanced_cluster", False)
    st.session_state.setdefault("cluster_choice", "(sin clusters)")
    st.session_state.setdefault("include_counts", False)
    st.session_state.setdefault("smote_random_state", 42)
    st.session_state.setdefault("smote_k_neighbors", 5)
    st.session_state.setdefault("smote_sampling_strategy", None)
    st.session_state.setdefault("test_size", 0.2)
    st.session_state.setdefault("balance_source", "Balancear nuevos datos")
    st.session_state.setdefault("variables_source", "Calcular nuevas variables")
    st.session_state.setdefault("optuna_best_smote_params", None)
    st.session_state.setdefault("optuna_best_model_params", None)
    st.session_state.setdefault("optuna_best_score", None)
    st.session_state.setdefault("optuna_best_model_choice", None)
    st.session_state.setdefault("optuna_best_settings", None)
    st.session_state.setdefault("optuna_best_search_space", None)
    st.session_state.setdefault("optuna_trials_df", None)
    st.session_state.setdefault("optuna_results_store", {})
    st.session_state.setdefault("optuna_active_key", None)
    st.session_state.setdefault("optuna_model_params_applied_signatures", {})
    # Aceptar fallback de calibración de Optuna en la pestaña Modelos.
    # Default False: el usuario debe marcar conscientemente el opt-in para
    # que Modelos use un Optuna cuya calibración no coincide con la pedida.
    st.session_state.setdefault("allow_optuna_calibration_fallback", False)
    st.session_state.setdefault("optuna_n_trials", 500)
    st.session_state.setdefault("optuna_timeout", 86400)
    st.session_state.setdefault("optuna_n_jobs", 1)
    st.session_state.setdefault("optuna_random_state", 42)
    st.session_state.setdefault("optuna_pruner_enabled", True)
    st.session_state.setdefault("optuna_pruner_startup_trials", 5)
    st.session_state.setdefault("far_target", 0.2)
    st.session_state.setdefault("val_size", 0.2)
    st.session_state.setdefault("model_feature_source", "feature_selection")
    st.session_state.setdefault("model_feature_source_config_label", None)
    st.session_state.setdefault("balance_last_stats", None)
    st.session_state.setdefault("balance_last_params", None)
    st.session_state.setdefault("history_entries", [])
    st.session_state.setdefault("xai_report_cache", {})
    st.session_state.setdefault("calibration_sweep_last_payload", None)


def _normalize_portico_code(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    text = text.upper()
    try:
        num = float(text.replace(",", "."))
    except ValueError:
        return text
    if num.is_integer():
        return str(int(num))
    return str(num)


def _normalize_portico_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.upper()
    invalid = (
        text.isna()
        | text.str.len().fillna(0).eq(0)
        | text.isin(["NAN", "NONE", "NULL"]).fillna(False)
    )
    numeric_text = text.str.replace(",", ".", regex=False)
    nums = pd.to_numeric(numeric_text, errors="coerce")
    result = text.copy()
    numeric_mask = nums.notna()
    if numeric_mask.any():
        int_mask = numeric_mask & np.isclose(nums, np.floor(nums))
        if int_mask.any():
            result.loc[int_mask] = nums.loc[int_mask].astype("Int64").astype("string")
        float_mask = numeric_mask & ~int_mask
        if float_mask.any():
            result.loc[float_mask] = nums.loc[float_mask].astype("string")
    result.loc[invalid] = pd.NA
    return result


def _max_optuna_parallel_jobs() -> int:
    return max(1, int(os.cpu_count() or 1))


def _build_optuna_int_grid(low: int, high: int, step: int = 1) -> List[int]:
    low_i = int(low)
    high_i = int(high)
    step_i = max(1, int(step))
    if high_i < low_i:
        high_i = low_i
    values = list(range(low_i, high_i + 1, step_i))
    if not values:
        return [low_i]
    if values[-1] != high_i:
        values.append(high_i)
    return values


def _suggest_optuna_discrete_int(
    trial: "optuna.Trial",
    name: str,
    low: int,
    high: int,
    *,
    step: int = 1,
) -> int:
    low_i = int(low)
    high_i = int(high)
    step_i = max(1, int(step))
    if high_i < low_i:
        high_i = low_i
    if step_i == 1 or (high_i - low_i) % step_i == 0:
        return int(trial.suggest_int(name, low_i, high_i, step=step_i))
    return int(
        trial.suggest_categorical(
            name,
            _build_optuna_int_grid(low_i, high_i, step_i),
        )
    )


def _render_optuna_n_jobs_input(
    label: str,
    *,
    key: str,
    default: int = 1,
) -> int:
    max_jobs = _max_optuna_parallel_jobs()
    raw_default = st.session_state.get(key, default)
    try:
        default_jobs = int(raw_default)
    except (TypeError, ValueError):
        default_jobs = int(default)
    default_jobs = max(1, min(max_jobs, default_jobs))
    return int(
        st.number_input(
            label,
            min_value=1,
            max_value=max_jobs,
            value=default_jobs,
            step=1,
            key=key,
            help=(
                "Paraleliza los trials de Optuna. No modifica `n_jobs` interno "
                "de Random Forest ni de XGBoost; esos hilos se controlan con los "
                "selectores de jobs del modelo."
            ),
        )
    )


def _render_model_n_jobs_input(
    label: str,
    *,
    key: str,
    default: int = 1,
    shared_key: Optional[str] = None,
) -> int:
    max_jobs = _max_optuna_parallel_jobs()
    raw_default = st.session_state.get(
        key,
        st.session_state.get(shared_key, default) if shared_key else default,
    )
    try:
        default_jobs = int(raw_default)
    except (TypeError, ValueError):
        default_jobs = int(default)
    default_jobs = max(1, min(max_jobs, default_jobs))
    value = int(
        st.number_input(
            label,
            min_value=1,
            max_value=max_jobs,
            value=default_jobs,
            step=1,
            key=key,
            help=(
                "Controla `n_jobs`, es decir, los hilos internos usados por el "
                "modelo durante el entrenamiento. Este valor de la UI prevalece "
                "sobre cualquier resultado previo de Optuna."
            ),
        )
    )
    if shared_key:
        st.session_state[shared_key] = int(value)
    return value


def _queue_controlled_job_config_apply(
    *,
    parallel_jobs: int,
    optuna_n_jobs: int,
    xgb_parallel_jobs: int,
    notice: str,
) -> None:
    st.session_state["exp_controlled_pending_job_config"] = {
        "parallel_jobs": int(parallel_jobs),
        "optuna_n_jobs": int(optuna_n_jobs),
        "xgb_parallel_jobs": int(xgb_parallel_jobs),
        "notice": str(notice),
    }
    st.rerun()


def _apply_pending_controlled_job_config() -> None:
    payload = st.session_state.pop("exp_controlled_pending_job_config", None)
    if not isinstance(payload, dict):
        return
    st.session_state["exp_controlled_parallel_jobs"] = int(
        payload.get("parallel_jobs", 1)
    )
    st.session_state["exp_controlled_optuna_n_jobs"] = int(
        payload.get("optuna_n_jobs", 1)
    )
    st.session_state["exp_controlled_xgb_parallel_jobs"] = int(
        payload.get("xgb_parallel_jobs", 1)
    )
    notice = str(payload.get("notice") or "").strip()
    if notice:
        st.session_state["exp_controlled_memory_notice"] = notice


def _format_bytes(value: object) -> str:
    if value is None:
        return "-"
    try:
        size = float(value)
    except Exception:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    return f"{size:,.2f} TB"


def _system_memory_snapshot() -> Dict[str, Optional[int]]:
    total_bytes: Optional[int] = None
    available_bytes: Optional[int] = None
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            total_bytes = int(vm.total)
            available_bytes = int(vm.available)
        except Exception:
            total_bytes = None
            available_bytes = None
    if total_bytes is None and hasattr(os, "sysconf"):
        try:
            total_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            total_bytes = None
    return {
        "total_bytes": total_bytes,
        "available_bytes": available_bytes,
    }


def _default_memory_budget_gb(snapshot: Dict[str, Optional[int]]) -> float:
    ref_bytes = snapshot.get("available_bytes") or snapshot.get("total_bytes")
    if not ref_bytes:
        return 8.0
    return max(1.0, round((float(ref_bytes) / float(1024 ** 3)) * 0.85, 1))


def _list_cluster_label_files() -> List[Path]:
    try:
        from src import cluster_visualization_app as cluster_vis
    except Exception:
        if not RESULTS_DIR.exists():
            return []
        candidates = sorted(RESULTS_DIR.glob("cluster_*.csv"))
        return [path for path in candidates if CLUSTER_LABEL_PATTERN.match(path.name)]
    return cluster_vis.list_cluster_files()


def _list_event_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    candidates = []
    for path in DATA_DIR.glob("*.csv"):
        if path.name.lower().startswith("eventos"):
            candidates.append(path)
    return sorted(candidates)


def _render_flow_summary() -> Optional[object]:
    try:
        summary = get_flow_db_summary()
    except Exception as exc:
        st.error(f"No se pudo leer la base de flujos: {exc}")
        return None
    if summary.row_count == 0:
        st.warning("La base de flujos esta vacia. Use Flow database para cargar CSVs.")
    st.caption(f"Archivo DuckDB: {summary.db_path}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{summary.row_count:,}")
    col2.metric(
        "Fecha min",
        summary.min_timestamp.strftime("%Y-%m-%d %H:%M")
        if summary.min_timestamp
        else "-",
    )
    col3.metric(
        "Fecha max",
        summary.max_timestamp.strftime("%Y-%m-%d %H:%M")
        if summary.max_timestamp
        else "-",
    )
    return summary


def _date_defaults(summary) -> Tuple[datetime.date, datetime.date]:
    today = datetime.today().date()
    if summary and summary.min_timestamp and summary.max_timestamp:
        return summary.min_timestamp.date(), summary.max_timestamp.date()
    return today, today


def _build_flow_sample_mode_selector(key_prefix: str) -> str:
    return st.radio(
        "Muestreo",
        ["Todo", "Rango de fechas", "Porcentaje"],
        horizontal=True,
        key=f"{key_prefix}_sample_mode",
    )


def _build_flow_sample_inputs(
    summary: Optional[object],
    mode: str,
    *,
    key_prefix: str,
) -> Tuple[FlowSampleSelection, bool, bool]:
    row_limit = None
    date_start = None
    date_end = None
    range_valid = True

    if mode == "Rango de fechas":
        default_start, default_end = _date_defaults(summary)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha inicio",
                value=default_start,
                key=f"{key_prefix}_start_date",
            )
        with col2:
            end_date = st.date_input(
                "Fecha fin",
                value=default_end,
                key=f"{key_prefix}_end_date",
            )
        use_time = st.checkbox(
            "Usar horas en el rango",
            value=False,
            key=f"{key_prefix}_use_time",
        )
        if use_time:
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.time_input(
                    "Hora inicio",
                    value=dt_time(0, 0),
                    key=f"{key_prefix}_start_time",
                )
            with col2:
                end_time = st.time_input(
                    "Hora fin",
                    value=dt_time(23, 59),
                    key=f"{key_prefix}_end_time",
                )
        else:
            start_time = dt_time(0, 0)
            end_time = dt_time(23, 59, 59)
        start_ts = pd.Timestamp(datetime.combine(start_date, start_time))
        end_ts = pd.Timestamp(datetime.combine(end_date, end_time))
        if end_ts <= start_ts:
            st.error("La fecha final debe ser posterior a la fecha de inicio.")
            range_valid = False
        else:
            date_start = start_ts
            date_end = end_ts
    elif mode == "Porcentaje":
        if summary is None or summary.row_count == 0:
            st.warning("No hay filas disponibles para muestrear.")
        else:
            percent = st.slider(
                "Porcentaje",
                min_value=1,
                max_value=100,
                value=10,
                key=f"{key_prefix}_percent",
            )
            row_limit = max(1, int(summary.row_count * (percent / 100.0)))
            st.caption(f"Se consultaran {row_limit:,} filas.")

    sample = FlowSampleSelection(
        date_start=date_start,
        date_end=date_end,
        row_limit=row_limit,
    )
    return sample, mode == "Porcentaje" and row_limit is not None, range_valid


def _build_tramo_selector(
    accidents_df: Optional[pd.DataFrame],
    *,
    date_start: Optional[pd.Timestamp],
    date_end: Optional[pd.Timestamp],
    allowed_porticos: Optional[set[str]] = None,
    key: str,
) -> Optional[Tuple[str, str, str, str]]:
    tramo_tuple: Optional[Tuple[str, str, str, str]] = None
    tramo_options = ["Toda la autopista"]
    tramo_lookup: Dict[str, Tuple[str, str]] = {}
    if accidents_df is None or accidents_df.empty:
        st.info("Cargue accidentes en la pestana Eventos para usar tramos.")
    elif not {"accidente_time", "ultimo_portico"}.issubset(accidents_df.columns):
        st.warning("Los accidentes no tienen accidente_time y ultimo_portico.")
    else:
        acc_filtered = accidents_df.copy()
        acc_filtered["_acc_time"] = pd.to_datetime(
            acc_filtered["accidente_time"], errors="coerce"
        )
        if date_start is not None and date_end is not None:
            acc_filtered = acc_filtered[
                (acc_filtered["_acc_time"] >= date_start)
                & (acc_filtered["_acc_time"] <= date_end)
            ]
        else:
            st.caption(
                "Muestreo sin rango temporal: conteo usa todos los accidentes."
            )
        st.caption(
            f"Accidentes considerados para tramo: {len(acc_filtered):,}"
        )
        try:
            porticos_df = load_porticos()
        except Exception as exc:
            st.warning(f"No se pudieron cargar los porticos: {exc}")
        else:
            porticos = porticos_df.copy()
            porticos["orden_num"] = pd.to_numeric(
                porticos["orden"], errors="coerce"
            )
            porticos["km_num"] = pd.to_numeric(
                porticos["km"], errors="coerce"
            )
            porticos["eje_norm"] = (
                porticos["eje"].astype(str).str.strip().str.upper()
            )
            porticos["calzada_norm"] = (
                porticos["calzada"].astype(str).str.strip().str.upper()
            )
            porticos = porticos.dropna(
                subset=["orden_num", "km_num", "eje_norm", "calzada_norm"]
            )

            segments: List[Dict[str, object]] = []
            for _, group in porticos.groupby(["eje_norm", "calzada_norm"]):
                group = group.sort_values("orden_num")
                for i in range(len(group) - 1):
                    start = group.iloc[i]
                    end = group.iloc[i + 1]
                    segments.append(
                        {
                            "Eje": start["eje"],
                            "Calzada": start["calzada"],
                            "orden_inicio": int(start["orden_num"]),
                            "portico_inicio": str(start["portico"]).strip(),
                            "km_inicio": float(start["km_num"]),
                            "portico_fin": str(end["portico"]).strip(),
                            "km_fin": float(end["km_num"]),
                        }
                    )
            segments_df = pd.DataFrame(segments)

            km_col = _find_match_column(acc_filtered, ["Km.", "Km", "Kilometro"])
            eje_col = _find_match_column(acc_filtered, ["Eje"])
            calzada_col = _find_match_column(acc_filtered, ["Calzada"])
            counts_df = pd.DataFrame(
                columns=[
                    "Eje",
                    "Calzada",
                    "portico_inicio",
                    "portico_fin",
                    "accidentes",
                ]
            )
            if km_col is None or eje_col is None or calzada_col is None:
                st.warning("No se encontraron columnas km/eje/calzada en accidentes.")
            else:
                acc_seg = acc_filtered[[eje_col, calzada_col, km_col]].copy()
                acc_seg = acc_seg.rename(
                    columns={
                        eje_col: "eje",
                        calzada_col: "calzada",
                        km_col: "km_acc",
                    }
                )
                acc_seg["km_acc"] = pd.to_numeric(
                    acc_seg["km_acc"].astype(str).str.replace(",", "."),
                    errors="coerce",
                )
                acc_seg = acc_seg.dropna(subset=["km_acc", "eje", "calzada"])

                segment_keys: List[Dict[str, object]] = []
                for row in acc_seg.itertuples(index=False):
                    try:
                        cand = find_candidate_porticos(
                            acc_km=row.km_acc,
                            porticos_df=porticos_df,
                            eje=row.eje,
                            calzada=row.calzada,
                        )
                    except Exception:
                        continue
                    posterior = cand.get("posterior")
                    cercano = cand.get("cercano")
                    if posterior is None or cercano is None:
                        continue
                    segment_keys.append(
                        {
                            "Eje": posterior["eje"],
                            "Calzada": posterior["calzada"],
                            "portico_inicio": str(posterior["portico"]).strip(),
                            "portico_fin": str(cercano["portico"]).strip(),
                        }
                    )

                if segment_keys:
                    counts_df = (
                        pd.DataFrame(segment_keys)
                        .groupby(
                            ["Eje", "Calzada", "portico_inicio", "portico_fin"],
                            dropna=False,
                        )
                        .size()
                        .reset_index(name="accidentes")
                    )

            if allowed_porticos is not None:
                allowed_clean = {
                    str(value).strip()
                    for value in allowed_porticos
                    if value is not None and str(value).strip()
                }
                if allowed_clean:
                    segments_df = segments_df[
                        segments_df["portico_inicio"].isin(allowed_clean)
                    ]
                else:
                    segments_df = segments_df.iloc[0:0]
            if not segments_df.empty:
                segments_df = segments_df.merge(
                    counts_df,
                    on=["Eje", "Calzada", "portico_inicio", "portico_fin"],
                    how="left",
                )
                segments_df["accidentes"] = (
                    segments_df["accidentes"].fillna(0).astype(int)
                )
                segments_df = segments_df.sort_values(
                    ["Eje", "Calzada", "orden_inicio"]
                ).reset_index(drop=True)
                for row in segments_df.itertuples(index=False):
                    label = (
                        f"{row.Eje} | {row.Calzada} | "
                        f"{row.portico_inicio} -> {row.portico_fin} "
                        f"({row.accidentes} accidentes)"
                    )
                    tramo_options.append(label)
                    tramo_lookup[label] = (
                        str(row.Eje),
                        str(row.Calzada),
                        str(row.portico_inicio),
                        str(row.portico_fin),
                    )
            elif allowed_porticos is not None:
                st.warning(
                    "No hay tramos con datos en el archivo seleccionado."
                )

    tramo_choice = st.selectbox(
        "Tramo",
        options=tramo_options,
        key=key,
    )
    if tramo_choice != "Toda la autopista":
        tramo_tuple = tramo_lookup.get(tramo_choice)
        if tramo_tuple:
            eje, calzada, p_start, p_end = tramo_tuple
            st.caption(
                f"Filtro activo: {eje} | {calzada} | {p_start} -> {p_end}"
            )
    return tramo_tuple


def _set_flow_tramo_selection(
    tramo_tuple: Optional[Tuple[str, str, str, str]],
) -> None:
    st.session_state["flow_features_tramo"] = tramo_tuple
    if tramo_tuple:
        eje, calzada, p_start, p_end = tramo_tuple
        st.session_state["flow_features_tramo_label"] = (
            f"{eje} | {calzada} | {p_start} -> {p_end}"
        )
    else:
        st.session_state["flow_features_tramo_label"] = "Toda la autopista"


def _duckdb_quote_identifier(name: str) -> str:
    safe = str(name).replace('"', '""')
    return f'"{safe}"'


def _pick_duckdb_table(tables: List[str], preferred: List[str]) -> Optional[str]:
    for name in preferred:
        if name in tables:
            return name
    return tables[0] if tables else None


def _build_tramo_duckdb_filters(
    tramo_tuple: Optional[Tuple[str, str, str, str]],
    columns: set[str],
) -> Tuple[List[str], List[object], bool]:
    if not tramo_tuple:
        return [], [], True
    eje_sel, calzada_sel, p_start, p_end = tramo_tuple
    clauses: List[str] = []
    params: List[object] = []
    has_segment_filter = False
    if {"portico_last", "portico_next"}.issubset(columns):
        clauses.extend(["portico_last = ?", "portico_next = ?"])
        params.extend([p_start, p_end])
        has_segment_filter = True
    elif {"portico_inicio", "portico_fin"}.issubset(columns):
        clauses.extend(["portico_inicio = ?", "portico_fin = ?"])
        params.extend([p_start, p_end])
        has_segment_filter = True
    elif "portico" in columns:
        clauses.append("portico = ?")
        params.append(p_start)
        has_segment_filter = True
    elif "ultimo_portico" in columns:
        clauses.append("ultimo_portico = ?")
        params.append(p_start)
        has_segment_filter = True

    if not has_segment_filter:
        return [], [], False

    if "eje" in columns and eje_sel not in (None, "") and not pd.isna(eje_sel):
        clauses.append("eje = ?")
        params.append(eje_sel)
    if (
        "calzada" in columns
        and calzada_sel not in (None, "")
        and not pd.isna(calzada_sel)
    ):
        clauses.append("calzada = ?")
        params.append(calzada_sel)

    return clauses, params, True


def _apply_tramo_filter_df(
    df: pd.DataFrame,
    tramo_tuple: Optional[Tuple[str, str, str, str]],
) -> Tuple[pd.DataFrame, bool]:
    if not tramo_tuple:
        return df, True
    eje_sel, calzada_sel, p_start, p_end = tramo_tuple
    start_norm = _normalize_portico_code(p_start)
    end_norm = _normalize_portico_code(p_end)
    mask = pd.Series(True, index=df.index)
    filter_ok = False

    if {"portico_last", "portico_next"}.issubset(df.columns):
        df = df.copy()
        df["portico_last"] = _normalize_portico_series(df["portico_last"])
        df["portico_next"] = _normalize_portico_series(df["portico_next"])
        mask &= df["portico_last"] == start_norm
        mask &= df["portico_next"] == end_norm
        filter_ok = True
    elif {"portico_inicio", "portico_fin"}.issubset(df.columns):
        df = df.copy()
        df["portico_inicio"] = _normalize_portico_series(df["portico_inicio"])
        df["portico_fin"] = _normalize_portico_series(df["portico_fin"])
        mask &= df["portico_inicio"] == start_norm
        mask &= df["portico_fin"] == end_norm
        filter_ok = True
    elif "portico" in df.columns:
        df = df.copy()
        df["portico"] = _normalize_portico_series(df["portico"])
        mask &= df["portico"] == start_norm
        filter_ok = True
    elif "ultimo_portico" in df.columns:
        df = df.copy()
        df["ultimo_portico"] = _normalize_portico_series(df["ultimo_portico"])
        mask &= df["ultimo_portico"] == start_norm
        filter_ok = True

    if not filter_ok:
        return df, False

    if "eje" in df.columns and eje_sel not in (None, "") and not pd.isna(eje_sel):
        eje_norm = str(eje_sel).strip()
        mask &= df["eje"].astype(str).str.strip().eq(eje_norm)
    if (
        "calzada" in df.columns
        and calzada_sel not in (None, "")
        and not pd.isna(calzada_sel)
    ):
        calzada_norm = str(calzada_sel).strip()
        mask &= df["calzada"].astype(str).str.strip().eq(calzada_norm)

    return df.loc[mask].copy(), True


def _load_porticos_from_feature_file(path: Path) -> Optional[set[str]]:
    cache = st.session_state.setdefault("flow_features_porticos_cache", {})
    key = str(path)
    if key in cache:
        cached = cache.get(key)
        if cached is None:
            return None
        return set(cached)
    
    if path.suffix.lower() == ".duckdb":
        if duckdb is None:
            return None
        try:
            con = duckdb.connect(str(path), read_only=True)
            table_rows = con.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in table_rows]
            table_name = _pick_duckdb_table(
                tables,
                ["flow_features", "cluster_features", "features"],
            )
            if not table_name:
                con.close()
                return set()
            table_ref = _duckdb_quote_identifier(table_name)
            
            # Feature engineering: check columns
            cols_info = con.execute(f"DESCRIBE {table_ref}").fetchall()
            cols = [r[0] for r in cols_info]
            
            candidate_cols = ["portico", "portico_last", "ultimo_portico", "portico_inicio", "portico_fin"]
            target_col = next((c for c in candidate_cols if c in cols), None)
            
            if not target_col:
                # If no known portico column, maybe return None?
                # or treat as no filtering possible (empty set means nothing passes if logic uses intersection, 
                # but upstream logic usually means empty set = nothing found).
                # Here we return None to indicate failure reading valid structure.
                con.close()
                return None
            
            col_ref = _duckdb_quote_identifier(target_col)
            rows = con.execute(
                f"SELECT DISTINCT {col_ref} FROM {table_ref}"
            ).fetchall()
            con.close()
            unique_porticos = sorted([str(r[0]).strip() for r in rows if r[0]])
            cache[key] = unique_porticos
            return set(unique_porticos)
        except Exception:
            # If any DB error, return None
            return None

    candidate_cols = ["portico", "portico_last", "ultimo_portico", "portico_inicio", "portico_fin"]
    try:
        # Detect separator
        import csv
        sample = ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(2048)
        except Exception:
             pass
        
        sep = ","
        if sample:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                sep = dialect.delimiter
            except csv.Error:
                pass
        
        # Read only header to find correct column
        header = pd.read_csv(path, sep=sep, nrows=0, engine="python")
        found_cols = [c for c in candidate_cols if c in header.columns]
        
        if not found_cols:
             raise ValueError(f"Columns not found. Available: {list(header.columns)}")
        
        portico_df = pd.read_csv(path, sep=sep, usecols=found_cols, engine="python")
        
        all_porticos = []
        for c in found_cols:
            all_porticos.extend(portico_df[c].dropna().astype(str).str.strip().tolist())
            
        unique_porticos = sorted(list(set(all_porticos)))
        cache[key] = unique_porticos
        return set(unique_porticos)

    except Exception:
        cache[key] = None
        return None


def _get_feature_date_range(path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    cache = st.session_state.setdefault("flow_features_date_range_cache", {})
    key = str(path)
    if key in cache:
        cached = cache.get(key)
        if cached is None:
            return None
        return cached

    if path.suffix.lower() == ".duckdb":
        if duckdb is None:
            cache[key] = None
            return None
        try:
            con = duckdb.connect(str(path), read_only=True)
            table_rows = con.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in table_rows]
            table_name = _pick_duckdb_table(
                tables,
                ["flow_features", "cluster_features", "features"],
            )
            if not table_name:
                con.close()
                cache[key] = None
                return None
            table_ref = _duckdb_quote_identifier(table_name)
            cols_info = con.execute(f"DESCRIBE {table_ref}").fetchall()
            columns = {row[0] for row in cols_info}
            if "interval_start" not in columns:
                con.close()
                cache[key] = None
                return None
            row = con.execute(
                f"SELECT MIN(interval_start), MAX(interval_start) FROM {table_ref}"
            ).fetchone()
            con.close()
            if not row:
                cache[key] = None
                return None
            min_ts, max_ts = row
        except Exception:
            cache[key] = None
            return None
    else:
        try:
            import csv
            sample = ""
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    sample = f.read(2048)
            except Exception:
                sample = ""
            sep = ","
            if sample:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                    sep = dialect.delimiter
                except csv.Error:
                    sep = ","
            header = pd.read_csv(path, sep=sep, nrows=0, engine="python")
            if "interval_start" not in header.columns:
                cache[key] = None
                return None
            df = pd.read_csv(
                path,
                sep=sep,
                usecols=["interval_start"],
                engine="python",
                low_memory=False,
            )
            min_ts = df["interval_start"].min()
            max_ts = df["interval_start"].max()
        except Exception:
            cache[key] = None
            return None

    min_ts = pd.to_datetime(min_ts, errors="coerce")
    max_ts = pd.to_datetime(max_ts, errors="coerce")
    if pd.isna(min_ts) or pd.isna(max_ts):
        cache[key] = None
        return None
    cache[key] = (min_ts, max_ts)
    return min_ts, max_ts


def _get_feature_max_window_days(path: Path) -> Optional[int]:
    date_range = _get_feature_date_range(path)
    if not date_range:
        return None
    min_ts, max_ts = date_range
    delta_days = (max_ts.normalize() - min_ts.normalize()).days + 1
    return max(1, int(delta_days))


def _load_accidents_for_event(path: Path) -> Optional[pd.DataFrame]:
    cache = st.session_state.setdefault("accidents_by_event_cache", {})
    key = str(path)
    if key in cache:
        return cache.get(key)
    try:
        raw_df = read_csv_with_progress(str(path))
        porticos_df = load_porticos()
        if porticos_df is None or porticos_df.empty:
            cache[key] = None
            return None
        acc_df, _ = process_accidentes_df(
            raw_df, porticos_df, return_excluded=True
        )
    except Exception:
        cache[key] = None
        return None
    cache[key] = acc_df
    return acc_df


def _build_batch_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    mode: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    if mode not in {"month", "week"}:
        raise ValueError("mode must be 'month' or 'week'")

    if mode == "month":
        range_start = start_ts.to_period("M").start_time.normalize()
        range_end = (end_ts + pd.offsets.MonthBegin(1)).normalize()
        boundaries = pd.date_range(start=range_start, end=range_end, freq="MS")
        ranges = [
            (boundaries[i], boundaries[i + 1], boundaries[i].strftime("%Y-%m"))
            for i in range(len(boundaries) - 1)
        ]
        return ranges

    range_start = start_ts.normalize() - pd.Timedelta(days=start_ts.weekday())
    range_end = end_ts.normalize() + pd.Timedelta(days=7)
    boundaries = pd.date_range(start=range_start, end=range_end, freq="7D")
    ranges = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        label = f"{start:%Y-%m-%d}_to_{(end - pd.Timedelta(days=1)):%Y-%m-%d}"
        ranges.append((start, end, label))
    return ranges


def _estimate_batch_ranges(
    summary: Optional[object],
    batch_mode: str,
    date_start: Optional[pd.Timestamp],
    date_end: Optional[pd.Timestamp],
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    if summary is None:
        return []
    if summary.min_timestamp is None or summary.max_timestamp is None:
        return []
    filter_start = date_start
    filter_end_exclusive = (
        date_end + pd.Timedelta(nanoseconds=1) if date_end is not None else None
    )
    range_start = summary.min_timestamp
    range_end = summary.max_timestamp
    if filter_start is not None:
        range_start = max(range_start, filter_start)
    if filter_end_exclusive is not None:
        range_end = min(range_end, filter_end_exclusive)
    if range_end <= range_start:
        return []
    return _build_batch_ranges(range_start, range_end, batch_mode)


def _class_distribution(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts().sort_index()
    total = int(counts.sum())
    df = pd.DataFrame(
        {
            "clase": counts.index.astype(int),
            "count": counts.values.astype(int),
        }
    )
    df["pct"] = (df["count"] / total * 100).round(2)
    return df


def _list_balanced_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("accident_balanced_*.csv"))


def _list_flow_feature_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    patterns = ["accident_flow_features_*.duckdb", "flow_features_*.duckdb"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(RESULTS_DIR.glob(pattern))
    unique = {path.name: path for path in files}
    return sorted(unique.values(), key=lambda path: path.name)


def _list_cluster_feature_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    patterns = ["accident_cluster_features_*.duckdb"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(RESULTS_DIR.glob(pattern))
    unique = {path.name: path for path in files}
    return sorted(unique.values(), key=lambda path: path.name)


def _list_experiment_result_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    patterns = [
        "experiments_results_*.csv",
        "find_samples_sizes_results_*.csv",
        "best_highway_section_results_*.csv",
        "best_highway_section_k_results_*.csv",
        "best_highway_section_controlled_summary_*.csv",
        "controlled_comparison_summary_*.csv",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(RESULTS_DIR.glob(pattern))

    calibration_root = RESULTS_DIR / "calibration_experiment_runs"
    if calibration_root.exists():
        for run_dir in calibration_root.glob("calibration_sweep_*"):
            if not run_dir.is_dir():
                continue
            history_file = _calibration_sweep_history_file_for_run(run_dir)
            if history_file is not None:
                files.append(history_file)

    unique = {str(path): path for path in files}
    return sorted(
        unique.values(),
        key=_experiment_result_sort_key,
        reverse=True,
    )


def _calibration_sweep_history_file_for_run(run_dir: Path) -> Optional[Path]:
    for filename in CALIBRATION_SWEEP_HISTORY_FILENAMES:
        candidate = run_dir / "results" / filename
        if candidate.exists():
            return candidate
    return None


def _calibration_sweep_run_dir_from_result_path(path: Path) -> Optional[Path]:
    path = Path(path)
    if path.name not in CALIBRATION_SWEEP_HISTORY_FILENAMES:
        return None
    if path.parent.name != "results":
        return None
    run_dir = path.parent.parent
    if not CALIBRATION_SWEEP_RUN_RE.match(run_dir.name):
        return None
    try:
        run_dir.resolve().relative_to(
            (RESULTS_DIR / "calibration_experiment_runs").resolve()
        )
    except ValueError:
        return None
    return run_dir


def _experiment_result_timestamp(path: Path) -> Optional[str]:
    match = EXPERIMENT_RESULT_TIMESTAMP_RE.search(Path(path).name)
    if match:
        return match.group(1)
    run_dir = _calibration_sweep_run_dir_from_result_path(path)
    if run_dir is not None:
        run_match = CALIBRATION_SWEEP_RUN_RE.match(run_dir.name)
        if run_match:
            return run_match.group(1)
    return None


def _experiment_result_sort_key(path: Path) -> str:
    timestamp = _experiment_result_timestamp(path)
    if timestamp:
        return timestamp
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime(
            "%Y%m%d_%H%M%S"
        )
    except OSError:
        return ""


def _read_json_file(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _calibration_sweep_manifest_for_result(path: Path) -> Dict[str, object]:
    run_dir = _calibration_sweep_run_dir_from_result_path(path)
    if run_dir is None:
        return {}
    return _read_json_file(run_dir / "manifest.json")


def _experiment_result_option_label(path: Path) -> str:
    run_dir = _calibration_sweep_run_dir_from_result_path(path)
    if run_dir is None:
        return Path(path).name

    manifest = _calibration_sweep_manifest_for_result(path)
    status = str(
        manifest.get("result_status") or manifest.get("status") or "disponible"
    ).strip()
    status_label = {
        "completed": "completado",
        "running": "en progreso",
        "failed": "fallido",
    }.get(status.lower(), status or "disponible")
    time_label = str(
        manifest.get("completed_at")
        or manifest.get("updated_at")
        or manifest.get("created_at")
        or _experiment_result_timestamp(path)
        or ""
    ).replace("T", " ")
    parts = ["Calibración score + threshold"]
    if time_label:
        parts.append(time_label)
    parts.extend([status_label, run_dir.name])
    return " | ".join(parts)


def _is_calibration_sweep_result_file(
    path: Path,
    result_df: Optional[pd.DataFrame] = None,
) -> bool:
    if _calibration_sweep_run_dir_from_result_path(path) is not None:
        return True
    if isinstance(result_df, pd.DataFrame):
        if "protocol_family" in result_df.columns and result_df[
            "protocol_family"
        ].astype(str).str.contains(
            "calibration_score_threshold",
            case=False,
            na=False,
        ).any():
            return True
        if "experiment" in result_df.columns and result_df["experiment"].astype(
            str
        ).str.contains(
            "calibration sweep",
            case=False,
            na=False,
        ).any():
            return True
    return False


def _calibration_sweep_result_state_from_path(
    path: Path,
    fallback_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    run_dir = _calibration_sweep_run_dir_from_result_path(path)
    if run_dir is not None:
        manifest = _calibration_sweep_manifest_for_result(path)
        manifest_path = run_dir / "manifest.json"
        return {
            "run_id": str(manifest.get("run_id") or run_dir.name),
            "checkpoint_run_dir": str(run_dir),
            "checkpoint_manifest_path": str(manifest_path)
            if manifest_path.exists()
            else None,
            "checkpoint_manifest": manifest,
            "result_status": str(
                manifest.get("result_status") or manifest.get("status") or ""
            ),
        }

    state: Dict[str, object] = {"run_id": Path(path).stem}
    if isinstance(fallback_df, pd.DataFrame):
        if Path(path).name == "best_summary.csv":
            state["best_summary_df"] = fallback_df
        elif Path(path).name == "leaderboard.csv":
            state["leaderboard_df"] = fallback_df
        elif Path(path).name == "pareto_front.csv":
            state["pareto_front_df"] = fallback_df
        elif Path(path).name == "grid_results.csv":
            state["grid_results_df"] = fallback_df
    return state


def _calibration_sweep_result_state_from_run_dir(
    run_dir: Path,
    *,
    loaded_from_selection: bool = False,
) -> Dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json_file(manifest_path)
    state = {
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "checkpoint_run_dir": str(run_dir),
        "checkpoint_manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "checkpoint_manifest": manifest,
        "result_status": str(manifest.get("result_status") or manifest.get("status") or ""),
    }
    if loaded_from_selection:
        state["loaded_from_selection"] = True
    return state


def _calibration_sweep_checkpoint_status_label(status: object) -> str:
    text = str(status or "disponible").strip()
    return {
        "completed": "completado",
        "running": "en progreso",
        "failed": "fallido",
    }.get(text.lower(), text or "disponible")


def _list_calibration_sweep_checkpoints(
    checkpoint_root: Optional[Path] = None,
) -> List[Dict[str, object]]:
    root = (
        Path(checkpoint_root)
        if checkpoint_root is not None
        else RESULTS_DIR / "calibration_experiment_runs"
    )
    if not root.exists():
        return []

    items: List[Dict[str, object]] = []
    for run_dir in root.glob("calibration_sweep_*"):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        manifest = _read_json_file(manifest_path)
        run_id = str(manifest.get("run_id") or run_dir.name)
        status = str(manifest.get("result_status") or manifest.get("status") or "missing")
        updated_at = str(
            manifest.get("completed_at")
            or manifest.get("updated_at")
            or manifest.get("created_at")
            or ""
        )
        history_file = _calibration_sweep_history_file_for_run(run_dir)
        label = (
            _experiment_result_option_label(history_file)
            if history_file is not None
            else " | ".join(
                part
                for part in [
                    "Calibración score + threshold",
                    updated_at.replace("T", " ") if updated_at else "",
                    _calibration_sweep_checkpoint_status_label(status),
                    run_id,
                ]
                if str(part).strip()
            )
        )
        progress = dict(manifest.get("progress") or {})
        items.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "status": status,
                "status_label": _calibration_sweep_checkpoint_status_label(status),
                "updated_at": updated_at,
                "label": label,
                "history_file": str(history_file) if history_file is not None else None,
                "completed_steps": int(progress.get("completed_steps") or 0),
                "total_steps": int(progress.get("total_steps") or 0),
                "current_step_id": progress.get("current_step_id"),
            }
        )

    def _sort_key(item: Dict[str, object]) -> Tuple[str, str]:
        updated_at = str(item.get("updated_at") or "")
        run_id = str(item.get("run_id") or "")
        return updated_at, run_id

    return sorted(items, key=_sort_key, reverse=True)


def _experiment_result_related_files(path: Path, timestamp: Optional[str]) -> List[Path]:
    run_dir = _calibration_sweep_run_dir_from_result_path(path)
    if run_dir is not None:
        return sorted(
            file_path for file_path in run_dir.rglob("*") if file_path.is_file()
        )
    if timestamp:
        return sorted(RESULTS_DIR.glob(f"*{timestamp}*"))
    return []


def _experiment_export_arcname(path: Path) -> str:
    try:
        return str(path.relative_to(RESULTS_DIR))
    except ValueError:
        return path.name


def _experiment_export_zip_name(path: Path, timestamp: Optional[str]) -> str:
    run_dir = _calibration_sweep_run_dir_from_result_path(path)
    if run_dir is not None:
        return f"{run_dir.name}.zip"
    if timestamp:
        return f"experiment_{timestamp}.zip"
    return f"experiment_{Path(path).stem}.zip"



def _cluster_choice_suffix(cluster_choice: Optional[str]) -> str:
    if not cluster_choice:
        return "sin_cluster"
    text = str(cluster_choice).strip()
    if text in {"(sin clusters)", "(sin cluster)", "(ninguno)"}:
        return "sin_cluster"
    try:
        stem = Path(text).stem
    except Exception:
        stem = text
    suffix = _slugify(stem)
    if not suffix or suffix == "unknown":
        return "sin_cluster"
    return suffix


def _write_df_to_duckdb(
    df: pd.DataFrame,
    path: Path,
    table_name: str,
) -> None:
    if duckdb is None:
        raise RuntimeError("duckdb no esta instalado.")
    con = duckdb.connect(str(path))
    try:
        con.register("df_view", df)
        table_ref = _duckdb_quote_identifier(table_name)
        con.execute(f"DROP TABLE IF EXISTS {table_ref}")
        con.execute(f"CREATE TABLE {table_ref} AS SELECT * FROM df_view")
    finally:
        con.close()


def _save_flow_features(
    df: pd.DataFrame,
    cluster_choice: Optional[str],
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _cluster_choice_suffix(cluster_choice)
    path = RESULTS_DIR / f"accident_flow_features_{suffix}_{stamp}.duckdb"
    _write_df_to_duckdb(df, path, "flow_features")
    return path


def _save_cluster_features(
    df: pd.DataFrame,
    cluster_choice: Optional[str],
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _cluster_choice_suffix(cluster_choice)
    path = RESULTS_DIR / f"accident_cluster_features_{suffix}_{stamp}.duckdb"
    _write_df_to_duckdb(df, path, "cluster_features")
    return path


def _dataset_content_fingerprint(features_df: pd.DataFrame) -> str:
    """Fingerprint determinístico de un DataFrame de features.

    Captura nombre+orden de columnas, dtypes y shape. Dos DataFrames con el
    mismo fingerprint son prácticamente idénticos a nivel de schema. Los
    valores no se incluyen por costo, pero los cambios que rompen la
    coherencia Optuna ↔ Modelos (columnas añadidas/removidas, dtype distinto,
    nº de filas distinto) sí quedan detectados.

    Se usa como:
      - desempate en `_feature_selection_key` cuando no hay `features_path`.
      - firma auxiliar en el store Optuna (`dataset_fingerprint`) para
        detectar drift aunque el path no cambie.
    """
    if not isinstance(features_df, pd.DataFrame):
        return "empty"
    if features_df.empty and len(features_df.columns) == 0:
        return "empty"
    try:
        cols = [str(c) for c in features_df.columns]
        sorted_cols = sorted(cols)
        dtypes = [str(features_df[col].dtype) for col in sorted_cols]
        payload = "|".join(
            [
                f"rows={int(len(features_df))}",
                f"cols={len(sorted_cols)}",
                "names=" + ",".join(sorted_cols),
                "dtypes=" + ",".join(dtypes),
            ]
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "error"


def _feature_selection_key(
    features_path: Optional[str],
    features_source: Optional[str],
    features_df: pd.DataFrame,
) -> str:
    if features_path:
        try:
            return str(Path(features_path).resolve())
        except Exception:
            return str(features_path)
    source = features_source or "memory"
    # Sin un path que identifique unívocamente la fuente, dos datasets con
    # mismo shape colisionaban silenciosamente. El fingerprint garantiza que
    # cambios de schema/shape produzcan un key distinto.
    fingerprint = _dataset_content_fingerprint(features_df)
    return f"{source}:{len(features_df)}:{len(features_df.columns)}:{fingerprint}"


def _feature_selection_id(
    features_path: Optional[str],
    features_source: Optional[str],
    features_df: pd.DataFrame,
) -> str:
    if features_path:
        base = Path(features_path).stem
    else:
        source = features_source or "memory"
        base = f"features_{source}_{len(features_df)}_{len(features_df.columns)}"
    return _slugify(base)


def _feature_selection_paths(feature_id: str) -> Tuple[Path, Path]:
    json_path = RESULTS_DIR / f"feature_selection_{feature_id}.json"
    csv_path = RESULTS_DIR / f"feature_selection_{feature_id}_importance.csv"
    return json_path, csv_path


def _feature_list_signature(features: List[str]) -> str:
    if not features:
        return "none"
    joined = "|".join(sorted(str(feature) for feature in features))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def _optuna_result_key(feature_key: str, feature_cols: List[str]) -> str:
    signature = _feature_list_signature(feature_cols)
    return f"{feature_key}|{signature}"


def _optuna_result_id(feature_id: str, feature_cols: List[str]) -> str:
    signature = _feature_list_signature(feature_cols)
    return _slugify(f"{feature_id}_{signature[:10]}")


def _optuna_result_paths(optuna_id: str) -> Tuple[Path, Path]:
    json_path = RESULTS_DIR / f"optuna_{optuna_id}.json"
    csv_path = RESULTS_DIR / f"optuna_{optuna_id}_trials.csv"
    return json_path, csv_path


def _normalize_optuna_balance_mode(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "sin smote": "none",
        "sin balance": "none",
        "no_smote": "none",
        "no-smote": "none",
        "class_weight": "none",
        "class weight": "none",
        "smote": "smote",
        "con smote": "smote",
        "with_smote": "smote",
        "with-smote": "smote",
    }
    return aliases.get(text, "none")


def _optuna_balance_mode_label(balance_mode: object) -> str:
    mode = _normalize_optuna_balance_mode(balance_mode)
    return OPTUNA_BALANCE_MODE_LABELS.get(mode, str(balance_mode))


def _ordered_calibration_methods(methods: Optional[List[object]] = None) -> List[str]:
    raw_methods = list(methods) if methods is not None else list(CALIBRATION_METHOD_ORDER)
    normalized = [
        _normalize_calibration_method(method)
        for method in raw_methods
    ]
    extras = [
        method
        for method in normalized
        if method not in CALIBRATION_METHOD_ORDER
    ]
    ordered = list(CALIBRATION_METHOD_ORDER) + sorted(set(extras))
    result: List[str] = []
    seen: set[str] = set()
    for method in ordered:
        if method not in normalized or method in seen:
            continue
        seen.add(method)
        result.append(method)
    return result


def _calibration_method_label(calibration_method: object) -> str:
    method = _normalize_calibration_method(calibration_method)
    return CALIBRATION_METHOD_LABELS.get(method, str(calibration_method))


def _calibration_method_options(
    methods: Optional[List[object]] = None,
) -> List[Tuple[str, str]]:
    return [
        (_calibration_method_label(method), method)
        for method in _ordered_calibration_methods(methods)
    ]


def _optuna_objective_option_label(metric: object) -> str:
    metric_key = _normalize_optuna_objective_metric(metric)
    label = OPTUNA_OBJECTIVE_LABELS.get(metric_key, str(metric).upper())
    if _optuna_objective_direction(metric_key) == "minimize":
        return f"{label} (menor es mejor)"
    return label


def _optuna_objective_options(
    metrics: Optional[List[object]] = None,
) -> Dict[str, Dict[str, str]]:
    metric_order = [
        "f1",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "recall",
        "precision",
        "fnr",
        "far_sens",
        "balanced_f1",
        "mcc",
        "brier_score",
        "recall_at_alerts_per_day",
        "operational_cost",
        "net_balanced_rate",
    ]
    requested = [
        _normalize_optuna_objective_metric(metric)
        for metric in (metrics if metrics is not None else metric_order)
    ]
    ordered = [metric for metric in metric_order if metric in requested]
    for metric in requested:
        if metric not in ordered:
            ordered.append(metric)
    return {
        _optuna_objective_option_label(metric): {
            "key": metric,
            "direction": _optuna_objective_direction(metric),
        }
        for metric in ordered
    }


def _optuna_objective_mode_options() -> Dict[str, str]:
    return {
        "Escalar legacy": CALIBRATION_SWEEP_OBJECTIVE_MODE_SCALAR,
        "Multiobjetivo Pareto": CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
    }


def _optuna_objective_mode_label(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode == CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE:
        return "Multiobjetivo Pareto"
    return "Escalar legacy"


def _format_optuna_multiobjective_values(
    values: Optional[Dict[str, object]],
) -> str:
    if not isinstance(values, dict) or not values:
        return "-"
    labels = {
        "mcc": "MCC",
        "pr_auc": "PR-AUC",
        "brier_score": "Brier",
        "recall_at_alerts_per_day": "Recall@N",
    }
    parts: List[str] = []
    for metric in CALIBRATION_SWEEP_MULTIOBJECTIVE_METRICS:
        value = values.get(metric)
        if value is None or pd.isna(value):
            continue
        parts.append(f"{labels.get(metric, metric)}={float(value):.4f}")
    return " | ".join(parts) if parts else "-"


def _controlled_objective_options() -> Dict[str, str]:
    options = _optuna_objective_options(
        [
            "pr_auc",
            "roc_auc",
            "balanced_f1",
            "f1",
            "mcc",
            "brier_score",
            "recall_at_alerts_per_day",
            "operational_cost",
        ]
    )
    return {label: cfg["key"] for label, cfg in options.items()}


def _calibration_sweep_optuna_objective_options(
    *,
    include_advanced: bool = False,
) -> Dict[str, str]:
    default_metrics = [
        "pr_auc",
        "mcc",
        "brier_score",
        "balanced_f1",
        "recall_at_alerts_per_day",
        "operational_cost",
        "far_sens",
    ]
    advanced_metrics = default_metrics + [
        "roc_auc",
        "f1",
        "accuracy",
        "recall",
        "precision",
        "fnr",
        "net_balanced_rate",
    ]
    options = _optuna_objective_options(
        advanced_metrics if include_advanced else default_metrics
    )
    return {label: cfg["key"] for label, cfg in options.items()}


def _calibration_sweep_threshold_objective_options() -> Dict[str, str]:
    return {
        "FAR": "far",
        "F1": "f1",
        "Balanced F1": "balanced_f1",
        "MCC": "mcc",
        "Recall@N alertas/día": "recall_at_alerts_per_day",
        "Costo operacional": "operational_cost",
    }


def _combined_threshold_field_visibility(
    objectives: Sequence[object],
) -> Dict[str, bool]:
    visible_fields = {
        "far_target": False,
        "alerts_per_day": False,
        "fn_cost": False,
        "fp_cost": False,
    }
    for objective in objectives:
        objective_visibility = _threshold_field_visibility_for_objective(objective)
        for field_name, is_visible in objective_visibility.items():
            visible_fields[field_name] = (
                visible_fields[field_name] or bool(is_visible)
            )
    return visible_fields


_THRESHOLD_CONFIG_FIELD_NAMES = (
    "far_target",
    "alerts_per_day",
    "fn_cost",
    "fp_cost",
)


def _threshold_field_visibility_for_objective(
    objective: object,
) -> Dict[str, bool]:
    objective_key = normalize_threshold_objective(objective)
    visible_fields = {
        "far": {"far_target"},
        "recall_at_alerts_per_day": {"alerts_per_day"},
        "operational_cost": {"alerts_per_day", "fn_cost", "fp_cost"},
    }.get(objective_key, set())
    return {
        field_name: field_name in visible_fields
        for field_name in _THRESHOLD_CONFIG_FIELD_NAMES
    }


def _threshold_field_visibility_for_strategy(
    strategy: object,
) -> Dict[str, bool]:
    strategy_key = str(strategy or "").strip().lower()
    aliases = {
        "optuna": "optuna",
        "optimizar threshold": "optuna",
        "far": "far",
        "calibrar por far": "far",
    }
    normalized_strategy = aliases.get(strategy_key, "optuna")
    visible_fields = {"far_target"} if normalized_strategy == "far" else set()
    return {
        field_name: field_name in visible_fields
        for field_name in _THRESHOLD_CONFIG_FIELD_NAMES
    }


def _option_value_from_state(
    options: Dict[str, str],
    state_key: str,
    *,
    default_label: Optional[str] = None,
) -> str:
    labels = list(options.keys())
    fallback_label = (
        default_label if default_label in options else (labels[0] if labels else "")
    )
    selected_label = str(st.session_state.get(state_key, fallback_label))
    if selected_label not in options:
        selected_label = fallback_label
    return str(options[selected_label])


def _persisted_widget_state_key(widget_key: str) -> str:
    return f"_persisted_widget_value::{widget_key}"


def _get_persisted_widget_value(widget_key: str, default: object) -> object:
    persisted_key = _persisted_widget_state_key(widget_key)
    if widget_key in st.session_state:
        st.session_state[persisted_key] = st.session_state[widget_key]
    elif persisted_key not in st.session_state:
        st.session_state[persisted_key] = default
    return st.session_state[persisted_key]


def _set_persisted_widget_value(widget_key: str, value: object) -> object:
    st.session_state[_persisted_widget_state_key(widget_key)] = value
    return value


def _render_conditional_slider(
    label: str,
    *,
    visible: bool,
    min_value: object,
    max_value: object,
    value: object,
    step: object,
    key: str,
    help: Optional[str] = None,
) -> object:
    current_value = _get_persisted_widget_value(key, value)
    if not visible:
        return _set_persisted_widget_value(key, current_value)
    rendered_value = st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=current_value,
        step=step,
        key=key,
        help=help,
    )
    return _set_persisted_widget_value(key, rendered_value)


def _render_conditional_number_input(
    label: str,
    *,
    visible: bool,
    min_value: object,
    value: object,
    step: object,
    key: str,
    max_value: Optional[object] = None,
    help: Optional[str] = None,
) -> object:
    current_value = _get_persisted_widget_value(key, value)
    if not visible:
        return _set_persisted_widget_value(key, current_value)
    kwargs = {
        "min_value": min_value,
        "value": current_value,
        "step": step,
        "key": key,
        "help": help,
    }
    if max_value is not None:
        kwargs["max_value"] = max_value
    rendered_value = st.number_input(
        label,
        **kwargs,
    )
    return _set_persisted_widget_value(key, rendered_value)


def _calibration_method_selectbox(
    label: str,
    *,
    key: str,
    default_method: str = "sigmoid",
    methods: Optional[List[object]] = None,
    help: Optional[str] = None,
) -> str:
    options = _calibration_method_options(methods)
    labels = [option_label for option_label, _ in options]
    mapping = {option_label: option_key for option_label, option_key in options}
    default_label = _calibration_method_label(default_method)
    index = labels.index(default_label) if default_label in labels else 0
    selected_label = st.selectbox(
        label,
        labels,
        index=index,
        key=key,
        help=help,
    )
    return mapping[selected_label]


def _calibration_method_multiselect(
    label: str,
    *,
    key: str,
    default_methods: Optional[List[object]] = None,
    methods: Optional[List[object]] = None,
    help: Optional[str] = None,
) -> List[str]:
    options = _calibration_method_options(methods)
    labels = [option_label for option_label, _ in options]
    mapping = {option_label: option_key for option_label, option_key in options}
    default_labels = [
        _calibration_method_label(method)
        for method in (
            default_methods if default_methods is not None else ["sigmoid", "isotonic"]
        )
        if _calibration_method_label(method) in mapping
    ]
    selected_labels = st.multiselect(
        label,
        labels,
        default=default_labels,
        key=key,
        help=help,
    )
    return [mapping[selected_label] for selected_label in selected_labels if selected_label in mapping]


def _infer_optuna_result_calibration_method(
    result: Optional[Dict[str, object]],
) -> str:
    if not isinstance(result, dict):
        return "none"
    settings = result.get("optuna_settings")
    if isinstance(settings, dict) and settings.get("calibration_method") is not None:
        return _normalize_calibration_method(settings.get("calibration_method"))
    if result.get("calibration_method") is not None:
        return _normalize_calibration_method(result.get("calibration_method"))
    return "none"


def _infer_optuna_result_balance_mode(result: Optional[Dict[str, object]]) -> str:
    if not isinstance(result, dict):
        return "none"
    settings = result.get("optuna_settings")
    if isinstance(settings, dict) and settings.get("balance_mode") is not None:
        return _normalize_optuna_balance_mode(settings.get("balance_mode"))
    best_smote_params = result.get("best_smote_params")
    if isinstance(best_smote_params, dict) and best_smote_params:
        return "smote"
    return "none"


def _normalize_optuna_variant_result(
    *,
    model_choice: str,
    result: Dict[str, object],
    balance_mode: Optional[str] = None,
    calibration_method: Optional[str] = None,
) -> Dict[str, object]:
    normalized_balance_mode = _normalize_optuna_balance_mode(
        balance_mode if balance_mode is not None else _infer_optuna_result_balance_mode(result)
    )
    normalized_calibration_method = _normalize_calibration_method(
        (
            calibration_method
            if calibration_method is not None
            else _infer_optuna_result_calibration_method(result)
        )
    )
    item = dict(result)
    settings = item.get("optuna_settings")
    if not isinstance(settings, dict):
        settings = {}
    else:
        settings = dict(settings)
    item["balance_mode"] = normalized_balance_mode
    item["balance_mode_label"] = _optuna_balance_mode_label(normalized_balance_mode)
    item["calibration_method"] = normalized_calibration_method
    item["calibration_method_label"] = _calibration_method_label(
        normalized_calibration_method
    )
    settings["balance_mode"] = normalized_balance_mode
    settings.setdefault(
        "balance_mode_label",
        _optuna_balance_mode_label(normalized_balance_mode),
    )
    settings["calibration_method"] = normalized_calibration_method
    settings.setdefault(
        "calibration_method_label",
        _calibration_method_label(normalized_calibration_method),
    )
    item["model_choice"] = str(item.get("model_choice") or model_choice)
    item["optuna_settings"] = settings
    item["search_space"] = dict(item.get("search_space") or {})
    return item


def _normalize_optuna_results_payload(
    results: Optional[Dict[str, object]],
) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    if not isinstance(results, dict):
        return normalized

    for key, raw_value in results.items():
        if not isinstance(raw_value, dict):
            continue
        model_choice = str(raw_value.get("model_choice") or key)
        container = normalized.setdefault(
            model_choice,
            {
                "model_choice": model_choice,
                "by_balance_mode": {},
            },
        )
        by_balance_mode = container.setdefault("by_balance_mode", {})

        raw_by_mode = raw_value.get("by_balance_mode")
        if isinstance(raw_by_mode, dict):
            for raw_mode, raw_mode_value in raw_by_mode.items():
                if not isinstance(raw_mode_value, dict):
                    continue
                mode = _normalize_optuna_balance_mode(raw_mode)
                mode_container = by_balance_mode.setdefault(
                    mode,
                    {
                        "balance_mode": mode,
                        "balance_mode_label": _optuna_balance_mode_label(mode),
                        "by_calibration_method": {},
                    },
                )
                by_calibration_method = mode_container.setdefault(
                    "by_calibration_method",
                    {},
                )
                raw_by_calibration: Dict[str, object]
                if any(
                    key in raw_mode_value
                    for key in [
                        "best_score",
                        "best_smote_params",
                        "best_model_params",
                        "optuna_settings",
                        "search_space",
                        "trials_csv",
                        "trials_df",
                    ]
                ):
                    raw_by_calibration = {
                        _infer_optuna_result_calibration_method(raw_mode_value): raw_mode_value
                    }
                elif isinstance(raw_mode_value.get("by_calibration_method"), dict):
                    raw_by_calibration = dict(
                        raw_mode_value.get("by_calibration_method") or {}
                    )
                else:
                    raw_by_calibration = raw_mode_value
                for raw_calibration_method, raw_variant in raw_by_calibration.items():
                    if not isinstance(raw_variant, dict):
                        continue
                    calibration_key = _normalize_calibration_method(
                        raw_calibration_method
                    )
                    by_calibration_method[calibration_key] = (
                        _normalize_optuna_variant_result(
                            model_choice=model_choice,
                            result=raw_variant,
                            balance_mode=mode,
                            calibration_method=calibration_key,
                        )
                    )
            continue

        legacy_mode = _infer_optuna_result_balance_mode(raw_value)
        legacy_calibration_method = _infer_optuna_result_calibration_method(
            raw_value
        )
        mode_container = by_balance_mode.setdefault(
            legacy_mode,
            {
                "balance_mode": legacy_mode,
                "balance_mode_label": _optuna_balance_mode_label(legacy_mode),
                "by_calibration_method": {},
            },
        )
        by_calibration_method = mode_container.setdefault("by_calibration_method", {})
        by_calibration_method[legacy_calibration_method] = (
            _normalize_optuna_variant_result(
                model_choice=model_choice,
                result=raw_value,
                balance_mode=legacy_mode,
                calibration_method=legacy_calibration_method,
            )
        )

    return normalized


def _get_optuna_model_result_container(
    results: Optional[Dict[str, object]],
    model_choice: str,
) -> Optional[Dict[str, object]]:
    normalized = _normalize_optuna_results_payload(results)
    container = normalized.get(str(model_choice))
    return container if isinstance(container, dict) else None


def _get_optuna_model_result_variant_match(
    results: Optional[Dict[str, object]],
    *,
    model_choice: str,
    balance_mode: str,
    calibration_method: Optional[str] = None,
    fallback_modes: Optional[List[str]] = None,
    fallback_calibration_methods: Optional[List[str]] = None,
    allow_any_calibration_within_mode: bool = False,
) -> Optional[Dict[str, object]]:
    container = _get_optuna_model_result_container(results, model_choice)
    if not isinstance(container, dict):
        return None
    by_balance_mode = container.get("by_balance_mode")
    if not isinstance(by_balance_mode, dict):
        return None
    requested_mode = _normalize_optuna_balance_mode(balance_mode)
    requested_calibration = (
        None
        if calibration_method is None
        else _normalize_calibration_method(calibration_method)
    )
    candidate_modes = [
        requested_mode,
        *[
            _normalize_optuna_balance_mode(mode)
            for mode in list(fallback_modes or [])
        ],
    ]
    seen_modes: set[str] = set()
    for candidate_mode in candidate_modes:
        if candidate_mode in seen_modes:
            continue
        seen_modes.add(candidate_mode)
        mode_container = by_balance_mode.get(candidate_mode)
        if not isinstance(mode_container, dict):
            continue
        by_calibration_method = mode_container.get("by_calibration_method")
        if not isinstance(by_calibration_method, dict):
            continue
        calibration_candidates: List[str] = []
        if requested_calibration is not None:
            calibration_candidates.append(requested_calibration)
        # `fallback_calibration_methods` explícitos siempre se respetan (el
        # caller los eligió intencionalmente). Si vienen como None, SOLO
        # expandir al resto cuando `allow_any_calibration_within_mode` está
        # activo. Antes se expandía incondicionalmente y el flag no tenía
        # efecto práctico, dejando pasar fallbacks silenciosos.
        if fallback_calibration_methods is not None:
            calibration_candidates.extend(
                _ordered_calibration_methods(fallback_calibration_methods)
            )
        if requested_calibration is None or allow_any_calibration_within_mode:
            calibration_candidates.extend(
                _ordered_calibration_methods(
                    list(by_calibration_method.keys())
                )
            )
        seen_calibration_methods: set[str] = set()
        for candidate_calibration in calibration_candidates:
            if candidate_calibration in seen_calibration_methods:
                continue
            seen_calibration_methods.add(candidate_calibration)
            variant = by_calibration_method.get(candidate_calibration)
            if isinstance(variant, dict):
                return {
                    "result": variant,
                    "requested_balance_mode": requested_mode,
                    "resolved_balance_mode": candidate_mode,
                    "requested_calibration_method": requested_calibration,
                    "resolved_calibration_method": candidate_calibration,
                    "used_fallback": (
                        candidate_mode != requested_mode
                        or (
                            requested_calibration is not None
                            and candidate_calibration != requested_calibration
                        )
                    ),
                }
    return None


def _get_optuna_model_result_variant(
    results: Optional[Dict[str, object]],
    *,
    model_choice: str,
    balance_mode: str,
    calibration_method: Optional[str] = None,
    fallback_modes: Optional[List[str]] = None,
    fallback_calibration_methods: Optional[List[str]] = None,
    allow_any_calibration_within_mode: bool = False,
) -> Optional[Dict[str, object]]:
    match = _get_optuna_model_result_variant_match(
        results,
        model_choice=model_choice,
        balance_mode=balance_mode,
        calibration_method=calibration_method,
        fallback_modes=fallback_modes,
        fallback_calibration_methods=fallback_calibration_methods,
        allow_any_calibration_within_mode=allow_any_calibration_within_mode,
    )
    if not isinstance(match, dict):
        return None
    result = match.get("result")
    return result if isinstance(result, dict) else None


def _optuna_model_tab_balance_mode(
    *,
    balance_strategy: str,
    use_balanced: bool,
) -> str:
    if bool(use_balanced):
        return "smote"
    if str(balance_strategy).strip().lower() == "smote":
        return "smote"
    return "none"


def _optuna_trials_path(
    optuna_id: str,
    model_choice: Optional[str] = None,
    balance_mode: Optional[str] = None,
    calibration_method: Optional[str] = None,
) -> Path:
    return _optuna_variant_frame_path(
        optuna_id,
        model_choice=model_choice,
        balance_mode=balance_mode,
        calibration_method=calibration_method,
        frame_kind="trials",
    )


def _optuna_variant_frame_path(
    optuna_id: str,
    *,
    model_choice: Optional[str] = None,
    balance_mode: Optional[str] = None,
    calibration_method: Optional[str] = None,
    frame_kind: str = "trials",
) -> Path:
    frame_suffix = _slugify(frame_kind) or "artifact"
    if model_choice:
        suffix = _slugify(model_choice)
        if balance_mode is not None:
            balance_suffix = _slugify(_normalize_optuna_balance_mode(balance_mode))
            if calibration_method is not None:
                calibration_suffix = _slugify(
                    _normalize_calibration_method(calibration_method)
                )
                return (
                    RESULTS_DIR
                    / f"optuna_{optuna_id}_{suffix}_{balance_suffix}_{calibration_suffix}_{frame_suffix}.csv"
                )
            return (
                RESULTS_DIR
                / f"optuna_{optuna_id}_{suffix}_{balance_suffix}_{frame_suffix}.csv"
            )
        return RESULTS_DIR / f"optuna_{optuna_id}_{suffix}_{frame_suffix}.csv"
    return RESULTS_DIR / f"optuna_{optuna_id}_{frame_suffix}.csv"


def _load_optuna_variant_frame(
    result: Optional[Dict[str, object]],
    *,
    frame_key: str,
    csv_key: str,
) -> Optional[pd.DataFrame]:
    if not isinstance(result, dict):
        return None
    frame = result.get(frame_key)
    if isinstance(frame, pd.DataFrame):
        return frame
    csv_path = result.get(csv_key)
    if not csv_path:
        return None
    try:
        candidate = Path(str(csv_path))
    except Exception:
        return None
    if not candidate.exists():
        return None
    try:
        frame = pd.read_csv(candidate)
    except Exception:
        return None
    result[frame_key] = frame
    return frame


def _load_optuna_result_from_disk(
    optuna_id: str,
) -> Tuple[Optional[Dict[str, object]], Optional[pd.DataFrame]]:
    json_path, csv_path = _optuna_result_paths(optuna_id)
    payload: Optional[Dict[str, object]] = None
    trials_df: Optional[pd.DataFrame] = None
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = None
    if payload and isinstance(payload.get("results"), dict):
        return payload, None
    if csv_path.exists():
        try:
            trials_df = pd.read_csv(csv_path)
        except Exception:
            trials_df = None
    return payload, trials_df


def _persist_optuna_results(
    *,
    optuna_key: str,
    optuna_id: str,
    feature_key: str,
    feature_id: str,
    features_path: Optional[str],
    features_source: Optional[str],
    features_df: pd.DataFrame,
    selected_features: Optional[List[str]],
    feature_cols: List[str],
    model_choice: str,
    balance_mode: str,
    calibration_method: str,
    best_score: float,
    best_smote_params: Dict[str, object],
    best_model_params: Dict[str, object],
    trials_df: Optional[pd.DataFrame],
    optuna_settings: Optional[Dict[str, object]],
    search_space: Dict[str, object],
    extra_result_fields: Optional[Dict[str, object]] = None,
    pareto_front_df: Optional[pd.DataFrame] = None,
) -> None:
    store = st.session_state.setdefault("optuna_results_store", {})
    entry = store.get(optuna_key, {})
    if not isinstance(optuna_settings, dict):
        optuna_settings = {}
    normalized_balance_mode = _normalize_optuna_balance_mode(balance_mode)
    normalized_calibration_method = _normalize_calibration_method(
        calibration_method
    )
    results = _normalize_optuna_results_payload(entry.get("results"))
    if not results:
        legacy_model = entry.get("model_choice")
        if legacy_model:
            results = _normalize_optuna_results_payload(
                {
                    str(legacy_model): {
                        "model_choice": legacy_model,
                        "best_score": entry.get("best_score"),
                        "best_smote_params": entry.get("best_smote_params", {}),
                        "best_model_params": entry.get("best_model_params", {}),
                        "optuna_settings": entry.get("optuna_settings", {}),
                        "search_space": entry.get("search_space", {}),
                        "saved_at": entry.get("saved_at"),
                        "trials_df": entry.get("trials_df"),
                        "trials_csv": entry.get("trials_csv"),
                    }
                }
            )

    model_container = results.setdefault(
        str(model_choice),
        {
            "model_choice": str(model_choice),
            "by_balance_mode": {},
        },
    )
    by_balance_mode = model_container.setdefault("by_balance_mode", {})
    mode_container = by_balance_mode.setdefault(
        normalized_balance_mode,
        {
            "balance_mode": normalized_balance_mode,
            "balance_mode_label": _optuna_balance_mode_label(normalized_balance_mode),
            "by_calibration_method": {},
        },
    )
    by_calibration_method = mode_container.setdefault("by_calibration_method", {})

    trials_csv = None
    if trials_df is not None and not trials_df.empty:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            trials_path = _optuna_trials_path(
                optuna_id,
                model_choice,
                normalized_balance_mode,
                normalized_calibration_method,
            )
            trials_df.to_csv(trials_path, index=False)
            trials_csv = str(trials_path)
        except Exception:
            trials_csv = None
    else:
        existing = by_calibration_method.get(normalized_calibration_method, {})
        trials_csv = existing.get("trials_csv")

    pareto_front_csv = None
    if pareto_front_df is not None and not pareto_front_df.empty:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            pareto_path = _optuna_variant_frame_path(
                optuna_id,
                model_choice=model_choice,
                balance_mode=normalized_balance_mode,
                calibration_method=normalized_calibration_method,
                frame_kind="pareto_front",
            )
            pareto_front_df.to_csv(pareto_path, index=False)
            pareto_front_csv = str(pareto_path)
        except Exception:
            pareto_front_csv = None
    else:
        existing = by_calibration_method.get(normalized_calibration_method, {})
        pareto_front_csv = existing.get("pareto_front_csv")

    variant_settings = dict(optuna_settings)
    variant_settings["balance_mode"] = normalized_balance_mode
    variant_settings.setdefault(
        "balance_mode_label",
        _optuna_balance_mode_label(normalized_balance_mode),
    )
    variant_settings["calibration_method"] = normalized_calibration_method
    variant_settings.setdefault(
        "calibration_method_label",
        _calibration_method_label(normalized_calibration_method),
    )

    result_entry = _normalize_optuna_variant_result(
        model_choice=str(model_choice),
        balance_mode=normalized_balance_mode,
        calibration_method=normalized_calibration_method,
        result={
            "model_choice": model_choice,
            "balance_mode": normalized_balance_mode,
            "balance_mode_label": _optuna_balance_mode_label(
                normalized_balance_mode
            ),
            "calibration_method": normalized_calibration_method,
            "calibration_method_label": _calibration_method_label(
                normalized_calibration_method
            ),
            "best_score": float(best_score),
            "best_smote_params": dict(best_smote_params),
            "best_model_params": dict(best_model_params),
            "optuna_settings": variant_settings,
            "search_space": dict(search_space),
            "saved_at": datetime.now().isoformat(),
            "trials_df": trials_df,
            "trials_csv": trials_csv,
            "pareto_front_df": pareto_front_df,
            "pareto_front_csv": pareto_front_csv,
        },
    )
    if isinstance(extra_result_fields, dict):
        result_entry.update(dict(extra_result_fields))
    by_calibration_method[normalized_calibration_method] = result_entry
    mode_container["balance_mode"] = normalized_balance_mode
    mode_container["balance_mode_label"] = _optuna_balance_mode_label(
        normalized_balance_mode
    )
    mode_container["by_calibration_method"] = by_calibration_method
    by_balance_mode[normalized_balance_mode] = mode_container
    model_container["model_choice"] = str(model_choice)
    model_container["by_balance_mode"] = by_balance_mode
    results[str(model_choice)] = model_container

    entry = {
        "optuna_id": optuna_id,
        "feature_key": feature_key,
        "feature_id": feature_id,
        "features_path": features_path,
        "features_source": features_source,
        "features_rows": int(len(features_df)),
        "features_cols": int(len(features_df.columns)),
        "dataset_fingerprint": _dataset_content_fingerprint(features_df),
        "selection_mode": "all" if selected_features is None else "selected",
        "selected_features": list(selected_features) if selected_features else [],
        "feature_cols": list(feature_cols),
        "results": results,
        "saved_at": datetime.now().isoformat(),
    }
    store[optuna_key] = entry

    json_path, _ = _optuna_result_paths(optuna_id)
    payload = dict(entry)
    payload_results: Dict[str, object] = {}
    for choice, data in results.items():
        if not isinstance(data, dict):
            continue
        result_payload = dict(data)
        raw_by_mode = result_payload.get("by_balance_mode")
        payload_by_mode: Dict[str, object] = {}
        if isinstance(raw_by_mode, dict):
            for mode_key, mode_data in raw_by_mode.items():
                if not isinstance(mode_data, dict):
                    continue
                mode_payload = dict(mode_data)
                raw_by_calibration = mode_payload.get("by_calibration_method")
                payload_by_calibration: Dict[str, object] = {}
                if isinstance(raw_by_calibration, dict):
                    for calibration_key, calibration_data in raw_by_calibration.items():
                        if not isinstance(calibration_data, dict):
                            continue
                        calibration_payload = dict(calibration_data)
                        calibration_payload.pop("trials_df", None)
                        calibration_payload.pop("pareto_front_df", None)
                        payload_by_calibration[str(calibration_key)] = calibration_payload
                mode_payload["by_calibration_method"] = payload_by_calibration
                mode_payload.pop("trials_df", None)
                payload_by_mode[str(mode_key)] = mode_payload
        result_payload["by_balance_mode"] = payload_by_mode
        payload_results[str(choice)] = result_payload
    payload["results"] = payload_results
    try:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                indent=2,
                default=_json_default,
            )
    except Exception:
        return


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    return str(value)


def _is_null_like(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _stringify_streamlit_cell(value: object) -> object:
    if _is_null_like(value):
        return None
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (dict, list, tuple, set, Path)):
        return json.dumps(value, default=_json_default, ensure_ascii=True, sort_keys=True)
    return value


def _prepare_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed object columns so Streamlit Arrow serialization stays stable."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    safe_df = df.copy()
    for column in safe_df.columns:
        series = safe_df[column]
        if not pd.api.types.is_object_dtype(series.dtype):
            continue
        non_null_values = [value for value in series.tolist() if not _is_null_like(value)]
        if not non_null_values:
            continue
        if any(isinstance(value, (dict, list, tuple, set, Path)) for value in non_null_values):
            safe_df[column] = series.map(_stringify_streamlit_cell)
            continue
        normalized_types = {
            str
            if isinstance(value, str)
            else bool
            if isinstance(value, (bool, np.bool_))
            else int
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            else float
            if isinstance(value, (float, np.floating))
            else type(value)
            for value in non_null_values[:50]
        }
        if len(normalized_types) > 1:
            safe_df[column] = series.map(
                lambda value: None if _is_null_like(value) else str(value)
            )
    return safe_df


def _init_experiment_db(
    experiment_name: str,
    meta: Optional[Dict[str, object]] = None,
) -> Optional[Path]:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(experiment_name) or "experiment"
        path = RESULTS_DIR / f"experiment_live_{slug}_{stamp}.sqlite"
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                experiment TEXT,
                payload_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS best (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                payload_json TEXT
            )
            """
        )
        base_meta = {
            "experiment": experiment_name,
            "created_at": datetime.now().isoformat(),
        }
        if meta:
            base_meta.update(meta)
        for key, value in base_meta.items():
            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (str(key), json.dumps(value, default=_json_default)),
            )
        con.commit()
        con.close()
        return path
    except Exception:
        return None


def _append_experiment_result(
    db_path: Optional[Path], payload: Dict[str, object]
) -> None:
    if not db_path:
        return
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        experiment_name = (
            payload.get("experiment")
            or payload.get("type")
            or "unknown"
        )
        cur.execute(
            "INSERT INTO results (created_at, experiment, payload_json) VALUES (?, ?, ?)",
            (
                datetime.now().isoformat(),
                str(experiment_name),
                json.dumps(payload, default=_json_default, ensure_ascii=True),
            ),
        )
        con.commit()
        con.close()
    except Exception:
        return


def _append_experiment_best(
    db_path: Optional[Path], payload: Dict[str, object]
) -> None:
    if not db_path:
        return
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO best (created_at, payload_json) VALUES (?, ?)",
            (
                datetime.now().isoformat(),
                json.dumps(payload, default=_json_default, ensure_ascii=True),
            ),
        )
        con.commit()
        con.close()
    except Exception:
        return


def _seed_controlled_comparison_live_db(
    db_path: Optional[Path],
    *,
    checkpoint_run_dir: Optional[object],
    dataset_name: str,
    features_name: str,
    segment_info: Optional[Dict[str, object]],
) -> int:
    if not db_path:
        return 0
    run_dir_text = str(checkpoint_run_dir or "").strip()
    if not run_dir_text:
        return 0

    run_dir = Path(run_dir_text)
    grid_results_path = _controlled_comparison_paths(run_dir)["grid_results"]
    if not grid_results_path.exists():
        return 0

    con = None
    try:
        con = sqlite3.connect(db_path)
        row_count = con.execute("SELECT COUNT(*) FROM results").fetchone()
        if row_count and int(row_count[0]) > 0:
            return 0
    except Exception:
        return 0
    finally:
        if con is not None:
            con.close()

    try:
        grid_results_df = pd.read_csv(grid_results_path)
    except Exception:
        return 0
    if grid_results_df.empty:
        return 0

    seeded = 0
    for record in grid_results_df.to_dict(orient="records"):
        payload = dict(record)
        payload["experiment"] = "Controlled comparison"
        payload["dataset_name"] = dataset_name
        payload["features_name"] = features_name
        payload["segment_info"] = dict(segment_info or {})
        _append_experiment_result(db_path, payload)
        seeded += 1
    return int(seeded)


def _append_history_entry(entry: Dict[str, object]) -> None:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(entry, ensure_ascii=True, default=_json_default)
                + "\n"
            )
        history = st.session_state.setdefault("history_entries", [])
        history.append(entry)
    except Exception:
        return


def _load_history_entries() -> List[Dict[str, object]]:
    if not HISTORY_PATH.exists():
        return []
    entries: List[Dict[str, object]] = []
    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
    except Exception:
        return entries
    return entries


def _delete_history_entry(run_id: Optional[str]) -> bool:
    if not run_id or not HISTORY_PATH.exists():
        return False
    try:
        lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    kept_lines: List[str] = []
    removed = False
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            kept_lines.append(line)
            continue
        if isinstance(entry, dict) and entry.get("run_id") == run_id:
            removed = True
            continue
        kept_lines.append(line)
    if not removed:
        return False
    HISTORY_PATH.write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""),
        encoding="utf-8",
    )
    history_state = st.session_state.get("history_entries")
    if isinstance(history_state, list):
        st.session_state["history_entries"] = [
            item
            for item in history_state
            if not (isinstance(item, dict) and item.get("run_id") == run_id)
        ]
    return True


def _summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {"rows": int(len(df))}
    summary["columns"] = int(len(df.columns))
    tramo_tuple = st.session_state.get("flow_features_tramo")
    tramo_label = st.session_state.get("flow_features_tramo_label")
    if tramo_tuple:
        eje, calzada, p_start, p_end = tramo_tuple
        tramo_info: Dict[str, object] = {
            "eje": str(eje),
            "calzada": str(calzada),
            "portico_inicio": str(p_start),
            "portico_fin": str(p_end),
        }
        if tramo_label:
            tramo_info["label"] = tramo_label
        summary["tramo"] = tramo_info
    elif tramo_label:
        summary["tramo"] = {"label": tramo_label}
    if "interval_start" in df.columns:
        times = pd.to_datetime(df["interval_start"], errors="coerce")
        times = times.dropna()
        if not times.empty:
            summary["fecha_min"] = times.min().isoformat()
            summary["fecha_max"] = times.max().isoformat()
    if "target" in df.columns:
        summary["accidentes"] = int(pd.to_numeric(df["target"], errors="coerce").fillna(0).sum())
    return summary


def _summarize_flow_settings(features_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    summary["features_path"] = st.session_state.get("flow_features_path")
    summary["features_source"] = st.session_state.get("flow_features_source")
    if isinstance(features_df, pd.DataFrame):
        summary["features_rows"] = int(len(features_df))
        summary["features_cols"] = int(len(features_df.columns))
    cluster_features_df = st.session_state.get("cluster_features_df")
    summary["cluster_features_path"] = st.session_state.get("cluster_features_path")
    summary["cluster_features_source"] = st.session_state.get("cluster_features_source")
    summary["cluster_choice"] = st.session_state.get("cluster_choice") or st.session_state.get(
        "acc_flow_cluster_choice"
    )
    if isinstance(cluster_features_df, pd.DataFrame):
        summary["cluster_features_rows"] = int(len(cluster_features_df))
        summary["cluster_features_cols"] = int(len(cluster_features_df.columns))
    summary["metrics"] = st.session_state.get("acc_flow_metrics")
    summary["categories"] = st.session_state.get("acc_flow_categories")
    summary["lanes"] = st.session_state.get("acc_flow_lanes")
    summary["include_cluster_vars"] = st.session_state.get(
        "acc_flow_include_cluster_vars"
    )
    summary["cluster_vars"] = st.session_state.get("acc_flow_cluster_vars")
    return summary


def _summarize_feature_selection(
    features_df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    if features_df is None or features_df.empty:
        return summary
    features_path = st.session_state.get("flow_features_path")
    features_source = st.session_state.get("flow_features_source")
    feature_key = _feature_selection_key(
        features_path, features_source, features_df
    )
    feature_id = _feature_selection_id(
        features_path, features_source, features_df
    )
    store = st.session_state.get("feature_selection_store", {})
    entry = store.get(feature_key, {}) if isinstance(store, dict) else {}
    selected_features = entry.get("selected_features")
    if selected_features is None:
        selected_features = st.session_state.get("selected_features")
    importance_df = entry.get("importance_df")
    if importance_df is None:
        importance_df = st.session_state.get("feature_importances_df")
    top_importance: List[Dict[str, object]] = []
    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        top = importance_df.head(25).copy()
        top_importance = top.to_dict(orient="records")
    _, csv_path = _feature_selection_paths(feature_id)
    summary["feature_id"] = feature_id
    summary["selected_features"] = list(selected_features or [])
    summary["selected_count"] = len(summary["selected_features"])
    summary["importance_top"] = top_importance
    summary["importance_csv"] = str(csv_path) if csv_path.exists() else None
    summary["params"] = entry.get("params", {})
    return summary


def _optuna_summary_from_results(
    results: Dict[str, object],
    *,
    optuna_key: str,
    optuna_id: str,
    feature_cols: List[str],
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    normalized_results = _normalize_optuna_results_payload(results)
    for choice, data in normalized_results.items():
        if not isinstance(data, dict):
            continue
        by_mode = data.get("by_balance_mode")
        if not isinstance(by_mode, dict):
            continue
        variants: Dict[str, object] = {}
        for balance_mode, mode_data in by_mode.items():
            if not isinstance(mode_data, dict):
                continue
            by_calibration_method = mode_data.get("by_calibration_method")
            if not isinstance(by_calibration_method, dict):
                continue
            calibration_variants: Dict[str, object] = {}
            for calibration_method, variant in by_calibration_method.items():
                if not isinstance(variant, dict):
                    continue
                calibration_variants[str(calibration_method)] = {
                    "balance_mode": str(balance_mode),
                    "balance_mode_label": _optuna_balance_mode_label(balance_mode),
                    "calibration_method": str(calibration_method),
                    "calibration_method_label": _calibration_method_label(
                        calibration_method
                    ),
                    "best_score": variant.get("best_score"),
                    "best_smote_params": variant.get("best_smote_params", {}),
                    "best_model_params": variant.get("best_model_params", {}),
                    "settings": variant.get("optuna_settings", {}),
                    "search_space": variant.get("search_space", {}),
                    "saved_at": variant.get("saved_at"),
                    "trials_csv": variant.get("trials_csv"),
                }
            variants[str(balance_mode)] = {
                "balance_mode": str(balance_mode),
                "balance_mode_label": _optuna_balance_mode_label(balance_mode),
                "by_calibration_method": calibration_variants,
            }
        models[str(choice)] = {"by_balance_mode": variants}
    return {
        "optuna_key": optuna_key,
        "optuna_id": optuna_id,
        "feature_cols": list(feature_cols),
        "models": models,
    }


def _summarize_optuna(
    *,
    feature_key: str,
    feature_id: str,
    base_feature_cols: List[str],
    cluster_only_feature_cols: Optional[List[str]] = None,
    cluster_feature_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "active_key": st.session_state.get("optuna_active_key")
    }
    store = st.session_state.get("optuna_results_store", {})

    base_key = _optuna_result_key(feature_key, base_feature_cols)
    base_id = _optuna_result_id(feature_id, base_feature_cols)
    base_entry = store.get(base_key)
    base_results: Optional[Dict[str, object]] = None
    if isinstance(base_entry, dict) and isinstance(base_entry.get("results"), dict):
        base_results = base_entry["results"]
    else:
        payload, _ = _load_optuna_result_from_disk(base_id)
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), dict):
                base_results = payload["results"]
            else:
                legacy_choice = payload.get("model_choice") or "legacy"
                base_results = {
                    str(legacy_choice): {
                        "model_choice": legacy_choice,
                        "best_score": payload.get("best_score"),
                        "best_smote_params": payload.get("best_smote_params", {}),
                        "best_model_params": payload.get("best_model_params", {}),
                        "optuna_settings": payload.get("optuna_settings", {}),
                        "search_space": payload.get("search_space", {}),
                        "saved_at": payload.get("saved_at"),
                        "trials_csv": payload.get("trials_csv"),
                    }
                }

    if isinstance(base_results, dict) and base_results:
        summary["base"] = _optuna_summary_from_results(
            base_results,
            optuna_key=base_key,
            optuna_id=base_id,
            feature_cols=base_feature_cols,
        )

    if (
        cluster_feature_cols
        and set(cluster_feature_cols) != set(base_feature_cols)
    ):
        cluster_key = _optuna_result_key(feature_key, cluster_feature_cols)
        cluster_id = _optuna_result_id(feature_id, cluster_feature_cols)
        cluster_entry = store.get(cluster_key)
        cluster_results: Optional[Dict[str, object]] = None
        if isinstance(cluster_entry, dict) and isinstance(
            cluster_entry.get("results"), dict
        ):
            cluster_results = cluster_entry["results"]
        else:
            payload, _ = _load_optuna_result_from_disk(cluster_id)
            if isinstance(payload, dict):
                if isinstance(payload.get("results"), dict):
                    cluster_results = payload["results"]
                else:
                    legacy_choice = payload.get("model_choice") or "legacy"
                    cluster_results = {
                        str(legacy_choice): {
                            "model_choice": legacy_choice,
                            "best_score": payload.get("best_score"),
                            "best_smote_params": payload.get("best_smote_params", {}),
                            "best_model_params": payload.get("best_model_params", {}),
                            "optuna_settings": payload.get("optuna_settings", {}),
                            "search_space": payload.get("search_space", {}),
                            "saved_at": payload.get("saved_at"),
                            "trials_csv": payload.get("trials_csv"),
                        }
                    }

        if isinstance(cluster_results, dict) and cluster_results:
            summary["base_cluster"] = _optuna_summary_from_results(
                cluster_results,
                optuna_key=cluster_key,
                optuna_id=cluster_id,
                feature_cols=cluster_feature_cols,
            )

    if (
        cluster_only_feature_cols
        and set(cluster_only_feature_cols) != set(base_feature_cols)
        and set(cluster_only_feature_cols) != set(cluster_feature_cols or [])
    ):
        cluster_only_key = _optuna_result_key(feature_key, cluster_only_feature_cols)
        cluster_only_id = _optuna_result_id(feature_id, cluster_only_feature_cols)
        cluster_only_entry = store.get(cluster_only_key)
        cluster_only_results: Optional[Dict[str, object]] = None
        if isinstance(cluster_only_entry, dict) and isinstance(
            cluster_only_entry.get("results"), dict
        ):
            cluster_only_results = cluster_only_entry["results"]
        else:
            payload, _ = _load_optuna_result_from_disk(cluster_only_id)
            if isinstance(payload, dict):
                if isinstance(payload.get("results"), dict):
                    cluster_only_results = payload["results"]
                else:
                    legacy_choice = payload.get("model_choice") or "legacy"
                    cluster_only_results = {
                        str(legacy_choice): {
                            "model_choice": legacy_choice,
                            "best_score": payload.get("best_score"),
                            "best_smote_params": payload.get("best_smote_params", {}),
                            "best_model_params": payload.get("best_model_params", {}),
                            "optuna_settings": payload.get("optuna_settings", {}),
                            "search_space": payload.get("search_space", {}),
                            "saved_at": payload.get("saved_at"),
                            "trials_csv": payload.get("trials_csv"),
                        }
                    }

        if isinstance(cluster_only_results, dict) and cluster_only_results:
            summary["cluster"] = _optuna_summary_from_results(
                cluster_only_results,
                optuna_key=cluster_only_key,
                optuna_id=cluster_only_id,
                feature_cols=cluster_only_feature_cols,
            )

    return summary


def _balance_stats_from_df(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if df is None or df.empty or "target" not in df.columns:
        return {}
    stats: Dict[str, object] = {}
    if "split" in df.columns:
        train_mask = df["split"] == "train"
        test_mask = df["split"] == "test"
        if train_mask.any():
            stats["train"] = _class_distribution(
                df.loc[train_mask, "target"]
            ).to_dict(orient="records")
        if test_mask.any():
            stats["test"] = _class_distribution(
                df.loc[test_mask, "target"]
            ).to_dict(orient="records")
    else:
        stats["all"] = _class_distribution(df["target"]).to_dict(orient="records")
    return stats


def _summarize_balance(
    *,
    base_df: Optional[pd.DataFrame],
    cluster_df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    params = st.session_state.get("balance_last_params")
    base_stats = _balance_stats_from_df(base_df)
    if not base_stats:
        base_stats = st.session_state.get("balance_last_stats") or {}
    summary["base"] = {"stats": base_stats, "params": params}
    if cluster_df is not None:
        cluster_stats = _balance_stats_from_df(cluster_df)
        summary["base_cluster"] = {
            "stats": cluster_stats,
            "params": params if cluster_stats else None,
        }
    return summary


def _save_model_bundle_artifact(
    *,
    model: object,
    run_id: str,
    label: str,
    model_name: str,
    model_params: Dict[str, object],
    feature_cols: List[str],
    result: Dict[str, object],
    feature_summary: Dict[str, object],
    feature_selection_summary: Dict[str, object],
    dataset_summary: Dict[str, object],
    use_balanced: bool,
) -> Dict[str, object]:
    if model is None:
        return {
            "bundle_path": None,
            "model_path": None,
            "manifest_path": None,
            "error": "Modelo no disponible para persistencia.",
        }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle_dir = MODELS_DIR / run_id / _slugify(label)
    manifest = {
        "run_id": run_id,
        "strategy_label": label,
        "model_name": model_name,
        "model_params": dict(model_params),
        "metrics": result.get("metrics", {}),
        "threshold": result.get("metrics", {}).get("threshold"),
        "split_info": result.get("split_info", {}),
        "features_path": feature_summary.get("features_path"),
        "features_source": feature_summary.get("features_source"),
        "cluster_features_path": feature_summary.get("cluster_features_path"),
        "cluster_features_source": feature_summary.get("cluster_features_source"),
        "cluster_choice": feature_summary.get("cluster_choice"),
        "selected_features": feature_selection_summary.get("selected_features", []),
        "use_balanced": bool(use_balanced),
        "saved_at": datetime.now().isoformat(),
        "dataset": dataset_summary,
    }
    try:
        saved_manifest = save_xai_bundle(
            bundle_dir,
            model=model,
            feature_cols=feature_cols,
            xai_payload=result.get("xai_payload"),
            manifest=manifest,
        )
    except Exception as exc:
        return {
            "bundle_path": None,
            "model_path": None,
            "manifest_path": None,
            "error": str(exc),
        }

    return {
        "bundle_path": str(bundle_dir),
        "model_path": saved_manifest.get("model_path"),
        "manifest_path": str(bundle_dir / "manifest.json"),
        "error": None,
    }


def _history_protocol_results_summary(
    protocol_results: Dict[str, object],
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for feature_set, value in dict(protocol_results or {}).items():
        if not isinstance(value, dict):
            continue
        protocol_summary: Dict[str, object] = {}
        for protocol, result in value.items():
            if not isinstance(result, dict):
                continue
            protocol_summary[str(protocol)] = {
                "metrics": result.get("metrics", {}),
                "validation_metrics": result.get("validation_metrics", {}),
                "confusion_matrix": result.get("confusion_matrix"),
                "split_info": result.get("split_info", {}),
                "threshold_info": result.get("threshold_info", {}),
                "note": result.get("note"),
            }
        summary[str(feature_set)] = protocol_summary
    return summary


def _record_experiment_history(
    *,
    base_df: pd.DataFrame,
    features_df: pd.DataFrame,
    balanced_df: Optional[pd.DataFrame],
    base_feature_cols: List[str],
    base_result: Dict[str, object],
    cluster_feature_cols: Optional[List[str]],
    cluster_result: Optional[Dict[str, object]],
    model_choice: str,
    model_params_base: Dict[str, object],
    model_params_cluster: Optional[Dict[str, object]],
    random_state: int,
    test_size: float,
    val_size: float,
    far_target: float,
    use_balanced: bool,
    protocol_results: Optional[Dict[str, object]] = None,
    threshold_protocols: Optional[List[str]] = None,
    threshold_objective: str = "far",
    calibration_method: str = "none",
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    robust_folds: int = 3,
    balance_strategy: str = "none",
    cluster_only_feature_cols: Optional[List[str]] = None,
    cluster_only_result: Optional[Dict[str, object]] = None,
    model_params_cluster_only: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    signature_payload = {
        "model_choice": model_choice,
        "model_params_base": model_params_base,
        "model_params_cluster_only": model_params_cluster_only,
        "model_params_cluster": model_params_cluster,
        "random_state": random_state,
        "time": time.time(),
    }
    signature = hashlib.md5(
        json.dumps(signature_payload, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()[:8]
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signature}"

    features_path = st.session_state.get("flow_features_path")
    features_source = st.session_state.get("flow_features_source")
    feature_key = _feature_selection_key(
        features_path, features_source, features_df
    )
    feature_id = _feature_selection_id(
        features_path, features_source, features_df
    )
    balanced_base_df = st.session_state.get("balanced_base_df")
    balanced_cluster_df = st.session_state.get("balanced_cluster_df")
    dataset_summary = _summarize_dataset(base_df)
    feature_summary = _summarize_flow_settings(features_df)
    feature_selection_summary = _summarize_feature_selection(features_df)
    optuna_summary = _summarize_optuna(
        feature_key=feature_key,
        feature_id=feature_id,
        base_feature_cols=base_feature_cols,
        cluster_only_feature_cols=cluster_only_feature_cols,
        cluster_feature_cols=cluster_feature_cols,
    )
    balance_summary = _summarize_balance(
        base_df=balanced_base_df,
        cluster_df=balanced_cluster_df,
    )

    models: Dict[str, object] = {}
    base_bundle = _save_model_bundle_artifact(
        model=base_result.get("model"),
        run_id=run_id,
        label="base",
        model_name=model_choice,
        model_params=model_params_base,
        feature_cols=base_feature_cols,
        result=base_result,
        feature_summary=feature_summary,
        feature_selection_summary=feature_selection_summary,
        dataset_summary=dataset_summary,
        use_balanced=use_balanced,
    )
    models["Base"] = {
        "model_name": model_choice,
        "model_params": dict(model_params_base),
        "metrics": base_result.get("metrics", {}),
        "confusion_matrix": base_result.get("confusion_matrix"),
        "model_path": base_bundle.get("model_path"),
        "bundle_path": base_bundle.get("bundle_path"),
        "manifest_path": base_bundle.get("manifest_path"),
        "xai_bundle_path": base_bundle.get("bundle_path"),
        "xai_error": base_bundle.get("error"),
        "feature_cols": list(base_feature_cols),
        "split_info": base_result.get("split_info", {}),
    }
    if cluster_only_result is not None and cluster_only_feature_cols is not None:
        cluster_only_bundle = _save_model_bundle_artifact(
            model=cluster_only_result.get("model"),
            run_id=run_id,
            label="cluster",
            model_name=model_choice,
            model_params=(
                dict(model_params_cluster_only) if model_params_cluster_only else {}
            ),
            feature_cols=cluster_only_feature_cols,
            result=cluster_only_result,
            feature_summary=feature_summary,
            feature_selection_summary=feature_selection_summary,
            dataset_summary=dataset_summary,
            use_balanced=use_balanced,
        )
        models["Cluster"] = {
            "model_name": model_choice,
            "model_params": (
                dict(model_params_cluster_only) if model_params_cluster_only else {}
            ),
            "metrics": cluster_only_result.get("metrics", {}),
            "confusion_matrix": cluster_only_result.get("confusion_matrix"),
            "model_path": cluster_only_bundle.get("model_path"),
            "bundle_path": cluster_only_bundle.get("bundle_path"),
            "manifest_path": cluster_only_bundle.get("manifest_path"),
            "xai_bundle_path": cluster_only_bundle.get("bundle_path"),
            "xai_error": cluster_only_bundle.get("error"),
            "feature_cols": list(cluster_only_feature_cols),
            "split_info": cluster_only_result.get("split_info", {}),
        }
    cluster_xai_summary = {
        "available": False,
        "bundle_path": None,
        "manifest_path": None,
        "error": None,
    }
    if cluster_result is not None and cluster_feature_cols is not None:
        cluster_bundle = _save_model_bundle_artifact(
            model=cluster_result.get("model"),
            run_id=run_id,
            label="base_cluster",
            model_name=model_choice,
            model_params=dict(model_params_cluster) if model_params_cluster else {},
            feature_cols=cluster_feature_cols,
            result=cluster_result,
            feature_summary=feature_summary,
            feature_selection_summary=feature_selection_summary,
            dataset_summary=dataset_summary,
            use_balanced=use_balanced,
        )
        models["Base + Cluster"] = {
            "model_name": model_choice,
            "model_params": dict(model_params_cluster) if model_params_cluster else {},
            "metrics": cluster_result.get("metrics", {}),
            "confusion_matrix": cluster_result.get("confusion_matrix"),
            "model_path": cluster_bundle.get("model_path"),
            "bundle_path": cluster_bundle.get("bundle_path"),
            "manifest_path": cluster_bundle.get("manifest_path"),
            "xai_bundle_path": cluster_bundle.get("bundle_path"),
            "xai_error": cluster_bundle.get("error"),
            "feature_cols": list(cluster_feature_cols),
            "split_info": cluster_result.get("split_info", {}),
        }
        cluster_xai_summary = {
            "available": bool(cluster_bundle.get("bundle_path")),
            "bundle_path": cluster_bundle.get("bundle_path"),
            "manifest_path": cluster_bundle.get("manifest_path"),
            "error": cluster_bundle.get("error"),
        }

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_summary,
        "training": {
            "use_balanced": bool(use_balanced),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "far_target": float(far_target),
            "random_state": int(random_state),
            "threshold_protocols": list(threshold_protocols or []),
            "threshold_objective": threshold_objective,
            "calibration_method": calibration_method,
            "alerts_per_day": float(alerts_per_day),
            "fn_cost": float(fn_cost),
            "fp_cost": float(fp_cost),
            "robust_folds": int(robust_folds),
            "balance_strategy": balance_strategy,
            "model_n_jobs": (
                int(model_params_cluster.get("n_jobs"))
                if isinstance(model_params_cluster, dict)
                and model_params_cluster.get("n_jobs") is not None
                else (
                    int(model_params_cluster_only.get("n_jobs"))
                    if isinstance(model_params_cluster_only, dict)
                    and model_params_cluster_only.get("n_jobs") is not None
                    else (
                        int(model_params_base.get("n_jobs"))
                        if model_params_base.get("n_jobs") is not None
                        else None
                    )
                )
            ),
        },
        "features": feature_summary,
        "feature_selection": feature_selection_summary,
        "optuna": optuna_summary,
        "balance": balance_summary,
        "xai": {"base_cluster": cluster_xai_summary},
        "models": models,
        "protocol_results": _history_protocol_results_summary(
            protocol_results or {}
        ),
    }
    _append_history_entry(entry)
    return entry


def _resolve_base_cluster_xai_info(
    entry: Dict[str, object],
) -> Tuple[Optional[str], Optional[str]]:
    models = entry.get("models", {})
    if isinstance(models, dict):
        cluster_entry = models.get("Base + Cluster")
        if isinstance(cluster_entry, dict):
            bundle_path = cluster_entry.get("xai_bundle_path") or cluster_entry.get(
                "bundle_path"
            )
            bundle_error = cluster_entry.get("xai_error")
            return (
                str(bundle_path) if bundle_path else None,
                str(bundle_error) if bundle_error else None,
            )
    xai = entry.get("xai", {})
    if isinstance(xai, dict):
        cluster_xai = xai.get("base_cluster")
        if isinstance(cluster_xai, dict):
            bundle_path = cluster_xai.get("bundle_path")
            bundle_error = cluster_xai.get("error")
            return (
                str(bundle_path) if bundle_path else None,
                str(bundle_error) if bundle_error else None,
            )
    return None, None


def _import_altair():
    try:
        import altair as alt  # type: ignore
    except Exception:
        return None
    return alt


def _parse_jsonish(value: object) -> object:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _jsonish_to_text(value: object) -> str:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    if parsed is None:
        return ""
    return str(parsed)


def _coerce_confusion_matrix(value: object) -> Optional[List[List[int]]]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, list) and len(parsed) == 4 and not isinstance(
        parsed[0], (list, tuple)
    ):
        try:
            tn, fp, fn, tp = [int(item) for item in parsed]
            return [[tn, fp], [fn, tp]]
        except Exception:
            return None
    if (
        isinstance(parsed, list)
        and len(parsed) == 2
        and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in parsed)
    ):
        try:
            return [[int(parsed[0][0]), int(parsed[0][1])], [int(parsed[1][0]), int(parsed[1][1])]]
        except Exception:
            return None
    return None


def _confusion_matrix_to_text(value: object) -> str:
    matrix = _coerce_confusion_matrix(value)
    if matrix is None:
        return _jsonish_to_text(value)
    return json.dumps(matrix, ensure_ascii=False)


def _inspect_controlled_feature_schema(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".duckdb":
        raise ValueError("El archivo de features debe ser .duckdb.")
    if duckdb is None:
        raise RuntimeError("duckdb no esta instalado.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        table_rows = con.execute("SHOW TABLES").fetchall()
        tables = [row[0] for row in table_rows]
        table_name = _pick_duckdb_table(
            tables,
            ["flow_features", "features", "cluster_features"],
        )
        if not table_name:
            raise ValueError("La base de datos de features esta vacia.")
        table_ref = _duckdb_quote_identifier(table_name)
        return con.execute(f"SELECT * FROM {table_ref} LIMIT 0").df()
    finally:
        con.close()


def _controlled_feature_timestamp_bounds(
    path: Path,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if path.suffix.lower() != ".duckdb" or duckdb is None:
        return None
    con = duckdb.connect(str(path), read_only=True)
    try:
        table_rows = con.execute("SHOW TABLES").fetchall()
        tables = [row[0] for row in table_rows]
        table_name = _pick_duckdb_table(
            tables,
            ["flow_features", "features", "cluster_features"],
        )
        if not table_name:
            return None
        table_ref = _duckdb_quote_identifier(table_name)
        cols_info = con.execute(f"DESCRIBE {table_ref}").fetchall()
        columns = {row[0] for row in cols_info}
        if "interval_start" not in columns:
            return None
        interval_ref = _duckdb_quote_identifier("interval_start")
        bounds = con.execute(
            "SELECT "
            f"MIN(TRY_CAST({interval_ref} AS TIMESTAMP)) AS min_ts, "
            f"MAX(TRY_CAST({interval_ref} AS TIMESTAMP)) AS max_ts "
            f"FROM {table_ref}"
        ).fetchone()
    finally:
        con.close()
    if not bounds or bounds[0] is None or bounds[1] is None:
        return None
    return pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])


def _sync_controlled_feature_date_defaults(
    path: Path,
    *,
    bounds: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
    key_prefix: str = "exp_controlled",
) -> None:
    if bounds is None:
        return
    state_path_key = f"{key_prefix}_feature_date_file"
    start_key = f"{key_prefix}_dataset_start_date"
    end_key = f"{key_prefix}_dataset_end_date"
    path_key = str(path)
    min_date = bounds[0].date()
    max_date = bounds[1].date()
    should_reset = st.session_state.get(state_path_key) != path_key
    if should_reset:
        st.session_state[start_key] = min_date
        st.session_state[end_key] = max_date
        st.session_state[state_path_key] = path_key
        return
    for key, fallback in [(start_key, min_date), (end_key, max_date)]:
        raw_value = st.session_state.get(key, fallback)
        try:
            current = pd.Timestamp(raw_value).date()
        except Exception:
            current = fallback
        st.session_state[key] = min(max(current, min_date), max_date)


def _render_controlled_feature_date_range_inputs(
    path: Path,
    *,
    key_prefix: str = "exp_controlled",
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool]:
    bounds = _controlled_feature_timestamp_bounds(path)
    if bounds is None:
        st.warning(
            "No se pudo inferir `interval_start` desde el archivo de features; "
            "la comparación usará el rango completo disponible."
        )
        return None, None, True

    _sync_controlled_feature_date_defaults(path, bounds=bounds, key_prefix=key_prefix)
    min_date = bounds[0].date()
    max_date = bounds[1].date()
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Fecha inicio dataset",
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_dataset_start_date",
            help=(
                "Define la primera fecha incluida desde el DuckDB de features. "
                f"Rango disponible: {min_date} a {max_date}. Afecta filas, "
                "accidentes considerados, split temporal, checkpoints y métricas. "
                "Evite seleccionar una ventana tan corta que deje una sola clase."
            ),
        )
    with col_end:
        end_date = st.date_input(
            "Fecha fin dataset",
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_dataset_end_date",
            help=(
                "Define la última fecha incluida desde el DuckDB de features. "
                f"Rango disponible: {min_date} a {max_date}. El filtro es "
                "inclusivo hasta el fin del día y cambia la grilla evaluada. "
                "No debe ser anterior a la fecha de inicio."
            ),
        )

    date_start = pd.Timestamp(datetime.combine(start_date, dt_time(0, 0)))
    date_end = pd.Timestamp(datetime.combine(end_date, dt_time(23, 59, 59)))
    if date_end < date_start:
        st.error("La fecha fin del dataset debe ser igual o posterior a la fecha inicio.")
        return date_start, date_end, False
    st.caption(
        "Rango temporal seleccionado: "
        f"{date_start:%Y-%m-%d} a {date_end:%Y-%m-%d} "
        f"(disponible: {bounds[0]:%Y-%m-%d %H:%M} a {bounds[1]:%Y-%m-%d %H:%M})."
    )
    return date_start, date_end, True


def _load_controlled_features_df(
    path: Path,
    tramo_tuple: Optional[Tuple[str, str, str, str]],
    *,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if path.suffix.lower() != ".duckdb":
        raise ValueError("El archivo de features debe ser .duckdb.")
    if duckdb is None:
        raise RuntimeError("duckdb no esta instalado.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        table_rows = con.execute("SHOW TABLES").fetchall()
        tables = [row[0] for row in table_rows]
        table_name = _pick_duckdb_table(
            tables,
            ["flow_features", "features", "cluster_features"],
        )
        if not table_name:
            raise ValueError("La base de datos de features esta vacia.")
        table_ref = _duckdb_quote_identifier(table_name)
        cols_info = con.execute(f"DESCRIBE {table_ref}").fetchall()
        columns = {row[0] for row in cols_info}
        clauses, params, filter_ok = _build_tramo_duckdb_filters(
            tramo_tuple, columns
        )
        if tramo_tuple and not filter_ok:
            raise ValueError(
                "El archivo de features no permite filtrar el tramo seleccionado."
            )
        if "interval_start" in columns:
            interval_ref = _duckdb_quote_identifier("interval_start")
            if date_start is not None:
                clauses.append(f"TRY_CAST({interval_ref} AS TIMESTAMP) >= ?")
                params.append(pd.Timestamp(date_start))
            if date_end is not None:
                clauses.append(f"TRY_CAST({interval_ref} AS TIMESTAMP) <= ?")
                params.append(pd.Timestamp(date_end))
        query = f"SELECT * FROM {table_ref}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        df = con.execute(query, params).df()
    finally:
        con.close()

    if {"portico_inicio", "portico_fin"}.issubset(df.columns) and not {
        "portico_last",
        "portico_next",
    }.issubset(df.columns):
        df = df.rename(
            columns={
                "portico_inicio": "portico_last",
                "portico_fin": "portico_next",
            }
        )
    if "interval_start" in df.columns:
        df["interval_start"] = pd.to_datetime(
            df["interval_start"], errors="coerce"
        )
    return df


def _prepare_controlled_comparison_base_df(
    *,
    accidents_df_for_tramo: Optional[pd.DataFrame],
    selected_features_path: Path,
    tramo_tuple: Tuple[str, str, str, str],
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    accidents_segment, filter_ok = _apply_tramo_filter_df(
        accidents_df_for_tramo
        if accidents_df_for_tramo is not None
        else pd.DataFrame(),
        tramo_tuple,
    )
    if not filter_ok:
        raise ValueError("No se pudo filtrar el tramo seleccionado en accidentes.")
    if accidents_segment.empty:
        raise ValueError("No hay accidentes para el tramo seleccionado.")
    if "accidente_time" in accidents_segment.columns and (
        date_start is not None or date_end is not None
    ):
        accidents_segment = accidents_segment.copy()
        accidents_segment["accidente_time"] = pd.to_datetime(
            accidents_segment["accidente_time"], errors="coerce"
        )
        if date_start is not None:
            accidents_segment = accidents_segment[
                accidents_segment["accidente_time"] >= pd.Timestamp(date_start)
            ]
        if date_end is not None:
            accidents_segment = accidents_segment[
                accidents_segment["accidente_time"] <= pd.Timestamp(date_end)
            ]
        if accidents_segment.empty:
            raise ValueError(
                "No hay accidentes para el tramo en el rango de fechas seleccionado."
            )

    features_df = _load_controlled_features_df(
        selected_features_path,
        tramo_tuple,
        date_start=date_start,
        date_end=date_end,
    )
    if features_df.empty:
        raise ValueError("No hay features para el tramo seleccionado.")
    if "interval_start" not in features_df.columns:
        raise ValueError("Las features no tienen interval_start.")

    features_df = features_df.dropna(subset=["interval_start"]).copy()
    if features_df.empty:
        raise ValueError("No hay timestamps válidos en las features del tramo.")
    if not _get_cluster_cols(features_df):
        raise ValueError(
            "El archivo seleccionado no contiene variables de cluster para este experimento."
        )

    base_df = add_accident_target(features_df, accidents_segment)
    if base_df.empty:
        raise ValueError("El dataset quedó vacío tras agregar el target.")
    if base_df["target"].astype(int).nunique() < 2:
        raise ValueError("El dataset del tramo no tiene ambas clases para entrenar.")
    return base_df


def _build_controlled_comparison_curve_chart(
    curves_df: pd.DataFrame,
    *,
    model_name: str,
    balance_mode: str,
    metric_col: str,
    metric_label: str,
):
    plot_df = curves_df.copy()
    if plot_df.empty:
        return None
    plot_df["k"] = pd.to_numeric(plot_df.get("k"), errors="coerce")
    plot_df[metric_col] = pd.to_numeric(
        plot_df.get(metric_col), errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["k", metric_col])
    plot_df = plot_df[
        (plot_df["model_name"].astype(str) == str(model_name))
        & (plot_df["balance_mode"].astype(str) == str(balance_mode))
    ].copy()
    if plot_df.empty:
        return None
    series_col = "feature_set"
    series_title = "Conjunto"
    use_ablation_series = {
        "params_source_feature_set",
        "target_feature_set",
    }.issubset(plot_df.columns)
    if use_ablation_series:
        plot_df["ablation_pair"] = (
            plot_df["params_source_feature_set"].astype(str)
            + " -> "
            + plot_df["target_feature_set"].astype(str)
        )
        series_col = "ablation_pair"
        series_title = "Fuente -> Target"
    plot_df = plot_df.sort_values([series_col, "k"]).reset_index(drop=True)

    alt = _import_altair()
    if alt is None:
        return None

    color_kwargs = {"title": series_title}
    if series_col == "feature_set":
        color_kwargs["scale"] = alt.Scale(
            domain=["Base", "Cluster", "Base + Cluster"],
            range=["#2f6c7a", "#c66a10", "#7a3e48"],
        )
    tooltips = [
        alt.Tooltip("model_name:N", title="Modelo"),
        alt.Tooltip("feature_set:N", title="Conjunto"),
        alt.Tooltip("balance_mode:N", title="Balance"),
        alt.Tooltip("threshold_protocol:N", title="Protocolo"),
        alt.Tooltip("k:Q", title="K"),
        alt.Tooltip(f"{metric_col}:Q", title=str(metric_label), format=".4f"),
    ]
    if use_ablation_series:
        tooltips.insert(2, alt.Tooltip("params_source_feature_set:N", title="Fuente params"))
        tooltips.insert(3, alt.Tooltip("target_feature_set:N", title="Target"))
    return (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", axis=alt.Axis(title="K")),
            y=alt.Y(
                f"{metric_col}:Q",
                axis=alt.Axis(title=str(metric_label)),
            ),
            color=alt.Color(
                f"{series_col}:N",
                **color_kwargs,
            ),
            tooltip=tooltips,
        )
        .properties(height=260)
        .interactive()
    )


def _controlled_comparison_metric_options(
    df: pd.DataFrame,
    *,
    objective_label: str,
) -> List[Tuple[str, str]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    catalog = [
        (f"Validación {objective_label} (objetivo)", "val_objective_score"),
        (f"Test {objective_label} (objetivo)", "test_objective_score"),
        ("Validación Accuracy", "val_accuracy"),
        ("Test Accuracy", "test_accuracy"),
        ("Validación Recall", "val_recall"),
        ("Test Recall", "test_recall"),
        ("Validación Sensibilidad", "val_sensitivity"),
        ("Test Sensibilidad", "test_sensitivity"),
        ("Validación ROC-AUC", "val_roc_auc"),
        ("Test ROC-AUC", "test_roc_auc"),
        ("Validación PR-AUC", "val_pr_auc"),
        ("Test PR-AUC", "test_pr_auc"),
        ("Validación Brier", "val_brier_score"),
        ("Test Brier", "test_brier_score"),
        ("Validación F1", "val_f1"),
        ("Test F1", "test_f1"),
        ("Validación F1 Global", "val_f1_global"),
        ("Test F1 Global", "test_f1_global"),
        ("Validación Balanced F1", "val_balanced_f1"),
        ("Test Balanced F1", "test_balanced_f1"),
        ("Validación F1 Clase 0", "val_f1_class_0"),
        ("Test F1 Clase 0", "test_f1_class_0"),
        ("Validación F1 Clase 1", "val_f1_class_1"),
        ("Test F1 Clase 1", "test_f1_class_1"),
        ("Validación MCC", "val_mcc"),
        ("Test MCC", "test_mcc"),
        ("Validación Alertas/día", "val_alerts_per_day"),
        ("Test Alertas/día", "test_alerts_per_day"),
        ("Validación Falsas alarmas/día", "val_false_alarms_per_day"),
        ("Test Falsas alarmas/día", "test_false_alarms_per_day"),
        ("Validación Recall evento aprox.", "val_event_recall_approx"),
        ("Test Recall evento aprox.", "test_event_recall_approx"),
        ("Validación Costo operacional", "val_operational_cost"),
        ("Test Costo operacional", "test_operational_cost"),
        ("Validación Costo/día", "val_cost_per_day"),
        ("Test Costo/día", "test_cost_per_day"),
        ("Validación Falsos Negativos", "val_false_negatives"),
        ("Test Falsos Negativos", "test_false_negatives"),
        ("Validación Falsos Positivos", "val_false_positives"),
        ("Test Falsos Positivos", "test_false_positives"),
        ("Validación Verdaderos Negativos", "val_true_negatives"),
        ("Test Verdaderos Negativos", "test_true_negatives"),
        ("Validación Verdaderos Positivos", "val_true_positives"),
        ("Test Verdaderos Positivos", "test_true_positives"),
        ("Threshold de decisión", "decision_threshold"),
    ]
    available: List[Tuple[str, str]] = []
    for label, col in catalog:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            available.append((label, col))
    return available


def _prepare_controlled_comparison_detail_display(detail_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(detail_df, pd.DataFrame) or detail_df.empty:
        return pd.DataFrame()
    display_df = detail_df.copy()
    for json_col in [
        "selected_features",
        "selected_features_global",
        "best_params",
        "effective_model_params",
        "smote_params",
    ]:
        if json_col in display_df.columns:
            display_df[json_col] = display_df[json_col].apply(_jsonish_to_text)
    for matrix_col in ["val_confusion_matrix", "test_confusion_matrix"]:
        if matrix_col in display_df.columns:
            display_df[matrix_col] = display_df[matrix_col].apply(
                _confusion_matrix_to_text
            )
    sort_cols = [
        col
        for col in [
            "model_name",
            "balance_mode",
            "params_source_feature_set",
            "target_feature_set",
            "feature_set",
            "k",
        ]
        if col in display_df.columns
    ]
    if sort_cols:
        display_df = display_df.sort_values(sort_cols).reset_index(drop=True)
    preferred_cols = [
        "protocol_family",
        "model_name",
        "feature_set",
        "ablation_phase",
        "params_source_feature_set",
        "target_feature_set",
        "source_combo_id",
        "frozen_tuning",
        "threshold_freeze_policy",
        "balance_mode",
        "threshold_protocol",
        "threshold_protocol_label",
        "threshold_objective",
        "threshold_objective_label",
        "calibration_method",
        "k",
        "k_global",
        "effective_k",
        "selected_base_feature_count",
        "selected_cluster_feature_count",
        "status",
        "objective_label",
        "objective_direction",
        "val_objective_score",
        "test_objective_score",
        "val_accuracy",
        "test_accuracy",
        "val_recall",
        "test_recall",
        "val_sensitivity",
        "test_sensitivity",
        "val_roc_auc",
        "test_roc_auc",
        "val_pr_auc",
        "test_pr_auc",
        "val_brier_score",
        "test_brier_score",
        "val_f1",
        "test_f1",
        "val_f1_global",
        "test_f1_global",
        "val_balanced_f1",
        "test_balanced_f1",
        "val_f1_class_0",
        "test_f1_class_0",
        "val_f1_class_1",
        "test_f1_class_1",
        "val_mcc",
        "test_mcc",
        "val_alerts_per_day",
        "test_alerts_per_day",
        "val_false_alarms_per_day",
        "test_false_alarms_per_day",
        "val_event_recall_approx",
        "test_event_recall_approx",
        "val_operational_cost",
        "test_operational_cost",
        "val_cost_per_day",
        "test_cost_per_day",
        "alerts_per_day_budget",
        "fn_cost",
        "fp_cost",
        "val_false_negatives",
        "test_false_negatives",
        "val_false_positives",
        "test_false_positives",
        "val_true_negatives",
        "test_true_negatives",
        "val_true_positives",
        "test_true_positives",
        "val_confusion_matrix",
        "test_confusion_matrix",
        "decision_threshold",
        "feature_ranking_mode",
        "ranking_protocol",
        "selected_features_global",
        "selected_features",
        "best_params",
        "effective_model_params",
        "effective_threshold_n_jobs",
        "smote_params",
        "error",
    ]
    visible_cols = [col for col in preferred_cols if col in display_df.columns]
    remaining_cols = [col for col in display_df.columns if col not in visible_cols]
    return display_df[visible_cols + remaining_cols]


def _render_controlled_comparison_results(
    summary_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    *,
    grid_results_df: Optional[pd.DataFrame] = None,
    ablation_deltas_df: Optional[pd.DataFrame] = None,
    key_prefix: str,
) -> None:
    objective_metric = "roc_auc"
    objective_label = "ROC-AUC"
    if (
        not summary_df.empty
        and "objective_metric" in summary_df.columns
        and summary_df["objective_metric"].dropna().astype(str).any()
    ):
        objective_metric = str(
            summary_df["objective_metric"].dropna().astype(str).iloc[0]
        )
    elif (
        not curves_df.empty
        and "objective_metric" in curves_df.columns
        and curves_df["objective_metric"].dropna().astype(str).any()
    ):
        objective_metric = str(
            curves_df["objective_metric"].dropna().astype(str).iloc[0]
        )
    objective_label_map = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "f1": "F1",
        "balanced_f1": "Balanced F1",
        "mcc": "MCC",
        "brier_score": "Brier",
        "recall_at_alerts_per_day": "Recall@N alertas/día",
        "operational_cost": "Costo operacional",
        "net_balanced_rate": "(TP-FP)/P + (TN-FN)/N",
    }
    objective_label = objective_label_map.get(objective_metric, objective_metric.upper())

    detail_df = (
        grid_results_df.copy()
        if isinstance(grid_results_df, pd.DataFrame)
        else pd.DataFrame()
    )
    plot_source_df = detail_df.copy() if not detail_df.empty else curves_df.copy()
    if "status" in plot_source_df.columns:
        plot_source_df = plot_source_df[
            plot_source_df["status"].astype(str).str.lower() == "completed"
        ].copy()
    metric_options = _controlled_comparison_metric_options(
        plot_source_df,
        objective_label=objective_label,
    )
    selected_metric_col = (
        "val_objective_score"
        if "val_objective_score" in plot_source_df.columns
        else ("val_roc_auc" if "val_roc_auc" in plot_source_df.columns else "")
    )
    selected_metric_label = (
        f"Validación {objective_label} (objetivo)"
        if selected_metric_col == "val_objective_score"
        else "Validación ROC-AUC"
    )
    if metric_options:
        metric_labels = [label for label, _ in metric_options]
        default_label = (
            f"Validación {objective_label} (objetivo)"
            if f"Validación {objective_label} (objetivo)" in metric_labels
            else metric_labels[0]
        )
        selected_metric_label = st.selectbox(
            "Métrica a graficar",
            metric_labels,
            index=metric_labels.index(default_label),
            key=f"{key_prefix}_metric_selector",
        )
        selected_metric_col = next(
            col for label, col in metric_options if label == selected_metric_label
        )

    summary_display = summary_df.copy()
    if not summary_display.empty:
        if "selected_features" in summary_display.columns:
            summary_display["selected_features"] = summary_display[
                "selected_features"
            ].apply(_jsonish_to_text)
        if "selected_features_global" in summary_display.columns:
            summary_display["selected_features_global"] = summary_display[
                "selected_features_global"
            ].apply(_jsonish_to_text)
        if "best_params" in summary_display.columns:
            summary_display["best_params"] = summary_display["best_params"].apply(
                _jsonish_to_text
            )
        if "effective_model_params" in summary_display.columns:
            summary_display["effective_model_params"] = summary_display[
                "effective_model_params"
            ].apply(_jsonish_to_text)
        if "smote_params" in summary_display.columns:
            summary_display["smote_params"] = summary_display["smote_params"].apply(
                _jsonish_to_text
            )
        if "best_test_confusion_matrix" in summary_display.columns:
            summary_display["best_test_confusion_matrix"] = summary_display[
                "best_test_confusion_matrix"
            ].apply(_confusion_matrix_to_text)
        if {"model_name", "feature_set"}.issubset(summary_display.columns):
            sort_cols = [
                col
                for col in [
                    "model_name",
                    "feature_set",
                    "balance_mode",
                    "threshold_protocol",
                ]
                if col in summary_display.columns
            ]
            summary_display = summary_display.sort_values(sort_cols).reset_index(
                drop=True
            )
        preferred_cols = [
            "protocol_family",
            "model_name",
            "feature_set",
            "ablation_phase",
            "params_source_feature_set",
            "target_feature_set",
            "source_combo_id",
            "frozen_tuning",
            "threshold_freeze_policy",
            "balance_mode",
            "threshold_protocol",
            "threshold_protocol_label",
            "threshold_objective",
            "threshold_objective_label",
            "calibration_method",
            "objective_label",
            "objective_direction",
            "val_objective_score",
            "test_objective_score",
            "best_test_accuracy",
            "best_test_recall",
            "best_test_sensitivity",
            "best_test_roc_auc",
            "best_test_pr_auc",
            "best_test_brier_score",
            "best_test_f1_global",
            "best_test_balanced_f1",
            "best_test_f1_class_0",
            "best_test_f1_class_1",
            "best_test_f1",
            "best_test_mcc",
            "best_test_alerts_per_day",
            "best_test_false_alarms_per_day",
            "best_test_event_recall_approx",
            "best_test_operational_cost",
            "best_test_cost_per_day",
            "best_test_false_negatives",
            "best_test_false_positives",
            "best_test_true_negatives",
            "best_test_true_positives",
            "best_test_confusion_matrix",
            "k_optimo",
            "k_global",
            "effective_k",
            "selected_base_feature_count",
            "selected_cluster_feature_count",
            "smote_optimo",
            "decision_threshold",
            "feature_ranking_mode",
            "ranking_protocol",
            "selected_features_global",
            "selected_features",
            "best_params",
            "effective_model_params",
            "effective_threshold_n_jobs",
            "smote_params",
            "status",
            "error",
        ]
        visible_cols = [
            col for col in preferred_cols if col in summary_display.columns
        ]
        remaining_cols = [
            col for col in summary_display.columns if col not in visible_cols
        ]
        summary_display = summary_display[visible_cols + remaining_cols]

    st.markdown("**Tabla resumen**")
    st.dataframe(summary_display, width="stretch")

    if isinstance(ablation_deltas_df, pd.DataFrame) and not ablation_deltas_df.empty:
        deltas_display = ablation_deltas_df.copy()
        preferred_delta_cols = [
            "effect_type",
            "comparison",
            "model_name",
            "balance_mode",
            "threshold_protocol",
            "objective_metric",
            "k",
            "params_source_feature_set",
            "target_feature_set",
            "delta_val_objective_score",
            "delta_test_objective_score",
            "delta_val_roc_auc",
            "delta_test_roc_auc",
            "delta_val_pr_auc",
            "delta_test_pr_auc",
            "delta_val_brier_score",
            "delta_test_brier_score",
            "delta_val_f1",
            "delta_test_f1",
            "delta_val_mcc",
            "delta_test_mcc",
            "delta_val_recall",
            "delta_test_recall",
            "delta_val_false_positives",
            "delta_test_false_positives",
            "delta_val_false_alarms_per_day",
            "delta_test_false_alarms_per_day",
            "delta_val_cost_per_day",
            "delta_test_cost_per_day",
            "baseline_combo_id",
            "comparison_combo_id",
        ]
        visible_delta_cols = [
            col for col in preferred_delta_cols if col in deltas_display.columns
        ]
        remaining_delta_cols = [
            col for col in deltas_display.columns if col not in visible_delta_cols
        ]
        st.markdown("**Deltas de ablación cruzada**")
        st.caption(
            "Feature effect compara target Base + Cluster menos target Base con la "
            "misma fuente de tuning. Tuning effect compara params Base + Cluster "
            "menos params Base sobre el mismo target."
        )
        st.dataframe(deltas_display[visible_delta_cols + remaining_delta_cols], width="stretch")

    if not summary_df.empty and "best_test_confusion_matrix" in summary_df.columns:
        with st.expander("Matrices de confusión de test", expanded=False):
            details_df = summary_df.copy()
            if {"model_name", "feature_set"}.issubset(details_df.columns):
                details_df = details_df.sort_values(
                    ["model_name", "feature_set"]
                ).reset_index(drop=True)
            for _, row in details_df.iterrows():
                matrix = _coerce_confusion_matrix(row.get("best_test_confusion_matrix"))
                if matrix is None:
                    continue
                title = (
                    f"{row.get('model_name', '-')} | {row.get('feature_set', '-')} "
                    f"| {row.get('balance_mode', '-')} | "
                    f"{row.get('threshold_protocol', '-')} "
                    f"| K={row.get('k_optimo', '-')}"
                )
                st.markdown(f"**{title}**")
                st.dataframe(
                    pd.DataFrame(
                        matrix,
                        index=["Actual 0", "Actual 1"],
                        columns=["Pred 0", "Pred 1"],
                    ),
                    width="stretch",
                )

    if plot_source_df.empty or not selected_metric_col:
        st.info("No hay datos suficientes para construir gráficos para esta corrida.")
    else:
        st.caption(
            "Las curvas usan todas las combinaciones completadas. "
            f"Métrica seleccionada: {selected_metric_label}."
        )
        tab_none, tab_smote = st.tabs(["Sin SMOTE", "Con SMOTE"])
        tab_specs = [
            ("none", tab_none),
            ("smote", tab_smote),
        ]
        for balance_mode, tab in tab_specs:
            with tab:
                mode_df = plot_source_df[
                    plot_source_df["balance_mode"].astype(str) == str(balance_mode)
                ].copy()
                if mode_df.empty:
                    st.info("No hay resultados para este modo de balance.")
                    continue
                mode_df["k"] = pd.to_numeric(mode_df.get("k"), errors="coerce")
                mode_df[selected_metric_col] = pd.to_numeric(
                    mode_df.get(selected_metric_col),
                    errors="coerce",
                )
                mode_df = mode_df.dropna(subset=["k", selected_metric_col])
                if mode_df.empty:
                    st.info("No hay datos numéricos para esta métrica en este balance.")
                    continue
                for model_name in ["Random Forest", "SVM", "XGBoost"]:
                    st.markdown(f"**{model_name}**")
                    chart = _build_controlled_comparison_curve_chart(
                        mode_df,
                        model_name=model_name,
                        balance_mode=balance_mode,
                        metric_col=selected_metric_col,
                        metric_label=selected_metric_label,
                    )
                    if chart is not None:
                        st.altair_chart(chart, width="stretch")
                        continue
                    fallback_df = mode_df[
                        mode_df["model_name"].astype(str) == str(model_name)
                    ].copy()
                    if fallback_df.empty:
                        st.info("No hay datos para este modelo.")
                        continue
                    pivot_column = "feature_set"
                    if {
                        "params_source_feature_set",
                        "target_feature_set",
                    }.issubset(fallback_df.columns):
                        fallback_df["ablation_pair"] = (
                            fallback_df["params_source_feature_set"].astype(str)
                            + " -> "
                            + fallback_df["target_feature_set"].astype(str)
                        )
                        pivot_column = "ablation_pair"
                    pivot_df = (
                        fallback_df.pivot_table(
                            index="k",
                            columns=pivot_column,
                            values=selected_metric_col,
                            aggfunc="max",
                        )
                        .sort_index()
                    )
                    st.line_chart(pivot_df, height=260)

    detail_source_df = detail_df.copy()
    if detail_source_df.empty and not curves_df.empty:
        detail_source_df = curves_df.copy()
    detail_display = _prepare_controlled_comparison_detail_display(detail_source_df)
    st.markdown("**Tabla completa por K / modelo / balanceo**")
    if detail_display.empty:
        st.info("No hay tabla detallada disponible para esta corrida.")
        return

    filtered_df = detail_display.copy()
    filter_specs = [
        ("model_name", "Modelo"),
        ("feature_set", "Conjunto"),
        ("params_source_feature_set", "Fuente params"),
        ("target_feature_set", "Target"),
        ("balance_mode", "Balanceo"),
        ("threshold_protocol", "Protocolo"),
        ("status", "Status"),
    ]
    filter_cols = st.columns(len(filter_specs))
    for idx, (column_name, label) in enumerate(filter_specs):
        if column_name not in filtered_df.columns:
            continue
        options = sorted(
            [
                str(value)
                for value in filtered_df[column_name]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if str(value).strip()
            ]
        )
        if not options:
            continue
        selected = filter_cols[idx].multiselect(
            label,
            options=options,
            default=options,
            key=f"{key_prefix}_{column_name}_filter",
        )
        if selected:
            filtered_df = filtered_df[
                filtered_df[column_name].astype(str).isin(selected)
            ].copy()
    st.dataframe(filtered_df, width="stretch")


def _render_controlled_comparison_results_panel(
    summary_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    *,
    grid_results_df: Optional[pd.DataFrame] = None,
    ablation_deltas_df: Optional[pd.DataFrame] = None,
    key_prefix: str,
) -> None:
    _render_controlled_comparison_results(
        summary_df,
        curves_df,
        grid_results_df=grid_results_df,
        ablation_deltas_df=ablation_deltas_df,
        key_prefix=key_prefix,
    )


if hasattr(st, "fragment"):
    _render_controlled_comparison_results_panel = st.fragment(
        _render_controlled_comparison_results_panel
    )


def _load_controlled_comparison_result_frames(
    result_state: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _load_frame(path_value: object, fallback: object) -> pd.DataFrame:
        path_text = str(path_value or "").strip()
        if path_text:
            path = Path(path_text)
            if path.exists():
                try:
                    return pd.read_csv(path)
                except Exception:
                    pass
        if isinstance(fallback, pd.DataFrame):
            return fallback
        return pd.DataFrame()

    summary_df = _load_frame(
        result_state.get("summary_path"),
        result_state.get("summary_df"),
    )
    curves_df = _load_frame(
        result_state.get("curves_path"),
        result_state.get("curves_df"),
    )
    grid_results_df = _load_frame(
        result_state.get("detail_path"),
        result_state.get("grid_results_df"),
    )
    ablation_deltas_df = _load_frame(
        result_state.get("ablation_deltas_path"),
        result_state.get("ablation_deltas_df"),
    )
    return summary_df, curves_df, grid_results_df, ablation_deltas_df


def _render_controlled_comparison_current_result(
    *,
    checkpoint_root: Path,
) -> None:
    result_state = st.session_state.get("exp_controlled_last_results")
    if not isinstance(result_state, dict) or not result_state:
        return

    (
        summary_df,
        curves_df,
        grid_results_df,
        ablation_deltas_df,
    ) = _load_controlled_comparison_result_frames(result_state)

    run_id = result_state.get("run_id")
    st.success(f"Comparación controlada finalizada. Run ID: {run_id}.")
    if result_state.get("loaded_from_checkpoint"):
        st.caption("Se cargó un checkpoint ya completado.")
    elif result_state.get("auto_resumed"):
        st.caption("La corrida se reanudó desde un checkpoint compatible.")
    st.caption(
        f"Checkpoint: {result_state.get('checkpoint_run_dir') or checkpoint_root}"
    )
    summary_name = result_state.get("summary_name") or "-"
    curves_name = result_state.get("curves_name") or "-"
    detail_name = result_state.get("detail_name") or "-"
    ablation_deltas_name = result_state.get("ablation_deltas_name") or ""
    st.caption(
        "Resumen: "
        f"{summary_name} | "
        f"Curvas: {curves_name} | "
        f"Detalle: {detail_name}"
        + (f" | Deltas: {ablation_deltas_name}" if ablation_deltas_name else "")
    )
    _render_controlled_comparison_results_panel(
        summary_df,
        curves_df,
        grid_results_df=grid_results_df,
        ablation_deltas_df=ablation_deltas_df,
        key_prefix=f"controlled_current_{run_id}",
    )


def _render_calibration_sweep_experiment() -> None:
    st.subheader("Calibración score + threshold")
    st.caption(
        "Barre, sobre un solo modelo, la combinación entre métrica objetivo de "
        "Optuna, calibración del score, threshold operacional y balance mode. "
        "El protocolo queda fijo en Robusto y el ranking oficial se arma con "
        "métricas de validación."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No hay archivos de eventos (accidents) en Datos.")
        return
    feature_files = _list_flow_feature_files()
    if not feature_files:
        st.warning("No hay archivos de features en Resultados.")
        return

    event_names = [p.name for p in event_files]
    feature_names = [p.name for p in feature_files]
    selected_event = st.selectbox(
        "Archivo de Eventos",
        event_names,
        key="exp_calibration_sweep_event_file",
    )
    selected_features_name = st.selectbox(
        "Archivo de Features",
        feature_names,
        key="exp_calibration_sweep_feature_file",
    )

    selected_event_path = next(
        (p for p in event_files if p.name == selected_event),
        None,
    )
    selected_features_path = next(
        (p for p in feature_files if p.name == selected_features_name),
        None,
    )
    if selected_event_path is None or selected_features_path is None:
        st.error("No se pudieron resolver los archivos seleccionados.")
        return

    dataset_date_start, dataset_date_end, dataset_date_valid = (
        _render_controlled_feature_date_range_inputs(
            selected_features_path,
            key_prefix="exp_calibration_sweep",
        )
    )
    if not dataset_date_valid:
        return

    try:
        schema_df = _inspect_controlled_feature_schema(selected_features_path)
    except Exception as exc:
        st.error(f"No se pudo inspeccionar el archivo de features: {exc}")
        return

    if not _get_cluster_cols(schema_df):
        st.error(
            "El archivo seleccionado no contiene variables de cluster. "
            "Este experimento requiere el mismo dataset enriquecido que Comparación controlada."
        )
        return

    accidents_df_for_tramo = _load_accidents_for_event(selected_event_path)
    allowed_porticos = _load_porticos_from_feature_file(selected_features_path)
    tramo_tuple = _build_tramo_selector(
        accidents_df_for_tramo,
        date_start=dataset_date_start,
        date_end=dataset_date_end,
        allowed_porticos=allowed_porticos,
        key="exp_calibration_sweep_tramo_choice",
    )
    if not tramo_tuple:
        st.info("Seleccione un tramo específico para ejecutar el experimento.")
        current_payload = st.session_state.get("calibration_sweep_last_payload")
        if isinstance(current_payload, dict):
            _render_calibration_sweep_results(
                current_payload,
                key_prefix="calibration_sweep_current",
            )
        return

    eje, calzada, p_start, p_end = tramo_tuple
    segment_info = {
        "eje": eje,
        "calzada": calzada,
        "portico_inicio": p_start,
        "portico_fin": p_end,
        "segment_label": f"{eje} | {calzada} | {p_start} -> {p_end}",
    }

    st.markdown("**Configuración general**")
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
            key="exp_calibration_sweep_random_state",
        )
    with cfg2:
        n_trials = st.number_input(
            "Optuna trials",
            min_value=1,
            value=25,
            step=1,
            key="exp_calibration_sweep_n_trials",
        )
    with cfg3:
        timeout = st.number_input(
            "Optuna timeout (seg)",
            min_value=1,
            value=1800,
            step=10,
            key="exp_calibration_sweep_timeout",
        )
    with cfg4:
        model_choice = st.selectbox(
            "Modelo",
            list(CONTROLLED_COMPARISON_MODELS),
            key="exp_calibration_sweep_model_choice",
        )

    source_options = {
        "Feature selection": "feature_selection",
        "Optuna (best_feature_cols)": "optuna",
    }
    reverse_source = {value: label for label, value in source_options.items()}
    current_source = str(
        st.session_state.get("calibration_sweep_feature_source", "feature_selection")
    )
    chosen_source_label = st.radio(
        "Origen de variables",
        list(source_options.keys()),
        index=list(source_options.keys()).index(
            reverse_source.get(current_source, "Feature selection")
        ),
        horizontal=True,
        key="exp_calibration_sweep_feature_source_radio",
        help=(
            "Feature selection rankea las variables numéricas y toma un K fijo. "
            "Optuna optimiza top_k en cada trial y guarda best_feature_cols."
        ),
    )
    st.session_state["calibration_sweep_feature_source"] = source_options[
        chosen_source_label
    ]
    calibration_feature_source = source_options[chosen_source_label]
    calibration_candidate_feature_count = len(_get_feature_cols(schema_df))
    if calibration_feature_source == "optuna":
        topk_col1, topk_col2, topk_col3 = st.columns(3)
        with topk_col1:
            calibration_k_min = st.number_input(
                "k_min",
                min_value=1,
                value=int(st.session_state.get("exp_calibration_sweep_k_min", 10)),
                step=1,
                key="exp_calibration_sweep_k_min",
            )
        with topk_col2:
            calibration_k_max = st.number_input(
                "k_max",
                min_value=1,
                value=int(st.session_state.get("exp_calibration_sweep_k_max", 100)),
                step=1,
                key="exp_calibration_sweep_k_max",
            )
        with topk_col3:
            calibration_k_step = st.number_input(
                "k_step",
                min_value=1,
                value=int(st.session_state.get("exp_calibration_sweep_k_step", 10)),
                step=1,
                key="exp_calibration_sweep_k_step",
            )
        calibration_top_k_grid = _k_grid_values(
            k_min=int(calibration_k_min),
            k_max=int(calibration_k_max),
            k_step=int(calibration_k_step),
            feature_count=int(calibration_candidate_feature_count),
        )
        if calibration_top_k_grid:
            st.caption(
                "Optuna explorará top_k="
                f"{calibration_top_k_grid} "
                f"(recortado a {calibration_candidate_feature_count} variables disponibles)."
            )
        calibration_feature_k_config = {
            "mode": "optuna_top_k",
            "k_min": int(calibration_k_min),
            "k_max": int(calibration_k_max),
            "k_step": int(calibration_k_step),
            "ranking_method": "rf",
        }
    else:
        calibration_fixed_k = st.number_input(
            "K features",
            min_value=1,
            value=int(st.session_state.get("exp_calibration_sweep_fixed_k", 20)),
            step=1,
            key="exp_calibration_sweep_fixed_k",
        )
        effective_fixed_k = (
            max(1, min(int(calibration_fixed_k), int(calibration_candidate_feature_count)))
            if calibration_candidate_feature_count > 0
            else int(calibration_fixed_k)
        )
        st.caption(
            f"Feature selection rankeará una vez y usará las top {effective_fixed_k} "
            f"de {calibration_candidate_feature_count} variables disponibles."
        )
        calibration_feature_k_config = {
            "mode": "fixed_top_k",
            "k": int(calibration_fixed_k),
            "ranking_method": "rf",
        }

    objective_mode_options = {
        "Multiobjetivo Pareto": CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
        "Escalar legacy": CALIBRATION_SWEEP_OBJECTIVE_MODE_SCALAR,
    }
    current_objective_mode = str(
        st.session_state.get(
            "exp_calibration_sweep_objective_mode",
            CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE,
        )
    )
    reverse_objective_mode = {
        value: label for label, value in objective_mode_options.items()
    }
    objective_mode_label = st.selectbox(
        "Modo objetivo Optuna",
        list(objective_mode_options.keys()),
        index=list(objective_mode_options.keys()).index(
            reverse_objective_mode.get(current_objective_mode, "Multiobjetivo Pareto")
        ),
        key="exp_calibration_sweep_objective_mode_label",
        help=(
            "Multiobjetivo Pareto optimiza MCC, PR-AUC, Brier y Recall@N alertas/día "
            "en un único estudio Optuna. Escalar legacy mantiene una métrica por estudio."
        ),
    )
    optuna_objective_mode = objective_mode_options[objective_mode_label]
    st.session_state["exp_calibration_sweep_objective_mode"] = optuna_objective_mode
    if optuna_objective_mode == CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE:
        objective_options = {
            CALIBRATION_SWEEP_MULTIOBJECTIVE_LABEL: CALIBRATION_SWEEP_MULTIOBJECTIVE_KEY
        }
        selected_objective_labels = [CALIBRATION_SWEEP_MULTIOBJECTIVE_LABEL]
        st.caption(
            "Objetivos fijos: MCC ↑, PR-AUC ↑, Brier ↓ y Recall@N alertas/día ↑. "
            "FAR se usa como gate operacional y penalización del proxy de pruning."
        )
    else:
        advanced_objectives = st.checkbox(
            "Mostrar catálogo avanzado de métricas objetivo de Optuna",
            value=bool(
                st.session_state.get(
                    "exp_calibration_sweep_show_advanced_objectives",
                    False,
                )
            ),
            key="exp_calibration_sweep_show_advanced_objectives",
        )
        objective_options = _calibration_sweep_optuna_objective_options(
            include_advanced=bool(advanced_objectives)
        )
        default_objective_keys = {
            "pr_auc",
            "mcc",
            "brier_score",
            "balanced_f1",
            "recall_at_alerts_per_day",
            "operational_cost",
            "far_sens",
        }
        default_objective_labels = [
            label
            for label, key in objective_options.items()
            if key in default_objective_keys
        ]
        selected_objective_labels = st.multiselect(
            "Métricas objetivo de Optuna",
            list(objective_options.keys()),
            default=default_objective_labels,
            key="exp_calibration_sweep_objectives",
            help=(
                "Optuna sigue siendo escalar por trial. Luego el ranking multiobjetivo "
                "se arma sobre las combinaciones finalistas usando métricas de validación."
            ),
        )

    calibration_methods = _calibration_method_multiselect(
        "Calibración del score",
        key="exp_calibration_sweep_calibration_methods",
        default_methods=["sigmoid", "isotonic", "none"],
        methods=["sigmoid", "isotonic", "none"],
    )

    threshold_objective_options = _calibration_sweep_threshold_objective_options()
    selected_threshold_labels = st.multiselect(
        "Objetivos operacionales de threshold",
        list(threshold_objective_options.keys()),
        default=list(threshold_objective_options.keys()),
        key="exp_calibration_sweep_threshold_objectives",
        help=(
            "Solo se incluyen objetivos operacionales. PR-AUC y ROC-AUC quedan "
            "fuera porque hoy no calibran threshold operativo."
        ),
    )
    selected_threshold_objectives = [
        threshold_objective_options[label]
        for label in selected_threshold_labels
        if label in threshold_objective_options
    ]

    st.caption("Threshold protocol fijo: Robusto.")
    threshold_visibility = _combined_threshold_field_visibility(
        selected_threshold_objectives
    )
    thr1, thr2, thr3, thr4 = st.columns(4)
    with thr1:
        far_target = float(
            _render_conditional_slider(
                "FAR target",
                visible=threshold_visibility["far_target"],
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="exp_calibration_sweep_far_target",
            )
        )
    with thr2:
        alerts_per_day = float(
            _render_conditional_number_input(
                "Alertas máximas por día",
                visible=threshold_visibility["alerts_per_day"],
                min_value=0.1,
                value=5.0,
                step=0.5,
                key="exp_calibration_sweep_alerts_per_day",
            )
        )
    with thr3:
        fn_cost = float(
            _render_conditional_number_input(
                "Costo FN",
                visible=threshold_visibility["fn_cost"],
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="exp_calibration_sweep_fn_cost",
            )
        )
    with thr4:
        fp_cost = float(
            _render_conditional_number_input(
                "Costo FP",
                visible=threshold_visibility["fp_cost"],
                min_value=0.0,
                value=1.0,
                step=0.5,
                key="exp_calibration_sweep_fp_cost",
            )
        )

    st.markdown("**Split y paralelización**")
    split1, split2, split3, split4 = st.columns(4)
    with split1:
        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="exp_calibration_sweep_test_size",
        )
    with split2:
        val_size = st.slider(
            "Validation size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="exp_calibration_sweep_val_size",
        )
    with split3:
        robust_folds = st.number_input(
            "Folds robustos",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="exp_calibration_sweep_robust_folds",
        )
    with split4:
        parallel_jobs = st.number_input(
            "Jobs paralelos RF/ranking",
            min_value=1,
            max_value=_max_optuna_parallel_jobs(),
            value=min(4, _max_optuna_parallel_jobs()),
            step=1,
            key="exp_calibration_sweep_parallel_jobs",
        )
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        optuna_n_jobs = _render_optuna_n_jobs_input(
            "Optuna jobs paralelos",
            key="exp_calibration_sweep_optuna_n_jobs",
            default=1,
        )
    with opt_col2:
        xgb_parallel_jobs = _render_model_n_jobs_input(
            "Jobs paralelos XGBoost",
            key="exp_calibration_sweep_xgb_parallel_jobs",
            default=1,
            shared_key="global_xgb_parallel_jobs",
        )

    with st.expander("Configuración avanzada de Optuna", expanded=False):
        pr_col1, pr_col2, pr_col3 = st.columns(3)
        with pr_col1:
            pruning_enabled = st.checkbox(
                "Activar pruning",
                value=True,
                key="exp_calibration_sweep_pruning_enabled",
                help=(
                    "Usa MedianPruner con reportes intermedios por trial para cortar "
                    "configuraciones claramente inferiores."
                ),
            )
        with pr_col2:
            pruning_startup_trials = st.number_input(
                "Pruning: startup trials",
                min_value=0,
                value=5,
                step=1,
                key="exp_calibration_sweep_pruning_startup_trials",
                disabled=not pruning_enabled,
                help="Cantidad de trials completos antes de permitir podado.",
            )
        with pr_col3:
            pruning_warmup_steps = st.number_input(
                "Pruning: warmup steps",
                min_value=0,
                value=1,
                step=1,
                key="exp_calibration_sweep_pruning_warmup_steps",
                disabled=not pruning_enabled,
                help="Cantidad de reportes intermedios antes de evaluar podado.",
            )
        pr_col4, pr_col5, pr_col6 = st.columns(3)
        with pr_col4:
            pruning_interval_steps = st.number_input(
                "Pruning: interval steps",
                min_value=1,
                value=1,
                step=1,
                key="exp_calibration_sweep_pruning_interval_steps",
                disabled=not pruning_enabled,
            )
        with pr_col5:
            pruning_intermediate_steps = st.number_input(
                "Reportes proxy por trial",
                min_value=0,
                max_value=4,
                value=2,
                step=1,
                key="exp_calibration_sweep_pruning_intermediate_steps",
                disabled=not pruning_enabled,
                help=(
                    "Entrena modelos proxy con menos filas/estimadores antes del fit "
                    "completo. Valores altos podan antes, pero agregan overhead."
                ),
            )
        with pr_col6:
            warm_start_enabled = st.checkbox(
                "Warm starts TPE",
                value=True,
                key="exp_calibration_sweep_warm_start_enabled",
                help="Encola 2 configuraciones base compatibles antes del muestreo TPE.",
            )
    optuna_pruning_config = {
        "enabled": bool(pruning_enabled),
        "type": "median",
        "n_startup_trials": int(pruning_startup_trials),
        "n_warmup_steps": int(pruning_warmup_steps),
        "interval_steps": int(pruning_interval_steps),
        "intermediate_steps": int(pruning_intermediate_steps),
        "warm_start": bool(warm_start_enabled),
    }

    search_space = _default_controlled_comparison_search_space()
    with st.expander("Rangos del barrido", expanded=False):
        st.json(search_space)

    selected_objective_metrics = [
        objective_options[label]
        for label in selected_objective_labels
        if label in objective_options
    ]
    checkpoint_root = RESULTS_DIR / "calibration_experiment_runs"
    checkpoint_entries = _list_calibration_sweep_checkpoints(
        checkpoint_root=checkpoint_root
    )
    checkpoint_entries_by_label = {
        str(entry.get("label") or entry.get("run_id") or ""): entry
        for entry in checkpoint_entries
    }
    selected_checkpoint_entry: Optional[Dict[str, object]] = None
    checkpoint_execution_mode = "Nuevo run"
    if checkpoint_entries_by_label:
        st.markdown("**Checkpoints guardados**")
        st.caption(
            "La selección es explícita: modificar los parámetros del formulario no "
            "busca checkpoints compatibles automáticamente."
        )
        checkpoint_selector_key = "exp_calibration_sweep_checkpoint_selector"
        checkpoint_selector_options = ["(sin checkpoint)"] + list(
            checkpoint_entries_by_label.keys()
        )
        selected_checkpoint_label = st.selectbox(
            "Checkpoint disponible",
            checkpoint_selector_options,
            key=checkpoint_selector_key,
            help=(
                "Permite cargar resultados previos para analizarlos y, si el "
                "formulario coincide con la corrida, reanudar o reiniciar ese checkpoint."
            ),
        )
        if selected_checkpoint_label != "(sin checkpoint)":
            selected_checkpoint_entry = checkpoint_entries_by_label.get(
                selected_checkpoint_label
            )
        if selected_checkpoint_entry is not None:
            progress_text = (
                f"{int(selected_checkpoint_entry.get('completed_steps') or 0)}/"
                f"{int(selected_checkpoint_entry.get('total_steps') or 0)}"
            )
            st.caption(
                f"Run ID: {selected_checkpoint_entry.get('run_id')} | "
                f"Estado: {selected_checkpoint_entry.get('status_label')} | "
                f"Actualizado: {selected_checkpoint_entry.get('updated_at') or '-'} | "
                f"Progreso: {progress_text}"
            )
            if st.button(
                "Cargar checkpoint seleccionado",
                key="exp_calibration_sweep_load_selected_checkpoint",
            ):
                st.session_state["calibration_sweep_last_payload"] = (
                    _calibration_sweep_result_state_from_run_dir(
                        Path(str(selected_checkpoint_entry.get("run_dir") or "")),
                        loaded_from_selection=True,
                    )
                )

            if str(selected_checkpoint_entry.get("status") or "").lower() == "completed":
                checkpoint_mode_options = [
                    "Nuevo run",
                    "Reiniciar checkpoint seleccionado",
                ]
            else:
                checkpoint_mode_options = [
                    "Nuevo run",
                    "Reanudar checkpoint seleccionado",
                    "Reiniciar checkpoint seleccionado",
                ]
            checkpoint_execution_key = (
                "exp_calibration_sweep_checkpoint_execution_mode"
            )
            if (
                st.session_state.get(checkpoint_execution_key)
                not in checkpoint_mode_options
            ):
                st.session_state[checkpoint_execution_key] = checkpoint_mode_options[0]
            checkpoint_execution_mode = st.radio(
                "Uso al ejecutar",
                checkpoint_mode_options,
                horizontal=True,
                key=checkpoint_execution_key,
            )

    if st.button(
        "Ejecutar experimento de calibración",
        key="exp_calibration_sweep_run",
    ):
        if not selected_objective_labels:
            st.error("Seleccione al menos una métrica objetivo de Optuna.")
            return
        if not calibration_methods:
            st.error("Seleccione al menos un método de calibración.")
            return
        if not selected_threshold_objectives:
            st.error("Seleccione al menos un objetivo operacional de threshold.")
            return
        if calibration_feature_source == "optuna":
            if int(calibration_k_min) > int(calibration_k_max):
                st.error("k_min no puede ser mayor que k_max.")
                return
            if int(calibration_k_step) < 1:
                st.error("k_step debe ser mayor o igual a 1.")
                return
        else:
            if int(calibration_fixed_k) < 1:
                st.error("K features debe ser mayor o igual a 1.")
                return

        try:
            base_df = _prepare_controlled_comparison_base_df(
                accidents_df_for_tramo=accidents_df_for_tramo,
                selected_features_path=selected_features_path,
                tramo_tuple=tramo_tuple,
                date_start=dataset_date_start,
                date_end=dataset_date_end,
            )
            features_df = _load_controlled_features_df(
                selected_features_path,
                tramo_tuple,
                date_start=dataset_date_start,
                date_end=dataset_date_end,
            )
        except Exception as exc:
            st.error(f"No se pudo preparar el dataset base: {exc}")
            return

        feature_resolution = _resolve_calibration_sweep_feature_selection(
            dataset_df=base_df,
            model_choice=str(model_choice),
            features_df=features_df,
            features_path=selected_features_path,
            features_source=selected_features_path.name,
        )
        feature_cols = list(feature_resolution.get("feature_cols") or [])
        if not feature_cols:
            st.error("No hay variables disponibles para ejecutar el experimento.")
            return

        selected_objective_metrics = [
            objective_options[label]
            for label in selected_objective_labels
            if label in objective_options
        ]
        total_combinations = (
            len(selected_objective_metrics)
            * len(calibration_methods)
            * len(selected_threshold_objectives)
            * 2
        )
        progress_bar = st.progress(
            0,
            text=(
                "Iniciando experimento de calibración... "
                f"0/{total_combinations} combinaciones"
            ),
        )
        progress_stats = st.empty()
        progress_last_combo = st.empty()
        progress_state = {
            "started_at": time.monotonic(),
            "processed": 0,
            "completed": 0,
            "failed": 0,
            "trial_completed": 0,
            "trial_pruned": 0,
            "trial_failed": 0,
            "trial_total": 0,
        }

        def _update_calibration_progress(
            payload_item: Optional[Dict[str, object]] = None,
        ) -> None:
            processed = int(progress_state["processed"])
            completed_ok = int(progress_state["completed"])
            failed = int(progress_state["failed"])
            trial_completed = int(progress_state["trial_completed"])
            trial_pruned = int(progress_state["trial_pruned"])
            trial_failed = int(progress_state["trial_failed"])
            trial_total = int(progress_state["trial_total"])
            pruning_rate = trial_pruned / max(1, trial_total)
            elapsed = time.monotonic() - float(progress_state["started_at"])
            avg_seconds = (
                elapsed / processed
                if processed > 0
                else None
            )
            remaining = max(0, int(total_combinations) - processed)
            eta_seconds = (
                remaining * avg_seconds
                if avg_seconds is not None
                else None
            )
            progress_pct = (
                int(round((processed / max(1, total_combinations)) * 100))
                if total_combinations > 0
                else 100
            )
            progress_bar.progress(
                min(100, max(0, progress_pct)),
                text=(
                    "Experimento de calibración en progreso... "
                    f"{processed}/{total_combinations} combinaciones"
                ),
            )
            progress_stats.caption(
                "Avance: "
                f"{processed}/{total_combinations} | "
                f"OK: {completed_ok} | "
                f"Fallidas: {failed} | "
                f"Trials OK: {trial_completed} | "
                f"Podados: {trial_pruned} | "
                f"Trials fallidos: {trial_failed} | "
                f"Tasa podado: {pruning_rate:.1%} | "
                f"Tiempo transcurrido: {_format_duration_compact(elapsed)} | "
                f"Promedio actual: {_format_duration_compact(avg_seconds)} por combinación | "
                f"ETA: {_format_duration_compact(eta_seconds)}"
            )
            if isinstance(payload_item, dict):
                combo_label = (
                    f"{payload_item.get('optuna_objective_metric') or payload_item.get('objective_metric')} | "
                    f"{payload_item.get('calibration_method')} | "
                    f"{payload_item.get('threshold_objective')} | "
                    f"{payload_item.get('balance_mode')}"
                )
                status_label = str(payload_item.get("status") or "").strip() or "-"
                progress_last_combo.caption(
                    f"Última combinación: {combo_label} | estado={status_label}"
                )

        def _progress_callback(payload_item: Dict[str, object]) -> None:
            progress_state["processed"] = int(progress_state["processed"]) + 1
            if str(payload_item.get("status") or "").strip().lower() == "completed":
                progress_state["completed"] = int(progress_state["completed"]) + 1
            else:
                progress_state["failed"] = int(progress_state["failed"]) + 1
            progress_state["trial_completed"] = int(
                progress_state["trial_completed"]
            ) + int(payload_item.get("optuna_trials_completed") or 0)
            progress_state["trial_pruned"] = int(
                progress_state["trial_pruned"]
            ) + int(payload_item.get("optuna_trials_pruned") or 0)
            progress_state["trial_failed"] = int(
                progress_state["trial_failed"]
            ) + int(payload_item.get("optuna_trials_failed") or 0)
            progress_state["trial_total"] = int(
                progress_state["trial_total"]
            ) + int(payload_item.get("optuna_trials_total") or 0)
            _update_calibration_progress(payload_item)

        _update_calibration_progress()
        runner = ExperimentsRunner(random_state=int(random_state))
        selected_checkpoint_run_id = None
        restart_selected_checkpoint = False
        if selected_checkpoint_entry is not None:
            if checkpoint_execution_mode == "Reanudar checkpoint seleccionado":
                selected_checkpoint_run_id_text = str(
                    selected_checkpoint_entry.get("run_id") or ""
                ).strip()
                selected_checkpoint_run_id = (
                    selected_checkpoint_run_id_text or None
                )
            elif checkpoint_execution_mode == "Reiniciar checkpoint seleccionado":
                selected_checkpoint_run_id_text = str(
                    selected_checkpoint_entry.get("run_id") or ""
                ).strip()
                selected_checkpoint_run_id = (
                    selected_checkpoint_run_id_text or None
                )
                restart_selected_checkpoint = selected_checkpoint_run_id is not None
        with st.spinner("Ejecutando barrido de calibración..."):
            try:
                payload = runner.run_calibration_sweep(
                    base_df,
                    model_name=str(model_choice),
                    selected_features=feature_cols,
                    objective_metrics=selected_objective_metrics,
                    calibration_methods=list(calibration_methods),
                    threshold_objectives=list(selected_threshold_objectives),
                    event_path=selected_event_path,
                    features_path=selected_features_path,
                    segment_info=segment_info,
                    dataset_date_start=dataset_date_start,
                    dataset_date_end=dataset_date_end,
                    feature_source=str(feature_resolution.get("feature_source") or ""),
                    test_size=float(test_size),
                    val_size=float(val_size),
                    n_trials=int(n_trials),
                    timeout=int(timeout),
                    optuna_n_jobs=int(optuna_n_jobs),
                    parallel_jobs=int(parallel_jobs),
                    xgb_parallel_jobs=int(xgb_parallel_jobs),
                    search_space_config=search_space,
                    far_target=float(far_target),
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                    robust_folds=int(robust_folds),
                    optuna_pruning_config=optuna_pruning_config,
                    feature_k_config=dict(calibration_feature_k_config),
                    optuna_objective_mode=str(optuna_objective_mode),
                    checkpoint_root=checkpoint_root,
                    auto_resume=False,
                    start_fresh=bool(restart_selected_checkpoint),
                    checkpoint_run_id_override=selected_checkpoint_run_id,
                    result_callback=_progress_callback,
                )
            except Exception as exc:
                _update_calibration_progress()
                st.error(f"El experimento falló: {exc}")
                return

        if bool(payload.get("loaded_from_checkpoint")):
            checkpoint_grid = payload.get("grid_results_df")
            if isinstance(checkpoint_grid, pd.DataFrame):
                completed_rows = int(
                    checkpoint_grid.get("status", pd.Series(dtype=str))
                    .astype(str)
                    .str.lower()
                    .eq("completed")
                    .sum()
                )
                failed_rows = int(
                    checkpoint_grid.get("status", pd.Series(dtype=str))
                    .astype(str)
                    .str.lower()
                    .eq("failed")
                    .sum()
                )
                progress_state["processed"] = int(len(checkpoint_grid))
                progress_state["completed"] = completed_rows
                progress_state["failed"] = failed_rows

        progress_bar.progress(
            100,
            text=(
                "Experimento de calibración finalizado. "
                f"{int(progress_state['processed'])}/{total_combinations} combinaciones"
            ),
        )
        _update_calibration_progress()
        payload["feature_source_note"] = feature_resolution.get("source_note")
        st.session_state["calibration_sweep_last_payload"] = payload

    current_payload = st.session_state.get("calibration_sweep_last_payload")
    if isinstance(current_payload, dict):
        _render_calibration_sweep_results(
            current_payload,
            key_prefix="calibration_sweep_current",
        )


def _render_controlled_comparison_memory_estimator(
    *,
    accidents_df_for_tramo: Optional[pd.DataFrame],
    selected_event_path: Path,
    selected_features_path: Path,
    tramo_tuple: Tuple[str, str, str, str],
    segment_info: Dict[str, object],
    dataset_date_start: Optional[pd.Timestamp],
    dataset_date_end: Optional[pd.Timestamp],
    test_size: float,
    val_size: float,
    k_min: int,
    k_max: int,
    k_step: int,
    xgb_parallel_jobs: int,
    selected_models: List[str],
    search_space: Dict[str, object],
) -> None:
    snapshot = _system_memory_snapshot()
    default_budget_gb = _default_memory_budget_gb(snapshot)
    total_bytes = snapshot.get("total_bytes")
    available_bytes = snapshot.get("available_bytes")
    max_budget_gb = (
        max(default_budget_gb, round(float(total_bytes) / float(1024 ** 3), 1))
        if total_bytes
        else max(64.0, default_budget_gb)
    )

    signature_payload = {
        "event_path": str(selected_event_path),
        "features_path": str(selected_features_path),
        "segment_info": dict(segment_info or {}),
        "dataset_date_start": (
            None if dataset_date_start is None else str(pd.Timestamp(dataset_date_start))
        ),
        "dataset_date_end": (
            None if dataset_date_end is None else str(pd.Timestamp(dataset_date_end))
        ),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "k_step": int(k_step),
        "selected_models": list(selected_models),
        "search_space": search_space,
        "memory_budget_gb": float(
            st.session_state.get("exp_controlled_memory_budget_gb", default_budget_gb)
        ),
    }

    notice = st.session_state.pop("exp_controlled_memory_notice", None)
    with st.expander("Calcular jobs seguros por memoria", expanded=False):
        if notice:
            st.success(str(notice))
        if total_bytes and available_bytes:
            st.caption(
                "RAM detectada: "
                f"disponible {_format_bytes(available_bytes)} / "
                f"total {_format_bytes(total_bytes)}."
            )
        elif total_bytes:
            st.caption(f"RAM total detectada: {_format_bytes(total_bytes)}.")
        else:
            st.caption(
                "No se pudo detectar la RAM del equipo. Ingrese manualmente el presupuesto."
            )

        memory_budget_gb = st.number_input(
            "Presupuesto máximo de RAM (GB)",
            min_value=0.5,
            max_value=float(max_budget_gb),
            value=float(
                st.session_state.get(
                    "exp_controlled_memory_budget_gb",
                    default_budget_gb,
                )
            ),
            step=0.5,
            key="exp_controlled_memory_budget_gb",
            help=(
                "Techo de RAM que el experimento puede consumir. "
                "Se usa para sugerir combinaciones seguras de "
                "`Jobs paralelos RF/ranking`, `Optuna jobs paralelos` "
                "y `Jobs paralelos XGBoost`."
            ),
        )
        st.caption(
            "El cálculo usa el dataset real del tramo, el mayor K evaluable y el peor caso "
            "de SMOTE/modelo dentro de la grilla actual."
        )

        signature_payload["memory_budget_gb"] = float(memory_budget_gb)
        estimate_signature = hashlib.sha1(
            json.dumps(signature_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        estimate_payload = None
        if (
            st.session_state.get("exp_controlled_memory_estimate_signature")
            == estimate_signature
        ):
            cached_payload = st.session_state.get("exp_controlled_memory_estimate")
            if isinstance(cached_payload, dict):
                estimate_payload = cached_payload

        if st.button(
            "Calcular jobs máximos seguros",
            key="exp_controlled_memory_estimate_run",
        ):
            try:
                with st.spinner("Estimando memoria para la comparación controlada..."):
                    base_df = _prepare_controlled_comparison_base_df(
                        accidents_df_for_tramo=accidents_df_for_tramo,
                        selected_features_path=selected_features_path,
                        tramo_tuple=tramo_tuple,
                        date_start=dataset_date_start,
                        date_end=dataset_date_end,
                    )
                    estimate_payload = estimate_controlled_comparison_parallelism(
                        base_df,
                        test_size=float(test_size),
                        val_size=float(val_size),
                        k_min=int(k_min),
                        k_max=int(k_max),
                        k_step=int(k_step),
                        search_space_config=search_space,
                        memory_budget_bytes=int(float(memory_budget_gb) * (1024 ** 3)),
                        xgb_parallel_jobs=int(xgb_parallel_jobs),
                        selected_models=list(selected_models),
                    )
                st.session_state["exp_controlled_memory_estimate"] = estimate_payload
                st.session_state[
                    "exp_controlled_memory_estimate_signature"
                ] = estimate_signature
            except Exception as exc:
                estimate_payload = None
                st.error(f"No se pudo calcular la frontera de memoria: {exc}")

        if not isinstance(estimate_payload, dict):
            st.info(
                "Ejecuta este cálculo para obtener: "
                "`max RF/ranking con Optuna=1 y XGBoost=1`, "
                "`max Optuna con RF=1 y XGBoost=1`, "
                "`max XGBoost con RF=1 y Optuna=1` y una frontera segura combinada."
            )
            return

        safe_frontier_df = estimate_payload.get("safe_frontier_df")
        frontier_df = estimate_payload.get("frontier_df")
        if not isinstance(safe_frontier_df, pd.DataFrame):
            safe_frontier_df = pd.DataFrame()
        if not isinstance(frontier_df, pd.DataFrame):
            frontier_df = pd.DataFrame()

        recommended_pair = (
            estimate_payload.get("recommended_pair")
            if isinstance(estimate_payload.get("recommended_pair"), dict)
            else None
        )
        max_parallel_jobs = int(
            estimate_payload.get("max_parallel_jobs_when_optuna_1") or 0
        )
        max_optuna_jobs = int(
            estimate_payload.get("max_optuna_jobs_when_parallel_1") or 0
        )
        max_xgb_jobs = int(
            estimate_payload.get("max_xgb_parallel_jobs_when_parallel_1_optuna_1")
            or 0
        )
        safe_pairs = int(len(safe_frontier_df))
        xgb_selected = "XGBoost" in selected_models

        kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)
        kpi_1.metric(
            "Max RF/ranking",
            str(max_parallel_jobs) if max_parallel_jobs > 0 else "0",
            "con Optuna=1, XGBoost=1",
        )
        kpi_2.metric(
            "Max Optuna",
            str(max_optuna_jobs) if max_optuna_jobs > 0 else "0",
            "con RF=1, XGBoost=1",
        )
        kpi_3.metric(
            "Max XGBoost",
            (
                str(max_xgb_jobs)
                if xgb_selected and max_xgb_jobs > 0
                else ("0" if xgb_selected else "N/A")
            ),
            "con RF=1, Optuna=1",
        )
        if recommended_pair:
            kpi_4.metric(
                "Config. segura",
                (
                    f"{int(recommended_pair['parallel_jobs'])} / "
                    f"{int(recommended_pair['optuna_n_jobs'])} / "
                    f"{int(recommended_pair['xgb_parallel_jobs'])}"
                ),
                str(recommended_pair.get("dominant_model") or "-"),
            )
        else:
            kpi_4.metric("Config. segura", "N/A")
        kpi_5.metric(
            "Presupuesto",
            f"{float(memory_budget_gb):.1f} GB",
            f"{safe_pairs} configuraciones seguras",
        )

        st.caption(
            "Lectura correcta: los máximos `RF/ranking`, `Optuna` y `XGBoost` son "
            "límites independientes con los otros ejes en 1. Si quieres aplicarlos a la vez, "
            "usa la `Config. segura` o la tabla de frontera."
        )

        if safe_frontier_df.empty:
            st.warning(
                "Con el presupuesto actual no se encontró una configuración segura estimada. "
                "Reduce `K`, acota la grilla o aumenta el presupuesto de RAM."
            )

        render_df = safe_frontier_df.copy() if not safe_frontier_df.empty else frontier_df.copy()
        if not render_df.empty:
            render_df = render_df.sort_values(
                ["parallel_jobs", "xgb_parallel_jobs", "max_optuna_jobs"]
            ).reset_index(drop=True)
            render_df["peak_gb"] = (
                pd.to_numeric(
                    render_df.get("optimization_peak_bytes"), errors="coerce"
                )
                / float(1024 ** 3)
            )
            render_df["ranking_peak_gb"] = (
                pd.to_numeric(
                    render_df.get("ranking_peak_bytes"), errors="coerce"
                )
                / float(1024 ** 3)
            )
            render_df["usage_pct"] = (
                pd.to_numeric(
                    render_df.get("optimization_usage_fraction"),
                    errors="coerce",
                )
                * 100.0
            )
            st.markdown("**Frontera segura combinada**")
            st.dataframe(
                render_df[
                    [
                        "parallel_jobs",
                        "xgb_parallel_jobs",
                        "max_optuna_jobs",
                        "cpu_limited_optuna_jobs",
                        "memory_limited_optuna_jobs",
                        "dominant_model",
                        "ranking_peak_gb",
                        "peak_gb",
                        "usage_pct",
                        "throughput_score",
                    ]
                ].rename(
                    columns={
                        "parallel_jobs": "RF/ranking jobs",
                        "xgb_parallel_jobs": "XGBoost jobs",
                        "max_optuna_jobs": "Optuna jobs",
                        "cpu_limited_optuna_jobs": "Tope CPU Optuna",
                        "memory_limited_optuna_jobs": "Limite RAM Optuna",
                        "dominant_model": "Modelo dominante",
                        "ranking_peak_gb": "Peak ranking RF (GB)",
                        "peak_gb": "Peak optimización (GB)",
                        "usage_pct": "Uso presupuesto (%)",
                        "throughput_score": "Score throughput",
                    }
                ),
                width="stretch",
            )

        button_col_1, button_col_2, button_col_3, button_col_4 = st.columns(4)
        with button_col_1:
            if recommended_pair and st.button(
                "Usar config. segura",
                key="exp_controlled_apply_safe_pair",
            ):
                _queue_controlled_job_config_apply(
                    parallel_jobs=int(recommended_pair["parallel_jobs"]),
                    optuna_n_jobs=int(recommended_pair["optuna_n_jobs"]),
                    xgb_parallel_jobs=int(recommended_pair["xgb_parallel_jobs"]),
                    notice="Se aplicó la configuración segura recomendada por memoria.",
                )
        with button_col_2:
            if max_parallel_jobs > 0 and st.button(
                "Usar max RF",
                key="exp_controlled_apply_parallel_max",
            ):
                _queue_controlled_job_config_apply(
                    parallel_jobs=int(max_parallel_jobs),
                    optuna_n_jobs=1,
                    xgb_parallel_jobs=1,
                    notice=(
                        "Se aplicó el máximo RF/ranking seguro con Optuna=1 y "
                        "XGBoost=1."
                    ),
                )
        with button_col_3:
            if max_optuna_jobs > 0 and st.button(
                "Usar max Optuna",
                key="exp_controlled_apply_optuna_max",
            ):
                _queue_controlled_job_config_apply(
                    parallel_jobs=1,
                    optuna_n_jobs=int(max_optuna_jobs),
                    xgb_parallel_jobs=1,
                    notice=(
                        "Se aplicó el máximo Optuna seguro con RF=1 y XGBoost=1."
                    ),
                )
        with button_col_4:
            if xgb_selected and max_xgb_jobs > 0 and st.button(
                "Usar max XGBoost",
                key="exp_controlled_apply_xgb_max",
            ):
                _queue_controlled_job_config_apply(
                    parallel_jobs=1,
                    optuna_n_jobs=1,
                    xgb_parallel_jobs=int(max_xgb_jobs),
                    notice=(
                        "Se aplicó el máximo XGBoost seguro con RF=1 y Optuna=1."
                    ),
                )

        detail_payload = {
            "estimator_version": estimate_payload.get("estimator_version"),
            "train_rows": int(estimate_payload.get("train_rows") or 0),
            "val_rows": int(estimate_payload.get("val_rows") or 0),
            "test_rows": int(estimate_payload.get("test_rows") or 0),
            "smote_train_rows_estimate": int(
                estimate_payload.get("smote_train_rows_estimate") or 0
            ),
            "feature_counts": estimate_payload.get("feature_counts") or {},
            "k_grid_by_set": estimate_payload.get("k_grid_by_set") or {},
            "trial_feature_count": int(
                estimate_payload.get("trial_feature_count") or 0
            ),
            "ranking_feature_count": int(
                estimate_payload.get("ranking_feature_count") or 0
            ),
            "xgb_parallel_jobs": int(
                estimate_payload.get("xgb_parallel_jobs") or 0
            ),
            "max_xgb_parallel_jobs_when_parallel_1_optuna_1": int(
                estimate_payload.get("max_xgb_parallel_jobs_when_parallel_1_optuna_1")
                or 0
            ),
        }
        components = dict(estimate_payload.get("components") or {})
        if components:
            detail_payload["componentes"] = {
                key: _format_bytes(value) if "bytes" in str(key) else value
                for key, value in components.items()
            }
        with st.expander("Detalle técnico de la estimación", expanded=False):
            st.json(detail_payload)


def _render_controlled_comparison_protocol_description(
    *,
    objective_label: str,
    selected_models: List[str],
    threshold_protocols: List[str],
    threshold_objective_label: str,
    calibration_methods: List[str],
    alerts_per_day: float,
    fn_cost: float,
    fp_cost: float,
    robust_folds: int,
    test_size: float,
    val_size: float,
    k_min: int,
    k_max: int,
    k_step: int,
    n_trials: int,
    timeout: int,
    optuna_n_jobs: int,
    parallel_jobs: int,
    xgb_parallel_jobs: int,
) -> None:
    selected_models_text = ", ".join(selected_models) if selected_models else "(ninguno)"
    protocol_text = ", ".join(threshold_protocols) if threshold_protocols else "(ninguno)"
    calibration_text = (
        ", ".join(_calibration_method_label(method) for method in calibration_methods)
        if calibration_methods
        else "(ninguna)"
    )
    with st.expander("Descripción detallada del experimento", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "1. **Selección de insumos.** El experimento parte de un archivo de eventos, un archivo de features y un tramo específico. El tramo define el subconjunto exacto del corredor que se evaluará.",
                    "2. **Validación de estructura.** Antes de correr, se inspecciona el archivo de features para confirmar que existen variables numéricas de flujo (`Base`) y variables de cluster (`Cluster`). Si faltan columnas de cluster, la corrida se bloquea porque este protocolo siempre compara `Base`, `Cluster` y `Base + Cluster`.",
                    "3. **Construcción del dataset etiquetado.** Se filtran las features al tramo, se convierten los timestamps, y luego se agrega el `target` de accidente mediante el archivo de eventos. Ese dataset etiquetado se construye una sola vez por corrida.",
                    "4. **Split temporal congelado.** El dataset se divide una sola vez en `train`, `val` y `test` respetando el orden temporal. `test_size` controla el holdout final y `val_size` controla la partición interna de validación dentro del bloque previo al test. Ese split se reutiliza en todos los modelos, conjuntos y valores de `K`.",
                    "5. **Sin leakage en feature selection.** El ranking de importancia se calcula únicamente sobre `train`, nunca sobre `val`, `test` ni sobre el dataset completo del tramo. Esto evita que la selección de variables vea información futura.",
                    "6. **Tres conjuntos de variables.** Se construyen y evalúan tres vistas distintas del mismo problema: `Base` (solo flujo), `Cluster` (solo variables de cluster) y `Base + Cluster` (todas las variables numéricas disponibles).",
                    "7. **Grilla efectiva de K.** La grilla configurada por el usuario (`min`, `max`, `step`) se recorta automáticamente al tamaño real de cada conjunto. Si el máximo real no cae justo en el paso, también se incluye para no perder el borde superior evaluable.",
                    "8. **Estimación opcional de memoria.** Antes de ejecutar, la UI puede calcular jobs seguros usando el dataset real del tramo, el mayor `K` evaluable y el peor caso estimado de SMOTE/modelo. Ese cálculo reporta máximos independientes y una frontera segura combinada para `RF/ranking jobs`, `Optuna jobs` y `XGBoost jobs` bajo un presupuesto explícito de RAM.",
                    "9. **Combinaciones evaluadas.** Para cada corrida se recorren todas las combinaciones `modelo × conjunto × balance_mode × calibración × protocolo_threshold × K` usando sólo los modelos, calibradores y protocolos seleccionados en la UI. Los modos de balance son `sin SMOTE` y `con SMOTE`.",
                    "10. **Optimización con Optuna.** Para cada combinación se ejecuta una búsqueda con Optuna usando exactamente el objetivo seleccionado en esta UI: "
                    f"`{objective_label}`. Se optimizan los hiperparámetros del modelo y, cuando corresponde, también los hiperparámetros de SMOTE.",
                    "11. **Separación objetivo/threshold.** Optuna optimiza la métrica de ranking o clasificación seleccionada. Después, el threshold operativo se escoge con el objetivo de threshold configurado, de modo que PR-AUC puede ordenar modelos sin imponer un umbral arbitrario.",
                    "12. **Uso correcto de SMOTE.** SMOTE se aplica solamente sobre el bloque de entrenamiento del trial. Nunca se usa sobre `val` ni sobre `test`, de modo que la evaluación se mantiene honesta.",
                    "13. **Protocolos de threshold.** Conservador entrena con `train`, calibra/elige threshold en `val` y evalúa ese mismo modelo en `test`. Robusto genera scores OOF temporales dentro de `train + val`, elige threshold con esos scores, reentrena en `train + val` y evalúa una sola vez en `test`.",
                    "14. **Selección del mejor K.** Dentro de cada grupo `modelo + conjunto + balance + protocolo`, el `K` óptimo se define por la mejor métrica objetivo en validación/OOF. En empate, se privilegia el `K` más pequeño para evitar complejidad innecesaria.",
                    "15. **Resultados reportados.** La tabla resumen muestra el mejor resultado por `modelo + conjunto + balance + protocolo`, indicando el `K` óptimo, threshold aplicado, PR-AUC, Balanced F1, falsas alarmas/día, recall aproximado por evento y costo operacional.",
                    "16. **Checkpoint y reanudación.** Cada corrida guarda checkpoints reanudables, estado live y artefactos intermedios. Si la configuración coincide exactamente con una corrida previa incompleta, la UI permite reanudar sin recalcular combinaciones ya resueltas.",
                    "17. **Paralelización.** `Random Forest` usa `n_jobs` para acelerar ranking y entrenamiento. `SVM` no expone `n_jobs`, por lo que su aceleración ocurre a nivel de trials de Optuna. `XGBoost` usa su propio `n_jobs`. La calibración de threshold reutiliza los jobs del modelo cuando aplica. En todos los casos, los valores declarados en la UI se respetan como fuente de verdad; Optuna no reduce automáticamente los jobs internos del modelo.",
                    "18. **Combinaciones inválidas.** Si una combinación falla por falta de clases, `K` imposible o configuración de SMOTE inválida para ese split, se registra como fallida y la corrida continúa con el resto de la matriz experimental.",
                ]
            )
        )
        st.caption(
            "Configuración actual: "
            f"modelos={selected_models_text} | "
            f"objetivo Optuna={objective_label} | protocolos={protocol_text} | "
            f"threshold={threshold_objective_label} | calibración={calibration_text} | "
            f"alertas/día={float(alerts_per_day):.1f} | costo FN/FP={float(fn_cost):.1f}/{float(fp_cost):.1f} | "
            f"folds robustos={int(robust_folds)} | test_size={float(test_size):.2f} | "
            f"val_size={float(val_size):.2f} | K=[{int(k_min)}, {int(k_max)}] paso {int(k_step)} | "
            f"trials={int(n_trials)} | timeout={int(timeout)} s | "
            f"Optuna jobs={int(optuna_n_jobs)} | RF/ranking jobs={int(parallel_jobs)} | "
            f"XGBoost jobs={int(xgb_parallel_jobs)}"
        )


def _build_xai_group_chart(group_df: pd.DataFrame):
    alt = _import_altair()
    if alt is None or group_df.empty:
        return None
    plot_df = group_df.copy()
    plot_df["contribution"] = "Contribucion total"
    return (
        alt.Chart(plot_df)
        .mark_bar(size=28)
        .encode(
            y=alt.Y("contribution:N", axis=None, title=None),
            x=alt.X(
                "total_mean_abs_shap:Q",
                stack="normalize",
                axis=alt.Axis(title="Participacion explicativa", format="%"),
            ),
            color=alt.Color(
                "feature_group:N",
                scale=alt.Scale(
                    domain=XAI_GROUP_COLOR_DOMAIN,
                    range=XAI_GROUP_COLOR_RANGE,
                ),
                legend=alt.Legend(title="Grupo"),
            ),
            tooltip=[
                alt.Tooltip("feature_group:N", title="Grupo"),
                alt.Tooltip("total_mean_abs_shap:Q", title="Contribucion total", format=".4f"),
                alt.Tooltip("share:Q", title="Share", format=".2%"),
            ],
        )
        .properties(height=70)
    )


def _build_xai_feature_bar_chart(
    feature_df: pd.DataFrame,
    *,
    x_field: str,
    x_title: str,
    use_group_colors: bool = True,
):
    alt = _import_altair()
    if alt is None or feature_df.empty:
        return None
    plot_df = feature_df.copy()
    order = list(plot_df["feature"].astype(str))
    color = alt.Color(
        "feature_group:N",
        scale=alt.Scale(
            domain=XAI_GROUP_COLOR_DOMAIN,
            range=XAI_GROUP_COLOR_RANGE,
        ),
        legend=alt.Legend(title="Grupo"),
    )
    if not use_group_colors:
        color = alt.value(XAI_GROUP_COLOR_RANGE[1])
    return (
        alt.Chart(plot_df)
        .mark_bar(size=16)
        .encode(
            x=alt.X(f"{x_field}:Q", axis=alt.Axis(title=x_title)),
            y=alt.Y("feature:N", sort=order, axis=alt.Axis(title=None)),
            color=color,
            tooltip=[
                alt.Tooltip("feature:N", title="Variable"),
                alt.Tooltip(f"{x_field}:Q", title=x_title, format=".4f"),
                alt.Tooltip("feature_group:N", title="Grupo"),
                alt.Tooltip("rank:Q", title="Rank"),
            ],
        )
        .properties(height=max(200, 24 * len(plot_df)))
    )


def _build_xai_beeswarm_chart(
    beeswarm_df: pd.DataFrame,
    *,
    feature_order: List[str],
):
    alt = _import_altair()
    if alt is None or beeswarm_df.empty:
        return None
    plot_df = beeswarm_df.copy()
    points = (
        alt.Chart(plot_df)
        .mark_circle(size=46, opacity=0.68)
        .encode(
            x=alt.X("shap_value:Q", axis=alt.Axis(title="SHAP value")),
            y=alt.Y(
                "jitter:Q",
                axis=None,
                scale=alt.Scale(domain=[-0.45, 0.45]),
            ),
            color=alt.Color(
                "feature_value_scaled:Q",
                scale=alt.Scale(
                    domain=[0.0, 0.5, 1.0],
                    range=XAI_FEATURE_VALUE_RANGE,
                ),
                legend=alt.Legend(title="Valor feature"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Variable"),
                alt.Tooltip("shap_value:Q", title="SHAP", format=".4f"),
                alt.Tooltip("feature_value:Q", title="Valor", format=".4f"),
                alt.Tooltip("score:Q", title="Score", format=".4f"),
                alt.Tooltip("pred:Q", title="Pred"),
                alt.Tooltip("target:Q", title="Target"),
                alt.Tooltip("case_hint:N", title="Caso"),
            ],
        )
    )
    zero_rule = (
        alt.Chart(plot_df)
        .transform_calculate(zero="0")
        .mark_rule(color="#6b7280", strokeDash=[4, 4], opacity=0.45)
        .encode(x="zero:Q")
    )
    return (
        alt.layer(zero_rule, points)
        .facet(
            row=alt.Row(
                "feature:N",
                sort=feature_order,
                header=alt.Header(title=None, labelAngle=0, labelFontWeight="bold"),
            )
        )
        .resolve_scale(x="shared")
        .properties(bounds="flush")
    )


def _build_xai_local_case_chart(case: Dict[str, object]):
    alt = _import_altair()
    all_contributions = case.get("all_contributions")
    if alt is None or not isinstance(all_contributions, pd.DataFrame) or all_contributions.empty:
        return None
    plot_df = all_contributions.head(10).copy()
    plot_df = plot_df.sort_values("shap_value", ascending=True).reset_index(drop=True)
    zero_df = pd.DataFrame({"x": [0.0]})
    order = list(plot_df["feature"].astype(str))
    bars = (
        alt.Chart(plot_df)
        .mark_bar(size=18)
        .encode(
            x=alt.X("shap_value:Q", axis=alt.Axis(title="SHAP value")),
            y=alt.Y("feature:N", sort=order, axis=alt.Axis(title=None)),
            color=alt.Color(
                "feature_group:N",
                scale=alt.Scale(
                    domain=XAI_GROUP_COLOR_DOMAIN,
                    range=XAI_GROUP_COLOR_RANGE,
                ),
                legend=alt.Legend(title="Grupo"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Variable"),
                alt.Tooltip("shap_value:Q", title="SHAP", format=".4f"),
                alt.Tooltip("value:Q", title="Valor", format=".4f"),
                alt.Tooltip("feature_group:N", title="Grupo"),
            ],
        )
    )
    zero_rule = alt.Chart(zero_df).mark_rule(color="#6b7280", strokeDash=[4, 4]).encode(
        x="x:Q"
    )
    return (
        alt.layer(zero_rule, bars)
        .properties(height=max(220, 24 * len(plot_df)))
    )


def _render_xai_report_content(report: Dict[str, object], *, key_prefix: str) -> None:
    st.caption(
        f"Explainer: {report.get('explainer_name', '-')} | "
        f"Filas explicadas: {report.get('rows_explained', 0)}"
    )
    tabs = st.tabs(["Resumen", "Importancia global", "Distribucion SHAP", "Casos locales"])
    group_df = report.get("group_summary")
    global_df = report.get("global_importance")
    cluster_top = report.get("cluster_top")
    beeswarm_df = report.get("beeswarm_points")
    feature_order = []
    if isinstance(global_df, pd.DataFrame) and not global_df.empty:
        feature_order = list(global_df["feature"].astype(str))

    with tabs[0]:
        if isinstance(group_df, pd.DataFrame) and not group_df.empty:
            group_show = group_df.copy()
            if "share" in group_show.columns:
                group_show["share_pct"] = (
                    pd.to_numeric(group_show["share"], errors="coerce").fillna(0.0)
                    * 100.0
                )
            metric_cols = st.columns(max(1, len(group_show)))
            for col, row in zip(metric_cols, group_show.itertuples(index=False)):
                col.metric(
                    str(getattr(row, "feature_group", "-")),
                    f"{float(getattr(row, 'share_pct', 0.0)):.1f}%",
                    f"{float(getattr(row, 'total_mean_abs_shap', 0.0)):.4f}",
                )
            st.caption("Contribucion agregada Base vs Cluster")
            chart = _build_xai_group_chart(group_show)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
            else:
                st.dataframe(group_show, width="stretch")
        else:
            st.info("No hay resumen Base vs Cluster disponible.")

        if isinstance(cluster_top, pd.DataFrame) and not cluster_top.empty:
            st.caption("Top variables de cluster")
            cluster_chart = _build_xai_feature_bar_chart(
                cluster_top,
                x_field="mean_abs_shap",
                x_title="mean(|SHAP|)",
                use_group_colors=False,
            )
            if cluster_chart is not None:
                st.altair_chart(cluster_chart, width="stretch")
            else:
                st.dataframe(cluster_top, width="stretch")
        else:
            st.info("No hay variables de cluster destacadas en esta corrida.")

    with tabs[1]:
        if isinstance(global_df, pd.DataFrame) and not global_df.empty:
            st.caption("Ranking global SHAP (mean(|SHAP|))")
            global_chart = _build_xai_feature_bar_chart(
                global_df,
                x_field="mean_abs_shap",
                x_title="mean(|SHAP|)",
            )
            if global_chart is not None:
                st.altair_chart(global_chart, width="stretch")
            else:
                st.dataframe(global_df, width="stretch")
        else:
            st.info("No hay importancia global disponible.")

    with tabs[2]:
        if isinstance(beeswarm_df, pd.DataFrame) and not beeswarm_df.empty:
            st.caption("Distribucion SHAP por variable")
            beeswarm_chart = _build_xai_beeswarm_chart(
                beeswarm_df,
                feature_order=feature_order,
            )
            if beeswarm_chart is not None:
                st.altair_chart(beeswarm_chart, width="stretch")
            else:
                st.dataframe(beeswarm_df, width="stretch")
        else:
            st.info("No hay puntos suficientes para la vista beeswarm.")

    local_cases = report.get("local_cases", [])
    with tabs[3]:
        if isinstance(local_cases, list) and local_cases:
            st.caption("Casos locales representativos")
            case_tabs = st.tabs(
                [
                    str(case.get("meta", {}).get("case_label", f"Caso {idx + 1}"))
                    for idx, case in enumerate(local_cases)
                ]
            )
            for idx, (tab, case) in enumerate(zip(case_tabs, local_cases), start=1):
                with tab:
                    meta = case.get("meta", {})
                    if isinstance(meta, dict):
                        meta_display = {
                            str(key): _json_default(value)
                            for key, value in meta.items()
                            if value is not None
                            and str(key)
                            in {
                                "case_label",
                                "case_key",
                                "score",
                                "threshold",
                                "pred",
                                "target",
                                "portico",
                                "interval_start",
                                "portico_last",
                                "portico_next",
                                "eje",
                                "calzada",
                                "source_index",
                            }
                        }
                        if meta_display:
                            st.json(meta_display)
                    local_chart = _build_xai_local_case_chart(case)
                    if local_chart is not None:
                        st.altair_chart(local_chart, width="stretch")
                    else:
                        col_up, col_down = st.columns(2)
                        with col_up:
                            st.caption("Variables que empujan al alza")
                            top_positive = case.get("top_positive")
                            if isinstance(top_positive, pd.DataFrame) and not top_positive.empty:
                                st.dataframe(top_positive, width="stretch")
                            else:
                                st.info("No hay contribuciones positivas destacadas.")
                        with col_down:
                            st.caption("Variables que empujan a la baja")
                            top_negative = case.get("top_negative")
                            if isinstance(top_negative, pd.DataFrame) and not top_negative.empty:
                                st.dataframe(top_negative, width="stretch")
                            else:
                                st.info("No hay contribuciones negativas destacadas.")
        else:
            st.info("No hay casos locales representativos disponibles.")


def _render_base_cluster_xai_block(
    entry: Dict[str, object],
    *,
    key_prefix: str,
    default_visible: bool,
) -> None:
    st.markdown("**XAI Base + Cluster**")
    bundle_path, bundle_error = _resolve_base_cluster_xai_info(entry)
    if bundle_error and not bundle_path:
        st.warning(
            "XAI no disponible para esta corrida. "
            f"Error al generar el bundle: {bundle_error}"
        )
        return
    if not bundle_path:
        st.info(
            "XAI no disponible para esta corrida. "
            "Las corridas antiguas no incluyen bundle reproducible."
        )
        return

    bundle_dir = Path(bundle_path)
    if not bundle_dir.exists():
        st.warning(f"El bundle XAI no existe en disco: {bundle_dir}")
        return

    visible_key = f"{key_prefix}_xai_visible"
    if default_visible and visible_key not in st.session_state:
        st.session_state[visible_key] = True

    if not st.session_state.get(visible_key, False):
        if st.button("Calcular SHAP", key=f"{key_prefix}_xai_show"):
            st.session_state[visible_key] = True
            st.rerun()
        st.caption(f"Bundle: {bundle_dir}")
        return

    if st.button("Recalcular SHAP", key=f"{key_prefix}_xai_refresh"):
        cache = st.session_state.setdefault("xai_report_cache", {})
        if isinstance(cache, dict):
            cache.pop(str(bundle_dir), None)
        st.rerun()

    cache = st.session_state.setdefault("xai_report_cache", {})
    report = cache.get(str(bundle_dir)) if isinstance(cache, dict) else None
    if report is None:
        try:
            with st.spinner("Calculando explicaciones SHAP..."):
                report = compute_xai_report(bundle_dir)
            if isinstance(cache, dict):
                cache[str(bundle_dir)] = report
        except ImportError as exc:
            st.warning(str(exc))
            return
        except Exception as exc:
            st.warning(f"No se pudo calcular XAI: {exc}")
            return
    _render_xai_report_content(report, key_prefix=key_prefix)

def _load_feature_selection_from_disk(
    feature_id: str,
) -> Tuple[Optional[Dict[str, object]], Optional[pd.DataFrame]]:
    json_path, csv_path = _feature_selection_paths(feature_id)
    payload: Optional[Dict[str, object]] = None
    importance_df: Optional[pd.DataFrame] = None
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            payload = None
    if csv_path.exists():
        try:
            importance_df = pd.read_csv(csv_path)
        except Exception:
            importance_df = None
    return payload, importance_df


def _persist_feature_selection(
    *,
    feature_key: str,
    feature_id: str,
    features_path: Optional[str],
    features_source: Optional[str],
    features_df: pd.DataFrame,
    selected_features: List[str],
    importance_df: Optional[pd.DataFrame],
    params: Dict[str, object],
) -> None:
    store = st.session_state.setdefault("feature_selection_store", {})
    prev = store.get(feature_key, {})
    prev_selected = prev.get("selected_features")
    prev_hash = prev.get("importance_hash")
    prev_importance = prev.get("importance_df")

    importance_hash = prev_hash
    if importance_df is None:
        importance_df = prev_importance
    elif importance_df is not None and not importance_df.empty:
        try:
            importance_hash = int(
                pd.util.hash_pandas_object(importance_df, index=True).sum()
            )
        except Exception:
            importance_hash = None

    entry = {
        "feature_id": feature_id,
        "features_path": features_path,
        "features_source": features_source,
        "features_rows": int(len(features_df)),
        "features_cols": int(len(features_df.columns)),
        "selected_features": list(selected_features),
        "importance_df": importance_df,
        "importance_hash": importance_hash,
        "params": dict(params),
        "saved_at": datetime.now().isoformat(),
    }
    store[feature_key] = entry

    selected_changed = prev_selected != selected_features
    importance_changed = (
        importance_df is not None and importance_hash != prev_hash
    )
    if not (selected_changed or importance_changed or prev == {}):
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = _feature_selection_paths(feature_id)
    payload = {
        "feature_key": feature_key,
        "feature_id": feature_id,
        "features_path": features_path,
        "features_source": features_source,
        "features_rows": int(len(features_df)),
        "features_cols": int(len(features_df.columns)),
        "selected_features": list(selected_features),
        "params": dict(params),
        "saved_at": datetime.now().isoformat(),
        "importance_csv": None,
    }
    if importance_df is not None and not importance_df.empty:
        try:
            importance_df.to_csv(csv_path, index=False)
            payload["importance_csv"] = str(csv_path)
        except Exception:
            payload["importance_csv"] = None
    else:
        if csv_path.exists():
            payload["importance_csv"] = str(csv_path)
    try:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
    except Exception:
        return


def _apply_smote_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    test_size: float,
    split_random_state: int,
    random_state: int,
    smote_k_neighbors: int,
    smote_sampling_strategy: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn no esta instalado. Ejecute `pip install imbalanced-learn`."
        ) from exc
    if not feature_cols:
        raise ValueError("No hay variables numericas para aplicar SMOTE.")
    if "interval_start" not in df.columns:
        raise ValueError(
            "No se encontro 'interval_start' para hacer split temporal."
        )

    train_df_raw, test_df_raw = _temporal_train_test_split(
        df, time_col="interval_start", test_size=float(test_size)
    )
    # Use float32 to reduce memory during resampling.
    X_train = train_df_raw[feature_cols].fillna(0).astype("float32")
    y_train = train_df_raw["target"].astype("int8")
    X_test = test_df_raw[feature_cols].fillna(0).astype("float32")
    y_test = test_df_raw["target"].astype("int8")
    if y_train.nunique() < 2:
        raise ValueError(
            "El split temporal dejo una sola clase en train. "
            "Ajuste el rango o el test_size."
        )

    dist_before = _class_distribution(y_train)
    dist_test = _class_distribution(y_test)
    min_count = int(y_train.value_counts().min())
    if min_count < 2:
        raise ValueError("No hay suficientes ejemplos minoritarios para SMOTE.")

    k_neighbors = max(1, min(int(smote_k_neighbors), min_count - 1))
    smote_kwargs: Dict[str, object] = {
        "k_neighbors": k_neighbors,
        "random_state": random_state,
    }
    if smote_sampling_strategy is not None:
        smote_kwargs["sampling_strategy"] = float(smote_sampling_strategy)
    smote = SMOTE(**smote_kwargs)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    dist_after = _class_distribution(y_res)

    original_train_count = len(X_train)
    resampled_count = len(X_res)
    synthetic_flags = np.zeros(resampled_count, dtype=bool)
    if resampled_count > original_train_count:
        synthetic_flags[original_train_count:] = True

    train_df = pd.DataFrame(X_res, columns=feature_cols)
    train_df["target"] = y_res
    train_df["split"] = "train"
    train_df["synthetic"] = synthetic_flags

    train_times = pd.to_datetime(
        train_df_raw["interval_start"], errors="coerce"
    ).reset_index(drop=True)
    synthetic_count = resampled_count - original_train_count
    if synthetic_count > 0:
        extra_times = pd.Series(
            [pd.NaT] * synthetic_count, dtype="datetime64[ns]"
        )
        train_times = pd.concat([train_times, extra_times], ignore_index=True)
    train_df["interval_start"] = train_times

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["target"] = y_test
    test_df["split"] = "test"
    test_df["synthetic"] = False
    test_df["interval_start"] = pd.to_datetime(
        test_df_raw["interval_start"], errors="coerce"
    ).reset_index(drop=True)

    balanced_df = pd.concat([train_df, test_df], ignore_index=True)
    return balanced_df, dist_before, dist_after, dist_test, k_neighbors


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col not in {"target", "synthetic"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def _rank_features_for_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    method: str,
    random_state: int,
) -> List[str]:
    """Devuelve las columnas de X ordenadas de mayor a menor importancia.

    Usa train (sin SMOTE) para evitar leakage. method in {"rf", "mutual_info"}.
    """
    cols = list(X.columns)
    if not cols:
        return []
    method_key = str(method or "rf").strip().lower()
    y_arr = pd.Series(y).astype(int)
    if y_arr.nunique() < 2:
        return cols
    X_arr = X.fillna(0).astype("float32")
    if method_key in {"mutual_info", "mi", "mutual_info_classif"}:
        from sklearn.feature_selection import mutual_info_classif

        scores = mutual_info_classif(
            X_arr, y_arr, random_state=int(random_state)
        )
    else:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=int(random_state),
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_arr, y_arr)
        scores = model.feature_importances_
    ranking_df = pd.DataFrame(
        {"variable": cols, "importance": scores}
    ).sort_values("importance", ascending=False, kind="mergesort")
    return ranking_df["variable"].tolist()















def _get_cluster_cols(df: pd.DataFrame) -> List[str]:
    cluster_prefixes = (
        "cluster_share_",
        "cluster_flow_",
        "cluster_count_",
        "cluster_speed_",
        "cluster_density_",
        "cluster_delta_speed_",
        "cluster_delta_density_",
        "cluster_entropy",
    )
    valid_cols = []
    for col in df.columns:
        # Check original
        if col.startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
        # Check segment prefixes
        if col.startswith("last_") and col[5:].startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
        if col.startswith("next_") and col[5:].startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
    return valid_cols


def _normalize_match_key(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def _select_detail_columns(
    df: pd.DataFrame, candidates: List[str]
) -> List[str]:
    normalized = {_normalize_match_key(col): col for col in df.columns}
    selected: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalize_match_key(candidate)
        col = normalized.get(key)
        if col and col not in seen:
            selected.append(col)
            seen.add(col)
    return selected


def _find_match_column(
    df: pd.DataFrame, candidates: List[str]
) -> Optional[str]:
    normalized = {_normalize_match_key(col): col for col in df.columns}
    for candidate in candidates:
        key = _normalize_match_key(candidate)
        col = normalized.get(key)
        if col:
            return col
    return None


def _resolve_feature_cols(
    df: pd.DataFrame,
    selected_features: Optional[List[str]],
    *,
    include_cluster_features: bool,
) -> Tuple[List[str], List[str]]:
    numeric_cols = _get_feature_cols(df)
    cluster_cols_set = set(_get_cluster_cols(df))
    cluster_cols = [col for col in numeric_cols if col in cluster_cols_set]
    allowed_cols = (
        numeric_cols
        if include_cluster_features
        else [col for col in numeric_cols if col not in cluster_cols]
    )
    if selected_features is None:
        return allowed_cols, []
    if not selected_features:
        return [], []
    selected = [col for col in selected_features if col in allowed_cols]
    if include_cluster_features:
        missing = [col for col in selected_features if col not in allowed_cols]
    else:
        missing = [
            col
            for col in selected_features
            if col not in allowed_cols and col not in cluster_cols
    ]
    return selected, missing


def _resolve_feature_group_cols(
    df: pd.DataFrame,
    selected_features: Optional[List[str]],
    *,
    feature_group: str,
) -> Tuple[List[str], List[str]]:
    numeric_cols = _get_feature_cols(df)
    cluster_cols_set = set(_get_cluster_cols(df))
    base_cols = [col for col in numeric_cols if col not in cluster_cols_set]
    cluster_cols = [col for col in numeric_cols if col in cluster_cols_set]
    if feature_group == "base":
        allowed_cols = base_cols
        ignored_cols = set(cluster_cols)
    elif feature_group == "cluster":
        allowed_cols = cluster_cols
        ignored_cols = set(base_cols)
    elif feature_group == "base_cluster":
        allowed_cols = numeric_cols
        ignored_cols = set()
    else:
        raise ValueError(f"Grupo de features no soportado: {feature_group}")

    if selected_features is None:
        return allowed_cols, []
    if not selected_features:
        return [], []
    selected = [col for col in selected_features if col in allowed_cols]
    missing = [
        col
        for col in selected_features
        if col not in allowed_cols and col not in ignored_cols
    ]
    return selected, missing


def _has_cluster_count_cols(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    return any(
        col.startswith(("cluster_flow_", "cluster_count_")) for col in df.columns
    )


def _render_selected_features_info() -> None:
    selected_features = st.session_state.get("selected_features")
    if selected_features is None:
        st.info(
            "Variables seleccionadas: todas "
            "(aun no se define una seleccion en Feature selection)."
        )
        return
    if not selected_features:
        st.info(
            "Variables seleccionadas: 0. Seleccione al menos una en Feature selection."
        )
        return
    max_items = 12
    preview = ", ".join(selected_features[:max_items])
    if len(selected_features) > max_items:
        preview = f"{preview} + {len(selected_features) - max_items} mas"
    st.caption(f"Variables seleccionadas ({len(selected_features)}): {preview}")


def _collect_optuna_best_feature_options(
    model_choice: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Devuelve las entradas del store de Optuna con best_feature_cols no vacio.

    Cada item incluye etiqueta amigable, lista de features, model_choice y metadata
    suficiente para mostrar en un selector.
    """
    store = st.session_state.get("optuna_results_store") or {}
    options: List[Dict[str, object]] = []
    for entry_key, entry in store.items():
        if not isinstance(entry, dict):
            continue
        results = _normalize_optuna_results_payload(entry.get("results"))
        for choice, container in results.items():
            if not isinstance(container, dict):
                continue
            if model_choice and str(choice) != str(model_choice):
                continue
            by_balance_mode = container.get("by_balance_mode")
            if not isinstance(by_balance_mode, dict):
                continue
            for balance_mode, mode_data in by_balance_mode.items():
                if not isinstance(mode_data, dict):
                    continue
                by_calibration_method = mode_data.get("by_calibration_method")
                if not isinstance(by_calibration_method, dict):
                    continue
                for calibration_method, result in by_calibration_method.items():
                    if not isinstance(result, dict):
                        continue
                    settings = result.get("optuna_settings") or {}
                    best_cols = settings.get("best_feature_cols")
                    if not best_cols:
                        # Fallback: si no hay best_feature_cols, usa el conjunto del config.
                        best_cols = entry.get("feature_cols")
                    if not best_cols:
                        continue
                    label_parts = []
                    if len(entry.get("feature_cols", [])) > 0:
                        label_parts.append(f"{len(entry['feature_cols'])} vars config")
                    if settings.get("best_top_k") is not None:
                        label_parts.append(f"top_k={int(settings['best_top_k'])}")
                    label_parts.append(f"{choice}")
                    label_parts.append(_optuna_balance_mode_label(balance_mode))
                    label_parts.append(
                        _calibration_method_label(calibration_method)
                    )
                    label = " | ".join(label_parts)
                    options.append(
                        {
                            "label": label,
                            "key": str(entry_key),
                            "model_choice": str(choice),
                            "balance_mode": str(balance_mode),
                            "calibration_method": str(calibration_method),
                            "best_feature_cols": list(best_cols),
                            "best_top_k": settings.get("best_top_k"),
                            "ranking_method": settings.get("ranking_method"),
                            "ranking_method_label": settings.get("ranking_method_label"),
                            "best_score": result.get("best_score"),
                            "objective_label": settings.get("objective_label"),
                        }
                    )
    return options


_MODEL_FEATURE_GROUP_OVERRIDE_KEYS: Dict[str, str] = {
    "base": "selected_features_override_base",
    "cluster": "selected_features_override_cluster",
    "base_cluster": "selected_features_override_base_cluster",
}


def _clear_model_feature_group_overrides() -> None:
    for state_key in _MODEL_FEATURE_GROUP_OVERRIDE_KEYS.values():
        st.session_state.pop(state_key, None)


def _lookup_optuna_best_feature_cols(
    *,
    store: Dict[str, object],
    feature_key: str,
    cols: List[str],
    model_choice: str,
    balance_mode: str,
    calibration_method: str,
    allow_calibration_fallback: bool = False,
) -> Optional[Dict[str, object]]:
    if not cols:
        return None
    key = _optuna_result_key(feature_key, cols)
    entry = store.get(key)
    if not isinstance(entry, dict):
        return None
    match = _get_optuna_model_result_variant_match(
        entry.get("results"),
        model_choice=model_choice,
        balance_mode=balance_mode,
        calibration_method=calibration_method,
        allow_any_calibration_within_mode=allow_calibration_fallback,
    )
    if not isinstance(match, dict):
        return None
    result = match.get("result")
    if not isinstance(result, dict):
        return None
    settings = result.get("optuna_settings") or {}
    best_cols = settings.get("best_feature_cols") or entry.get("feature_cols")
    if not best_cols:
        return None
    return {
        "best_feature_cols": list(best_cols),
        "best_top_k": settings.get("best_top_k"),
        "ranking_method_label": settings.get("ranking_method_label"),
        "best_score": result.get("best_score"),
        "objective_label": settings.get("objective_label"),
        "balance_mode": settings.get("balance_mode"),
        "balance_mode_label": settings.get("balance_mode_label"),
        "calibration_method": settings.get("calibration_method"),
        "calibration_method_label": settings.get("calibration_method_label"),
        "requested_calibration_method": _normalize_calibration_method(
            calibration_method
        ),
        "used_fallback": bool(match.get("used_fallback")),
        # Fingerprint del dataset usado cuando se corrió Optuna. Permite al
        # consumidor detectar drift comparando contra el dataset actual.
        "dataset_fingerprint": entry.get("dataset_fingerprint"),
        "features_rows": entry.get("features_rows"),
        "features_cols": entry.get("features_cols"),
        "saved_at": entry.get("saved_at"),
    }


def _diagnose_optuna_key_mismatch(
    *,
    store: Dict[str, object],
    expected_key: str,
    active_key: Optional[str],
    current_fingerprint: Optional[str],
) -> Dict[str, object]:
    """Diagnóstico legible de por qué un `optuna_key` no coincide.

    Reutilizable en cualquier tab que consuma Optuna (Balance, Modelos, etc.)
    para explicar *por qué* el Optuna guardado ya no aplica, en lugar del
    clásico warning genérico "no coinciden".

    Devuelve:
      - ``has_match``: hay entry exacto para `expected_key`.
      - ``reasons``: lista de razones en español (dataset distinto, features
        distintas, drift de contenido, etc.).
      - ``dataset_drift``: el entry existe pero el fingerprint del dataset
        actual difiere del guardado.
      - ``stored_fingerprint``: fingerprint guardado en el entry (si aplica).
    """
    store_dict = store if isinstance(store, dict) else {}
    expected_entry = store_dict.get(expected_key)
    has_exact_entry = isinstance(expected_entry, dict)
    reasons: List[str] = []
    dataset_drift = False
    stored_fingerprint: Optional[str] = None

    if has_exact_entry:
        stored_fingerprint = expected_entry.get("dataset_fingerprint")
        if (
            stored_fingerprint
            and current_fingerprint
            and stored_fingerprint != current_fingerprint
        ):
            dataset_drift = True
            reasons.append(
                "el contenido del dataset cambió desde la última corrida "
                "(schema o shape distinto)"
            )
        return {
            "has_match": True,
            "reasons": reasons,
            "dataset_drift": dataset_drift,
            "stored_fingerprint": stored_fingerprint,
        }

    # No hay entry exacto: comparamos expected_key vs active_key por sus
    # dos partes (feature_key|feature_list_signature).
    expected_parts = str(expected_key).rsplit("|", 1)
    active_str = str(active_key or "").strip()
    active_parts = active_str.rsplit("|", 1) if active_str else []
    if len(expected_parts) == 2 and len(active_parts) == 2:
        expected_feature_key, expected_sig = expected_parts
        active_feature_key, active_sig = active_parts
        if expected_feature_key != active_feature_key:
            reasons.append(
                "el dataset activo (path, fuente o schema) difiere del "
                "que usó Optuna"
            )
        if expected_sig != active_sig:
            reasons.append(
                "las variables seleccionadas cambiaron respecto a las "
                "optimizadas por Optuna"
            )
    elif not active_str:
        reasons.append("no hay resultado de Optuna activo en esta sesión")
    else:
        reasons.append(
            "el identificador de Optuna tiene un formato inesperado "
            "(puede ser un resultado legacy)"
        )

    # Si existe un entry activo (no exacto), detectar drift de contenido
    # usando su fingerprint — ayuda a distinguir "cambiaste las features"
    # de "regeneraste el dataset con el mismo path".
    active_entry = store_dict.get(active_str) if active_str else None
    if isinstance(active_entry, dict):
        active_stored_fp = active_entry.get("dataset_fingerprint")
        if (
            active_stored_fp
            and current_fingerprint
            and active_stored_fp != current_fingerprint
        ):
            reasons.append(
                "el contenido del dataset activo cambió respecto al "
                "guardado por Optuna"
            )

    # Deduplicar preservando orden.
    seen = set()
    deduped: List[str] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        deduped.append(reason)

    return {
        "has_match": False,
        "reasons": deduped,
        "dataset_drift": False,
        "stored_fingerprint": stored_fingerprint,
    }


def _get_active_optuna_best(
    *,
    store: Optional[Dict[str, object]] = None,
    active_key: Optional[str] = None,
    model_choice: Optional[str] = None,
    calibration_method: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Lee del store Optuna la vista "primary" para ``(active_key, model_choice)``.

    Sustituto canónico de los keys legacy ``optuna_best_*`` en session_state:
    en lugar de replicar los parámetros ganadores en múltiples keys top-level,
    esta función los lee directamente del store
    (``optuna_results_store[active_key]``) y devuelve una vista unificada.

    Política de selección (mirror del disk-reloader en ``_render_optuna_tab``,
    L9095-9139):
      - ``balance_mode=smote`` gana sobre ``balance_mode=none``.
      - ``calibration_method`` es la seleccionada en UI
        (``session_state["optuna_calibration_method"]``); no hace fallback a
        otras calibraciones dentro del mismo modo.
      - ``best_smote_params`` proviene *solo* del variante SMOTE estricto
        (no se rellena con ``none``); es ``None`` si SMOTE no corrió para esta
        calibración.

    Parameters
    ----------
    store : dict, optional
        Store de Optuna. Si es ``None`` lo lee de
        ``session_state["optuna_results_store"]``.
    active_key : str, optional
        Key activo. Si es ``None`` lo lee de
        ``session_state["optuna_active_key"]``.
    model_choice : str, optional
        Modelo. Si es ``None`` lo lee de
        ``session_state["optuna_model_choice"]``.
    calibration_method : str, optional
        Calibración preferida. Si es ``None`` lo lee de
        ``session_state["optuna_calibration_method"]``.

    Returns
    -------
    dict or None
        ``None`` si no hay resultado activo para ``(active_key, model_choice)``.
        Caso contrario, un dict con ``best_smote_params``, ``best_model_params``,
        ``best_score``, ``model_choice``, ``balance_mode``,
        ``calibration_method``, ``optuna_settings``, ``search_space``,
        ``trials_df``, ``active_key``, ``entry``.
    """
    if store is None:
        store = st.session_state.get("optuna_results_store") or {}
    if not isinstance(store, dict) or not store:
        return None

    if active_key is None:
        active_key = st.session_state.get("optuna_active_key")
    active_key_str = str(active_key or "").strip()
    if not active_key_str:
        return None

    entry = store.get(active_key_str)
    if not isinstance(entry, dict):
        return None

    if model_choice is None:
        model_choice = st.session_state.get("optuna_model_choice")
    model_choice_str = str(model_choice or "").strip()
    if not model_choice_str:
        return None

    if calibration_method is None:
        calibration_method = st.session_state.get("optuna_calibration_method")
    calibration_method_str = _normalize_calibration_method(calibration_method)
    if not calibration_method_str:
        calibration_method_str = "sigmoid"

    results = entry.get("results")

    # Selección principal: prefiere SMOTE sobre `none`; respeta la calibración
    # pedida sin fallback (mismo contrato que el disk-reloader original).
    primary = _get_optuna_model_result_variant(
        results,
        model_choice=model_choice_str,
        balance_mode="smote",
        calibration_method=calibration_method_str,
        fallback_modes=["none"],
    )
    if not isinstance(primary, dict):
        return None

    # SMOTE estricto: solo si el variante SMOTE *real* existe para esta
    # calibración. Devuelve None cuando el primario vino de balance_mode=none.
    smote_variant = _get_optuna_model_result_variant(
        results,
        model_choice=model_choice_str,
        balance_mode="smote",
        calibration_method=calibration_method_str,
    )
    if isinstance(smote_variant, dict):
        smote_params = smote_variant.get("best_smote_params") or None
    else:
        smote_params = None

    settings = primary.get("optuna_settings") if isinstance(primary, dict) else None
    balance_mode_used: Optional[str] = None
    calibration_used: Optional[str] = None
    if isinstance(settings, dict):
        bm = settings.get("balance_mode")
        if bm:
            balance_mode_used = str(bm)
        cm = settings.get("calibration_method")
        if cm:
            calibration_used = str(cm)

    # Hidrata ``trials_df`` desde CSV si no está en memoria — mirror del
    # disk-reloader para mantener paridad con el key legacy ``optuna_trials_df``.
    trials_df = primary.get("trials_df")
    trials_csv = primary.get("trials_csv")
    if trials_df is None and trials_csv:
        try:
            if Path(str(trials_csv)).exists():
                trials_df = pd.read_csv(trials_csv)
        except Exception:
            trials_df = None

    return {
        "best_smote_params": smote_params,
        "best_model_params": primary.get("best_model_params"),
        "best_score": primary.get("best_score"),
        "model_choice": model_choice_str,
        "balance_mode": balance_mode_used,
        "calibration_method": calibration_used or calibration_method_str,
        "optuna_settings": primary.get("optuna_settings"),
        "search_space": primary.get("search_space"),
        "trials_df": trials_df,
        "active_key": active_key_str,
        "entry": entry,
    }


# =============================================================================
# Contratos de estado por tab (punto 5 del hardening de session_state)
# =============================================================================
#
# Declaración explícita de qué keys cada tab (Optuna / Balance / Modelos) lee
# y escribe. Convierte las dependencias implícitas entre tabs en un contrato
# testeable:
#
#   - ``required``: keys que DEBEN existir y no ser None para que la tab
#     pueda operar (p.ej. ``accidents_df``, ``flow_features_df``).
#   - ``optional``: keys que la tab lee pero tolera que falten.
#   - ``produces``: keys que la tab ESCRIBE como output. Usados por el botón
#     "Reset state" del diagnóstico para volver a correr la tab desde cero.
#
# Las funciones ``_validate_tab_state``, ``_reset_tab_state_keys`` y
# ``_render_state_diagnostics`` consumen este dict para dar visibilidad al
# usuario sobre por qué una tab puede estar fallando.
_TAB_STATE_CONTRACTS: Dict[str, Dict[str, List[str]]] = {
    "optuna": {
        "required": [
            "accidents_df",
            "flow_features_df",
            "selected_features",
        ],
        "optional": [
            "flow_features_path",
            "flow_features_source",
            "optuna_results_store",
            "optuna_active_key",
            "optuna_model_choice",
            "optuna_calibration_method",
        ],
        "produces": [
            "optuna_results_store",
            "optuna_active_key",
            # Keys legacy top-level — ver DEPRECATED comments en
            # _render_optuna_tab. Los dejamos en ``produces`` para que el
            # botón Reset los borre y el estado quede limpio.
            "optuna_best_smote_params",
            "optuna_best_model_params",
            "optuna_best_score",
            "optuna_best_model_choice",
            "optuna_best_settings",
            "optuna_best_search_space",
            "optuna_trials_df",
        ],
    },
    "balance": {
        "required": [
            "accidents_df",
            "flow_features_df",
        ],
        "optional": [
            "flow_features_path",
            "flow_features_source",
            "selected_features",
            "optuna_results_store",
            "optuna_active_key",
            "balance_source",
            "test_size",
        ],
        "produces": [
            "balanced_base_df",
            "balanced_cluster_df",
            "balanced_cluster_only_df",
            "balance_last_stats",
            "balance_last_params",
            "smote_k_neighbors",
            "smote_sampling_strategy",
            "smote_random_state",
        ],
    },
    "modelos": {
        "required": [
            "accidents_df",
            "flow_features_df",
        ],
        "optional": [
            "cluster_features_df",
            "balanced_base_df",
            "balanced_cluster_df",
            "balanced_cluster_only_df",
            "selected_features",
            "optuna_results_store",
            "optuna_active_key",
            "test_size",
            "val_size",
            "model_choice",
            "allow_optuna_calibration_fallback",
        ],
        "produces": [
            "history_entries",
            "feature_importances_df",
        ],
    },
}


def _session_state_value(session_state: object, key: str) -> object:
    """Accede a session_state tolerando dict o streamlit.SessionState."""
    if session_state is None:
        return None
    if hasattr(session_state, "get"):
        try:
            return session_state.get(key)
        except Exception:
            pass
    try:
        return session_state[key]  # type: ignore[index]
    except Exception:
        return None


def _validate_tab_state(
    tab: str,
    *,
    session_state: Optional[object] = None,
) -> List[Dict[str, object]]:
    """Valida el estado de una tab. Retorna lista de issues.

    Cada issue es un dict con:
      - ``level``: ``"error"``, ``"warning"`` o ``"info"``.
      - ``key``: el key de session_state implicado (o ``__contract__`` si
        la tab es desconocida).
      - ``message``: explicación legible en español.

    Política:
      - ``error``: falta un key obligatorio, DataFrame requerido vacío,
        lista requerida vacía.
      - ``warning``: inconsistencias entre keys que la tab lee (p.ej.
        ``optuna_active_key`` apunta a un key inexistente en el store).

    No muta ``session_state``; es una función pura.
    """
    if session_state is None:
        session_state = st.session_state

    contract = _TAB_STATE_CONTRACTS.get(tab)
    if contract is None:
        return [
            {
                "level": "error",
                "key": "__contract__",
                "message": f"tab desconocida: {tab!r}",
            }
        ]

    issues: List[Dict[str, object]] = []

    for key in contract.get("required", []):
        value = _session_state_value(session_state, key)
        if value is None:
            issues.append(
                {
                    "level": "error",
                    "key": key,
                    "message": (
                        f"falta el key obligatorio `{key}` — "
                        "verifique tabs anteriores (Eventos, Feature "
                        "engineering, Feature selection)"
                    ),
                }
            )
            continue
        if isinstance(value, pd.DataFrame) and value.empty:
            issues.append(
                {
                    "level": "error",
                    "key": key,
                    "message": f"`{key}` está vacío (DataFrame sin filas)",
                }
            )
            continue
        if key == "selected_features" and hasattr(value, "__len__") and not len(value):
            issues.append(
                {
                    "level": "error",
                    "key": key,
                    "message": (
                        "`selected_features` está vacío — seleccione al menos "
                        "una variable en Feature selection"
                    ),
                }
            )

    # Coherencia Optuna: el active_key debe existir en el store.
    store = _session_state_value(session_state, "optuna_results_store")
    active_key = _session_state_value(session_state, "optuna_active_key")
    if (
        active_key
        and isinstance(store, dict)
        and str(active_key).strip()
        and active_key not in store
    ):
        issues.append(
            {
                "level": "warning",
                "key": "optuna_active_key",
                "message": (
                    f"`optuna_active_key` apunta a un key ausente del store "
                    f"(`{active_key}`). Es probable que el store se haya "
                    "reseteado sin limpiar el active key."
                ),
            }
        )

    # Coherencia Balance → Modelos: si la tab es Modelos, avisar cuando no
    # hay dataset balanceado disponible.
    if tab == "modelos":
        balanced_base = _session_state_value(session_state, "balanced_base_df")
        if balanced_base is None:
            issues.append(
                {
                    "level": "warning",
                    "key": "balanced_base_df",
                    "message": (
                        "no hay dataset balanceado en memoria — la tab Modelos "
                        "funcionará solo con datos sin balancear. Ejecute la "
                        "tab Balance para habilitar SMOTE."
                    ),
                }
            )

    return issues


def _reset_tab_state_keys(
    tab: str,
    *,
    session_state: Optional[object] = None,
) -> List[str]:
    """Borra los keys producidos por una tab. Retorna la lista de keys borrados.

    Pensado para el botón "Reset state de esta tab" de
    ``_render_state_diagnostics``. Útil cuando la UI cayó en un estado
    inconsistente y el usuario quiere volver a correr la tab desde cero sin
    reiniciar la sesión entera.

    No borra keys ``required``; solo los listados en ``produces``.
    """
    if session_state is None:
        session_state = st.session_state
    contract = _TAB_STATE_CONTRACTS.get(tab)
    if contract is None:
        return []
    removed: List[str] = []
    for key in contract.get("produces", []):
        try:
            if key in session_state:  # type: ignore[operator]
                del session_state[key]  # type: ignore[index]
                removed.append(key)
        except Exception:
            continue
    return removed


def _render_state_diagnostics(tab: str) -> None:
    """Renderiza un expander con el diagnóstico de estado de la tab.

    Muestra:
      - los ``issues`` de ``_validate_tab_state``.
      - qué keys producidos por esta tab están presentes actualmente.
      - un botón "Reset state de esta tab" que invoca
        ``_reset_tab_state_keys`` y recarga la página.

    Se auto-expande cuando hay errors o warnings para hacerlos visibles.
    """
    issues = _validate_tab_state(tab)
    contract = _TAB_STATE_CONTRACTS.get(tab, {})
    produces = contract.get("produces", [])

    error_count = sum(1 for i in issues if i.get("level") == "error")
    warning_count = sum(1 for i in issues if i.get("level") == "warning")

    if error_count:
        title = (
            f"⚠️ Estado ({error_count} "
            f"{'errores' if error_count != 1 else 'error'})"
        )
    elif warning_count:
        title = (
            f"ℹ️ Estado ({warning_count} "
            f"{'avisos' if warning_count != 1 else 'aviso'})"
        )
    else:
        title = "✅ Estado"

    with st.expander(title, expanded=bool(error_count or warning_count)):
        if issues:
            for issue in issues:
                level = issue.get("level", "info")
                message = issue.get("message", "")
                if level == "error":
                    st.error(f"• {message}")
                elif level == "warning":
                    st.warning(f"• {message}")
                else:
                    st.info(f"• {message}")
        else:
            st.success("Todos los keys obligatorios están presentes.")

        present_produced = [
            k
            for k in produces
            if _session_state_value(st.session_state, k) is not None
        ]
        if produces:
            st.caption(
                f"Keys producidos por esta tab: "
                f"{len(present_produced)}/{len(produces)} presentes."
            )
            if present_produced:
                st.caption(", ".join(f"`{k}`" for k in present_produced))

        if st.button(
            "Reset state de esta tab",
            key=f"_reset_state_btn_{tab}",
            help=(
                "Borra los keys producidos por esta tab, dejando intactos los "
                "datasets cargados (accidentes, features, etc)."
            ),
        ):
            removed = _reset_tab_state_keys(tab)
            if removed:
                st.success(
                    f"Borrados {len(removed)} keys: {', '.join(removed)}"
                )
                try:
                    st.rerun()
                except Exception:
                    # Algunos entornos de test no soportan rerun.
                    pass
            else:
                st.info("No había keys producidos que borrar.")


def _resolve_calibration_sweep_feature_selection(
    *,
    dataset_df: pd.DataFrame,
    model_choice: str,
    features_df: pd.DataFrame,
    features_path: Optional[Path] = None,
    features_source: Optional[str] = None,
) -> Dict[str, object]:
    source = str(
        st.session_state.get(
            "calibration_sweep_feature_source",
            "feature_selection",
        )
    )
    numeric_cols = _get_feature_cols(dataset_df)
    feature_selection_cols = list(numeric_cols)
    if source != "optuna":
        return {
            "feature_source": "feature_selection",
            "feature_source_label": "Feature selection",
            "feature_cols": list(feature_selection_cols),
            "source_note": (
                f"{len(feature_selection_cols)} variables candidatas; se rankean "
                "en train y se toma el K fijo configurado."
            ),
        }
    return {
        "feature_source": "optuna",
        "feature_source_label": "Optuna (best_feature_cols)",
        "feature_cols": list(feature_selection_cols),
        "source_note": (
            f"{len(feature_selection_cols)} variables candidatas; Optuna optimiza "
            "top_k y genera best_feature_cols en esta corrida."
        ),
    }


def _calibration_sweep_protocol_preview(
    *,
    model_name: str,
    feature_source: str,
    optuna_objective_mode: str,
    candidate_feature_cols: Sequence[str],
    feature_k_config: Dict[str, object],
    objective_metrics: Sequence[str],
    calibration_methods: Sequence[str],
    threshold_objectives: Sequence[str],
    test_size: float,
    val_size: float,
    n_trials: int,
    timeout: int,
    optuna_n_jobs: int,
    parallel_jobs: int,
    xgb_parallel_jobs: int,
    far_target: float,
    alerts_per_day: float,
    fn_cost: float,
    fp_cost: float,
    robust_folds: int,
    search_space: Dict[str, object],
    optuna_pruning_config: Dict[str, object],
    random_state: int,
    segment_info: Dict[str, object],
    event_path: Path,
    features_path: Path,
    dataset_date_start: Optional[object],
    dataset_date_end: Optional[object],
) -> Dict[str, object]:
    objective_mode = (
        CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
        if str(optuna_objective_mode).strip().lower()
        == CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
        else CALIBRATION_SWEEP_OBJECTIVE_MODE_SCALAR
    )
    protocol_version = (
        CALIBRATION_SWEEP_MULTIOBJECTIVE_PROTOCOL_VERSION
        if objective_mode == CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
        else CALIBRATION_SWEEP_PROTOCOL_VERSION
    )
    feature_k_config_payload = dict(feature_k_config or {})
    feature_k_mode = str(
        feature_k_config_payload.get("mode") or "fixed_feature_list"
    ).strip().lower()
    ranking_method = str(
        feature_k_config_payload.get("ranking_method") or "rf"
    ).strip().lower()
    candidate_cols = [str(col) for col in candidate_feature_cols]
    top_k_min_value: Optional[int] = None
    top_k_max_value: Optional[int] = None
    top_k_step_value: Optional[int] = None
    if feature_k_mode == "optuna_top_k":
        top_k_grid = _k_grid_values(
            k_min=int(feature_k_config_payload.get("k_min", 10)),
            k_max=int(feature_k_config_payload.get("k_max", 100)),
            k_step=int(feature_k_config_payload.get("k_step", 10)),
            feature_count=len(candidate_cols),
        )
        if top_k_grid:
            top_k_min_value = int(top_k_grid[0])
            top_k_max_value = int(top_k_grid[-1])
            top_k_step_value = int(feature_k_config_payload.get("k_step", 10))
    elif feature_k_mode != "fixed_top_k":
        feature_k_mode = "fixed_feature_list"

    return {
        "protocol_family": CALIBRATION_SWEEP_PROTOCOL_FAMILY,
        "protocol_version": protocol_version,
        "model_name": str(model_name),
        "threshold_protocol": "robust",
        "optuna_objective_mode": objective_mode,
        "feature_source": str(feature_source),
        "selected_features": list(candidate_cols),
        "selected_feature_count": int(len(candidate_cols)),
        "candidate_feature_count": int(len(candidate_cols)),
        "feature_k_mode": str(feature_k_mode),
        "feature_k_config": dict(feature_k_config_payload),
        "ranking_method": ranking_method
        if feature_k_mode in {"fixed_top_k", "optuna_top_k"}
        else None,
        "top_k_min": top_k_min_value,
        "top_k_max": top_k_max_value,
        "top_k_step": top_k_step_value,
        "objective_metrics": list(objective_metrics),
        "multiobjective_metrics": list(CALIBRATION_SWEEP_MULTIOBJECTIVE_METRICS)
        if objective_mode == CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
        else [],
        "multiobjective_directions": list(CALIBRATION_SWEEP_MULTIOBJECTIVE_DIRECTIONS)
        if objective_mode == CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
        else [],
        "calibration_methods": list(calibration_methods),
        "threshold_objectives": list(threshold_objectives),
        "balance_modes": list(CALIBRATION_SWEEP_BALANCE_MODES),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "n_trials": int(n_trials),
        "timeout": int(timeout),
        "optuna_n_jobs": int(optuna_n_jobs),
        "parallel_jobs": int(parallel_jobs),
        "xgb_parallel_jobs": int(xgb_parallel_jobs),
        "far_target": float(far_target),
        "alerts_per_day": float(alerts_per_day),
        "fn_cost": float(fn_cost),
        "fp_cost": float(fp_cost),
        "robust_folds": int(robust_folds),
        "search_space": dict(search_space),
        "optuna_pruning": dict(optuna_pruning_config),
        "random_state": int(random_state),
        "segment_info": dict(segment_info or {}),
        "event_path": str(event_path or ""),
        "features_path": str(features_path or ""),
        "dataset_date_start": (
            None
            if dataset_date_start is None
            else str(pd.Timestamp(dataset_date_start))
        ),
        "dataset_date_end": (
            None if dataset_date_end is None else str(pd.Timestamp(dataset_date_end))
        ),
    }


def _load_calibration_sweep_result_frames(
    result_state: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir_text = str(result_state.get("checkpoint_run_dir") or "").strip()
    run_dir = Path(run_dir_text) if run_dir_text else None

    def _load_frame(filename: str, fallback: object) -> pd.DataFrame:
        if run_dir is not None:
            candidate = run_dir / "results" / filename
            if candidate.exists():
                try:
                    return pd.read_csv(candidate)
                except Exception:
                    pass
        if isinstance(fallback, pd.DataFrame):
            return fallback
        return pd.DataFrame()

    best_summary_df = _load_frame(
        "best_summary.csv",
        result_state.get("best_summary_df"),
    )
    leaderboard_df = _load_frame(
        "leaderboard.csv",
        result_state.get("leaderboard_df"),
    )
    pareto_front_df = _load_frame(
        "pareto_front.csv",
        result_state.get("pareto_front_df"),
    )
    grid_results_df = _load_frame(
        "grid_results.csv",
        result_state.get("grid_results_df"),
    )
    return best_summary_df, leaderboard_df, pareto_front_df, grid_results_df


def _calibration_sweep_manifest_from_state(
    result_state: Dict[str, object],
) -> Dict[str, object]:
    manifest = result_state.get("checkpoint_manifest")
    if isinstance(manifest, dict):
        return manifest

    manifest_path_text = str(result_state.get("checkpoint_manifest_path") or "").strip()
    if manifest_path_text:
        manifest_path = Path(manifest_path_text)
        if manifest_path.exists():
            manifest = _read_json_file(manifest_path)
            if manifest:
                return manifest

    run_dir_text = str(result_state.get("checkpoint_run_dir") or "").strip()
    if run_dir_text:
        manifest_path = Path(run_dir_text) / "manifest.json"
        if manifest_path.exists():
            manifest = _read_json_file(manifest_path)
            if manifest:
                return manifest
    return {}


def _first_frame_value(df: pd.DataFrame, column: str) -> Optional[object]:
    if not isinstance(df, pd.DataFrame) or df.empty or column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    value = values.iloc[0]
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _metadata_value(value: object) -> object:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _calibration_sweep_metadata_rows(
    result_state: Dict[str, object],
    grid_results_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    manifest = _calibration_sweep_manifest_from_state(result_state)
    protocol = result_state.get("protocol")
    if not isinstance(protocol, dict):
        protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        protocol = {}

    def _first(*values: object) -> Optional[object]:
        for value in values:
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    rows: List[Dict[str, object]] = []

    def _add(group: str, field: str, *values: object) -> None:
        value = _first(*values)
        if value is None:
            return
        rows.append(
            {
                "grupo": group,
                "campo": field,
                "valor": _metadata_value(value),
            }
        )

    _add("Corrida", "run_id", manifest.get("run_id"), result_state.get("run_id"))
    _add("Corrida", "computed_run_id", manifest.get("computed_run_id"))
    _add("Corrida", "status", manifest.get("status"))
    _add("Corrida", "result_status", manifest.get("result_status"))
    _add("Corrida", "created_at", manifest.get("created_at"))
    _add("Corrida", "updated_at", manifest.get("updated_at"))
    _add("Corrida", "completed_at", manifest.get("completed_at"))
    _add("Corrida", "last_error", manifest.get("last_error"))

    progress = manifest.get("progress")
    if isinstance(progress, dict):
        _add("Progreso", "completed_steps", progress.get("completed_steps"))
        _add("Progreso", "total_steps", progress.get("total_steps"))
        _add("Progreso", "current_step_id", progress.get("current_step_id"))

    _add(
        "Insumos",
        "event_path",
        _first_frame_value(grid_results_df, "event_path"),
    )
    _add(
        "Insumos",
        "features_path",
        _first_frame_value(grid_results_df, "features_path"),
    )
    _add(
        "Insumos",
        "feature_source",
        protocol.get("feature_source"),
        _first_frame_value(grid_results_df, "feature_source"),
    )
    _add(
        "Insumos",
        "selected_feature_count",
        protocol.get("selected_feature_count"),
        _first_frame_value(grid_results_df, "selected_feature_count"),
    )
    _add("Insumos", "segment_info", protocol.get("segment_info"))
    _add("Insumos", "dataset_date_start", protocol.get("dataset_date_start"))
    _add("Insumos", "dataset_date_end", protocol.get("dataset_date_end"))

    for key in ("train_rows", "val_rows", "test_rows"):
        _add("Split", key, _first_frame_value(grid_results_df, key))

    _add("Protocolo", "protocol_family", manifest.get("protocol_family"), protocol.get("protocol_family"))
    _add("Protocolo", "protocol_version", manifest.get("protocol_version"), protocol.get("protocol_version"))
    _add("Protocolo", "model_name", protocol.get("model_name"), _first_frame_value(grid_results_df, "model_name"))
    _add("Protocolo", "threshold_protocol", protocol.get("threshold_protocol"))
    _add("Protocolo", "objective_metrics", protocol.get("objective_metrics"))
    _add("Protocolo", "calibration_methods", protocol.get("calibration_methods"))
    _add("Protocolo", "threshold_objectives", protocol.get("threshold_objectives"))
    _add("Protocolo", "balance_modes", protocol.get("balance_modes"))
    _add("Protocolo", "test_size", protocol.get("test_size"))
    _add("Protocolo", "val_size", protocol.get("val_size"))
    _add("Protocolo", "n_trials", protocol.get("n_trials"))
    _add("Protocolo", "timeout", protocol.get("timeout"))
    _add("Protocolo", "optuna_n_jobs", protocol.get("optuna_n_jobs"))
    _add("Protocolo", "parallel_jobs", protocol.get("parallel_jobs"))
    _add("Protocolo", "xgb_parallel_jobs", protocol.get("xgb_parallel_jobs"))
    _add("Protocolo", "far_target", protocol.get("far_target"))
    _add("Protocolo", "alerts_per_day", protocol.get("alerts_per_day"))
    _add("Protocolo", "fn_cost", protocol.get("fn_cost"))
    _add("Protocolo", "fp_cost", protocol.get("fp_cost"))
    _add("Protocolo", "robust_folds", protocol.get("robust_folds"))

    return rows


def _calibration_sweep_selected_features(
    result_state: Dict[str, object],
    grid_results_df: pd.DataFrame,
) -> List[str]:
    manifest = _calibration_sweep_manifest_from_state(result_state)
    protocol = result_state.get("protocol")
    if not isinstance(protocol, dict):
        protocol = manifest.get("protocol")
    if isinstance(protocol, dict):
        selected_features = protocol.get("selected_features")
        if isinstance(selected_features, list):
            return [str(feature) for feature in selected_features]

    raw_features = _first_frame_value(grid_results_df, "selected_features")
    if raw_features is None:
        return []
    if isinstance(raw_features, list):
        return [str(feature) for feature in raw_features]
    if isinstance(raw_features, str):
        try:
            parsed = json.loads(raw_features)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(feature) for feature in parsed]
        return [raw_features]
    return []


def _feature_list_from_jsonish(value: object) -> List[str]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return []


def _calibration_sweep_best_feature_cols(
    *,
    best_summary_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    grid_results_df: pd.DataFrame,
) -> List[str]:
    for frame in (best_summary_df, leaderboard_df, grid_results_df):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        work = frame.copy()
        if "status" in work.columns:
            completed = work[
                work["status"].astype(str).str.lower().eq("completed")
            ]
            if not completed.empty:
                work = completed
        if "rank" in work.columns:
            work["_rank_sort"] = pd.to_numeric(work["rank"], errors="coerce")
            work = work.sort_values("_rank_sort", na_position="last")
        if work.empty:
            continue
        row = work.iloc[0]
        for column in ("best_feature_cols", "selected_features"):
            if column not in work.columns:
                continue
            features = _feature_list_from_jsonish(row.get(column))
            if features:
                return features
    return []


def _calibration_sweep_split_db_path(
    result_state: Dict[str, object],
) -> Optional[Path]:
    manifest = _calibration_sweep_manifest_from_state(result_state)
    steps_index = manifest.get("steps_index")
    if isinstance(steps_index, dict):
        split_step = steps_index.get("split_freeze")
        if isinstance(split_step, dict):
            artifact_paths = split_step.get("artifact_paths")
            if isinstance(artifact_paths, dict):
                path_text = str(artifact_paths.get("splits_duckdb") or "").strip()
                if path_text:
                    path = Path(path_text)
                    if path.exists():
                        return path

    run_dir_text = str(result_state.get("checkpoint_run_dir") or "").strip()
    if run_dir_text:
        candidate = Path(run_dir_text) / "dataset" / "splits.duckdb"
        if candidate.exists():
            return candidate
    return None


def _load_calibration_sweep_split_metadata(
    split_db_path: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    if split_db_path is None:
        return pd.DataFrame(), pd.DataFrame(), "No hay splits.duckdb registrado."
    if duckdb is None:
        return pd.DataFrame(), pd.DataFrame(), "duckdb no está disponible."
    if not split_db_path.exists():
        return pd.DataFrame(), pd.DataFrame(), f"No existe {split_db_path}."

    try:
        con = duckdb.connect(str(split_db_path), read_only=True)
        try:
            table_names = [str(row[0]) for row in con.execute("SHOW TABLES").fetchall()]
            split_tables = [
                table_name
                for table_name in ("train", "val", "test")
                if table_name in table_names
            ]
            split_rows: List[Dict[str, object]] = []
            table_columns: Dict[str, List[str]] = {}
            for table_name in split_tables:
                table_ref = _duckdb_quote_identifier(table_name)
                columns = con.execute(f"DESCRIBE {table_ref}").fetchdf()[
                    "column_name"
                ].tolist()
                table_columns[table_name] = [str(column) for column in columns]
                row: Dict[str, object] = {
                    "split": table_name,
                    "rows": int(
                        con.execute(f"SELECT COUNT(*) FROM {table_ref}").fetchone()[0]
                    ),
                    "columns": int(len(columns)),
                }
                if "interval_start" in columns:
                    min_ts, max_ts = con.execute(
                        f"SELECT MIN(interval_start), MAX(interval_start) FROM {table_ref}"
                    ).fetchone()
                    row["interval_start_min"] = min_ts
                    row["interval_start_max"] = max_ts
                if {"portico_last", "portico_next"}.issubset(columns):
                    row["segments"] = int(
                        con.execute(
                            "SELECT COUNT(DISTINCT "
                            "CAST(portico_last AS VARCHAR) || ' -> ' || "
                            f"CAST(portico_next AS VARCHAR)) FROM {table_ref}"
                        ).fetchone()[0]
                    )
                if "km_last" in columns:
                    row["km_last_min"], row["km_last_max"] = con.execute(
                        f"SELECT MIN(km_last), MAX(km_last) FROM {table_ref}"
                    ).fetchone()
                if "km_next" in columns:
                    row["km_next_min"], row["km_next_max"] = con.execute(
                        f"SELECT MIN(km_next), MAX(km_next) FROM {table_ref}"
                    ).fetchone()
                split_rows.append(row)

            segment_selects: List[str] = []
            for table_name in split_tables:
                columns = table_columns.get(table_name, [])
                if not {"portico_last", "portico_next"}.issubset(columns):
                    continue
                table_ref = _duckdb_quote_identifier(table_name)
                km_last_expr = "km_last" if "km_last" in columns else "NULL"
                km_next_expr = "km_next" if "km_next" in columns else "NULL"
                segment_selects.append(
                    "SELECT "
                    "CAST(portico_last AS VARCHAR) AS portico_last, "
                    "CAST(portico_next AS VARCHAR) AS portico_next, "
                    f"{km_last_expr} AS km_last, "
                    f"{km_next_expr} AS km_next "
                    f"FROM {table_ref}"
                )

            segment_df = pd.DataFrame()
            if segment_selects:
                union_sql = " UNION ALL ".join(segment_selects)
                segment_df = con.execute(
                    "SELECT portico_last, portico_next, "
                    "MIN(km_last) AS km_last, MIN(km_next) AS km_next, "
                    "COUNT(*) AS rows "
                    f"FROM ({union_sql}) "
                    "GROUP BY portico_last, portico_next "
                    "ORDER BY rows DESC "
                    "LIMIT 50"
                ).fetchdf()

            return pd.DataFrame(split_rows), segment_df, None
        finally:
            con.close()
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), str(exc)


def _calibration_sweep_artifact_rows(
    result_state: Dict[str, object],
) -> List[Dict[str, object]]:
    manifest = _calibration_sweep_manifest_from_state(result_state)
    artifact_rows: List[Dict[str, object]] = []

    def _add(source: str, name: str, value: object) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        artifact_rows.append({"origen": source, "artefacto": name, "path": text})

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for name, value in artifacts.items():
            _add("manifest.artifacts", str(name), value)

    steps_index = manifest.get("steps_index")
    if isinstance(steps_index, dict):
        for step_name, step_payload in steps_index.items():
            if not isinstance(step_payload, dict):
                continue
            artifact_paths = step_payload.get("artifact_paths")
            if not isinstance(artifact_paths, dict):
                continue
            for name, value in artifact_paths.items():
                _add(f"steps_index.{step_name}", str(name), value)

    return artifact_rows


def _render_calibration_sweep_metadata(
    result_state: Dict[str, object],
    *,
    best_summary_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    grid_results_df: pd.DataFrame,
) -> None:
    with st.expander("Metadata disponible", expanded=False):
        metadata_rows = _calibration_sweep_metadata_rows(result_state, grid_results_df)
        if metadata_rows:
            st.markdown("**Manifest, protocolo e insumos**")
            st.dataframe(
                _prepare_dataframe_for_streamlit(pd.DataFrame(metadata_rows)),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No hay metadata general disponible.")

        selected_features = _calibration_sweep_selected_features(
            result_state,
            grid_results_df,
        )
        if selected_features:
            st.markdown("**Pool candidato**")
            st.caption(f"Total: {len(selected_features):,}")
            st.dataframe(
                pd.DataFrame({"feature": selected_features}),
                width="stretch",
                hide_index=True,
            )

        best_feature_cols = _calibration_sweep_best_feature_cols(
            best_summary_df=best_summary_df,
            leaderboard_df=leaderboard_df,
            grid_results_df=grid_results_df,
        )
        if best_feature_cols:
            st.markdown("**Variables efectivas del mejor resultado**")
            st.caption(f"Total: {len(best_feature_cols):,}")
            st.dataframe(
                pd.DataFrame({"feature": best_feature_cols}),
                width="stretch",
                hide_index=True,
            )

        split_db_path = _calibration_sweep_split_db_path(result_state)
        split_df, segment_df, split_error = _load_calibration_sweep_split_metadata(
            split_db_path
        )
        st.markdown("**Porticos, tramo y splits**")
        if split_db_path is not None:
            st.caption(f"Split congelado: {split_db_path}")
        st.caption(
            "`segment_info` y el rango temporal quedan en el protocolo del checkpoint; "
            "los porticos también se pueden auditar desde `splits.duckdb` cuando "
            "existen columnas `portico_last` y `portico_next`."
        )
        if isinstance(split_df, pd.DataFrame) and not split_df.empty:
            st.dataframe(split_df, width="stretch", hide_index=True)
        elif split_error:
            st.info(split_error)
        if isinstance(segment_df, pd.DataFrame) and not segment_df.empty:
            st.markdown("**Tramos inferidos desde splits**")
            st.dataframe(segment_df, width="stretch", hide_index=True)

        artifact_rows = _calibration_sweep_artifact_rows(result_state)
        if artifact_rows:
            st.markdown("**Artefactos registrados**")
            st.dataframe(
                pd.DataFrame(artifact_rows),
                width="stretch",
                hide_index=True,
            )


def _render_calibration_sweep_results(
    result_state: Dict[str, object],
    *,
    key_prefix: str,
) -> None:
    if not isinstance(result_state, dict) or not result_state:
        return

    (
        best_summary_df,
        leaderboard_df,
        pareto_front_df,
        grid_results_df,
    ) = _load_calibration_sweep_result_frames(result_state)

    run_id = str(result_state.get("run_id") or "").strip()
    status = str(result_state.get("result_status") or "").strip().lower()
    if run_id:
        if status == "completed":
            st.success(f"Experimento de calibración finalizado. Run ID: {run_id}.")
        else:
            st.info(f"Experimento de calibración disponible. Run ID: {run_id}.")

    run_dir = result_state.get("checkpoint_run_dir")
    manifest_path = result_state.get("checkpoint_manifest_path")
    if run_dir:
        st.caption(f"Checkpoint: {run_dir}")
    if manifest_path:
        st.caption(f"Manifest: {manifest_path}")
    if bool(result_state.get("loaded_from_selection")):
        st.caption("Resultado cargado desde el selector de checkpoints.")
    elif bool(result_state.get("loaded_from_checkpoint")):
        st.caption("Resultado cargado desde un checkpoint compatible completado.")
    elif bool(result_state.get("auto_resumed")):
        st.caption("Corrida reanudada desde un checkpoint compatible incompleto.")
    source_note = str(result_state.get("feature_source_note") or "").strip()
    if source_note:
        st.caption(source_note)

    _render_calibration_sweep_metadata(
        result_state,
        best_summary_df=best_summary_df,
        leaderboard_df=leaderboard_df,
        grid_results_df=grid_results_df,
    )

    best_columns = [
        "selection_scope",
        "rank",
        "balance_mode",
        "optuna_objective_metric",
        "optuna_objective_mode",
        "calibration_method",
        "threshold_objective",
        "feature_k_mode",
        "best_top_k",
        "selected_feature_count",
        "best_feature_cols",
        "stability_score",
        "pareto_front",
        "far_gate_pass",
        "far_gate_fallback",
        "pruning_proxy_score",
        "val_mcc",
        "val_brier_score",
        "val_pr_auc",
        "val_recall_at_alerts_per_day",
        "val_true_positives",
        "val_false_negatives",
        "val_far",
        "test_mcc",
        "test_brier_score",
        "test_pr_auc",
        "test_recall_at_alerts_per_day",
        "test_true_positives",
        "test_false_negatives",
        "test_far",
    ]
    if isinstance(best_summary_df, pd.DataFrame) and not best_summary_df.empty:
        st.markdown("**Mejores combinaciones**")
        st.dataframe(
            best_summary_df[
                [col for col in best_columns if col in best_summary_df.columns]
            ],
            width="stretch",
        )

    leaderboard_columns = [
        "rank",
        "pareto_front",
        "stability_score",
        "rankable",
        "model_name",
        "balance_mode",
        "optuna_objective_metric",
        "optuna_objective_mode",
        "calibration_method",
        "threshold_objective",
        "threshold_protocol",
        "feature_k_mode",
        "best_top_k",
        "selected_feature_count",
        "best_feature_cols",
        "decision_threshold",
        "far_gate_pass",
        "far_gate_fallback",
        "pruning_proxy_score",
        "val_mcc",
        "val_brier_score",
        "val_pr_auc",
        "val_recall_at_alerts_per_day",
        "val_true_positives",
        "val_false_negatives",
        "val_far",
        "val_positive_support",
        "val_tp_capture",
        "val_fn_rate",
        "test_mcc",
        "test_brier_score",
        "test_pr_auc",
        "test_recall_at_alerts_per_day",
        "test_true_positives",
        "test_false_negatives",
        "test_far",
        "test_positive_support",
        "test_tp_capture",
        "test_fn_rate",
    ]
    grid_columns = [
        "status",
        "error",
        "model_name",
        "balance_mode",
        "optuna_objective_metric",
        "optuna_objective_mode",
        "calibration_method",
        "threshold_objective",
        "threshold_protocol",
        "feature_k_mode",
        "best_top_k",
        "selected_feature_count",
        "best_feature_cols",
        "rankable",
        "pareto_front",
        "stability_score",
        "far_gate_pass",
        "far_gate_fallback",
        "pruning_proxy_score",
        "val_mcc",
        "val_brier_score",
        "val_pr_auc",
        "val_recall_at_alerts_per_day",
        "val_true_positives",
        "val_false_negatives",
        "val_far",
        "test_mcc",
        "test_brier_score",
        "test_pr_auc",
        "test_recall_at_alerts_per_day",
        "test_true_positives",
        "test_false_negatives",
        "test_far",
    ]
    tab_leaderboard, tab_pareto, tab_grid = st.tabs(
        ["Leaderboard", "Pareto", "Grid completo"]
    )
    with tab_leaderboard:
        if isinstance(leaderboard_df, pd.DataFrame) and not leaderboard_df.empty:
            st.dataframe(
                leaderboard_df[
                    [col for col in leaderboard_columns if col in leaderboard_df.columns]
                ],
                width="stretch",
            )
        else:
            st.info("No hay leaderboard disponible.")
    with tab_pareto:
        if isinstance(pareto_front_df, pd.DataFrame) and not pareto_front_df.empty:
            st.dataframe(
                pareto_front_df[
                    [col for col in leaderboard_columns if col in pareto_front_df.columns]
                ],
                width="stretch",
            )
        else:
            st.info("No hay combinaciones no dominadas disponibles.")
    with tab_grid:
        if isinstance(grid_results_df, pd.DataFrame) and not grid_results_df.empty:
            st.dataframe(
                grid_results_df[
                    [col for col in grid_columns if col in grid_results_df.columns]
                ],
                width="stretch",
            )
        else:
            st.info("No hay resultados completos disponibles.")


def _apply_feature_source_for_model_tab(
    *,
    model_choice: str,
    balance_mode: str,
    calibration_method: str,
    base_df: Optional[pd.DataFrame] = None,
    features_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, object]]:
    """Radio para elegir origen de variables en la pestana Modelos.

    Si el usuario elige Optuna, matchea automaticamente los best_feature_cols
    de cada corrida de Optuna (Base, Cluster, Base + Cluster) con el modelo
    correspondiente y la variante sin/con SMOTE que se entrena en Modelos.
    """
    source = st.session_state.get("model_feature_source", "feature_selection")
    source_options = {
        "Feature selection": "feature_selection",
        "Optuna (best_feature_cols)": "optuna",
    }
    reverse_source = {v: k for k, v in source_options.items()}
    current_label = reverse_source.get(source, "Feature selection")
    chosen_label = st.radio(
        "Origen de variables",
        list(source_options.keys()),
        index=list(source_options.keys()).index(current_label),
        horizontal=True,
        key="model_feature_source_radio",
        help=(
            "Feature selection: usa la seleccion manual de la pestana Feature "
            "selection (ranking sobre train). Optuna: matchea automaticamente "
            "los best_feature_cols de Optuna con los modelos Base, Cluster y "
            "Base + Cluster."
        ),
    )
    st.session_state["model_feature_source"] = source_options[chosen_label]

    if st.session_state["model_feature_source"] != "optuna":
        _clear_model_feature_group_overrides()
        return None

    # Opt-in explícito para fallback de calibración. Por default Modelos
    # exige que Optuna haya optimizado EXACTAMENTE la calibración elegida
    # aquí; si el usuario quiere aceptar un match de otra calibración
    # dentro del mismo balance_mode, debe marcarlo conscientemente.
    allow_calibration_fallback = st.checkbox(
        "Aceptar fallback de calibración de Optuna",
        value=bool(
            st.session_state.get("allow_optuna_calibration_fallback", False)
        ),
        key="allow_optuna_calibration_fallback",
        help=(
            "Cuando Optuna no optimizó exactamente la calibración elegida "
            "aquí (p. ej. optimizó Platt pero aquí pediste Isotonic), "
            "aceptar la mejor disponible dentro del mismo balance_mode. "
            "Desactivado por default: entrenar con parámetros ajustados "
            "a otra calibración cambia el tradeoff y puede degradar la "
            "calidad del threshold."
        ),
    )

    dataset_df = base_df if isinstance(base_df, pd.DataFrame) else features_df
    if not isinstance(dataset_df, pd.DataFrame) or dataset_df.empty:
        _clear_model_feature_group_overrides()
        st.warning(
            "No hay dataset cargado para calcular el match automatico con Optuna."
        )
        return None

    features_df_state = (
        features_df
        if isinstance(features_df, pd.DataFrame)
        else st.session_state.get("flow_features_df")
    )
    features_path = st.session_state.get("flow_features_path")
    features_source = st.session_state.get("flow_features_source")
    feature_key = _feature_selection_key(
        features_path,
        features_source,
        features_df_state if isinstance(features_df_state, pd.DataFrame) else dataset_df,
    )

    selected_features = st.session_state.get("selected_features")
    numeric_cols = _get_feature_cols(dataset_df)
    numeric_cols_set = set(numeric_cols)
    cluster_cols_set = set(_get_cluster_cols(dataset_df))
    if selected_features is None:
        cols_all = list(numeric_cols)
    else:
        cols_all = [col for col in selected_features if col in numeric_cols]
    cols_base = [col for col in cols_all if col not in cluster_cols_set]
    cols_cluster_only = [col for col in cols_all if col in cluster_cols_set]

    # Fingerprint del dataset actual para detectar drift respecto al dataset
    # usado cuando se entrenó Optuna.
    current_fingerprint = _dataset_content_fingerprint(
        features_df_state if isinstance(features_df_state, pd.DataFrame) else dataset_df
    )

    store = st.session_state.get("optuna_results_store") or {}

    groups = [
        ("base", "Base", cols_base),
        ("cluster", "Cluster", cols_cluster_only),
        ("base_cluster", "Base + Cluster", cols_all),
    ]

    summary_lines: List[str] = []
    any_match = False
    fallback_groups: List[str] = []
    missing_cols_by_group: Dict[str, List[str]] = {}
    drift_groups: List[str] = []
    empty_after_filter_groups: List[str] = []
    for group_key, group_label, cols in groups:
        state_key = _MODEL_FEATURE_GROUP_OVERRIDE_KEYS[group_key]

        if group_key == "cluster" and not cols_cluster_only:
            st.session_state.pop(state_key, None)
            summary_lines.append(f"{group_label}: sin variables de cluster")
            continue
        if (
            group_key == "base_cluster"
            and cols_cluster_only == []
            and set(cols_all) == set(cols_base)
        ):
            st.session_state.pop(state_key, None)
            summary_lines.append(f"{group_label}: igual a Base")
            continue

        match = _lookup_optuna_best_feature_cols(
            store=store,
            feature_key=feature_key,
            cols=cols,
            model_choice=model_choice,
            balance_mode=balance_mode,
            calibration_method=calibration_method,
            allow_calibration_fallback=allow_calibration_fallback,
        )
        if match is None:
            st.session_state.pop(state_key, None)
            # Distinguir "no hay Optuna" vs "sólo hay con otra calibración
            # y el usuario no aceptó fallback". Esto último es ahora el
            # motivo más frecuente si el default queda en False.
            if not allow_calibration_fallback:
                alt_match = _lookup_optuna_best_feature_cols(
                    store=store,
                    feature_key=feature_key,
                    cols=cols,
                    model_choice=model_choice,
                    balance_mode=balance_mode,
                    calibration_method=calibration_method,
                    allow_calibration_fallback=True,
                )
                if alt_match is not None:
                    alt_label = (
                        alt_match.get("calibration_method_label")
                        or alt_match.get("calibration_method")
                        or "otra calibración"
                    )
                    summary_lines.append(
                        f"✗ {group_label}: Optuna optimizó {alt_label} "
                        "(active fallback para usarlo)"
                    )
                    continue
            summary_lines.append(f"✗ {group_label}: sin Optuna")
            continue

        # Validar que best_feature_cols existan en el dataset actual. Si
        # Optuna corrió con otro esquema (columnas renombradas, archivo
        # recalculado, selección distinta) este filtro lo detecta en lugar
        # de fallar al entrenar.
        raw_best_cols = [str(col) for col in match.get("best_feature_cols") or []]
        valid_cols = [col for col in raw_best_cols if col in numeric_cols_set]
        missing_cols = [col for col in raw_best_cols if col not in numeric_cols_set]
        if missing_cols:
            missing_cols_by_group[group_label] = missing_cols

        if not valid_cols:
            # Todas las cols de Optuna están ausentes → no podemos aplicar
            # el override, caemos a Feature selection para este grupo.
            st.session_state.pop(state_key, None)
            empty_after_filter_groups.append(group_label)
            summary_lines.append(
                f"✗ {group_label}: Optuna propone {len(raw_best_cols)} vars "
                "ausentes del dataset"
            )
            continue

        st.session_state[state_key] = list(valid_cols)
        any_match = True

        # Detectar drift del dataset (mismo feature_key pero fingerprint distinto).
        stored_fingerprint = match.get("dataset_fingerprint")
        if (
            stored_fingerprint
            and current_fingerprint
            and stored_fingerprint != current_fingerprint
        ):
            drift_groups.append(group_label)

        parts = [f"{len(valid_cols)} vars"]
        if missing_cols:
            parts.append(f"{len(missing_cols)} ignoradas")
        if match.get("best_top_k") is not None:
            parts.append(f"top_k={int(match['best_top_k'])}")
        if match.get("best_score") is not None:
            label = match.get("objective_label") or "score"
            parts.append(f"best {label}={float(match['best_score']):.4f}")
        resolved_calibration_label = match.get("calibration_method_label")
        if resolved_calibration_label:
            parts.append(f"calibración={resolved_calibration_label}")
        if bool(match.get("used_fallback")):
            fallback_groups.append(group_label)
            parts.append("fallback")
        if group_label in drift_groups:
            parts.append("drift")
        summary_lines.append(f"✓ {group_label}: " + ", ".join(parts))

    if not any_match:
        st.warning(
            "No se encontraron resultados de Optuna para el modelo "
            f"'{model_choice}' ({_optuna_balance_mode_label(balance_mode)} | "
            f"{_calibration_method_label(calibration_method)}) que "
            "coincidan con los grupos Base, Cluster o Base + Cluster. "
            "Ejecute Optuna primero o cambie el modelo."
        )
    else:
        if empty_after_filter_groups:
            st.error(
                "Optuna tiene resultados para "
                + ", ".join(empty_after_filter_groups)
                + " pero sus `best_feature_cols` no existen en el dataset "
                "actual. Ese/esos grupos volverán a usar Feature selection. "
                "Ejecute Optuna nuevamente sobre este dataset o elija otro "
                "origen de variables."
            )
        if missing_cols_by_group:
            detail_lines = []
            for group_label, missing in missing_cols_by_group.items():
                preview = ", ".join(missing[:5])
                if len(missing) > 5:
                    preview = f"{preview}, … (+{len(missing) - 5})"
                detail_lines.append(f"- **{group_label}**: {preview}")
            st.warning(
                "Algunas columnas que Optuna seleccionó no están en el "
                "dataset actual y fueron ignoradas:\n"
                + "\n".join(detail_lines)
            )
        if drift_groups:
            st.warning(
                "⚠️ Dataset drift detectado en: "
                + ", ".join(drift_groups)
                + ". El schema (columnas/dtypes/filas) actual difiere del "
                "usado cuando se entrenó Optuna. Los resultados pueden no "
                "ser representativos; considere re-ejecutar Optuna."
            )
        if fallback_groups:
            # Si el usuario aceptó el fallback opt-in, el mensaje es
            # informativo, no un warning.
            fallback_message = (
                "Fallback de calibración aceptado para: "
                + ", ".join(fallback_groups)
                + f". No existe match exacto para "
                f"{_calibration_method_label(calibration_method)} dentro de "
                f"{_optuna_balance_mode_label(balance_mode)}; se usó la "
                "mejor calibración disponible dentro del mismo balance_mode."
            )
            if allow_calibration_fallback:
                st.info(fallback_message)
            else:
                st.warning(fallback_message)

    st.caption(
        "Match Optuna por grupo "
        f"({_optuna_balance_mode_label(balance_mode)} | "
        f"{_calibration_method_label(calibration_method)}) — "
        + "   ".join(summary_lines)
    )
    return None


def _render_flow_features_preview(features_df: pd.DataFrame) -> None:
    if features_df is None or features_df.empty:
        st.info("No hay variables en memoria.")
        return
    source = st.session_state.get("flow_features_source") or "-"
    path = st.session_state.get("flow_features_path")
    st.caption(
        f"Fuente: {source} | Filas: {len(features_df):,} | "
        f"Columnas: {len(features_df.columns)}"
    )
    if path:
        st.caption(f"Archivo: {path}")
    st.dataframe(features_df.head(50), width="stretch")


def _render_cluster_features_preview(features_df: pd.DataFrame) -> None:
    if features_df is None or features_df.empty:
        st.info("No hay variables de cluster en memoria.")
        return
    source = st.session_state.get("cluster_features_source") or "-"
    path = st.session_state.get("cluster_features_path")
    st.caption(
        f"Fuente: {source} | Filas: {len(features_df):,} | "
        f"Columnas: {len(features_df.columns)}"
    )
    if path:
        st.caption(f"Archivo: {path}")
    st.dataframe(features_df.head(50), width="stretch")


def _render_cluster_features_section(
    *,
    flow_df: Optional[pd.DataFrame],
    flow_batch_paths: Optional[List[str]],
    cluster_choice: str,
    include_counts: bool,
    key_prefix: str,
) -> None:
    features_df = st.session_state.get("cluster_features_df")
    has_memory = isinstance(features_df, pd.DataFrame) and not features_df.empty

    source_options = ["Cargar existentes", "Calcular nuevas", "En memoria"]
    source_key = f"{key_prefix}_cluster_features_source"
    if source_key not in st.session_state or st.session_state[source_key] not in source_options:
        st.session_state[source_key] = "En memoria" if has_memory else "Calcular nuevas"
    source = st.radio(
        "Fuente de variables de cluster",
        source_options,
        horizontal=True,
        key=source_key,
    )

    if source == "En memoria":
        if not has_memory:
            st.info("No hay variables de cluster en memoria.")
            return
        _render_cluster_features_preview(features_df)
        st.subheader("Exportar variables de cluster")
        export_name = st.text_input(
            "Nombre de archivo (sin .duckdb)",
            value="accident_cluster_features_export",
            key=f"{key_prefix}_cluster_export_name",
        )
        if st.button(
            "Exportar variables de cluster",
            key=f"{key_prefix}_cluster_export_btn",
        ):
            out_path = RESULTS_DIR / f"{export_name.strip()}.duckdb"
            try:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                _write_df_to_duckdb(features_df, out_path, "cluster_features")
            except Exception as exc:
                st.error(f"No se pudo exportar: {exc}")
            else:
                st.success(f"Exportado en {out_path}")
        return

    if source == "Cargar existentes":
        feature_files = _list_cluster_feature_files()
        if not feature_files:
            st.warning(
                "No se encontraron archivos accident_cluster_features_*.duckdb en Resultados."
            )
            return
        names = [path.name for path in feature_files]
        selected = st.selectbox(
            "Archivo de variables de cluster",
            options=["(ninguno)"] + names,
            key=f"{key_prefix}_cluster_features_file",
        )
        if st.button(
            "Cargar variables de cluster",
            key=f"{key_prefix}_cluster_load_btn",
        ):
            if selected == "(ninguno)":
                st.warning("Seleccione un archivo de Resultados.")
            else:
                progress = st.progress(0)
                try:
                    with st.spinner("Cargando variables de cluster existentes..."):
                        progress.progress(10)
                        path = RESULTS_DIR / selected
                        if path.suffix.lower() != ".duckdb":
                            st.error("Solo se permiten archivos .duckdb.")
                            return
                        if duckdb is None:
                            st.error("duckdb no esta instalado.")
                            return
                        con = None
                        try:
                            con = duckdb.connect(str(path), read_only=True)
                            table_rows = con.execute("SHOW TABLES").fetchall()
                            tables = [row[0] for row in table_rows]
                            table_name = _pick_duckdb_table(
                                tables, ["cluster_features", "features"]
                            )
                            if not table_name:
                                st.error("La base de datos esta vacia.")
                                return
                            table_ref = _duckdb_quote_identifier(table_name)
                            progress.progress(40)
                            loaded_df = con.execute(
                                f"SELECT * FROM {table_ref}"
                            ).df()
                        except Exception as exc:
                            st.error(f"No se pudo cargar {selected}: {exc}")
                            return
                        finally:
                            if con is not None:
                                con.close()
                        progress.progress(60)
                        if "interval_start" in loaded_df.columns:
                            loaded_df["interval_start"] = pd.to_datetime(
                                loaded_df["interval_start"], errors="coerce"
                            )
                        if "portico" in loaded_df.columns:
                            loaded_df["portico"] = (
                                loaded_df["portico"].astype(str).str.strip()
                            )
                        if not {"portico", "interval_start"}.issubset(
                            loaded_df.columns
                        ):
                            st.warning(
                                "El archivo no contiene portico e interval_start."
                            )
                            return
                        progress.progress(85)
                        st.session_state["cluster_features_df"] = loaded_df
                        st.session_state["cluster_features_path"] = str(path)
                        st.session_state["cluster_features_source"] = "duckdb"
                        progress.progress(100)
                        st.success(
                            f"Variables de cluster cargadas: {len(loaded_df):,} filas"
                        )
                finally:
                    progress.empty()
        return

    if cluster_choice == "(sin clusters)":
        st.warning("Seleccione un archivo de etiquetas de cluster.")
        return

    has_batches = bool(flow_batch_paths)
    if (flow_df is None or flow_df.empty) and not has_batches:
        st.warning("No hay flujos ni lotes para calcular variables de cluster.")
        return

    flow_rows = int(st.session_state.get("flow_rows_loaded", 0))
    st.caption(
        f"Flujos en memoria: {0 if flow_df is None else len(flow_df):,} | "
        f"Lotes disponibles: {len(flow_batch_paths or []):,} | "
        f"Filas en lotes: {flow_rows:,}"
    )
    if st.button(
        "Calcular variables de cluster",
        key=f"{key_prefix}_cluster_calc_btn",
    ):
        cluster_path = RESULTS_DIR / cluster_choice
        try:
            cluster_labels = _load_cluster_labels(cluster_path)
            if flow_df is None or flow_df.empty:
                batch_paths = [Path(path) for path in (flow_batch_paths or [])]
                cluster_features = _compute_cluster_features_from_batches(
                    batch_paths,
                    cluster_labels,
                    include_counts=include_counts,
                )
            else:
                cluster_features = _call_compute_cluster_features(
                    flow_df,
                    cluster_labels,
                    include_counts=include_counts,
                )
        except Exception as exc:
            st.error(f"No se pudieron calcular variables de cluster: {exc}")
            return

        if cluster_features.empty:
            st.warning("No se pudieron generar variables de cluster.")
            return

        if "interval_start" in cluster_features.columns:
            cluster_features["interval_start"] = pd.to_datetime(
                cluster_features["interval_start"], errors="coerce"
            )
        if "portico" in cluster_features.columns:
            cluster_features["portico"] = (
                cluster_features["portico"].astype(str).str.strip()
            )
        st.session_state["cluster_features_df"] = cluster_features
        st.session_state["cluster_features_source"] = "calculadas"
        try:
            saved_path = _save_cluster_features(
                cluster_features, cluster_choice=cluster_choice
            )
        except Exception as exc:
            st.session_state["cluster_features_path"] = None
            st.warning(f"No se pudieron guardar las variables de cluster: {exc}")
            st.success(f"Variables de cluster calculadas: {len(cluster_features):,} filas")
        else:
            st.session_state["cluster_features_path"] = str(saved_path)
            st.success(f"Variables de cluster calculadas: {len(cluster_features):,} filas")
            st.caption(f"Guardadas en {saved_path}")

def _split_balanced_dataset(
    df: pd.DataFrame,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    if "split" not in df.columns:
        return None
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        return None
    return train_df, test_df


def _cluster_selector(key_prefix: str) -> Tuple[str, bool]:
    cluster_files = _list_cluster_label_files()
    cluster_names = [path.name for path in cluster_files]
    options = ["(sin clusters)"] + cluster_names
    default_choice = st.session_state.get("cluster_choice", "(sin clusters)")
    if default_choice not in options:
        default_choice = "(sin clusters)"
    selected = st.selectbox(
        "Archivo de etiquetas de cluster",
        options=options,
        index=options.index(default_choice),
        key=f"{key_prefix}_cluster_choice",
    )
    include_default = bool(st.session_state.get("include_counts", False))
    include_counts = st.checkbox(
        "Incluir Flow por cluster",
        value=include_default,
        key=f"{key_prefix}_include_counts",
    )
    st.session_state["cluster_choice"] = selected
    st.session_state["include_counts"] = include_counts
    return selected, include_counts


def _load_cluster_labels(cluster_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(cluster_path, usecols=["plate", "cluster_label"])
    except ValueError:
        return pd.read_csv(cluster_path)


def _call_compute_cluster_features(
    flows_df: pd.DataFrame,
    cluster_labels_df: pd.DataFrame,
    **kwargs: object,
) -> pd.DataFrame:
    try:
        sig = inspect.signature(compute_cluster_features)
    except (TypeError, ValueError):
        return compute_cluster_features(flows_df, cluster_labels_df, **kwargs)
    allowed = {key: value for key, value in kwargs.items() if key in sig.parameters}
    missing = set(kwargs.keys()) - set(allowed.keys())
    unsupported = [
        key for key in missing if kwargs.get(key) not in (None, False)
    ]
    if unsupported:
        st.warning(
            "La version de utils.py no soporta: "
            + ", ".join(sorted(unsupported))
            + ". Reinicie la app o actualice el codigo."
        )
    return compute_cluster_features(flows_df, cluster_labels_df, **allowed)


def _compute_cluster_features_from_batches(
    batch_paths: List[Path],
    cluster_labels: pd.DataFrame,
    *,
    include_counts: bool,
    include_entropy: bool = False,
    include_speed: bool = False,
    include_density: bool = False,
    include_delta_speed: bool = False,
    include_delta_density: bool = False,
    interval_minutes: int = 5,
    lanes: int = 3,
) -> pd.DataFrame:
    if not batch_paths:
        return pd.DataFrame()

    need_speed = (
        include_speed
        or include_density
        or include_delta_speed
        or include_delta_density
    )
    compute_speed = include_speed or include_delta_speed
    compute_density = include_density or include_delta_density
    frames: List[pd.DataFrame] = []
    progress = _StreamlitProgress(total=len(batch_paths))
    for idx, path in enumerate(batch_paths, start=1):
        progress.set_description(f"Lote {idx}/{len(batch_paths)}: clusters")
        usecols = ["FECHA", "PORTICO", "MATRICULA"]
        if need_speed:
            usecols.append("VELOCIDAD")
        batch_df = pd.read_csv(path, usecols=usecols)
        if not batch_df.empty:
            batch_features = _call_compute_cluster_features(
                batch_df,
                cluster_labels,
                interval_minutes=interval_minutes,
                include_counts=include_counts,
                include_entropy=include_entropy,
                include_speed=compute_speed,
                include_density=compute_density,
                include_delta_speed=False,
                include_delta_density=False,
                lanes=lanes,
            )
            if not batch_features.empty:
                frames.append(batch_features)
        progress.update(1)
    progress.close()

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    if "interval_start" in result.columns:
        result["interval_start"] = pd.to_datetime(
            result["interval_start"], errors="coerce"
        )
    if "portico" in result.columns:
        result["portico"] = result["portico"].astype(str).str.strip()

    if include_delta_speed or include_delta_density:
        result = result.sort_values(["portico", "interval_start"]).reset_index(
            drop=True
        )
    if include_delta_speed:
        speed_cols = [
            col for col in result.columns if col.startswith("cluster_speed_")
        ]
        if speed_cols:
            delta_speed = (
                result.groupby("portico")[speed_cols].diff().fillna(0)
            )
            delta_speed.columns = [
                col.replace("cluster_speed_", "cluster_delta_speed_")
                for col in speed_cols
            ]
            result = pd.concat([result, delta_speed], axis=1)
    if include_delta_density:
        density_cols = [
            col for col in result.columns if col.startswith("cluster_density_")
        ]
        if density_cols:
            delta_density = (
                result.groupby("portico")[density_cols].diff().fillna(0)
            )
            delta_density.columns = [
                col.replace("cluster_density_", "cluster_delta_density_")
                for col in density_cols
            ]
            result = pd.concat([result, delta_density], axis=1)

    if not include_speed:
        speed_cols = [
            col for col in result.columns if col.startswith("cluster_speed_")
        ]
        if speed_cols:
            result = result.drop(columns=speed_cols)
    if not include_density:
        density_cols = [
            col for col in result.columns if col.startswith("cluster_density_")
        ]
        if density_cols:
            result = result.drop(columns=density_cols)

    return result


def _build_cluster_dataset(
    base_df: pd.DataFrame,
    *,
    cluster_features_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    base_cluster_cols = _get_cluster_cols(base_df)
    if base_cluster_cols:
        merged = base_df.copy()
        numeric_cols = _get_feature_cols(merged)
        merged[numeric_cols] = merged[numeric_cols].fillna(0)
        return merged
    if cluster_features_df is None or cluster_features_df.empty:
        st.warning(
            "Cargue variables de cluster en la pestana Feature engineering."
        )
        return None
    cluster_features = cluster_features_df.copy()
    if not {"portico", "interval_start"}.issubset(cluster_features.columns):
        st.warning(
            "Las variables de cluster cargadas no contienen portico e interval_start."
        )
        return None
    cluster_features["interval_start"] = pd.to_datetime(
        cluster_features["interval_start"], errors="coerce"
    )
    cluster_features["portico"] = (
        cluster_features["portico"].astype(str).str.strip()
    )
    merged = base_df.merge(
        cluster_features,
        how="left",
        on=["portico", "interval_start"],
    )
    numeric_cols = _get_feature_cols(merged)
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    return merged





def _render_event_tab() -> None:
    st.subheader("Eventos (accidentes)")
    st.markdown(
        "Selecciona uno o varios archivos de eventos desde la carpeta Datos."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No se encontraron archivos de eventos en la carpeta Datos.")
        return

    selected_names = st.multiselect(
        "Archivos de eventos disponibles",
        [path.name for path in event_files],
        default=[path.name for path in event_files],
    )

    if st.button("Procesar eventos"):
        if not selected_names:
            st.warning("Seleccione al menos un archivo de eventos.")
            return

        try:
            porticos_df = load_porticos()
            porticos_source = "Datos/Porticos.csv"
        except FileNotFoundError:
            st.error(
                "No se encontro Porticos.csv en la carpeta Datos. "
                "Agreguelo antes de continuar."
            )
            return
        except Exception as exc:
            st.error(f"No se pudieron cargar los porticos: {exc}")
            return

        frames: List[pd.DataFrame] = []
        for name in selected_names:
            path = DATA_DIR / name
            try:
                frames.append(
                    pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
                )
            except UnicodeDecodeError:
                frames.append(
                    pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
                )
            except Exception as exc:
                st.error(f"No se pudo leer {name}: {exc}")
                return

        raw_df = pd.concat(frames, ignore_index=True)
        try:
            acc_df, excluded = process_accidentes_df(
                raw_df, porticos_df, return_excluded=True
            )
        except Exception as exc:
            st.error(f"No se pudieron procesar los eventos: {exc}")
            return

        st.session_state["accidents_df"] = acc_df
        st.session_state["accident_files"] = selected_names
        st.session_state["porticos_source"] = porticos_source

        st.success(
            f"Accidentes procesados: {len(acc_df):,} | "
            f"Excluidos sin portico: {len(excluded):,}"
        )

    accidents_df = st.session_state.get("accidents_df")
    if accidents_df is None or accidents_df.empty:
        st.info("No hay accidentes cargados.")
        return

    st.caption(
        f"Archivos: {', '.join(st.session_state.get('accident_files', []))} | "
        f"Porticos: {st.session_state.get('porticos_source') or '-'}"
    )

    preview_rows = st.slider(
        "Filas de vista previa",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        key="events_preview_rows",
    )

    preview_cols = [
        col
        for col in [
            "accidente_time",
            "ultimo_portico",
            "proximo_portico",
            "duracion_accidente",
            "severidad",
        ]
        if col in accidents_df.columns
    ]
    st.dataframe(
        accidents_df[preview_cols].head(preview_rows),
        width="stretch",
    )

    def _clean_p(val):
        s = str(val).strip()
        return s[:-2] if s.endswith(".0") else s

    st.subheader("Secuencia de porticos")
    st.caption(
        "Secuencia ordenada por eje/calzada segun el archivo Porticos.csv."
    )
    try:
        porticos_df = load_porticos()
    except Exception as exc:
        st.warning(f"No se pudieron cargar los porticos: {exc}")
        return
    if porticos_df is None or porticos_df.empty:
        st.warning("No hay porticos disponibles.")
        return

    porticos = porticos_df.copy()
    porticos["orden_num"] = pd.to_numeric(porticos["orden"], errors="coerce")
    porticos["km_num"] = pd.to_numeric(porticos["km"], errors="coerce")
    porticos["eje_norm"] = (
        porticos["eje"].astype(str).str.strip().str.upper()
    )
    porticos["calzada_norm"] = (
        porticos["calzada"].astype(str).str.strip().str.upper()
    )
    porticos = porticos.dropna(
        subset=["orden_num", "km_num", "eje_norm", "calzada_norm"]
    )

    sequence_rows: List[Dict[str, object]] = []
    for _, group in porticos.groupby(["eje_norm", "calzada_norm"]):
        group = group.sort_values("orden_num")
        sequence = [
            f"{_clean_p(row['portico'])}({row['km_num']:g})"
            for _, row in group.iterrows()
        ]
        if not sequence:
            continue
        sequence_rows.append(
            {
                "Eje": group["eje"].iloc[0],
                "Calzada": group["calzada"].iloc[0],
                "Secuencia": " -> ".join(sequence),
            }
        )
    if not sequence_rows:
        st.info("No se pudo construir la secuencia de porticos.")
    else:
        sequence_df = (
            pd.DataFrame(sequence_rows)
            .sort_values(["Eje", "Calzada"])
            .reset_index(drop=True)
        )
        st.dataframe(sequence_df, width="stretch")

    st.subheader("Accidentes por tramo")
    st.caption(
        "Conteo de accidentes entre porticos consecutivos segun el orden."
    )

    km_col = _find_match_column(accidents_df, ["Km.", "Km", "Kilometro"])
    eje_col = _find_match_column(accidents_df, ["Eje"])
    calzada_col = _find_match_column(accidents_df, ["Calzada"])
    if km_col is None or eje_col is None or calzada_col is None:
        st.warning(
            "No se encontraron columnas de km/eje/calzada en accidentes."
        )
        return

    segments: List[Dict[str, object]] = []
    for _, group in porticos.groupby(["eje_norm", "calzada_norm"]):
        group = group.sort_values("orden_num")
        for i in range(len(group) - 1):
            start = group.iloc[i]
            end = group.iloc[i + 1]
            segments.append(
                {
                    "Eje": start["eje"],
                    "Calzada": start["calzada"],
                    "orden_inicio": int(start["orden_num"]),
                    "portico_inicio": str(start["portico"]).strip(),
                    "km_inicio": float(start["km_num"]),
                    "orden_fin": int(end["orden_num"]),
                    "portico_fin": str(end["portico"]).strip(),
                    "km_fin": float(end["km_num"]),
                }
            )
    segments_df = pd.DataFrame(segments)
    if segments_df.empty:
        st.info("No se pudieron construir los tramos de porticos.")
        return

    acc_seg = accidents_df[[eje_col, calzada_col, km_col]].copy()
    acc_seg = acc_seg.rename(
        columns={eje_col: "eje", calzada_col: "calzada", km_col: "km_acc"}
    )
    acc_seg["km_acc"] = pd.to_numeric(
        acc_seg["km_acc"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    acc_seg = acc_seg.dropna(subset=["km_acc", "eje", "calzada"])

    segment_keys: List[Dict[str, object]] = []
    assigned_indices = set()
    for row in acc_seg.itertuples():
        try:
            cand = find_candidate_porticos(
                acc_km=row.km_acc,
                porticos_df=porticos_df,
                eje=row.eje,
                calzada=row.calzada,
            )
        except Exception:
            continue
        posterior = cand.get("posterior")
        cercano = cand.get("cercano")
        if posterior is None or cercano is None:
            continue
        assigned_indices.add(row.Index)
        segment_keys.append(
            {
                "Eje": posterior["eje"],
                "Calzada": posterior["calzada"],
                "portico_inicio": _clean_p(posterior["portico"]),
                "portico_fin": _clean_p(cercano["portico"]),
            }
        )

    if segment_keys:
        counts_df = (
            pd.DataFrame(segment_keys)
            .groupby(
                ["Eje", "Calzada", "portico_inicio", "portico_fin"],
                dropna=False,
            )
            .size()
            .reset_index(name="accidentes")
        )
    else:
        counts_df = pd.DataFrame(
            columns=[
                "Eje",
                "Calzada",
                "portico_inicio",
                "portico_fin",
                "accidentes",
            ]
        )

    segments_df = segments_df.merge(
        counts_df,
        on=["Eje", "Calzada", "portico_inicio", "portico_fin"],
        how="left",
    )
    segments_df["accidentes"] = (
        segments_df["accidentes"].fillna(0).astype(int)
    )
    segments_df = segments_df.sort_values(
        ["Eje", "Calzada", "orden_inicio"]
    ).reset_index(drop=True)

    display_cols = [
        "Eje",
        "Calzada",
        "portico_inicio",
        "km_inicio",
        "portico_fin",
        "km_fin",
        "accidentes",
    ]
    st.dataframe(segments_df[display_cols], width="stretch")

    missing_info = int(len(accidents_df) - len(acc_seg))
    unassigned = int(len(acc_seg) - len(segment_keys))
    if missing_info > 0:
        st.caption(
            f"Accidentes sin km/eje/calzada: {missing_info:,}"
        )
    if unassigned > 0:
        st.caption(
            f"Accidentes sin tramo asignado: {unassigned:,}"
        )
        unassigned_indices = acc_seg.index.difference(assigned_indices)
        if not unassigned_indices.empty:
            st.markdown("**Detalle de accidentes sin tramo asignado**")
            cols_to_show = [
                col
                for col in ["accidente_time", eje_col, calzada_col, km_col, "Descripcion", "SubTipo"]
                if col in accidents_df.columns
            ]
            st.dataframe(accidents_df.loc[unassigned_indices, cols_to_show], width="stretch")


def _render_match_tab() -> None:
    st.subheader("Match accidentes vs features")

    accidents_df = st.session_state.get("accidents_df")
    if accidents_df is None or accidents_df.empty:
        st.info("Cargue accidentes en la pestana Eventos.")
        return

    features_df = st.session_state.get("flow_features_df")
    if features_df is None or features_df.empty:
        st.info(
            "Calcule variables en la pestana Feature engineering para comparar."
        )
        return

    if not {"portico", "interval_start"}.issubset(features_df.columns):
        if not {"portico_last", "interval_start"}.issubset(features_df.columns):
            st.warning(
                "Las variables no tienen portico (o portico_last) e interval_start para hacer el match."
            )
            return
        # Use portico_last as portico for matching
        features_df = features_df.rename(columns={"portico_last": "portico"})

    if not {"accidente_time", "ultimo_portico"}.issubset(accidents_df.columns):
        st.warning(
            "Los accidentes no tienen accidente_time y ultimo_portico."
        )
        return

    acc = accidents_df.copy()
    acc["_acc_time"] = pd.to_datetime(
        acc["accidente_time"], errors="coerce"
    )
    acc["_acc_portico"] = acc["ultimo_portico"].astype(str).str.strip()
    invalid_tokens = {"", "nan", "none", "null"}
    acc["_acc_portico"] = acc["_acc_portico"].where(
        ~acc["_acc_portico"].str.lower().isin(invalid_tokens), None
    )
    interval_minutes = DEFAULT_INTERVAL_MINUTES
    acc["intervalo_accidente"] = acc["_acc_time"].dt.floor(
        f"{interval_minutes}min"
    ) - pd.Timedelta(minutes=interval_minutes)

    features = features_df[["portico", "interval_start"]].copy()
    features["interval_start"] = pd.to_datetime(
        features["interval_start"], errors="coerce"
    )
    features["portico"] = features["portico"].astype(str).str.strip()
    features = features.dropna(subset=["portico", "interval_start"])
    features = features.drop_duplicates(subset=["portico", "interval_start"])
    if features.empty:
        st.warning("No hay pares portico/interval_start en variables.")
        return

    match_index = pd.MultiIndex.from_frame(
        features[["portico", "interval_start"]]
    )
    acc_index = pd.MultiIndex.from_frame(
        acc[["_acc_portico", "intervalo_accidente"]].rename(
            columns={
                "_acc_portico": "portico",
                "intervalo_accidente": "interval_start",
            }
        )
    )
    acc["matched"] = acc_index.isin(match_index)
    acc["missing_time"] = acc["_acc_time"].isna()
    acc["missing_portico"] = acc["_acc_portico"].isna()

    features_porticos = set(features["portico"].unique())
    acc["portico_in_features"] = acc["_acc_portico"].isin(features_porticos)

    features_min = features["interval_start"].min()
    features_max = features["interval_start"].max()
    if pd.isna(features_min) or pd.isna(features_max):
        acc["out_of_range"] = False
    else:
        range_end = features_max + pd.Timedelta(minutes=interval_minutes)
        acc["out_of_range"] = acc["_acc_time"].notna() & (
            (acc["_acc_time"] < features_min)
            | (acc["_acc_time"] > range_end)
        )

    acc["match_estado"] = np.select(
        [
            acc["matched"],
            acc["missing_time"],
            acc["missing_portico"],
            acc["out_of_range"],
            ~acc["portico_in_features"],
        ],
        [
            "con features",
            "sin fecha/hora",
            "sin ultimo_portico",
            "fuera de rango de features",
            "portico sin datos en features",
        ],
        default="intervalo sin datos en features",
    )

    total_acc = int(len(acc))
    matched_count = int(acc["matched"].sum())
    unmatched_count = total_acc - matched_count
    col1, col2, col3 = st.columns(3)
    col1.metric("Accidentes", f"{total_acc:,}")
    col2.metric("Con features", f"{matched_count:,}")
    col3.metric("Sin features", f"{unmatched_count:,}")

    acc_min = acc["_acc_time"].min()
    acc_max = acc["_acc_time"].max()
    if pd.notna(acc_min) and pd.notna(acc_max):
        st.caption(
            f"Accidentes: {acc_min:%Y-%m-%d %H:%M} a {acc_max:%Y-%m-%d %H:%M}"
        )
    if pd.notna(features_min) and pd.notna(features_max):
        st.caption(
            f"Features: {features_min:%Y-%m-%d %H:%M} a {features_max:%Y-%m-%d %H:%M}"
        )

    st.subheader("Resumen de match")
    summary_df = (
        acc["match_estado"]
        .value_counts()
        .rename_axis("estado")
        .reset_index(name="count")
    )
    st.dataframe(summary_df, width="stretch")

    detail_candidates = [
        "accidente_time",
        "intervalo_accidente",
        "ultimo_portico",
        "Km.",
        "Km",
        "Eje",
        "Calzada",
        "Via",
        "Descripcion",
        "SubTipo",
        "duracion_accidente",
        "severidad",
    ]
    detail_cols = _select_detail_columns(acc, detail_candidates)
    if "match_estado" not in detail_cols:
        detail_cols.append("match_estado")

    rows_to_show = st.slider(
        "Filas por tabla",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="match_rows",
    )

    st.subheader("Accidentes fuera de rango de features")
    st.caption(
        "Accidentes con fecha fuera del rango temporal cubierto por las features."
    )
    out_of_range_df = acc.loc[acc["match_estado"] == "fuera de rango de features"]
    if out_of_range_df.empty:
        st.info("No hay accidentes fuera de rango.")
    else:
        st.dataframe(
            out_of_range_df[detail_cols].head(rows_to_show),
            width="stretch",
        )

    st.subheader("Accidentes con portico sin datos en features")
    st.caption(
        "El portico del accidente no existe en la columna portico de features."
    )
    missing_portico_df = acc.loc[
        acc["match_estado"] == "portico sin datos en features"
    ]
    if missing_portico_df.empty:
        st.info("No hay accidentes con portico sin datos en features.")
    else:
        st.dataframe(
            missing_portico_df[detail_cols].head(rows_to_show),
            width="stretch",
        )

    st.subheader("Accidentes en intervalo sin datos en features")
    st.caption(
        "Portico y fecha en rango, pero falta el registro en el intervalo exacto."
    )
    missing_interval_df = acc.loc[
        acc["match_estado"] == "intervalo sin datos en features"
    ]
    if missing_interval_df.empty:
        st.info("No hay accidentes con intervalo sin datos en features.")
    else:
        st.dataframe(
            missing_interval_df[detail_cols].head(rows_to_show),
            width="stretch",
        )


def _render_variables_tab() -> None:
    st.subheader("Feature engineering")

    features_df = st.session_state.get("flow_features_df")
    flow_df = st.session_state.get("flow_df")
    flow_batch_paths = st.session_state.get("flow_batch_paths")
    cluster_features_state = st.session_state.get("cluster_features_df")
    if isinstance(features_df, pd.DataFrame) and not features_df.empty:
        cluster_cols = _get_cluster_cols(features_df)
        if cluster_cols and (cluster_features_state is None or cluster_features_state.empty):
            if {"portico", "interval_start"}.issubset(features_df.columns):
                st.session_state["cluster_features_df"] = features_df[
                    ["portico", "interval_start"] + cluster_cols
                ].copy()
                st.session_state["cluster_features_source"] = "integradas"
                st.session_state["cluster_features_path"] = None
        elif (
            not cluster_cols
            and isinstance(cluster_features_state, pd.DataFrame)
            and not cluster_features_state.empty
        ):
            st.session_state["cluster_features_df"] = None
            st.session_state["cluster_features_source"] = None
            st.session_state["cluster_features_path"] = None
            st.session_state["cluster_choice"] = "(sin clusters)"

    

    has_memory = isinstance(features_df, pd.DataFrame) and not features_df.empty

    source_options = [
        "Cargar existentes",
        "Calcular nuevas",
        "En memoria",
    ]
    source_key = "variables_source"
    if source_key not in st.session_state or st.session_state[source_key] not in source_options:
        st.session_state[source_key] = (
            "En memoria" if has_memory else "Calcular nuevas"
        )
    source = st.radio(
        "Fuente",
        source_options,
        horizontal=True,
        key=source_key,
    )

    if source == "En memoria":
        if not has_memory:
            st.info("No hay variables en memoria.")
            return
        _render_flow_features_preview(features_df)
        st.subheader("Exportar variables")
        default_name = "accident_flow_features_export"
        export_name = st.text_input(
            "Nombre de archivo (sin .duckdb)",
            value=default_name,
            key="flow_features_export_name",
        )
        if st.button("Exportar variables"):
            out_path = RESULTS_DIR / f"{export_name.strip()}.duckdb"
            try:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                _write_df_to_duckdb(features_df, out_path, "flow_features")
            except Exception as exc:
                st.error(f"No se pudo exportar: {exc}")
            else:
                st.success(f"Exportado en {out_path}")
        return

    if source == "Cargar existentes":
        feature_files = _list_flow_feature_files()
        if not feature_files:
            st.warning(
                "No se encontraron archivos accident_flow_features_*.duckdb en Resultados."
            )
        else:
            names = [path.name for path in feature_files]
            selected = st.selectbox(
                "Archivo de variables",
                options=["(ninguno)"] + names,
                key="flow_features_file",
            )
            allowed_porticos: Optional[set[str]] = None
            if selected != "(ninguno)":
                selected_path = RESULTS_DIR / selected
                allowed_porticos = _load_porticos_from_feature_file(
                    selected_path
                )
                if allowed_porticos is None:
                    st.warning(
                        "No se pudo leer porticos del archivo para filtrar tramos."
                    )
                    allowed_porticos = set()
            accidents_df_existing = st.session_state.get("accidents_df")
            date_start_ts = None
            date_end_inclusive = None
            date_end_exclusive = None
            if (
                isinstance(accidents_df_existing, pd.DataFrame)
                and not accidents_df_existing.empty
                and "accidente_time" in accidents_df_existing.columns
            ):
                acc_times = pd.to_datetime(
                    accidents_df_existing["accidente_time"], errors="coerce"
                ).dropna()
                if not acc_times.empty:
                    min_date = acc_times.min().date()
                    max_date = acc_times.max().date()
                    c_date1, c_date2 = st.columns(2)
                    with c_date1:
                        date_start_input = st.date_input(
                            "Fecha inicio",
                            value=min_date,
                            key="acc_flow_existing_date_start",
                        )
                    with c_date2:
                        date_end_input = st.date_input(
                            "Fecha fin",
                            value=max_date,
                            key="acc_flow_existing_date_end",
                        )
                    if date_start_input and date_end_input:
                        if date_start_input > date_end_input:
                            st.error(
                                "La fecha de inicio no puede ser mayor que la fecha final."
                            )
                        else:
                            date_start_ts = pd.Timestamp(date_start_input)
                            date_end_exclusive = (
                                pd.Timestamp(date_end_input) + pd.Timedelta(days=1)
                            )
                            date_end_inclusive = date_end_exclusive - pd.Timedelta(
                                nanoseconds=1
                            )
            else:
                st.info(
                    "Cargue accidentes en la pestana Eventos para habilitar "
                    "el filtro de fechas."
                )
            tramo_tuple = _build_tramo_selector(
                accidents_df_existing,
                date_start=date_start_ts,
                date_end=date_end_inclusive,
                allowed_porticos=allowed_porticos,
                key="acc_flow_tramo_choice_existing",
            )
            if st.button("Cargar variables"):
                if selected == "(ninguno)":
                    st.warning("Seleccione un archivo de Resultados.")
                else:
                    progress = st.progress(0)
                    try:
                        with st.spinner("Cargando features existentes..."):
                            progress.progress(5)
                            path = RESULTS_DIR / selected
                            if path.suffix.lower() != ".duckdb":
                                st.error("Solo se permiten archivos .duckdb.")
                                return
                            if duckdb is None:
                                st.error("duckdb no esta instalado.")
                                return
                            con = None
                            try:
                                con = duckdb.connect(str(path), read_only=True)
                                table_rows = con.execute("SHOW TABLES").fetchall()
                                tables = [row[0] for row in table_rows]
                                table_name = _pick_duckdb_table(
                                    tables, ["flow_features", "features"]
                                )
                                if not table_name:
                                    st.error("La base de datos esta vacia.")
                                    return
                                table_ref = _duckdb_quote_identifier(table_name)
                                progress.progress(25)
                                cols_info = con.execute(
                                    f"DESCRIBE {table_ref}"
                                ).fetchall()
                                columns = {row[0] for row in cols_info}
                                clauses, params, filter_ok = _build_tramo_duckdb_filters(
                                    tramo_tuple, columns
                                )
                                if not filter_ok:
                                    st.warning(
                                        "El archivo no contiene columnas para filtrar por tramo "
                                        "(se buscaron: portico, portico_last/portico_next, "
                                        "portico_inicio/portico_fin, ultimo_portico)."
                                    )
                                    return
                                if "interval_start" in columns:
                                    if date_start_ts is not None:
                                        clauses.append("interval_start >= ?")
                                        params.append(date_start_ts)
                                    if date_end_exclusive is not None:
                                        clauses.append("interval_start < ?")
                                        params.append(date_end_exclusive)
                                query = f"SELECT * FROM {table_ref}"
                                if clauses:
                                    query += " WHERE " + " AND ".join(clauses)
                                progress.progress(45)
                                loaded_df = con.execute(query, params).df()
                                progress.progress(55)
                            except Exception as exc:
                                st.error(f"No se pudo cargar {selected}: {exc}")
                                return
                            finally:
                                if con is not None:
                                    con.close()

                            if "interval_start" in loaded_df.columns:
                                loaded_df["interval_start"] = pd.to_datetime(
                                    loaded_df["interval_start"], errors="coerce"
                                )
                            has_segment_cols = {
                                "portico_last",
                                "portico_next",
                            }.issubset(loaded_df.columns)
                            has_alt_segment_cols = {
                                "portico_inicio",
                                "portico_fin",
                            }.issubset(loaded_df.columns)
                            if not has_segment_cols and not has_alt_segment_cols:
                                portico_col_found = next(
                                    (
                                        c
                                        for c in [
                                            "portico",
                                            "portico_last",
                                            "ultimo_portico",
                                            "portico_inicio",
                                        ]
                                        if c in loaded_df.columns
                                    ),
                                    None,
                                )
                                if portico_col_found and portico_col_found != "portico":
                                    loaded_df = loaded_df.rename(
                                        columns={portico_col_found: "portico"}
                                    )
                            if "portico" in loaded_df.columns:
                                loaded_df["portico"] = (
                                    loaded_df["portico"].astype(str).str.strip()
                                )
                            progress.progress(70)
                            if loaded_df.empty:
                                if tramo_tuple:
                                    st.warning(
                                        "No se encontraron variables para el tramo seleccionado."
                                    )
                                else:
                                    st.warning("El archivo de features esta vacio.")
                                return
                            progress.progress(85)
                            st.session_state["flow_features_df"] = loaded_df
                            st.session_state["flow_features_path"] = str(path)
                            st.session_state["flow_features_source"] = "duckdb"
                            _set_flow_tramo_selection(tramo_tuple)
                            cluster_cols = _get_cluster_cols(loaded_df)
                            if (
                                cluster_cols
                                and {"portico", "interval_start"}.issubset(
                                    loaded_df.columns
                                )
                            ):
                                st.session_state["cluster_features_df"] = loaded_df[
                                    ["portico", "interval_start"] + cluster_cols
                                ].copy()
                                st.session_state["cluster_features_source"] = "integradas"
                                st.session_state["cluster_features_path"] = None
                                st.session_state["cluster_choice"] = "(sin clusters)"
                            else:
                                st.session_state["cluster_features_df"] = None
                                st.session_state["cluster_features_source"] = None
                                st.session_state["cluster_features_path"] = None
                                st.session_state["cluster_choice"] = "(sin clusters)"
                            progress.progress(100)
                            st.success(
                                f"Variables cargadas: {len(loaded_df):,} filas"
                            )
                    finally:
                        progress.empty()

        return

    summary = _render_flow_summary()
    if summary is None:
        return

    mode = _build_flow_sample_mode_selector(key_prefix="acc_flow")
    
    # Dynamic toggles outside form
    use_batches = st.checkbox(
        "Procesar por lotes (mes/semana)",
        value=True,
        key="acc_flow_use_batches",
    )
    
    include_cluster_vars = st.checkbox(
        "Incluir variables de cluster",
        value=bool(st.session_state.get("acc_flow_include_cluster_vars", True)),
        key="acc_flow_include_cluster_vars",
    )

    with st.form("acc_flow_features_form"):
        sample, percent_mode, range_valid = _build_flow_sample_inputs(
            summary, mode, key_prefix="acc_flow"
        )
        if percent_mode and use_batches:
             st.warning("El muestreo por porcentaje no ignora la opcion de lotes (no compatible).")

        accidents_df = st.session_state.get("accidents_df")
        tramo_tuple = _build_tramo_selector(
            accidents_df,
            date_start=sample.date_start,
            date_end=sample.date_end,
            allowed_porticos=None,
            key="acc_flow_tramo_choice",
        )

        batch_mode = "month"
        if use_batches:
            batch_mode = st.radio(
                "Modo de lotes",
                ["month", "week"],
                horizontal=True,
                key="acc_flow_batch_mode",
            )
        
        keep_flow_in_memory = st.checkbox(
            "Mantener flujos en memoria (usa RAM)",
            value=False,
            disabled=not use_batches,
            key="acc_flow_keep_flows",
        )

        metric_options = {
            "Flow": "flow",
            "Speed": "speed",
            "Speed_std": "speed_std",
            "Density": "density",
            "Delta.Speed": "delta_speed",
            "Delta.Density": "delta_density",
        }
        metrics_selected = st.multiselect(
            "Variables",
            list(metric_options.keys()),
            default=list(metric_options.keys()),
            key="acc_flow_metrics",
        )
        metrics = [metric_options[key] for key in metrics_selected]

        category_options = ["Light", "Heavy", "Motorcycles"]
        categories = st.multiselect(
            "Tipos de vehiculo",
            category_options,
            default=category_options,
            key="acc_flow_categories",
        )

        lanes = st.number_input(
            "Carriles para normalizar Flow",
            min_value=1,
            value=3,
            step=1,
            key="acc_flow_lanes",
        )

        cluster_choice = "(sin clusters)"
        cluster_vars: List[str] = []
        
        if include_cluster_vars:
            cluster_files = _list_cluster_label_files()
            if not cluster_files:
                st.warning("No se encontraron archivos cluster_*.csv en Resultados.")
            else:
                cluster_names = [path.name for path in cluster_files]
                cluster_choice = st.selectbox(
                    "Archivo de etiquetas de cluster",
                    options=["(ninguno)"] + cluster_names,
                    key="acc_flow_cluster_choice",
                )
            cluster_var_options = [
                "Proporciones por cluster",
                "Flow por tipo de cluster",
                "Entropia de cluster",
                "Speed por tipo de cluster",
                "Density por tipo de cluster",
                "Delta.Speed por tipo de cluster",
                "Delta.Density por tipo de cluster",
            ]
            existing_vars = st.session_state.get("acc_flow_cluster_vars")
            default_vars = cluster_var_options
            if isinstance(existing_vars, list):
                normalized: List[str] = []
                for item in existing_vars:
                    if item in {
                        "Conteos por cluster",
                        "Conteo por cluster",
                        "Conteo por tipo de cluster",
                    }:
                        normalized.append("Flow por tipo de cluster")
                    elif item in {"Speed por cluster"}:
                        normalized.append("Speed por tipo de cluster")
                    elif item in {"Density por cluster"}:
                        normalized.append("Density por tipo de cluster")
                    elif item in {
                        "Delta-Speed por tipo de cluster",
                        "Delta.Speed por cluster",
                        "Delta-Speed por cluster",
                    }:
                        normalized.append("Delta.Speed por tipo de cluster")
                    elif item in {
                        "Delta-Density por tipo de cluster",
                        "Delta.Density por cluster",
                        "Delta-Density por cluster",
                    }:
                        normalized.append("Delta.Density por tipo de cluster")
                    else:
                        normalized.append(item)
                st.session_state["acc_flow_cluster_vars"] = normalized
                default_vars = normalized
            multiselect_kwargs = {
                "label": "Variables de cluster",
                "options": cluster_var_options,
                "key": "acc_flow_cluster_vars",
            }
            if "acc_flow_cluster_vars" not in st.session_state:
                multiselect_kwargs["default"] = default_vars
            cluster_vars = st.multiselect(**multiselect_kwargs)

        col_upd, col_run = st.columns(2)
        with col_upd:
            update_filters = st.form_submit_button("Actualizar filtros")
        with col_run:
            run_calculation = st.form_submit_button("Calcular features (5 min)", disabled=not range_valid)

    if run_calculation:
        if duckdb is None:
            st.error("duckdb no esta instalado. Ejecute `pip install duckdb`.")
            return

        if include_cluster_vars:
            if not cluster_vars:
                st.warning("Seleccione al menos una variable de cluster.")
                return
            if cluster_choice in {"(sin clusters)", "(ninguno)"}:
                st.warning("Seleccione un archivo de etiquetas de cluster.")
                return

        # Step 0: Create temp DB and load filtered data
        temp_db_path = RESULTS_DIR / "temp_work_features.duckdb"
        if temp_db_path.exists():
            temp_db_path.unlink()

        con_temp = duckdb.connect(str(temp_db_path))
        try:
            flow_summary = get_flow_db_summary()
            flow_db_path = flow_summary.db_path
            con_temp.execute(f"ATTACH '{flow_db_path}' AS flow_db (READ_ONLY)")
            
            query = "CREATE TABLE work_flujos AS SELECT * FROM flow_db.flujos_duckdb WHERE 1=1"
            params = []
            if sample.date_start:
                query += " AND FECHA >= ?"
                params.append(sample.date_start)
            if sample.date_end:
                query += " AND FECHA <= ?"
                params.append(sample.date_end)
            
            with st.spinner("Creando base de trabajo temporal..."):
                con_temp.execute(query, params)
        except Exception as exc:
            st.error(f"Error creando base temporal: {exc}")
            con_temp.close()
            return

        # Step 1: Create persistent DB
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = _cluster_choice_suffix(cluster_choice) if include_cluster_vars else "sin_cluster"
        features_db_name = f"accident_flow_features_{suffix}_{stamp}.duckdb"
        features_db_path = RESULTS_DIR / features_db_name
        
        con_feat = duckdb.connect(str(features_db_path))

        # Step 2: Batches
        min_max = con_temp.execute("SELECT MIN(FECHA), MAX(FECHA) FROM work_flujos").fetchone()
        if min_max[0] is None:
            st.warning("No se encontraron datos en el rango seleccionado.")
            con_temp.close()
            con_feat.close()
            temp_db_path.unlink()
            return
            
        ranges = _build_batch_ranges(pd.Timestamp(min_max[0]), pd.Timestamp(min_max[1]), batch_mode)
        
        cluster_labels_df = None
        if include_cluster_vars:
            try:
                cluster_labels_df = _load_cluster_labels(RESULTS_DIR / cluster_choice)
            except Exception as exc:
                st.error(f"Error cargando clusters: {exc}")
                con_temp.close()
                con_feat.close()
                return

        # Prepare segments
        try:
            porticos_df = load_porticos()
            all_segments = get_portico_segments(porticos_df)
        except Exception as exc:
            st.error(f"Error cargando segmentos: {exc}")
            con_temp.close()
            con_feat.close()
            return

        target_segments = all_segments
        if tramo_tuple:
            eje_sel, calzada_sel, p_start, p_end = tramo_tuple
            target_segments = all_segments[
                (all_segments["eje"] == eje_sel)
                & (all_segments["calzada"] == calzada_sel)
                & (all_segments["portico_last"] == p_start)
                & (all_segments["portico_next"] == p_end)
            ].copy()
            if target_segments.empty:
                st.warning("El tramo seleccionado no es valido en la configuracion actual de porticos.")
                con_temp.close()
                con_feat.close()
                return

        target_segments["portico_last"] = _normalize_portico_series(
            target_segments["portico_last"]
        )
        target_segments["portico_next"] = _normalize_portico_series(
            target_segments["portico_next"]
        )

        # Prepare cluster args
        include_shares = "Proporciones por cluster" in cluster_vars
        include_flow = "Flow por tipo de cluster" in cluster_vars
        include_speed = "Speed por tipo de cluster" in cluster_vars
        include_density = "Density por tipo de cluster" in cluster_vars
        include_delta_speed = "Delta.Speed por tipo de cluster" in cluster_vars
        include_delta_density = "Delta.Density por tipo de cluster" in cluster_vars
        include_entropy = "Entropia de cluster" in cluster_vars

        progress = _StreamlitProgress(total=len(ranges))
        total_rows = 0
        table_created = False
        diagnostics = {
            "input_rows": 0,
            "feature_rows": 0,
            "step1_rows": 0,
            "final_rows": 0,
            "porticos_features": set(),
            "porticos_segments": set(
                target_segments["portico_last"].dropna().astype(str).tolist()
                + target_segments["portico_next"].dropna().astype(str).tolist()
            ),
        }

        # Step 3 & 4: Process and Store
        for idx, (start, end, label) in enumerate(ranges, start=1):
            progress.set_description(f"Procesando lote {idx}/{len(ranges)}")
            
            # Load batch (all porticos to ensure we have neighbors)
            df_batch = con_temp.execute(
                "SELECT * FROM work_flujos WHERE FECHA >= ? AND FECHA < ?",
                [start, end]
            ).df()
            
            diagnostics["input_rows"] += len(df_batch)
            if not df_batch.empty:
                # Calculate flow features for ALL porticos
                feat_batch = compute_flow_features(
                    df_batch,
                        interval_minutes=5,
                        lanes=int(lanes),
                        metrics=metrics,
                        categories=categories,
                        progress=None,
                    )
                
                if not feat_batch.empty:
                    feat_batch["portico"] = _normalize_portico_series(
                        feat_batch["portico"]
                    )
                    diagnostics["feature_rows"] += len(feat_batch)
                    if len(diagnostics["porticos_features"]) < 2000:
                        diagnostics["porticos_features"].update(
                            feat_batch["portico"].dropna().astype(str).tolist()
                        )
                    if include_cluster_vars and cluster_labels_df is not None:
                        clust_batch = _call_compute_cluster_features(
                            df_batch,
                            cluster_labels_df,
                            interval_minutes=5,
                            include_counts=include_flow,
                            include_entropy=include_entropy,
                            include_speed=include_speed,
                            include_density=include_density,
                            include_delta_speed=include_delta_speed,
                            include_delta_density=include_delta_density,
                            lanes=int(lanes),
                        )
                        if not clust_batch.empty:
                            clust_batch["portico"] = _normalize_portico_series(
                                clust_batch["portico"]
                            )
                            feat_batch = feat_batch.merge(
                                clust_batch,
                                on=["portico", "interval_start"],
                                how="left"
                            )
                            # Fillna for numeric cols
                            num_cols = _get_feature_cols(feat_batch)
                            feat_batch[num_cols] = feat_batch[num_cols].fillna(0)
                            
                            if not include_shares:
                                share_cols = [c for c in feat_batch.columns if c.startswith("cluster_share_")]
                                if share_cols:
                                    feat_batch = feat_batch.drop(columns=share_cols)

                    # --- Transform to Segment Features (Last/Next) ---
                    # feat_batch has [portico, interval_start, ...features...]
                    
                    # 1. Prepare Last
                    df_last = feat_batch.add_prefix("last_")
                    df_last = df_last.rename(columns={"last_interval_start": "interval_start"})
                    
                    # 2. Prepare Next
                    df_next = feat_batch.add_prefix("next_")
                    df_next = df_next.rename(columns={"next_interval_start": "interval_start"})
                    
                    # 3. Join with Segments
                    # Merge segments with Last (on portico_last)
                    # result has: [eje, calzada, portico_last, km_last, portico_next, km_next, interval_start, last_features...]
                    step1 = target_segments.merge(
                        df_last,
                        left_on="portico_last",
                        right_on="last_portico",
                        how="inner"
                    )
                    diagnostics["step1_rows"] += len(step1)
                    
                    # 4. Join with Next (on portico_next AND interval_start)
                    # result has: [..., interval_start, last_features..., next_features...]
                    final_batch = step1.merge(
                        df_next,
                        left_on=["portico_next", "interval_start"],
                        right_on=["next_portico", "interval_start"],
                        how="inner"
                    )
                    diagnostics["final_rows"] += len(final_batch)
                    
                    # Cleanup key columns if redundant
                    # We keep portico_last/next from segments. last_portico/next_portico from features are redundant.
                    final_batch = final_batch.drop(columns=["last_portico", "next_portico"], errors="ignore")

                    if not final_batch.empty:
                        # Store
                        if not table_created:
                            con_feat.execute("CREATE TABLE flow_features AS SELECT * FROM final_batch")
                            table_created = True
                        else:
                            con_feat.execute("INSERT INTO flow_features SELECT * FROM final_batch")
                        
                        total_rows += len(final_batch)
            
            progress.update()
            
        progress.close()
        con_temp.close()
        if temp_db_path.exists():
            temp_db_path.unlink()
        
        # Step 5: Load result
        if table_created:
            with st.spinner("Cargando resultados en memoria..."):
                final_df = con_feat.execute("SELECT * FROM flow_features").df()
            con_feat.close()
            
            if "interval_start" in final_df.columns:
                final_df["interval_start"] = pd.to_datetime(final_df["interval_start"], errors="coerce")
            
            # Normalize strings
            for col in ["portico_last", "portico_next"]:
                if col in final_df.columns:
                    final_df[col] = final_df[col].astype(str).str.strip()

            st.session_state["flow_features_df"] = final_df
            st.session_state["flow_features_path"] = str(features_db_path)
            st.session_state["flow_features_source"] = "calculadas (DB)"
            _set_flow_tramo_selection(tramo_tuple)
            
            # Update cluster features state if included
            # With segment features, we might consider all numeric columns as features
            # or split them. For now, we will store everything in flow_features_df.
            st.session_state["cluster_features_df"] = None 
            st.session_state["cluster_choice"] = cluster_choice if include_cluster_vars else "(sin clusters)"

            st.success(f"Variables calculadas y guardadas en {features_db_name}: {total_rows:,} filas")
        else:
            con_feat.close()
            if features_db_path.exists():
                features_db_path.unlink()
            st.warning("No se generaron variables.")
            seg_porticos = diagnostics["porticos_segments"]
            feat_porticos = diagnostics["porticos_features"]
            if seg_porticos or feat_porticos:
                intersection = len(seg_porticos.intersection(feat_porticos))
                st.info(
                    "Diagnostico: "
                    f"filas flujos={diagnostics['input_rows']:,}, "
                    f"filas features={diagnostics['feature_rows']:,}, "
                    f"segmentos={len(seg_porticos):,}, "
                    f"match porticos={intersection:,}, "
                    f"match last+next={diagnostics['final_rows']:,}."
                )


def _render_feature_selection_tab() -> None:
    st.subheader("Feature selection")

    accidents_df = st.session_state.get("accidents_df")
    features_df = st.session_state.get("flow_features_df")

    if accidents_df is None or accidents_df.empty:
        st.info("Cargue accidentes en la pestana Eventos.")
        return
    if features_df is None or features_df.empty:
        st.info("Calcule variables de flujo en la pestana Feature engineering.")
        return

    base_df = add_accident_target(features_df, accidents_df)
    if base_df.empty:
        st.warning("No se pudo preparar el dataset base.")
        return

    features_path = st.session_state.get("flow_features_path")
    features_source = st.session_state.get("flow_features_source")
    feature_key = _feature_selection_key(
        features_path, features_source, features_df
    )
    feature_id = _feature_selection_id(
        features_path, features_source, features_df
    )
    active_key = st.session_state.get("feature_selection_active_key")
    if active_key != feature_key:
        st.session_state["feature_selection_active_key"] = feature_key
        store = st.session_state.get("feature_selection_store", {})
        entry = store.get(feature_key)
        if entry is None:
            payload, importance_df = _load_feature_selection_from_disk(feature_id)
            if payload or importance_df is not None:
                entry = {
                    "feature_id": feature_id,
                    "selected_features": payload.get("selected_features")
                    if payload
                    else None,
                    "importance_df": importance_df,
                    "importance_hash": None,
                    "params": payload.get("params") if payload else {},
                }
                store[feature_key] = entry
                st.session_state["feature_selection_store"] = store
        if entry:
            if entry.get("importance_df") is not None:
                st.session_state["feature_importances_df"] = entry.get(
                    "importance_df"
                )
            if entry.get("selected_features") is not None:
                st.session_state["selected_features"] = entry.get(
                    "selected_features"
                )
        else:
            st.session_state["selected_features"] = None
            st.session_state["feature_importances_df"] = None

    if features_path:
        st.caption(f"Archivo de features: {features_path}")
    else:
        st.caption("Archivo de features: (sin archivo)")

    feature_cols = _get_feature_cols(base_df)
    if not feature_cols:
        st.warning("No hay variables numericas disponibles.")
        return

    st.caption(
        f"Filas: {len(base_df):,} | Variables numericas: {len(feature_cols)}"
    )

    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.number_input(
            "n_estimators",
            min_value=50,
            value=200,
            step=50,
            key="fs_n_estimators",
        )
    with col2:
        max_depth = st.number_input(
            "max_depth (0 = sin limite)",
            min_value=0,
            value=0,
            step=1,
            key="fs_max_depth",
        )
    random_state = st.number_input(
        "random_state",
        min_value=0,
        value=42,
        step=1,
        key="fs_random_state",
    )

    col_split_a, col_split_b = st.columns(2)
    with col_split_a:
        fs_test_size = st.slider(
            "Test size (para excluir test del ranking)",
            min_value=0.05,
            max_value=0.4,
            value=float(st.session_state.get("test_size", 0.2)),
            step=0.05,
            key="fs_test_size",
            help=(
                "Fraccion temporal reservada como test. El ranking se calcula "
                "sobre train (o train + val) para evitar leakage."
            ),
        )
    with col_split_b:
        fs_use_val = st.checkbox(
            "Excluir tambien validacion",
            value=bool(st.session_state.get("fs_use_val", False)),
            key="fs_use_val",
            help=(
                "Si se activa, el ranking se calcula solo sobre train "
                "(train/val/test). Si se deja desactivado, se usa train + val."
            ),
        )
    fs_val_size = float(st.session_state.get("val_size", 0.2))
    if fs_use_val:
        fs_val_size = st.slider(
            "Validation size",
            min_value=0.05,
            max_value=0.4,
            value=float(st.session_state.get("val_size", 0.2)),
            step=0.05,
            key="fs_val_size",
        )

    if st.button("Calcular importancia"):
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            st.error(
                "scikit-learn no esta instalado. Ejecute `pip install scikit-learn`."
            )
            return

        progress = st.progress(0)
        try:
            # Split temporal: test siempre excluido; val excluido si el usuario lo pide.
            try:
                train_val_df, _test_df = _temporal_train_test_split(
                    base_df,
                    time_col="interval_start",
                    test_size=float(fs_test_size),
                )
            except ValueError as exc:
                st.warning(f"No se pudo hacer split temporal: {exc}")
                return
            if fs_use_val:
                try:
                    fit_df, _val_df = _temporal_train_test_split(
                        train_val_df,
                        time_col="interval_start",
                        test_size=float(fs_val_size),
                    )
                except ValueError as exc:
                    st.warning(f"No se pudo crear validacion temporal: {exc}")
                    return
            else:
                fit_df = train_val_df

            progress.progress(10)
            X = fit_df[feature_cols].fillna(0)
            progress.progress(20)
            y = fit_df["target"].astype(int)
            progress.progress(25)
            if y.nunique() < 2:
                st.warning(
                    "No hay dos clases en el target del split de entrenamiento. "
                    "Ajuste test_size/val_size."
                )
                return
            with st.spinner("Calculando importancia..."):
                progress.progress(30)
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth) if max_depth else None,
                    criterion="gini",
                    random_state=int(random_state),
                    class_weight="balanced",
                    n_jobs=-1,
                )
                model.fit(X, y)
                progress.progress(80)
            importance_df = pd.DataFrame(
                {
                    "variable": feature_cols,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            importance_df = importance_df.reset_index(drop=True)
            progress.progress(95)
            st.session_state["feature_importances_df"] = importance_df
            fit_label = "train" if fs_use_val else "train + val"
            st.session_state["feature_importances_fit_label"] = fit_label
            st.session_state["feature_importances_fit_rows"] = int(len(fit_df))
            progress.progress(100)
            st.success(
                f"Importancias calculadas sobre {fit_label} "
                f"({len(fit_df):,} filas). Test excluido."
            )
        finally:
            progress.empty()

    importance_df = st.session_state.get("feature_importances_df")
    ordered_vars = feature_cols
    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        importance_df = importance_df[
            importance_df["variable"].isin(feature_cols)
        ].copy()
        if importance_df.empty:
            st.session_state["feature_importances_df"] = None
            st.info("Calcule la importancia para ordenar las variables.")
        else:
            st.session_state["feature_importances_df"] = importance_df
            fit_label = st.session_state.get("feature_importances_fit_label")
            fit_rows = st.session_state.get("feature_importances_fit_rows")
            if fit_label:
                st.caption(
                    f"Ranking calculado sobre **{fit_label}** "
                    f"({int(fit_rows or 0):,} filas, test excluido)."
                )
            st.dataframe(importance_df, width="stretch")
            ordered_vars = importance_df["variable"].tolist()
    else:
        st.info("Calcule la importancia para ordenar las variables.")

    selected_features = st.session_state.get("selected_features")
    if selected_features is None:
        selected_features = list(ordered_vars)
    else:
        selected_features = [
            feature for feature in selected_features if feature in ordered_vars
        ]

    col_imp, col_btn = st.columns([2, 1])
    with col_imp:
        if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
            min_imp = float(importance_df["importance"].min())
            max_imp = float(importance_df["importance"].max())
            range_span = max_imp - min_imp
            slider_kwargs = {
                "min_value": min_imp,
                "max_value": max_imp,
                "value": min_imp,
                "format": "%.6f",
            }
            if range_span > 0:
                step = max(range_span / 1000.0, 1e-6)
                if step > range_span:
                    step = range_span
                slider_kwargs["step"] = step
            threshold = st.slider(
                "Seleccionar por umbral de importancia (> value)",
                **slider_kwargs,
            )
            if st.button("Seleccionar > Umbral"):
                subset = importance_df[importance_df["importance"] > threshold]
                new_selection = subset["variable"].tolist()
                st.session_state["selected_features"] = new_selection
                # Explicitly set widget states
                for idx, feature in enumerate(ordered_vars):
                    safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
                    k = f"feature_sel_{idx}_{safe_key}"
                    st.session_state[k] = feature in new_selection
                st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Seleccionar todas"):
            st.session_state["selected_features"] = list(ordered_vars)
            for idx, feature in enumerate(ordered_vars):
                safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
                k = f"feature_sel_{idx}_{safe_key}"
                st.session_state[k] = True
            st.rerun()
    with col_b:
        if st.button("Limpiar seleccion"):
            st.session_state["selected_features"] = []
            for idx, feature in enumerate(ordered_vars):
                safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
                k = f"feature_sel_{idx}_{safe_key}"
                st.session_state[k] = False
            st.rerun()

    st.caption("Selecciona variables para usar en Balance y Modelos.")
    selected = []
    for idx, feature in enumerate(ordered_vars):
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", feature)
        checked = st.checkbox(
            feature,
            value=feature in selected_features,
            key=f"feature_sel_{idx}_{key}",
        )
        if checked:
            selected.append(feature)
    st.session_state["selected_features"] = selected
    st.caption(f"Variables seleccionadas: {len(selected)}")
    st.markdown("**Resumen de variables seleccionadas**")
    if selected:
        st.dataframe(pd.DataFrame({"Variable": selected}), width="stretch")
    else:
        st.info("No hay variables seleccionadas.")

    fs_params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth) if max_depth else None,
        "random_state": int(random_state),
        "test_size": float(fs_test_size),
        "use_val": bool(fs_use_val),
        "val_size": float(fs_val_size) if fs_use_val else None,
        "fit_rows": int(st.session_state.get("feature_importances_fit_rows") or 0),
        "fit_label": st.session_state.get("feature_importances_fit_label"),
    }
    _persist_feature_selection(
        feature_key=feature_key,
        feature_id=feature_id,
        features_path=features_path,
        features_source=features_source,
        features_df=features_df,
        selected_features=selected,
        importance_df=importance_df if isinstance(importance_df, pd.DataFrame) else None,
        params=fs_params,
    )


def _render_optuna_tab() -> None:
    st.subheader("Optuna")
    _render_selected_features_info()
    _render_state_diagnostics("optuna")

    accidents_df = st.session_state.get("accidents_df")
    features_df = st.session_state.get("flow_features_df")

    if accidents_df is None or accidents_df.empty:
        st.info("Cargue accidentes en la pestana Eventos.")
        return
    if features_df is None or features_df.empty:
        st.info("Calcule variables de flujo en la pestana Feature engineering.")
        return

    base_df = add_accident_target(features_df, accidents_df)
    if base_df.empty:
        st.warning("No se pudo preparar el dataset base.")
        return

    features_path = st.session_state.get("flow_features_path")
    features_source = st.session_state.get("flow_features_source")
    feature_key = _feature_selection_key(
        features_path, features_source, features_df
    )
    feature_id = _feature_selection_id(
        features_path, features_source, features_df
    )

    numeric_cols = _get_feature_cols(base_df)
    cluster_cols = _get_cluster_cols(base_df)
    selected_features = st.session_state.get("selected_features")
    if selected_features is None:
        st.warning(
            "Seleccione variables en Feature selection para Optuna."
        )
        return
    if not selected_features:
        st.warning(
            "Seleccione al menos una variable en Feature selection."
        )
        return

    selected_in_numeric = [
        col for col in selected_features if col in numeric_cols
    ]
    missing = [
        col for col in selected_features if col not in numeric_cols
    ]
    
    feature_cols_base = [
        col for col in selected_in_numeric if col not in cluster_cols
    ]
    feature_cols_cluster = list(selected_in_numeric)

    selected_cluster_cols = [
        col for col in selected_in_numeric if col in cluster_cols
    ]
    
    configs = []
    configs.append({
        "label": "Base",
        "cols": feature_cols_base,
        "key": _optuna_result_key(feature_key, feature_cols_base),
        "id": _optuna_result_id(feature_id, feature_cols_base)
    })
    
    if set(feature_cols_cluster) != set(feature_cols_base):
        if selected_cluster_cols:
            configs.append({
                "label": "Cluster",
                "cols": selected_cluster_cols,
                "key": _optuna_result_key(feature_key, selected_cluster_cols),
                "id": _optuna_result_id(feature_id, selected_cluster_cols),
            })
        configs.append({
            "label": "Base + Cluster",
            "cols": feature_cols_cluster,
            "key": _optuna_result_key(feature_key, feature_cols_cluster),
            "id": _optuna_result_id(feature_id, feature_cols_cluster)
        })

    if missing:
        st.warning(
            "Variables seleccionadas no estan en el dataset: "
            + ", ".join(missing)
        )
    if not feature_cols_base:
        st.warning("No hay variables numericas para Optuna.")
        return

    store = st.session_state.get("optuna_results_store", {})
    active_optuna_key = st.session_state.get("optuna_active_key")
    
    primary_config = configs[-1]
    primary_key = primary_config["key"]
    
    if active_optuna_key != primary_key:
        st.session_state["optuna_active_key"] = primary_key
        
        for cfg in configs:
            c_key = cfg["key"]
            c_id = cfg["id"]
            c_cols = cfg["cols"]
            
            entry = store.get(c_key)
            if entry is None:
                payload, trials_df = _load_optuna_result_from_disk(c_id)
                if payload or trials_df is not None:
                    results: Dict[str, object] = _normalize_optuna_results_payload(
                        payload.get("results") if payload else None
                    )
                    if not results and payload:
                        legacy_choice = payload.get("model_choice") or "legacy"
                        _, legacy_csv = _optuna_result_paths(c_id)
                        trials_csv = payload.get("trials_csv")
                        if not trials_csv and legacy_csv.exists():
                            trials_csv = str(legacy_csv)
                        legacy_result = {
                            "model_choice": legacy_choice,
                            "best_score": payload.get("best_score"),
                            "best_smote_params": payload.get("best_smote_params", {}),
                            "best_model_params": payload.get("best_model_params", {}),
                            "optuna_settings": payload.get("optuna_settings", {}),
                            "search_space": payload.get("search_space", {}),
                            "saved_at": payload.get("saved_at"),
                            "trials_csv": trials_csv,
                        }
                        if trials_df is not None:
                            legacy_result["trials_df"] = trials_df
                        results = _normalize_optuna_results_payload(
                            {str(legacy_choice): legacy_result}
                        )

                    entry = {
                        "optuna_id": payload.get("optuna_id", c_id)
                        if payload
                        else c_id,
                        "feature_key": payload.get("feature_key", feature_key)
                        if payload
                        else feature_key,
                        "feature_id": payload.get("feature_id", feature_id)
                        if payload
                        else feature_id,
                        "features_path": payload.get("features_path", features_path)
                        if payload
                        else features_path,
                        "features_source": payload.get(
                            "features_source", features_source
                        )
                        if payload
                        else features_source,
                        "features_rows": payload.get("features_rows", len(features_df))
                        if payload
                        else int(len(features_df)),
                        "features_cols": payload.get(
                            "features_cols", len(features_df.columns)
                        )
                        if payload
                        else int(len(features_df.columns)),
                        "dataset_fingerprint": payload.get(
                            "dataset_fingerprint",
                            _dataset_content_fingerprint(features_df),
                        )
                        if payload
                        else _dataset_content_fingerprint(features_df),
                        "selection_mode": payload.get(
                            "selection_mode",
                            "all" if selected_features is None else "selected",
                        )
                        if payload
                        else ("all" if selected_features is None else "selected"),
                        "selected_features": payload.get(
                            "selected_features",
                            list(selected_features) if selected_features else [],
                        )
                        if payload
                        else list(selected_features) if selected_features else [],
                        "feature_cols": payload.get(
                            "feature_cols", list(c_cols)
                        )
                        if payload
                        else list(c_cols),
                        "results": results,
                        "saved_at": payload.get("saved_at") if payload else None,
                    }
                    store[c_key] = entry
        
        st.session_state["optuna_results_store"] = store
        st.session_state["optuna_best_smote_params"] = None
        st.session_state["optuna_best_model_params"] = None
        st.session_state["optuna_best_score"] = None
        st.session_state["optuna_best_model_choice"] = None
        st.session_state["optuna_trials_df"] = None
        st.session_state["optuna_best_settings"] = None
        st.session_state["optuna_best_search_space"] = None

    # Ensure legacy entries in store are normalized
    for cfg in configs:
        c_key = cfg["key"]
        c_id = cfg["id"]
        c_cols = cfg["cols"]
        entry = store.get(c_key)
        if entry and not isinstance(entry.get("results"), dict):
            legacy_choice = entry.get("model_choice") or "legacy"
            _, legacy_csv = _optuna_result_paths(c_id)
            trials_csv = entry.get("trials_csv")
            if not trials_csv and legacy_csv.exists():
                trials_csv = str(legacy_csv)
            results = {
                str(legacy_choice): {
                    "model_choice": legacy_choice,
                    "best_score": entry.get("best_score"),
                    "best_smote_params": entry.get("best_smote_params", {}),
                    "best_model_params": entry.get("best_model_params", {}),
                    "optuna_settings": entry.get("optuna_settings", {}),
                    "search_space": entry.get("search_space", {}),
                    "saved_at": entry.get("saved_at"),
                    "trials_df": entry.get("trials_df"),
                    "trials_csv": trials_csv,
                }
            }
            normalized_results = _normalize_optuna_results_payload(results)
            entry = {
                "optuna_id": entry.get("optuna_id", c_id),
                "feature_key": entry.get("feature_key", feature_key),
                "feature_id": entry.get("feature_id", feature_id),
                "features_path": entry.get("features_path", features_path),
                "features_source": entry.get("features_source", features_source),
                "features_rows": entry.get("features_rows", int(len(features_df))),
                "features_cols": entry.get(
                    "features_cols", int(len(features_df.columns))
                ),
                "dataset_fingerprint": entry.get(
                    "dataset_fingerprint",
                    _dataset_content_fingerprint(features_df),
                ),
                "selection_mode": entry.get(
                    "selection_mode",
                    "all" if selected_features is None else "selected",
                ),
                "selected_features": entry.get(
                    "selected_features",
                    list(selected_features) if selected_features else [],
                ),
                "feature_cols": entry.get("feature_cols", list(c_cols)),
                "results": normalized_results,
                "saved_at": entry.get("saved_at"),
            }
            store[c_key] = entry
            st.session_state["optuna_results_store"] = store
        elif entry and isinstance(entry.get("results"), dict):
            normalized_results = _normalize_optuna_results_payload(
                entry.get("results")
            )
            entry = dict(entry)
            entry["results"] = normalized_results
            store[c_key] = entry
            st.session_state["optuna_results_store"] = store

    st.caption(
        f"Filas: {len(base_df):,} | Variables numericas (Base): {len(feature_cols_base)}"
    )
    if len(configs) > 1:
        st.caption(
            f"Variables de cluster seleccionadas: {len(selected_cluster_cols)} (se optimizara con y sin ellas)"
        )
    objective_mode_options = _optuna_objective_mode_options()
    objective_mode_label = st.selectbox(
        "Modo objetivo Optuna",
        list(objective_mode_options.keys()),
        key="optuna_objective_mode_label",
        help=(
            "Escalar legacy optimiza una sola métrica por estudio. "
            "Multiobjetivo Pareto usa MCC, PR-AUC, Brier y Recall@N alertas/día "
            "en el mismo study, replicando el criterio del experimento "
            "`Calibración score + threshold`."
        ),
    )
    optuna_objective_mode = objective_mode_options[objective_mode_label]
    is_multiobjective_optuna = (
        optuna_objective_mode
        == CALIBRATION_SWEEP_OBJECTIVE_MODE_MULTIOBJECTIVE
    )
    if is_multiobjective_optuna:
        objective_label = CALIBRATION_SWEEP_MULTIOBJECTIVE_LABEL
        objective_key = CALIBRATION_SWEEP_MULTIOBJECTIVE_KEY
        objective_direction = "multiobjective"
        st.caption(
            "Objetivos fijos: MCC ↑, PR-AUC ↑, Brier ↓ y Recall@N alertas/día ↑. "
            "FAR actúa como gate operacional para elegir el trial dentro del frente Pareto."
        )
    else:
        objective_options = _optuna_objective_options()
        objective_label = st.selectbox(
            "Metrica objetivo",
            list(objective_options.keys()),
            key="optuna_objective_metric",
        )
        objective_key = objective_options[objective_label]["key"]
        objective_direction = objective_options[objective_label]["direction"]
        objective_verb = (
            "minimiza" if objective_direction == "minimize" else "optimiza"
        )
        st.caption(
            f"Optuna {objective_verb} {objective_label} en el set de validacion "
            "usando el criterio de threshold seleccionado (test queda como hold-out final)."
        )
    entry = store.get(primary_key)

    model_choice = st.selectbox(
        "Modelo",
        ["XGBoost", "Random Forest", "SVM", "Neural Network"],
        key="optuna_model_choice",
    )
    optuna_calibration_options = _calibration_method_options()
    optuna_calibration_labels = [
        label for label, _ in optuna_calibration_options
    ]
    optuna_calibration_map = {
        label: key for label, key in optuna_calibration_options
    }
    optuna_calibration_label = st.selectbox(
        "Calibración",
        optuna_calibration_labels,
        index=0,
        key="optuna_calibration_method",
        help=(
            "Transforma scores antes de seleccionar el threshold operativo en "
            "validación. Default: Platt scaling (sigmoid)."
        ),
    )
    optuna_calibration_method = optuna_calibration_map[optuna_calibration_label]
    model_result: Optional[Dict[str, object]] = None
    if entry and isinstance(entry.get("results"), dict):
        model_result = _get_optuna_model_result_variant(
            entry.get("results"),
            model_choice=model_choice,
            balance_mode="smote",
            calibration_method=optuna_calibration_method,
            fallback_modes=["none"],
        )
        smote_result = _get_optuna_model_result_variant(
            entry.get("results"),
            model_choice=model_choice,
            balance_mode="smote",
            calibration_method=optuna_calibration_method,
        )
        if isinstance(model_result, dict):
            trials_df = model_result.get("trials_df")
            trials_csv = model_result.get("trials_csv")
            if (
                trials_df is None
                and trials_csv
                and Path(str(trials_csv)).exists()
            ):
                try:
                    trials_df = pd.read_csv(trials_csv)
                    model_result["trials_df"] = trials_df
                except Exception:
                    trials_df = None
            # DEPRECATED: replicación de valores del store en keys top-level
            # ``optuna_best_*``. Consumidores nuevos deben usar
            # ``_get_active_optuna_best()`` que lee directamente del
            # ``optuna_results_store``. Se mantiene por compatibilidad con
            # flujos legacy que aún leen estos keys (disk reloaders, tests
            # viejos).
            st.session_state["optuna_best_smote_params"] = (
                smote_result.get("best_smote_params")
                if isinstance(smote_result, dict)
                else None
            )
            st.session_state["optuna_best_model_params"] = model_result.get(
                "best_model_params"
            )
            st.session_state["optuna_best_score"] = model_result.get("best_score")
            st.session_state["optuna_best_model_choice"] = model_choice
            st.session_state["optuna_trials_df"] = trials_df
            st.session_state["optuna_best_settings"] = model_result.get(
                "optuna_settings"
            )
            st.session_state["optuna_best_search_space"] = model_result.get(
                "search_space"
            )
        else:
            # DEPRECATED: reset de keys top-level ``optuna_best_*`` (legacy).
            st.session_state["optuna_best_smote_params"] = None
            st.session_state["optuna_best_model_params"] = None
            st.session_state["optuna_best_score"] = None
            st.session_state["optuna_best_model_choice"] = None
            st.session_state["optuna_trials_df"] = None
            st.session_state["optuna_best_settings"] = None
            st.session_state["optuna_best_search_space"] = None

    n_trials = st.number_input(
        "n_trials",
        min_value=5,
        max_value=1000,
        value=int(st.session_state.get("optuna_n_trials", 500)),
        step=5,
        key="optuna_n_trials",
    )
    timeout = st.number_input(
        "timeout (segundos)",
        min_value=60,
        max_value=99999999999,
        value=int(st.session_state.get("optuna_timeout", 86400)),
        step=60,
        key="optuna_timeout",
    )
    optuna_n_jobs = _render_optuna_n_jobs_input(
        "Optuna jobs paralelos",
        key="optuna_n_jobs",
        default=1,
    )
    optuna_rf_n_jobs: Optional[int] = None
    optuna_xgb_n_jobs: Optional[int] = None
    max_internal_jobs = _max_optuna_parallel_jobs()
    if model_choice in {"Random Forest", "Balanced Random Forest"}:
        optuna_rf_n_jobs = _render_model_n_jobs_input(
            "Random Forest n_jobs (trials)",
            key="optuna_rf_n_jobs",
            default=max_internal_jobs,
        )
    elif model_choice == "XGBoost":
        optuna_xgb_n_jobs = _render_model_n_jobs_input(
            "XGBoost n_jobs (trials)",
            key="optuna_xgb_n_jobs",
            default=max_internal_jobs,
        )
    if optuna_rf_n_jobs is not None or optuna_xgb_n_jobs is not None:
        st.caption(
            "Hilos internos del modelo dentro de cada trial. "
            f"CPUs detectadas: {max_internal_jobs}. "
            "Si `Optuna jobs paralelos` > 1, los trials compiten por esos hilos; "
            "considere bajar uno de los dos para evitar oversuscripcion."
        )
    optuna_random_state = st.number_input(
        "random_state",
        min_value=0,
        value=int(st.session_state.get("optuna_random_state", 42)),
        step=1,
        key="optuna_random_state",
    )
    st.markdown("**Poda (pruning)**")
    pruner_enabled = st.checkbox(
        "Activar poda (MedianPruner)",
        value=bool(st.session_state.get("optuna_pruner_enabled", True)),
        key="optuna_pruner_enabled",
    )
    pruner_startup_trials = st.number_input(
        "Trials iniciales sin poda",
        min_value=0,
        value=int(st.session_state.get("optuna_pruner_startup_trials", 5)),
        step=1,
        key="optuna_pruner_startup_trials",
        disabled=not pruner_enabled,
    )
    if is_multiobjective_optuna and pruner_enabled:
        st.info(
            "En multiobjetivo el study corre sin `MedianPruner`; se guarda el "
            "proxy Pareto para ranking y selección del trial."
        )
    optuna_test_size = st.slider(
        "Test size",
        min_value=0.1,
        max_value=0.4,
        value=float(st.session_state.get("test_size", 0.2)),
        step=0.05,
        key="optuna_test_size",
    )
    st.session_state["test_size"] = float(optuna_test_size)

    st.markdown("**Calibracion de umbral**")
    optuna_threshold_objective_options = {
        "FAR": "far",
        "F1": "f1",
        "Balanced F1": "balanced_f1",
        "MCC": "mcc",
        "Recall@N alertas/dia": "recall_at_alerts_per_day",
        "Costo operacional": "operational_cost",
        "PR-AUC": "pr_auc",
        "ROC-AUC": "roc_auc",
    }
    optuna_threshold_objective_label = st.selectbox(
        "Criterio de threshold",
        list(optuna_threshold_objective_options.keys()),
        index=0,
        key="optuna_threshold_objective",
        help=(
            "Metrica con la que se escoge el umbral operativo sobre val dentro "
            "de cada trial. Se aplica el mismo criterio que en la pestana de "
            "Modelos. Para PR-AUC/ROC-AUC el umbral queda en 0.5 (metricas de "
            "ranking)."
        ),
    )
    optuna_threshold_objective = optuna_threshold_objective_options[
        optuna_threshold_objective_label
    ]
    optuna_threshold_visibility = _threshold_field_visibility_for_objective(
        optuna_threshold_objective
    )
    optuna_far_target = float(
        _render_conditional_slider(
            "FAR (False alarm rate) target",
            visible=optuna_threshold_visibility["far_target"],
            min_value=0.0,
            max_value=0.5,
            value=float(st.session_state.get("far_target", 0.2)),
            step=0.01,
            key="optuna_far_target",
        )
    )
    optuna_alerts_per_day = float(
        _render_conditional_number_input(
            "Alertas maximas por dia",
            visible=optuna_threshold_visibility["alerts_per_day"],
            min_value=0.1,
            max_value=50.0,
            value=float(st.session_state.get("optuna_alerts_per_day", 5.0)),
            step=0.5,
            key="optuna_alerts_per_day",
            help=(
                "Presupuesto diario de alertas para Recall@N y costo operacional. "
                "Se ignora para criterios que no lo usen."
            ),
        )
    )
    if (
        optuna_threshold_visibility["fn_cost"]
        or optuna_threshold_visibility["fp_cost"]
    ):
        col_cost_a, col_cost_b = st.columns(2)
        with col_cost_a:
            optuna_fn_cost = float(
                _render_conditional_number_input(
                    "Costo FN",
                    visible=optuna_threshold_visibility["fn_cost"],
                    min_value=0.0,
                    value=float(st.session_state.get("optuna_fn_cost", 10.0)),
                    step=1.0,
                    key="optuna_fn_cost",
                    help=(
                        "Costo de no alertar un accidente real "
                        "(usado por costo operacional)."
                    ),
                )
            )
        with col_cost_b:
            optuna_fp_cost = float(
                _render_conditional_number_input(
                    "Costo FP",
                    visible=optuna_threshold_visibility["fp_cost"],
                    min_value=0.0,
                    value=float(st.session_state.get("optuna_fp_cost", 1.0)),
                    step=0.5,
                    key="optuna_fp_cost",
                    help="Costo de una falsa alarma (usado por costo operacional).",
                )
            )
    else:
        optuna_fn_cost = float(
            _render_conditional_number_input(
                "Costo FN",
                visible=False,
                min_value=0.0,
                value=float(st.session_state.get("optuna_fn_cost", 10.0)),
                step=1.0,
                key="optuna_fn_cost",
            )
        )
        optuna_fp_cost = float(
            _render_conditional_number_input(
                "Costo FP",
                visible=False,
                min_value=0.0,
                value=float(st.session_state.get("optuna_fp_cost", 1.0)),
                step=0.5,
                key="optuna_fp_cost",
            )
        )
    optuna_val_size = st.slider(
        "Validation size",
        min_value=0.05,
        max_value=0.4,
        value=float(st.session_state.get("val_size", 0.2)),
        step=0.05,
        key="optuna_val_size",
    )
    st.session_state["far_target"] = float(optuna_far_target)
    st.session_state["val_size"] = float(optuna_val_size)

    st.markdown("**Rangos y pasos de optimizacion**")
    st.caption("Ajuste los rangos que Optuna puede explorar.")

    st.markdown("**Numero de variables (top-K del ranking)**")
    optuna_tune_topk = st.checkbox(
        "Optimizar K (numero de variables)",
        value=bool(st.session_state.get("optuna_tune_topk", False)),
        key="optuna_tune_topk",
        help=(
            "Si se activa, Optuna elige cuantas variables usar (top-K segun un "
            "ranking calculado sobre train real). Si se desactiva, se usan "
            "todas las variables del config."
        ),
    )
    optuna_ranking_method_label = st.selectbox(
        "Metodo de ranking",
        ["Random Forest (importancia)", "Mutual information"],
        index=0,
        key="optuna_ranking_method_label",
        disabled=not optuna_tune_topk,
        help=(
            "Metodo para rankear variables una vez sobre train (sin SMOTE). "
            "Random Forest usa importancia por impureza; Mutual information "
            "usa dependencia no lineal con el target."
        ),
    )
    optuna_ranking_method = (
        "mutual_info"
        if optuna_ranking_method_label.startswith("Mutual")
        else "rf"
    )
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        optuna_k_min = st.number_input(
            "k_min",
            min_value=1,
            value=int(st.session_state.get("optuna_k_min", 3)),
            step=1,
            key="optuna_k_min",
            disabled=not optuna_tune_topk,
        )
    with col_k2:
        optuna_k_max = st.number_input(
            "k_max",
            min_value=1,
            value=int(st.session_state.get("optuna_k_max", 20)),
            step=1,
            key="optuna_k_max",
            disabled=not optuna_tune_topk,
        )
    with col_k3:
        optuna_k_step = st.number_input(
            "k_step",
            min_value=1,
            value=int(st.session_state.get("optuna_k_step", 1)),
            step=1,
            key="optuna_k_step",
            disabled=not optuna_tune_topk,
        )
    if optuna_tune_topk:
        st.caption(
            "k_max se ajusta automaticamente al tamano de cada config "
            "(Base / Cluster / Base + Cluster)."
        )

    st.markdown("**SMOTE**")
    col1, col2, col3 = st.columns(3)
    with col1:
        smote_k_min = st.number_input(
            "smote_k_min",
            min_value=1,
            value=1,
            step=1,
            key="optuna_smote_k_min",
        )
    with col2:
        smote_k_max = st.number_input(
            "smote_k_max",
            min_value=1,
            value=10,
            step=1,
            key="optuna_smote_k_max",
        )
    with col3:
        smote_k_step = st.number_input(
            "smote_k_step",
            min_value=1,
            value=1,
            step=1,
            key="optuna_smote_k_step",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        smote_sampling_min = st.number_input(
            "smote_sampling_min",
            min_value=0.01,
            max_value=1.0,
            value=0.01,
            step=0.01,
            format="%.2f",
            key="optuna_smote_sampling_min",
        )
    with col2:
        smote_sampling_max = st.number_input(
            "smote_sampling_max",
            min_value=0.05,
            max_value=1.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            key="optuna_smote_sampling_max",
        )
    with col3:
        smote_sampling_step = st.number_input(
            "smote_sampling_step",
            min_value=0.01,
            max_value=0.5,
            value=0.01,
            step=0.01,
            format="%.2f",
            key="optuna_smote_sampling_step",
        )

    if model_choice == "Random Forest":
        st.markdown("**Random Forest**")
        col1, col2, col3 = st.columns(3)
        with col1:
            rf_n_min = st.number_input(
                "rf_n_estimators_min",
                min_value=10,
                value=100,
                step=10,
                key="optuna_rf_n_min",
            )
        with col2:
            rf_n_max = st.number_input(
                "rf_n_estimators_max",
                min_value=10,
                value=500,
                step=10,
                key="optuna_rf_n_max",
            )
        with col3:
            rf_n_step = st.number_input(
                "rf_n_estimators_step",
                min_value=1,
                value=50,
                step=1,
                key="optuna_rf_n_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            rf_depth_min = st.number_input(
                "rf_max_depth_min (0 = None)",
                min_value=0,
                value=0,
                step=1,
                key="optuna_rf_depth_min",
            )
        with col2:
            rf_depth_max = st.number_input(
                "rf_max_depth_max (0 = None)",
                min_value=0,
                value=20,
                step=1,
                key="optuna_rf_depth_max",
            )
        with col3:
            rf_depth_step = st.number_input(
                "rf_max_depth_step",
                min_value=1,
                value=1,
                step=1,
                key="optuna_rf_depth_step",
            )
    elif model_choice == "XGBoost":
        st.markdown("**XGBoost**")
        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_n_min = st.number_input(
                "xgb_n_estimators_min",
                min_value=10,
                value=100,
                step=10,
                key="optuna_xgb_n_min",
            )
        with col2:
            xgb_n_max = st.number_input(
                "xgb_n_estimators_max",
                min_value=10,
                value=500,
                step=10,
                key="optuna_xgb_n_max",
            )
        with col3:
            xgb_n_step = st.number_input(
                "xgb_n_estimators_step",
                min_value=1,
                value=50,
                step=1,
                key="optuna_xgb_n_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_depth_min = st.number_input(
                "xgb_max_depth_min",
                min_value=1,
                value=2,
                step=1,
                key="optuna_xgb_depth_min",
            )
        with col2:
            xgb_depth_max = st.number_input(
                "xgb_max_depth_max",
                min_value=1,
                value=10,
                step=1,
                key="optuna_xgb_depth_max",
            )
        with col3:
            xgb_depth_step = st.number_input(
                "xgb_max_depth_step",
                min_value=1,
                value=1,
                step=1,
                key="optuna_xgb_depth_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_lr_min = st.number_input(
                "xgb_learning_rate_min",
                min_value=0.001,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.3f",
                key="optuna_xgb_lr_min",
            )
        with col2:
            xgb_lr_max = st.number_input(
                "xgb_learning_rate_max",
                min_value=0.001,
                max_value=1.0,
                value=0.3,
                step=0.001,
                format="%.3f",
                key="optuna_xgb_lr_max",
            )
        with col3:
            xgb_lr_step = st.number_input(
                "xgb_learning_rate_step",
                min_value=0.001,
                max_value=0.5,
                value=0.01,
                step=0.001,
                format="%.3f",
                key="optuna_xgb_lr_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_sub_min = st.number_input(
                "xgb_subsample_min",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.05,
                format="%.2f",
                key="optuna_xgb_sub_min",
            )
        with col2:
            xgb_sub_max = st.number_input(
                "xgb_subsample_max",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.05,
                format="%.2f",
                key="optuna_xgb_sub_max",
            )
        with col3:
            xgb_sub_step = st.number_input(
                "xgb_subsample_step",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.2f",
                key="optuna_xgb_sub_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_col_min = st.number_input(
                "xgb_colsample_min",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.05,
                format="%.2f",
                key="optuna_xgb_col_min",
            )
        with col2:
            xgb_col_max = st.number_input(
                "xgb_colsample_max",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.05,
                format="%.2f",
                key="optuna_xgb_col_max",
            )
        with col3:
            xgb_col_step = st.number_input(
                "xgb_colsample_step",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01,
                format="%.2f",
                key="optuna_xgb_col_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_reg_alpha_min = st.number_input(
                "xgb_reg_alpha_min",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="optuna_xgb_reg_alpha_min",
            )
        with col2:
            xgb_reg_alpha_max = st.number_input(
                "xgb_reg_alpha_max",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                key="optuna_xgb_reg_alpha_max",
            )
        with col3:
            xgb_reg_alpha_step = st.number_input(
                "xgb_reg_alpha_step",
                min_value=0.01,
                max_value=5.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="optuna_xgb_reg_alpha_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_reg_lambda_min = st.number_input(
                "xgb_reg_lambda_min",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="optuna_xgb_reg_lambda_min",
            )
        with col2:
            xgb_reg_lambda_max = st.number_input(
                "xgb_reg_lambda_max",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.1,
                format="%.2f",
                key="optuna_xgb_reg_lambda_max",
            )
        with col3:
            xgb_reg_lambda_step = st.number_input(
                "xgb_reg_lambda_step",
                min_value=0.01,
                max_value=5.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="optuna_xgb_reg_lambda_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            xgb_gamma_min = st.number_input(
                "xgb_gamma_min",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="optuna_xgb_gamma_min",
            )
        with col2:
            xgb_gamma_max = st.number_input(
                "xgb_gamma_max",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                key="optuna_xgb_gamma_max",
            )
        with col3:
            xgb_gamma_step = st.number_input(
                "xgb_gamma_step",
                min_value=0.01,
                max_value=5.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="optuna_xgb_gamma_step",
            )
    elif model_choice == "Neural Network":
        st.markdown("**Neural Network**")
        col1, col2, col3 = st.columns(3)
        with col1:
            nn_hidden_min = st.number_input(
                "nn_hidden_dim_min",
                min_value=32,
                value=64,
                step=32,
                key="optuna_nn_hidden_min",
            )
        with col2:
            nn_hidden_max = st.number_input(
                "nn_hidden_dim_max",
                min_value=32,
                value=512,
                step=32,
                key="optuna_nn_hidden_max",
            )
        with col3:
            nn_hidden_step = st.number_input(
                "nn_hidden_dim_step",
                min_value=1,
                value=64,
                step=1,
                key="optuna_nn_hidden_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            nn_layers_min = st.number_input(
                "nn_num_layers_min",
                min_value=1,
                value=1,
                step=1,
                key="optuna_nn_layers_min",
            )
        with col2:
            nn_layers_max = st.number_input(
                "nn_num_layers_max",
                min_value=1,
                value=4,
                step=1,
                key="optuna_nn_layers_max",
            )
        with col3:
            nn_layers_step = st.number_input(
                "nn_num_layers_step",
                min_value=1,
                value=1,
                step=1,
                key="optuna_nn_layers_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            nn_dropout_min = st.number_input(
                "nn_dropout_min",
                min_value=0.0,
                max_value=0.9,
                value=0.0,
                step=0.05,
                format="%.2f",
                key="optuna_nn_dropout_min",
            )
        with col2:
            nn_dropout_max = st.number_input(
                "nn_dropout_max",
                min_value=0.0,
                max_value=0.9,
                value=0.5,
                step=0.05,
                format="%.2f",
                key="optuna_nn_dropout_max",
            )
        with col3:
            nn_dropout_step = st.number_input(
                "nn_dropout_step",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="optuna_nn_dropout_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            nn_lr_min = st.number_input(
                "nn_learning_rate_min",
                min_value=0.00001,
                max_value=0.1,
                value=0.0001,
                step=0.0001,
                format="%.5f",
                key="optuna_nn_lr_min",
            )
        with col2:
            nn_lr_max = st.number_input(
                "nn_learning_rate_max",
                min_value=0.00001,
                max_value=0.1,
                value=0.01,
                step=0.0001,
                format="%.5f",
                key="optuna_nn_lr_max",
            )
        with col3:
            nn_lr_step = st.number_input(
                "nn_learning_rate_step",
                min_value=0.00001,
                max_value=0.01,
                value=0.0001,
                step=0.00001,
                format="%.5f",
                key="optuna_nn_lr_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            nn_wd_min = st.number_input(
                "nn_weight_decay_min",
                min_value=0.0,
                max_value=0.1,
                value=1e-6,
                step=1e-6,
                format="%.7f",
                key="optuna_nn_wd_min",
            )
        with col2:
            nn_wd_max = st.number_input(
                "nn_weight_decay_max",
                min_value=0.0,
                max_value=0.1,
                value=1e-3,
                step=1e-6,
                format="%.7f",
                key="optuna_nn_wd_max",
            )
        with col3:
            nn_wd_step = st.number_input(
                "nn_weight_decay_step",
                min_value=1e-7,
                max_value=1e-2,
                value=1e-6,
                step=1e-7,
                format="%.7f",
                key="optuna_nn_wd_step",
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            nn_pw_min = st.number_input(
                "nn_pos_weight_min",
                min_value=0.1,
                max_value=500.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                key="optuna_nn_pw_min",
            )
        with col2:
            nn_pw_max = st.number_input(
                "nn_pos_weight_max",
                min_value=0.1,
                max_value=500.0,
                value=50.0,
                step=0.1,
                format="%.2f",
                key="optuna_nn_pw_max",
            )
        with col3:
            nn_pw_step = st.number_input(
                "nn_pos_weight_step",
                min_value=0.1,
                max_value=50.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                key="optuna_nn_pw_step",
            )

        nn_batch_options = st.multiselect(
            "nn_batch_size",
            [128, 256, 512, 1024, 2048, 4096, 8192],
            default=[1024, 2048, 4096],
            key="optuna_nn_batch_sizes",
        )
        st.caption(
            "epochs no se optimiza aqui: cada trial entrena hasta convergencia "
            "con early stopping (paciencia 5). El maximo de epochs se "
            "configura en la pestana Modelos."
        )
        # Hints de GPU / paralelismo
        try:
            import torch as _torch

            if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                _device_name = "MPS (Apple Silicon)"
                _on_gpu = True
            elif _torch.cuda.is_available():
                _device_name = f"CUDA ({_torch.cuda.get_device_name(0)})"
                _on_gpu = True
            else:
                _device_name = "CPU"
                _on_gpu = False
        except Exception:
            _device_name = "CPU"
            _on_gpu = False
        if _on_gpu:
            st.info(
                f"Dispositivo NN detectado: **{_device_name}**. "
                "Para maximizar la GPU: usa batch_size grandes (>=1024) y "
                "deja **Optuna n_jobs = 1** (trials paralelos comparten el "
                "mismo device y se serializan, sin ganancia de velocidad). "
                "El wrapper precarga train/val en device para evitar H2D "
                "por batch."
            )
            if int(optuna_n_jobs) > 1:
                st.warning(
                    "⚠️ Optuna n_jobs > 1 con Neural Network en GPU no "
                    "acelera: los trials se encolan en el mismo dispositivo. "
                    "Se recomienda n_jobs=1."
                )
        else:
            st.caption(
                f"Dispositivo NN detectado: {_device_name}. "
                "Instale PyTorch con MPS/CUDA para acelerar con GPU."
            )
    elif model_choice == "SVM":
        st.markdown("**SVM**")
        kernel_options = ["rbf", "linear", "poly", "sigmoid"]
        svm_kernels = st.multiselect(
            "svm_kernels",
            kernel_options,
            default=kernel_options,
            key="optuna_svm_kernels",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            svm_c_min = st.number_input(
                "svm_C_min",
                min_value=0.01,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="optuna_svm_c_min",
            )
        with col2:
            svm_c_max = st.number_input(
                "svm_C_max",
                min_value=0.01,
                value=50.0,
                step=0.1,
                format="%.2f",
                key="optuna_svm_c_max",
            )
        with col3:
            svm_c_step = st.number_input(
                "svm_C_step",
                min_value=0.01,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="optuna_svm_c_step",
            )

    if st.button("Ejecutar Optuna"):
        try:
            import optuna  # type: ignore
        except ImportError:
            st.error(
                "optuna no esta instalado. Ejecute `pip install optuna`."
            )
            return
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
            smote_import_error = None
        except ImportError:
            SMOTE = None  # type: ignore[assignment]
            smote_import_error = (
                "imbalanced-learn no esta instalado. La variante Con SMOTE se omitirá."
            )
        if model_choice == "XGBoost":
            try:
                import xgboost as xgb  # noqa: F401
            except ImportError:
                st.error(
                    "xgboost no esta instalado. Ejecute `pip install xgboost`."
                )
                return

        y = base_df["target"].astype("int8")
        if y.nunique() < 2:
            st.warning("No hay dos clases en el target para Optuna.")
            return
        try:
            train_val_df, test_df = _temporal_train_test_split(
                base_df,
                time_col="interval_start",
                test_size=float(optuna_test_size),
            )
        except ValueError as exc:
            st.warning(f"No se pudo hacer split temporal: {exc}")
            return
        try:
            train_df, val_df = _temporal_train_test_split(
                train_val_df,
                time_col="interval_start",
                test_size=float(optuna_val_size),
            )
        except ValueError as exc:
            st.warning(f"No se pudo crear validacion temporal: {exc}")
            return

        X_train = train_df[numeric_cols].fillna(0).astype("float32")
        y_train = train_df["target"].astype("int8")
        X_val = val_df[numeric_cols].fillna(0).astype("float32")
        y_val = val_df["target"].astype("int8")
        X_test = test_df[numeric_cols].fillna(0).astype("float32")
        y_test = test_df["target"].astype("int8")
        if y_train.nunique() < 2:
            st.warning(
                "El split temporal dejo una sola clase en train. "
                "Ajuste el rango o el test_size."
            )
            return
        if y_val.nunique() < 2:
            st.warning(
                "El split temporal dejo una sola clase en validacion. "
                "Ajuste val_size o test_size."
            )
            return
        if y_test.nunique() < 2:
            st.warning(
                "El split temporal dejo una sola clase en test. "
                "Ajuste el rango o el test_size."
            )
            return

        if optuna_tune_topk:
            if optuna_k_min > optuna_k_max:
                st.warning("k_min no puede ser mayor que k_max.")
                return
            if optuna_k_step <= 0:
                st.warning("k_step debe ser mayor a 0.")
                return

        if smote_k_min > smote_k_max:
            st.warning("smote_k_min no puede ser mayor que smote_k_max.")
            return
        if smote_sampling_min > smote_sampling_max:
            st.warning(
                "smote_sampling_min no puede ser mayor que smote_sampling_max."
            )
            return
        if smote_sampling_step <= 0:
            st.warning("smote_sampling_step debe ser mayor a 0.")
            return

        min_count = int(pd.Series(y_train).value_counts().min())
        smote_skip_reason: Optional[str] = None
        k_low = max(1, int(smote_k_min))
        k_high = int(smote_k_max)
        if SMOTE is None:
            smote_skip_reason = smote_import_error
        elif min_count < 2:
            smote_skip_reason = "No hay suficientes ejemplos minoritarios para SMOTE."
        else:
            max_k = max(1, min_count - 1)
            k_high = min(int(smote_k_max), max_k)
            if k_high < k_low:
                smote_skip_reason = (
                    "El rango de smote_k no es valido para este dataset."
                )

        if model_choice == "Random Forest":
            if rf_n_min > rf_n_max:
                st.warning("rf_n_estimators_min > rf_n_estimators_max.")
                return
            if rf_n_step <= 0:
                st.warning("rf_n_estimators_step debe ser mayor a 0.")
                return
            if rf_depth_min > rf_depth_max:
                st.warning("rf_max_depth_min > rf_max_depth_max.")
                return
            if rf_depth_step <= 0:
                st.warning("rf_max_depth_step debe ser mayor a 0.")
                return
        elif model_choice == "XGBoost":
            if xgb_n_min > xgb_n_max:
                st.warning("xgb_n_estimators_min > xgb_n_estimators_max.")
                return
            if xgb_n_step <= 0:
                st.warning("xgb_n_estimators_step debe ser mayor a 0.")
                return
            if xgb_depth_min > xgb_depth_max:
                st.warning("xgb_max_depth_min > xgb_max_depth_max.")
                return
            if xgb_depth_step <= 0:
                st.warning("xgb_max_depth_step debe ser mayor a 0.")
                return
            if xgb_lr_min > xgb_lr_max:
                st.warning("xgb_learning_rate_min > xgb_learning_rate_max.")
                return
            if xgb_lr_step <= 0:
                st.warning("xgb_learning_rate_step debe ser mayor a 0.")
                return
            if xgb_sub_min > xgb_sub_max:
                st.warning("xgb_subsample_min > xgb_subsample_max.")
                return
            if xgb_sub_step <= 0:
                st.warning("xgb_subsample_step debe ser mayor a 0.")
                return
            if xgb_col_min > xgb_col_max:
                st.warning("xgb_colsample_min > xgb_colsample_max.")
                return
            if xgb_col_step <= 0:
                st.warning("xgb_colsample_step debe ser mayor a 0.")
                return
            if xgb_reg_alpha_min > xgb_reg_alpha_max:
                st.warning("xgb_reg_alpha_min > xgb_reg_alpha_max.")
                return
            if xgb_reg_alpha_step <= 0:
                st.warning("xgb_reg_alpha_step debe ser mayor a 0.")
                return
            if xgb_reg_lambda_min > xgb_reg_lambda_max:
                st.warning("xgb_reg_lambda_min > xgb_reg_lambda_max.")
                return
            if xgb_reg_lambda_step <= 0:
                st.warning("xgb_reg_lambda_step debe ser mayor a 0.")
                return
            if xgb_gamma_min > xgb_gamma_max:
                st.warning("xgb_gamma_min > xgb_gamma_max.")
                return
            if xgb_gamma_step <= 0:
                st.warning("xgb_gamma_step debe ser mayor a 0.")
                return
        elif model_choice == "Neural Network":
            if nn_hidden_min > nn_hidden_max:
                st.warning("nn_hidden_dim_min > nn_hidden_dim_max.")
                return
            if nn_layers_min > nn_layers_max:
                st.warning("nn_num_layers_min > nn_num_layers_max.")
                return
            if nn_dropout_min > nn_dropout_max:
                st.warning("nn_dropout_min > nn_dropout_max.")
                return
            if nn_lr_min > nn_lr_max:
                st.warning("nn_learning_rate_min > nn_learning_rate_max.")
                return
            if nn_wd_min > nn_wd_max:
                st.warning("nn_weight_decay_min > nn_weight_decay_max.")
                return
            if nn_wd_step <= 0:
                st.warning("nn_weight_decay_step debe ser mayor a 0.")
                return
            if nn_pw_min > nn_pw_max:
                st.warning("nn_pos_weight_min > nn_pos_weight_max.")
                return
            if nn_pw_step <= 0:
                st.warning("nn_pos_weight_step debe ser mayor a 0.")
                return
            if not nn_batch_options:
                st.warning("Seleccione al menos un batch_size para Neural Network.")
                return
        elif model_choice == "SVM":
            if not svm_kernels:
                st.warning("Seleccione al menos un kernel para SVM.")
                return
            if svm_c_min > svm_c_max:
                st.warning("svm_C_min > svm_C_max.")
                return
            if svm_c_step <= 0:
                st.warning("svm_C_step debe ser mayor a 0.")
                return

        balance_modes_to_run = ["none"]
        if smote_skip_reason is None:
            balance_modes_to_run.append("smote")
        else:
            st.warning(
                "No se pudo ejecutar la variante Con SMOTE. "
                f"Motivo: {smote_skip_reason}"
            )

        def _run_optimization(
            cols: List[str],
            label: str,
            *,
            balance_mode: str,
        ):
            balance_mode = _normalize_optuna_balance_mode(balance_mode)
            balance_label = _optuna_balance_mode_label(balance_mode)
            st.markdown(f"**Optimizando: {label} | {balance_label}**")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            effective_optuna_n_jobs = max(1, int(optuna_n_jobs))
            sampler = optuna.samplers.TPESampler(
                seed=int(optuna_random_state)
            )
            if pruner_enabled:
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=int(pruner_startup_trials),
                    n_warmup_steps=0,
                    interval_steps=1,
                )
            else:
                pruner = optuna.pruners.NopPruner()

            study = optuna.create_study(
                direction=objective_direction,
                sampler=sampler,
                pruner=pruner,
            )

            X_train_run = X_train[cols].fillna(0).astype("float32")
            X_val_run = X_val[cols].fillna(0).astype("float32")

            ranked_cols: List[str] = list(cols)
            effective_k_low = 0
            effective_k_high = 0
            effective_k_step = max(1, int(optuna_k_step)) if optuna_tune_topk else 1
            if optuna_tune_topk:
                try:
                    with st.spinner(
                        f"Calculando ranking ({optuna_ranking_method_label}) sobre train para {label} | {balance_label}..."
                    ):
                        ranked_cols = _rank_features_for_optuna(
                            X_train_run,
                            y_train,
                            method=optuna_ranking_method,
                            random_state=int(optuna_random_state),
                        )
                except Exception as exc:
                    st.warning(
                        f"No se pudo calcular el ranking para {label} | {balance_label}: {exc}. "
                        "Se usaran todas las variables del config."
                    )
                    ranked_cols = list(cols)
                total_cols = len(ranked_cols)
                effective_k_low = max(1, min(int(optuna_k_min), total_cols))
                effective_k_high = max(1, min(int(optuna_k_max), total_cols))
                if effective_k_high < effective_k_low:
                    effective_k_high = effective_k_low
                st.caption(
                    f"[{label} | {balance_label}] Ranking calculado sobre {total_cols} variables. "
                    f"top_k explorado en [{effective_k_low}, {effective_k_high}] "
                    f"paso {effective_k_step}."
                )

            def objective(trial: "optuna.Trial") -> float:
                if optuna_tune_topk and ranked_cols:
                    top_k = _suggest_optuna_discrete_int(
                        trial,
                        "top_k",
                        int(effective_k_low),
                        int(effective_k_high),
                        step=int(effective_k_step),
                    )
                    trial_cols = ranked_cols[: int(top_k)]
                else:
                    trial_cols = list(cols)
                if not trial_cols:
                    raise optuna.TrialPruned("trial_cols vacio.")
                X_train_trial = X_train_run[trial_cols]
                X_val_trial = X_val_run[trial_cols]

                if balance_mode == "smote":
                    smote_k = _suggest_optuna_discrete_int(
                        trial,
                        "smote_k_neighbors",
                        int(k_low),
                        int(k_high),
                        step=int(smote_k_step),
                    )
                    smote_sampling = trial.suggest_float(
                        "smote_sampling_strategy",
                        float(smote_sampling_min),
                        float(smote_sampling_max),
                        step=float(smote_sampling_step),
                    )
                    smote = SMOTE(
                        k_neighbors=int(smote_k),
                        sampling_strategy=float(smote_sampling),
                        random_state=int(optuna_random_state),
                    )
                    try:
                        X_res, y_res = smote.fit_resample(X_train_trial, y_train)
                    except ValueError as exc:
                        raise optuna.TrialPruned(str(exc)) from exc
                else:
                    X_res, y_res = X_train_trial, y_train

                model_params: Dict[str, object]
                if model_choice == "Random Forest":
                    n_estimators = _suggest_optuna_discrete_int(
                        trial,
                        "rf_n_estimators",
                        int(rf_n_min),
                        int(rf_n_max),
                        step=int(rf_n_step),
                    )
                    max_depth = _suggest_optuna_discrete_int(
                        trial,
                        "rf_max_depth",
                        int(rf_depth_min),
                        int(rf_depth_max),
                        step=int(rf_depth_step),
                    )
                    model_params = {
                        "n_estimators": int(n_estimators),
                        "max_depth": None if max_depth == 0 else int(max_depth),
                    }
                    if optuna_rf_n_jobs is not None:
                        model_params["n_jobs"] = int(optuna_rf_n_jobs)
                elif model_choice == "XGBoost":
                    n_estimators = _suggest_optuna_discrete_int(
                        trial,
                        "xgb_n_estimators",
                        int(xgb_n_min),
                        int(xgb_n_max),
                        step=int(xgb_n_step),
                    )
                    max_depth = _suggest_optuna_discrete_int(
                        trial,
                        "xgb_max_depth",
                        int(xgb_depth_min),
                        int(xgb_depth_max),
                        step=int(xgb_depth_step),
                    )
                    learning_rate = trial.suggest_float(
                        "xgb_learning_rate",
                        float(xgb_lr_min),
                        float(xgb_lr_max),
                        step=float(xgb_lr_step),
                    )
                    subsample = trial.suggest_float(
                        "xgb_subsample",
                        float(xgb_sub_min),
                        float(xgb_sub_max),
                        step=float(xgb_sub_step),
                    )
                    colsample = trial.suggest_float(
                        "xgb_colsample_bytree",
                        float(xgb_col_min),
                        float(xgb_col_max),
                        step=float(xgb_col_step),
                    )
                    reg_alpha = trial.suggest_float(
                        "xgb_reg_alpha",
                        float(xgb_reg_alpha_min),
                        float(xgb_reg_alpha_max),
                        step=float(xgb_reg_alpha_step),
                    )
                    reg_lambda = trial.suggest_float(
                        "xgb_reg_lambda",
                        float(xgb_reg_lambda_min),
                        float(xgb_reg_lambda_max),
                        step=float(xgb_reg_lambda_step),
                    )
                    gamma = trial.suggest_float(
                        "xgb_gamma",
                        float(xgb_gamma_min),
                        float(xgb_gamma_max),
                        step=float(xgb_gamma_step),
                    )
                    model_params = {
                        "n_estimators": int(n_estimators),
                        "max_depth": int(max_depth),
                        "learning_rate": float(learning_rate),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample),
                        "reg_alpha": float(reg_alpha),
                        "reg_lambda": float(reg_lambda),
                        "gamma": float(gamma),
                    }
                    if optuna_xgb_n_jobs is not None:
                        model_params["n_jobs"] = int(optuna_xgb_n_jobs)
                elif model_choice == "Neural Network":
                    hidden_dim = _suggest_optuna_discrete_int(
                        trial,
                        "nn_hidden_dim",
                        int(nn_hidden_min),
                        int(nn_hidden_max),
                        step=int(nn_hidden_step),
                    )
                    num_layers = _suggest_optuna_discrete_int(
                        trial,
                        "nn_num_layers",
                        int(nn_layers_min),
                        int(nn_layers_max),
                        step=int(nn_layers_step),
                    )
                    dropout = trial.suggest_float(
                        "nn_dropout",
                        float(nn_dropout_min),
                        float(nn_dropout_max),
                        step=float(nn_dropout_step),
                    )
                    learning_rate = trial.suggest_float(
                        "nn_learning_rate",
                        float(nn_lr_min),
                        float(nn_lr_max),
                        step=float(nn_lr_step),
                    )
                    weight_decay = trial.suggest_float(
                        "nn_weight_decay",
                        float(nn_wd_min),
                        float(nn_wd_max),
                        step=float(nn_wd_step),
                    )
                    pos_weight = trial.suggest_float(
                        "nn_pos_weight",
                        float(nn_pw_min),
                        float(nn_pw_max),
                        step=float(nn_pw_step),
                    )
                    batch_size = trial.suggest_categorical(
                        "nn_batch_size", list(nn_batch_options)
                    )
                    model_params = {
                        "hidden_dim": int(hidden_dim),
                        "num_layers": int(num_layers),
                        "dropout": float(dropout),
                        "learning_rate": float(learning_rate),
                        "weight_decay": float(weight_decay),
                        "pos_weight": float(pos_weight),
                        "batch_size": int(batch_size),
                        "epochs": 100,
                        "early_stopping_patience": 5,
                    }
                else:
                    kernel = trial.suggest_categorical(
                        "svm_kernel", list(svm_kernels)
                    )
                    c_value = trial.suggest_float(
                        "svm_C",
                        float(svm_c_min),
                        float(svm_c_max),
                        step=float(svm_c_step),
                    )
                    model_params = {"kernel": kernel, "C": float(c_value)}

                try:
                    model = _build_model(
                        model_choice, model_params, int(optuna_random_state)
                    )
                    model.fit(X_res, y_res)
                    raw_scores_val = _get_model_scores(model, X_val_trial)
                    calibrator = _fit_score_calibrator(
                        y_val.to_numpy(),
                        raw_scores_val,
                        method=optuna_calibration_method,
                    )
                    scores_val = calibrator.transform(raw_scores_val)
                    scored = _score_optuna_objective(
                        y_val.to_numpy(),
                        scores_val,
                        objective_metric=objective_key,
                        threshold_objective=str(optuna_threshold_objective),
                        eval_df=val_df,
                        far_target=float(optuna_far_target),
                        alerts_per_day=float(optuna_alerts_per_day),
                        fn_cost=float(optuna_fn_cost),
                        fp_cost=float(optuna_fp_cost),
                    )
                    score = float(scored.get("score", float("nan")))
                except Exception as exc:
                    raise optuna.TrialPruned(str(exc)) from exc

                if pd.isna(score):
                    raise optuna.TrialPruned(
                        f"{objective_key} invalido en validacion."
                    )
                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned("Pruned by MedianPruner")
                return score

            start_time = time.monotonic()
            status_placeholder = st.empty()

            def _format_best_params(params: Dict[str, object]) -> str:
                if not params:
                    return "-"
                parts = []
                for key in sorted(params.keys()):
                    value = params[key]
                    parts.append(f"{key}={value}")
                return ", ".join(parts)

            def _render_optuna_progress(study: "optuna.Study", trial) -> None:
                elapsed = time.monotonic() - start_time
                completed_trials = [
                    t
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None
                ]
                completed = len(completed_trials)
                best_score = None
                best_params: Dict[str, object] = {}
                if completed_trials:
                    if objective_direction == "minimize":
                        best_trial = min(
                            completed_trials, key=lambda t: float(t.value)
                        )
                    else:
                        best_trial = max(
                            completed_trials, key=lambda t: float(t.value)
                        )
                    best_score = float(best_trial.value)
                    best_params = dict(best_trial.params)
                pruned = sum(
                    1
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                )
                total = len(study.trials)
                best_prefix = (
                    "Menor" if objective_direction == "minimize" else "Mejor"
                )
                lines = [
                    f"Modo balanceo: {balance_label}",
                    f"Tiempo transcurrido: {elapsed:.1f}s",
                    f"Trials: {completed} completados | {pruned} podados | {total} total",
                    f"Optuna n_jobs: {effective_optuna_n_jobs}",
                    (
                        f"{best_prefix} "
                        f"{objective_label}: {best_score:.4f}"
                        if best_score is not None
                        else f"{best_prefix} {objective_label}: -"
                    ),
                    "Mejores parametros:",
                    _format_best_params(best_params),
                ]
                if effective_optuna_n_jobs > 1:
                    lines.append(
                        "Modo paralelo: el progreso por trial se refresca al finalizar."
                    )
                status_placeholder.code("\n".join(lines), language="text")

            _render_optuna_progress(study, None)
            optuna_callbacks = None
            if effective_optuna_n_jobs == 1:
                optuna_callbacks = [_render_optuna_progress]
            with st.spinner(f"Optuna ({label} | {balance_label}) en ejecucion..."):
                study.optimize(
                    objective,
                    n_trials=int(n_trials),
                    timeout=int(timeout),
                    n_jobs=effective_optuna_n_jobs,
                    callbacks=optuna_callbacks,
                )
            _render_optuna_progress(study, None)

            if not study.trials:
                st.warning(
                    f"Optuna ({label} | {balance_label}) no genero resultados."
                )
                return None

            completed_trials = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
            ]
            if not completed_trials:
                st.warning(
                    f"Optuna ({label} | {balance_label}) no genero trials completos."
                )
                return None

            if objective_direction == "minimize":
                best_trial = min(completed_trials, key=lambda t: float(t.value))
            else:
                best_trial = max(completed_trials, key=lambda t: float(t.value))
            best_params = dict(best_trial.params)
            best_score = float(best_trial.value)
            smote_params: Dict[str, object] = {}
            if balance_mode == "smote":
                smote_params = {
                    "smote_k_neighbors": int(best_params["smote_k_neighbors"]),
                    "smote_sampling_strategy": float(
                        best_params["smote_sampling_strategy"]
                    ),
                }
            if model_choice == "Random Forest":
                max_depth = int(best_params["rf_max_depth"])
                model_params = {
                    "n_estimators": int(best_params["rf_n_estimators"]),
                    "max_depth": None if max_depth == 0 else max_depth,
                }
            elif model_choice == "XGBoost":
                model_params = {
                    "n_estimators": int(best_params["xgb_n_estimators"]),
                    "max_depth": int(best_params["xgb_max_depth"]),
                    "learning_rate": float(best_params["xgb_learning_rate"]),
                    "subsample": float(best_params["xgb_subsample"]),
                    "colsample_bytree": float(best_params["xgb_colsample_bytree"]),
                    "reg_alpha": float(best_params["xgb_reg_alpha"]),
                    "reg_lambda": float(best_params["xgb_reg_lambda"]),
                    "gamma": float(best_params["xgb_gamma"]),
                }
            elif model_choice == "Neural Network":
                model_params = {
                    "hidden_dim": int(best_params["nn_hidden_dim"]),
                    "num_layers": int(best_params["nn_num_layers"]),
                    "dropout": float(best_params["nn_dropout"]),
                    "learning_rate": float(best_params["nn_learning_rate"]),
                    "weight_decay": float(best_params["nn_weight_decay"]),
                    "pos_weight": float(best_params["nn_pos_weight"]),
                    "batch_size": int(best_params["nn_batch_size"]),
                }
            else:
                model_params = {
                    "kernel": best_params["svm_kernel"],
                    "C": float(best_params["svm_C"]),
                }

            trials_df = study.trials_dataframe(
                attrs=("number", "value", "params", "state")
            )
            trials_df = trials_df.sort_values(
                "value", ascending=objective_direction == "minimize"
            ).reset_index(drop=True)

            if optuna_tune_topk and "top_k" in best_params:
                best_top_k = int(best_params["top_k"])
                best_top_k = max(1, min(best_top_k, len(ranked_cols)))
                best_feature_cols = list(ranked_cols[:best_top_k])
            else:
                best_top_k = len(cols)
                best_feature_cols = list(cols)

            return {
                "balance_mode": balance_mode,
                "balance_mode_label": balance_label,
                "calibration_method": optuna_calibration_method,
                "calibration_method_label": _calibration_method_label(
                    optuna_calibration_method
                ),
                "best_score": best_score,
                "smote_params": smote_params,
                "model_params": model_params,
                "trials_df": trials_df,
                "best_top_k": int(best_top_k),
                "best_feature_cols": best_feature_cols,
                "ranked_cols": list(ranked_cols),
            }

        # Flag local para "ya promovimos un primary en esta corrida". Sustituye
        # la lectura de ``session_state["optuna_best_model_params"]`` (legacy)
        # que podía contaminarse con resultados de corridas previas de Optuna.
        primary_promoted_this_run = False

        for cfg in configs:
            for balance_mode in balance_modes_to_run:
                res = _run_optimization(
                    cfg["cols"],
                    cfg["label"],
                    balance_mode=balance_mode,
                )
                if not res:
                    continue

                should_promote_primary = (
                    cfg["key"] == primary_key
                    and (
                        res["balance_mode"] == "smote"
                        or not primary_promoted_this_run
                    )
                )
                if should_promote_primary:
                    # DEPRECATED: escritura de keys top-level ``optuna_best_*``.
                    # Mantenida por compatibilidad hacia atrás; los consumidores
                    # nuevos deben usar ``_get_active_optuna_best()`` que lee
                    # directamente del ``optuna_results_store``.
                    st.session_state["optuna_best_smote_params"] = (
                        res["smote_params"] if res["balance_mode"] == "smote" else None
                    )
                    st.session_state["optuna_best_model_params"] = res["model_params"]
                    st.session_state["optuna_best_score"] = res["best_score"]
                    st.session_state["optuna_best_model_choice"] = model_choice
                    st.session_state["optuna_trials_df"] = res["trials_df"]
                    primary_promoted_this_run = True
                    if res["balance_mode"] == "smote" and res["smote_params"]:
                        st.session_state["smote_k_neighbors"] = res["smote_params"][
                            "smote_k_neighbors"
                        ]
                        st.session_state["smote_sampling_strategy"] = res["smote_params"][
                            "smote_sampling_strategy"
                        ]

                search_space: Dict[str, object] = {}
                if res["balance_mode"] == "smote":
                    search_space["smote"] = {
                        "k_neighbors": {
                            "min": int(smote_k_min),
                            "max": int(smote_k_max),
                            "step": int(smote_k_step),
                        },
                        "sampling_strategy": {
                            "min": float(smote_sampling_min),
                            "max": float(smote_sampling_max),
                            "step": float(smote_sampling_step),
                        },
                    }
                if optuna_tune_topk:
                    search_space["top_k"] = {
                        "min": int(optuna_k_min),
                        "max": int(optuna_k_max),
                        "step": int(optuna_k_step),
                        "ranking_method": str(optuna_ranking_method),
                        "ranking_method_label": optuna_ranking_method_label,
                    }
                if model_choice == "Random Forest":
                    search_space["model"] = {
                        "n_estimators": {
                            "min": int(rf_n_min),
                            "max": int(rf_n_max),
                            "step": int(rf_n_step),
                        },
                        "max_depth": {
                            "min": int(rf_depth_min),
                            "max": int(rf_depth_max),
                            "step": int(rf_depth_step),
                        },
                    }
                elif model_choice == "XGBoost":
                    search_space["model"] = {
                        "n_estimators": {
                            "min": int(xgb_n_min),
                            "max": int(xgb_n_max),
                            "step": int(xgb_n_step),
                        },
                        "max_depth": {
                            "min": int(xgb_depth_min),
                            "max": int(xgb_depth_max),
                            "step": int(xgb_depth_step),
                        },
                        "learning_rate": {
                            "min": float(xgb_lr_min),
                            "max": float(xgb_lr_max),
                            "step": float(xgb_lr_step),
                        },
                        "subsample": {
                            "min": float(xgb_sub_min),
                            "max": float(xgb_sub_max),
                            "step": float(xgb_sub_step),
                        },
                        "colsample_bytree": {
                            "min": float(xgb_col_min),
                            "max": float(xgb_col_max),
                            "step": float(xgb_col_step),
                        },
                        "reg_alpha": {
                            "min": float(xgb_reg_alpha_min),
                            "max": float(xgb_reg_alpha_max),
                            "step": float(xgb_reg_alpha_step),
                        },
                        "reg_lambda": {
                            "min": float(xgb_reg_lambda_min),
                            "max": float(xgb_reg_lambda_max),
                            "step": float(xgb_reg_lambda_step),
                        },
                        "gamma": {
                            "min": float(xgb_gamma_min),
                            "max": float(xgb_gamma_max),
                            "step": float(xgb_gamma_step),
                        },
                    }
                elif model_choice == "Neural Network":
                    search_space["model"] = {
                        "hidden_dim": {
                            "min": int(nn_hidden_min),
                            "max": int(nn_hidden_max),
                            "step": int(nn_hidden_step),
                        },
                        "num_layers": {
                            "min": int(nn_layers_min),
                            "max": int(nn_layers_max),
                            "step": int(nn_layers_step),
                        },
                        "dropout": {
                            "min": float(nn_dropout_min),
                            "max": float(nn_dropout_max),
                            "step": float(nn_dropout_step),
                        },
                        "learning_rate": {
                            "min": float(nn_lr_min),
                            "max": float(nn_lr_max),
                            "step": float(nn_lr_step),
                        },
                        "weight_decay": {
                            "min": float(nn_wd_min),
                            "max": float(nn_wd_max),
                            "step": float(nn_wd_step),
                        },
                        "pos_weight": {
                            "min": float(nn_pw_min),
                            "max": float(nn_pw_max),
                            "step": float(nn_pw_step),
                        },
                        "batch_size": list(nn_batch_options),
                    }
                else:
                    search_space["model"] = {
                        "kernel": list(svm_kernels),
                        "C": {
                            "min": float(svm_c_min),
                            "max": float(svm_c_max),
                            "step": float(svm_c_step),
                        },
                    }

                optuna_settings_payload = {
                    "n_trials": int(n_trials),
                    "timeout": int(timeout),
                    "n_jobs": int(optuna_n_jobs),
                    "random_state": int(optuna_random_state),
                    "test_size": float(optuna_test_size),
                    "val_size": float(optuna_val_size),
                    "far_target": float(optuna_far_target),
                    "threshold_objective": str(optuna_threshold_objective),
                    "threshold_objective_label": optuna_threshold_objective_label,
                    "calibration_method": str(optuna_calibration_method),
                    "calibration_method_label": _calibration_method_label(
                        optuna_calibration_method
                    ),
                    "alerts_per_day": float(optuna_alerts_per_day),
                    "fn_cost": float(optuna_fn_cost),
                    "fp_cost": float(optuna_fp_cost),
                    "objective_metric": objective_key,
                    "objective_label": objective_label,
                    "objective_direction": objective_direction,
                    "objective_eval_set": "val",
                    "balance_mode": str(res["balance_mode"]),
                    "balance_mode_label": str(res["balance_mode_label"]),
                    "tune_topk": bool(optuna_tune_topk),
                    "ranking_method": str(optuna_ranking_method)
                    if optuna_tune_topk
                    else None,
                    "ranking_method_label": optuna_ranking_method_label
                    if optuna_tune_topk
                    else None,
                    "k_min": int(optuna_k_min) if optuna_tune_topk else None,
                    "k_max": int(optuna_k_max) if optuna_tune_topk else None,
                    "k_step": int(optuna_k_step) if optuna_tune_topk else None,
                    "best_top_k": int(res["best_top_k"]),
                    "best_feature_cols": list(res["best_feature_cols"]),
                    "ranked_cols": list(res["ranked_cols"]),
                    "pruner": {
                        "enabled": bool(pruner_enabled),
                        "type": "MedianPruner" if pruner_enabled else "NopPruner",
                        "startup_trials": int(pruner_startup_trials),
                    },
                }
                if should_promote_primary:
                    # DEPRECATED: escrituras top-level ``optuna_best_settings`` /
                    # ``optuna_best_search_space``; preferir lectura vía
                    # ``_get_active_optuna_best()``.
                    st.session_state["optuna_best_settings"] = optuna_settings_payload
                    st.session_state["optuna_best_search_space"] = search_space

                _persist_optuna_results(
                    optuna_key=cfg["key"],
                    optuna_id=cfg["id"],
                    feature_key=feature_key,
                    feature_id=feature_id,
                    features_path=features_path,
                    features_source=features_source,
                    features_df=features_df,
                    selected_features=selected_features,
                    feature_cols=cfg["cols"],
                    model_choice=model_choice,
                    balance_mode=str(res["balance_mode"]),
                    calibration_method=str(res["calibration_method"]),
                    best_score=res["best_score"],
                    best_smote_params=res["smote_params"],
                    best_model_params=res["model_params"],
                    trials_df=res["trials_df"],
                    optuna_settings=optuna_settings_payload,
                    search_space=search_space,
                )
                if optuna_tune_topk:
                    st.success(
                        f"Optuna ({cfg['label']} | {res['balance_mode_label']}) finalizado. "
                        f"{'Menor' if objective_direction == 'minimize' else 'Mejor'} "
                        f"{objective_label}: {res['best_score']:.4f} | "
                        f"best top_k = {res['best_top_k']} / {len(res['ranked_cols'])}"
                    )
                else:
                    st.success(
                        f"Optuna ({cfg['label']} | {res['balance_mode_label']}) finalizado. "
                        f"{'Menor' if objective_direction == 'minimize' else 'Mejor'} "
                        f"{objective_label}: {res['best_score']:.4f}"
                    )

        st.rerun()

    st.subheader("Resultados guardados")
    res_tabs = st.tabs([c["label"] for c in configs])
    for idx, cfg in enumerate(configs):
        with res_tabs[idx]:
            entry_cfg = store.get(cfg["key"])
            results_cfg = (
                entry_cfg.get("results")
                if entry_cfg and isinstance(entry_cfg.get("results"), dict)
                else None
            )
            variants_present = [
                mode
                for mode in OPTUNA_BALANCE_MODE_ORDER
                if isinstance(
                    (
                        (_get_optuna_model_result_container(results_cfg, model_choice) or {})
                        .get("by_balance_mode", {})
                        .get(mode, {})
                    ).get("by_calibration_method"),
                    dict,
                )
            ]
            if not variants_present:
                st.info(
                    f"No hay resultados guardados para {cfg['label']} con {model_choice}."
                )
                continue

            variant_tabs = st.tabs(
                [_optuna_balance_mode_label(mode) for mode in variants_present]
            )
            for variant_idx, balance_mode in enumerate(variants_present):
                with variant_tabs[variant_idx]:
                    model_container = _get_optuna_model_result_container(
                        results_cfg,
                        model_choice,
                    )
                    mode_container = (
                        dict(model_container or {}).get("by_balance_mode", {}).get(
                            balance_mode
                        )
                    )
                    by_calibration_method = (
                        mode_container.get("by_calibration_method")
                        if isinstance(mode_container, dict)
                        else None
                    )
                    if not isinstance(by_calibration_method, dict) or not by_calibration_method:
                        st.info("Sin resultados para esta variante.")
                        continue
                    calibration_methods = _ordered_calibration_methods(
                        list(by_calibration_method.keys())
                    )
                    calibration_tabs = st.tabs(
                        [
                            _calibration_method_label(method)
                            for method in calibration_methods
                        ]
                    )
                    for calibration_idx, calibration_method in enumerate(
                        calibration_methods
                    ):
                        with calibration_tabs[calibration_idx]:
                            res_cfg = by_calibration_method.get(calibration_method)
                            if not isinstance(res_cfg, dict):
                                st.info("Sin resultados para este calibrador.")
                                continue
                            trials_df_cfg = res_cfg.get("trials_df")
                            trials_csv_cfg = res_cfg.get("trials_csv")
                            if (
                                trials_df_cfg is None
                                and trials_csv_cfg
                                and Path(str(trials_csv_cfg)).exists()
                            ):
                                try:
                                    trials_df_cfg = pd.read_csv(trials_csv_cfg)
                                    res_cfg["trials_df"] = trials_df_cfg
                                except Exception:
                                    pass

                            saved_score = res_cfg.get("best_score")
                            saved_settings = res_cfg.get("optuna_settings")
                            metric_label = "F1"
                            if isinstance(saved_settings, dict) and saved_settings:
                                metric_label = saved_settings.get(
                                    "objective_label", metric_label
                                )
                            if saved_score is not None:
                                st.metric(metric_label, f"{float(saved_score):.4f}")
                            if isinstance(saved_settings, dict) and saved_settings:
                                st.caption("Configuración Optuna")
                                st.json(saved_settings)
                            saved_space = res_cfg.get("search_space")
                            if isinstance(saved_space, dict) and saved_space:
                                st.caption("Rangos y pasos usados")
                                st.json(saved_space)
                            saved_smote = res_cfg.get("best_smote_params")
                            if isinstance(saved_smote, dict) and saved_smote:
                                st.caption("Mejor SMOTE")
                                st.json(saved_smote)
                            saved_model = res_cfg.get("best_model_params")
                            if isinstance(saved_model, dict) and saved_model:
                                st.caption("Mejor modelo")
                                st.json(saved_model)
                            saved_trials = res_cfg.get("trials_df")
                            if isinstance(saved_trials, pd.DataFrame) and not saved_trials.empty:
                                st.caption("Top trials")
                                st.dataframe(saved_trials.head(20), width="stretch")








def _render_balance_tab() -> None:
    st.subheader("Balance con SMOTE")
    _render_selected_features_info()
    _render_state_diagnostics("balance")

    accidents_df = st.session_state.get("accidents_df")
    features_df = st.session_state.get("flow_features_df")

    if accidents_df is None or accidents_df.empty:
        st.info("Cargue accidentes en la pestana Eventos.")
        return
    if features_df is None or features_df.empty:
        st.info("Calcule variables de flujo en la pestana Feature engineering.")
        return

    base_df = add_accident_target(features_df, accidents_df)
    if base_df.empty:
        st.warning("No se pudo preparar el dataset base.")
        return

    balance_sources = [
        "Cargar dataset balanceado",
        "Balancear nuevos datos",
        "Dataset balanceado en memoria",
    ]
    balance_source_key = "balance_source"
    if (
        balance_source_key in st.session_state
        and st.session_state[balance_source_key] == "Datasets balanceados en memoria"
    ):
        st.session_state[balance_source_key] = "Dataset balanceado en memoria"
    if (
        balance_source_key in st.session_state
        and st.session_state[balance_source_key] not in balance_sources
    ):
        st.session_state[balance_source_key] = balance_sources[0]
    balance_source = st.radio(
        "Fuente de balanceo",
        balance_sources,
        horizontal=True,
        key=balance_source_key,
    )

    if balance_source == "Cargar dataset balanceado":
        balanced_files = _list_balanced_files()
        if not balanced_files:
            st.warning("No se encontraron archivos accident_balanced_*.csv en Resultados.")
        else:
            balanced_names = [path.name for path in balanced_files]
            selected_balanced = st.selectbox(
                "Archivo de dataset balanceado",
                options=["(ninguno)"] + balanced_names,
                key="balance_load_file",
            )
            if st.button("Cargar dataset balanceado"):
                if selected_balanced == "(ninguno)":
                    st.warning("Seleccione un archivo de Resultados.")
                else:
                    try:
                        loaded_df = pd.read_csv(RESULTS_DIR / selected_balanced)
                    except Exception as exc:
                        st.error(f"No se pudo cargar {selected_balanced}: {exc}")
                    else:
                        if (
                            "target" not in loaded_df.columns
                            or "split" not in loaded_df.columns
                        ):
                            st.warning(
                                "El archivo seleccionado debe incluir columnas target y split."
                            )
                        else:
                            splits = set(loaded_df["split"].dropna().unique().tolist())
                            if not {"train", "test"}.issubset(splits):
                                st.warning(
                                    "El archivo no contiene split train/test valido."
                                )
                            else:
                                if "synthetic" not in loaded_df.columns:
                                    loaded_df["synthetic"] = False
                                    st.info(
                                        "El dataset cargado no tiene columna synthetic; "
                                        "se marco todo como False."
                                    )
                                st.session_state["balanced_base_df"] = loaded_df
                                st.session_state["balanced_cluster_only_df"] = None
                                st.session_state["balanced_cluster_df"] = None
                                st.session_state["balance_last_stats"] = (
                                    _balance_stats_from_df(loaded_df)
                                )
                                st.session_state["balance_last_params"] = {
                                    "source": "archivo",
                                    "file": selected_balanced,
                                }
                                st.success(
                                    f"Dataset balanceado cargado: {selected_balanced}"
                                )
    elif balance_source == "Balancear nuevos datos":
        st.caption("Distribucion de clases en el dataset.")
        dist_total = _class_distribution(base_df["target"])
        st.dataframe(dist_total, width="stretch")

        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.4,
            value=float(st.session_state.get("test_size", 0.2)),
            step=0.05,
            key="balance_test_size",
        )
        st.session_state["test_size"] = float(test_size)

        # Lee ``best_smote_params`` directamente del store vía el helper
        # canónico ``_get_active_optuna_best`` en lugar del key legacy
        # ``optuna_best_smote_params``. Esto protege al Balance tab de leer
        # un valor stale si alguien mutó el top-level key sin actualizar el
        # store (p.ej. legacy disk reloader).
        _active_optuna_best = _get_active_optuna_best()
        optuna_smote = (
            _active_optuna_best.get("best_smote_params")
            if isinstance(_active_optuna_best, dict)
            else None
        )
        optuna_active_key = st.session_state.get("optuna_active_key")
        features_path = st.session_state.get("flow_features_path")
        features_source = st.session_state.get("flow_features_source")
        feature_key = _feature_selection_key(
            features_path, features_source, features_df
        )
        selected_features = st.session_state.get("selected_features")
        numeric_cols = _get_feature_cols(base_df)
        if selected_features is None:
            optuna_feature_cols = numeric_cols
        else:
            optuna_feature_cols = [
                col for col in selected_features if col in numeric_cols
            ]
        optuna_key = _optuna_result_key(feature_key, optuna_feature_cols)
        optuna_matches = optuna_active_key == optuna_key
        use_optuna_smote = False
        if isinstance(optuna_smote, dict) and optuna_smote:
            if optuna_matches:
                use_optuna_smote = st.checkbox(
                    "Usar parametros Optuna para SMOTE",
                    value=True,
                    key="use_optuna_smote",
                )
            else:
                diagnosis = _diagnose_optuna_key_mismatch(
                    store=st.session_state.get("optuna_results_store") or {},
                    expected_key=optuna_key,
                    active_key=optuna_active_key,
                    current_fingerprint=_dataset_content_fingerprint(features_df),
                )
                reasons = diagnosis.get("reasons") or []
                if reasons:
                    reasons_text = "; ".join(reasons)
                else:
                    reasons_text = (
                        "no se pudo determinar la causa exacta del desajuste"
                    )
                st.warning(
                    "⚠️ Los parámetros de Optuna no aplican al dataset / "
                    f"selección actual: {reasons_text}. "
                    "Re-ejecute Optuna sobre este dataset y selección para "
                    "actualizar los parámetros SMOTE sugeridos."
                )
        else:
            st.info(
                "ℹ️ No hay resultados de Optuna disponibles para sugerir parametros."
            )

        col1, col2 = st.columns(2)
        with col1:
            smote_random_state = st.number_input(
                "SMOTE random_state",
                min_value=0,
                value=int(st.session_state.get("smote_random_state", 42)),
                step=1,
                key="smote_random_state_input",
            )
        with col2:
            smote_k = st.number_input(
                "SMOTE k_neighbors",
                min_value=1,
                value=int(st.session_state.get("smote_k_neighbors", 5)),
                step=1,
                key="smote_k_neighbors_input",
                disabled=use_optuna_smote,
            )
        st.session_state["smote_random_state"] = int(smote_random_state)
        smote_sampling_strategy: Optional[float] = None
        if use_optuna_smote and isinstance(optuna_smote, dict):
            smote_k = int(
                optuna_smote.get("smote_k_neighbors", smote_k)
            )
            smote_sampling_strategy = optuna_smote.get(
                "smote_sampling_strategy"
            )
            if smote_sampling_strategy is not None:
                smote_sampling_strategy = float(smote_sampling_strategy)
            st.caption(
                "Optuna SMOTE: "
                f"k_neighbors={smote_k} | "
                f"sampling_strategy="
                f"{smote_sampling_strategy if smote_sampling_strategy is not None else 'auto'}"
            )
        st.session_state["smote_k_neighbors"] = int(smote_k)
        st.session_state["smote_sampling_strategy"] = smote_sampling_strategy

        if st.button("Aplicar SMOTE"):
            progress = _StreamlitProgress(total=5)
            with st.spinner("Aplicando SMOTE..."):
                progress.set_description("Preparando dataset")
                dataset_df = base_df
                progress.update(1)

                try:
                    progress.set_description("Aplicando SMOTE (Base)")
                    selected_features = st.session_state.get("selected_features")

                    # 1. Base (Flow only) — alineado con el modelo "Base" de Optuna
                    feature_cols_base, missing_base = _resolve_feature_group_cols(
                        dataset_df,
                        selected_features,
                        feature_group="base",
                    )

                    balanced_base_df = None
                    dist_before_base = dist_after_base = dist_test_base = None
                    k_used_base = None

                    if feature_cols_base:
                        (
                            balanced_base_df,
                            dist_before_base,
                            dist_after_base,
                            dist_test_base,
                            k_used_base,
                        ) = _apply_smote_dataset(
                            dataset_df,
                            feature_cols_base,
                            test_size=float(test_size),
                            split_random_state=int(smote_random_state),
                            random_state=int(smote_random_state),
                            smote_k_neighbors=int(smote_k),
                            smote_sampling_strategy=smote_sampling_strategy,
                        )
                    progress.update(1)

                    # 2. Cluster (cluster-only) — alineado con el modelo "Cluster" de Optuna
                    progress.set_description("Aplicando SMOTE (Cluster)")
                    balanced_cluster_only_df = None
                    dist_before_cluster_only = dist_after_cluster_only = dist_test_cluster_only = None
                    k_used_cluster_only = None

                    has_cluster_cols = bool(_get_cluster_cols(dataset_df))
                    feature_cols_cluster_only: List[str] = []
                    if has_cluster_cols:
                        feature_cols_cluster_only, _ = _resolve_feature_group_cols(
                            dataset_df,
                            selected_features,
                            feature_group="cluster",
                        )
                        if feature_cols_cluster_only:
                            (
                                balanced_cluster_only_df,
                                dist_before_cluster_only,
                                dist_after_cluster_only,
                                dist_test_cluster_only,
                                k_used_cluster_only,
                            ) = _apply_smote_dataset(
                                dataset_df,
                                feature_cols_cluster_only,
                                test_size=float(test_size),
                                split_random_state=int(smote_random_state),
                                random_state=int(smote_random_state),
                                smote_k_neighbors=int(smote_k),
                                smote_sampling_strategy=smote_sampling_strategy,
                            )
                    progress.update(1)

                    # 3. Base + Cluster — alineado con el modelo "Base + Cluster" de Optuna
                    progress.set_description("Aplicando SMOTE (Base + Cluster)")
                    balanced_cluster_df = None

                    if has_cluster_cols:
                        feature_cols_cluster, missing_cluster = _resolve_feature_group_cols(
                            dataset_df,
                            selected_features,
                            feature_group="base_cluster",
                        )
                        # Solo si hay diferencia real con Base (es decir, existen cluster cols seleccionadas)
                        if set(feature_cols_cluster) != set(feature_cols_base):
                            (
                                balanced_cluster_df,
                                _,
                                _,
                                _,
                                _,
                            ) = _apply_smote_dataset(
                                dataset_df,
                                feature_cols_cluster,
                                test_size=float(test_size),
                                split_random_state=int(smote_random_state),
                                random_state=int(smote_random_state),
                                smote_k_neighbors=int(smote_k),
                                smote_sampling_strategy=smote_sampling_strategy,
                            )
                    progress.update(1)

                except Exception as exc:
                    progress.close()
                    st.error(f"No se pudo aplicar SMOTE: {exc}")
                else:
                    progress.set_description("Finalizando")
                    st.session_state["balanced_base_df"] = balanced_base_df
                    st.session_state["balanced_cluster_only_df"] = balanced_cluster_only_df
                    st.session_state["balanced_cluster_df"] = balanced_cluster_df

                    # Store stats for base (primary)
                    st.session_state["balance_last_stats"] = {
                        "train_before": dist_before_base.to_dict(orient="records") if dist_before_base is not None else [],
                        "train_after": dist_after_base.to_dict(orient="records") if dist_after_base is not None else [],
                        "test": dist_test_base.to_dict(orient="records") if dist_test_base is not None else [],
                    }
                    st.session_state["balance_last_params"] = {
                        "source": "smote",
                        "test_size": float(test_size),
                        "random_state": int(smote_random_state),
                        "k_neighbors": int(k_used_base) if k_used_base else 0,
                        "sampling_strategy": smote_sampling_strategy,
                    }

                    msg = []
                    if balanced_base_df is not None:
                        msg.append(f"Base: {len(balanced_base_df):,} filas")
                    if balanced_cluster_only_df is not None:
                        msg.append(f"Cluster: {len(balanced_cluster_only_df):,} filas")
                    if balanced_cluster_df is not None:
                        msg.append(f"Base + Cluster: {len(balanced_cluster_df):,} filas")

                    st.success(f"Datasets balanceados generados. {', '.join(msg)}")

                    if dist_before_base is not None:
                        st.caption(f"SMOTE k_neighbors usado (Base): {k_used_base}")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.caption("Train antes SMOTE")
                            st.dataframe(dist_before_base, width="stretch")
                        with col_b:
                            st.caption("Train despues SMOTE")
                            st.dataframe(dist_after_base, width="stretch")
                        with col_c:
                            st.caption("Test (sin balancear)")
                            st.dataframe(dist_test_base, width="stretch")

                    progress.update(1)
                    progress.close()

    else:
        st.subheader("Datasets balanceados en memoria")
        balanced_base = st.session_state.get("balanced_base_df")
        balanced_cluster_only = st.session_state.get("balanced_cluster_only_df")
        balanced_cluster = st.session_state.get("balanced_cluster_df")

        if (
            balanced_base is None
            and balanced_cluster_only is None
            and balanced_cluster is None
        ):
            st.info("No hay dataset balanceado en memoria.")
        else:
            # Validate consistency with selected features
            selected_features = st.session_state.get("selected_features", [])
            if selected_features:
                # Identify cluster cols in the standard set to distinguish flow vs cluster
                all_cluster_cols = set(_get_cluster_cols(features_df))

                if balanced_base is not None:
                    missing_inv = [
                        c
                        for c in selected_features
                        if c not in balanced_base.columns and c not in all_cluster_cols
                    ]
                    if missing_inv:
                        st.warning(
                            "⚠️ El dataset Base en memoria no contiene variables de flujo "
                            f"seleccionadas: {', '.join(missing_inv)}"
                        )

                if balanced_cluster_only is not None:
                    cluster_selected = [
                        c for c in selected_features if c in all_cluster_cols
                    ]
                    missing_co = [
                        c for c in cluster_selected if c not in balanced_cluster_only.columns
                    ]
                    if missing_co:
                        st.warning(
                            "⚠️ El dataset Cluster en memoria no contiene variables "
                            f"de cluster seleccionadas: {', '.join(missing_co)}"
                        )

                if balanced_cluster is not None:
                    missing_all = [
                        c for c in selected_features if c not in balanced_cluster.columns
                    ]
                    if missing_all:
                         st.warning(
                            "⚠️ El dataset Base + Cluster en memoria no contiene variables "
                            f"seleccionadas: {', '.join(missing_all)}"
                        )

            tabs = st.tabs(["Base (Flujo)", "Cluster", "Base + Cluster"])

            def _show_balanced_info(df: pd.DataFrame, label: str):
                if df is None:
                    st.info(f"No hay dataset {label} balanceado.")
                    return
                st.caption(f"Dataset {label}: {len(df):,} filas")
                if "target" in df.columns and "split" in df.columns:
                    train_mask = df["split"] == "train"
                    test_mask = df["split"] == "test"
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption("Train")
                        st.dataframe(_class_distribution(df.loc[train_mask, "target"]), width="stretch")
                    with col_b:
                        st.caption("Test")
                        st.dataframe(_class_distribution(df.loc[test_mask, "target"]), width="stretch")
                st.dataframe(df.head(20), width="stretch")

            with tabs[0]:
                _show_balanced_info(balanced_base, "Base")
            with tabs[1]:
                _show_balanced_info(balanced_cluster_only, "Cluster")
            with tabs[2]:
                _show_balanced_info(balanced_cluster, "Base + Cluster")

        if (
            balanced_base is not None
            or balanced_cluster_only is not None
            or balanced_cluster is not None
        ):
            st.subheader("Exportar dataset balanceado")
            export_name = st.text_input(
                "Nombre de archivo (sin .csv)",
                value="accident_balanced",
                key="export_balanced_name",
            )
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                if balanced_base is not None and st.button("Exportar Base"):
                    out_path = RESULTS_DIR / f"{export_name.strip()}_base.csv"
                    try:
                        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                        balanced_base.to_csv(out_path, index=False)
                        st.success(f"Base exportado en {out_path}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")
            with col_exp2:
                if balanced_cluster_only is not None and st.button("Exportar Cluster"):
                    out_path = RESULTS_DIR / f"{export_name.strip()}_cluster_only.csv"
                    try:
                        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                        balanced_cluster_only.to_csv(out_path, index=False)
                        st.success(f"Cluster exportado en {out_path}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")
            with col_exp3:
                if balanced_cluster is not None and st.button("Exportar Base + Cluster"):
                    out_path = RESULTS_DIR / f"{export_name.strip()}_base_cluster.csv"
                    try:
                        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                        balanced_cluster.to_csv(out_path, index=False)
                        st.success(f"Base + Cluster exportado en {out_path}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    use_balanced = bool(
        st.session_state.get("use_balanced_base", False)
        or st.session_state.get("use_balanced_cluster", False)
    )
    st.session_state["use_balanced_base"] = use_balanced
    st.session_state["use_balanced_cluster"] = use_balanced


def _render_model_params_ui(model_choice: str, prefix: str) -> Dict[str, object]:
    params = {}
    if model_choice in {"Random Forest", "Balanced Random Forest"}:
        n_estimators = st.number_input(
            "n_estimators",
            min_value=50,
            value=200,
            step=50,
            key=f"{prefix}model_rf_n_estimators",
        )
        max_depth = st.number_input(
            "max_depth (0 = sin limite)",
            min_value=0,
            value=0,
            step=1,
            key=f"{prefix}model_rf_max_depth",
        )
        class_weight = None
        if model_choice == "Random Forest":
            class_weight_label = st.selectbox(
                "class_weight",
                ["balanced", "None"],
                key=f"{prefix}model_rf_class_weight",
                help=(
                    "Peso de clases para Random Forest. Valores: balanced o None. "
                    "Default: balanced. Balanced compensa el desbalance; usar None "
                    "puede favorecer la clase mayoritaria."
                ),
            )
            class_weight = None if class_weight_label == "None" else class_weight_label
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth) if max_depth else None,
        }
        if model_choice == "Random Forest":
            params["class_weight"] = class_weight
    elif model_choice == "XGBoost":
        n_estimators = st.number_input(
            "n_estimators",
            min_value=50,
            value=300,
            step=50,
            key=f"{prefix}model_xgb_n_estimators",
        )
        max_depth = st.number_input(
            "max_depth",
            min_value=2,
            value=6,
            step=1,
            key=f"{prefix}model_xgb_max_depth",
        )
        learning_rate = st.number_input(
            "learning_rate",
            min_value=0.01,
            value=0.1,
            step=0.01,
            format="%.2f",
            key=f"{prefix}model_xgb_learning_rate",
        )
        subsample = st.number_input(
            "subsample",
            min_value=0.5,
            value=1.0,
            step=0.1,
            format="%.2f",
            key=f"{prefix}model_xgb_subsample",
        )
        colsample = st.number_input(
            "colsample_bytree",
            min_value=0.5,
            value=1.0,
            step=0.1,
            format="%.2f",
            key=f"{prefix}model_xgb_colsample",
        )
        reg_alpha = st.number_input(
            "reg_alpha",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"{prefix}model_xgb_reg_alpha",
        )
        reg_lambda = st.number_input(
            "reg_lambda",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.2f",
            key=f"{prefix}model_xgb_reg_lambda",
        )
        gamma = st.number_input(
            "gamma",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            key=f"{prefix}model_xgb_gamma",
        )
        scale_pos_weight = st.number_input(
            "scale_pos_weight (0 = auto)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            key=f"{prefix}model_xgb_scale_pos_weight",
            help=(
                "Peso de la clase positiva para XGBoost. Use 0 para calcular "
                "negativos/positivos automáticamente cuando la estrategia de "
                "desbalance lo requiera; valores manuales altos suben recall pero "
                "pueden disparar falsas alarmas."
            ),
        )
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample),
            "reg_alpha": float(reg_alpha),
            "reg_lambda": float(reg_lambda),
            "gamma": float(gamma),
        }
        if float(scale_pos_weight) > 0:
            params["scale_pos_weight"] = float(scale_pos_weight)
        else:
            params["scale_pos_weight"] = "auto"
    elif model_choice == "Neural Network":
        hidden_dim = st.number_input(
            "hidden_dim",
            min_value=32,
            value=256,
            step=32,
            key=f"{prefix}model_nn_hidden_dim",
        )
        num_layers = st.number_input(
            "num_layers",
            min_value=1,
            value=2,
            step=1,
            key=f"{prefix}model_nn_num_layers",
        )
        dropout = st.number_input(
            "dropout",
            min_value=0.0,
            value=0.2,
            step=0.05,
            format="%.2f",
            key=f"{prefix}model_nn_dropout",
        )
        learning_rate = st.number_input(
            "learning_rate",
            min_value=0.0001,
            value=0.001,
            step=0.0001,
            format="%.4f",
            key=f"{prefix}model_nn_learning_rate",
        )
        weight_decay = st.number_input(
            "weight_decay",
            min_value=0.0,
            value=0.00001,
            step=0.00001,
            format="%.6f",
            key=f"{prefix}model_nn_weight_decay",
        )
        batch_size = st.selectbox(
            "batch_size",
            [256, 512, 1024, 2048, 4096, 8192],
            index=2,
            key=f"{prefix}model_nn_batch_size",
            help=(
                "Batch mayor = mejor ocupacion de GPU (MPS/CUDA). "
                "El wrapper precarga train/val en device y evita copias H2D "
                "por batch; elige >=1024 si tienes GPU disponible."
            ),
        )
        epochs = st.number_input(
            "epochs (maximo)",
            min_value=5,
            value=100,
            step=5,
            key=f"{prefix}model_nn_epochs",
            help=(
                "Maximo de epocas de entrenamiento. El early stopping "
                "(paciencia 5) cortara antes si la metrica de validacion "
                "deja de mejorar."
            ),
        )
        early_stopping_patience = 5
        st.caption(
            f"Early stopping activo | paciencia = {early_stopping_patience} "
            "| val_fraction = 0.15"
        )
        pos_weight = st.number_input(
            "pos_weight (0 = auto)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            key=f"{prefix}model_nn_pos_weight",
            help=(
                "Peso de la clase positiva en CrossEntropyLoss. "
                "Use 0 para calcular negativos/positivos automáticamente "
                "cuando la estrategia de desbalance lo requiera."
            ),
        )
        params = {
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "early_stopping_patience": int(early_stopping_patience),
        }
        if float(pos_weight) > 0:
            params["pos_weight"] = float(pos_weight)
        else:
            params["pos_weight"] = "auto"
    elif model_choice == "SVM":
        kernel = st.selectbox(
            "kernel",
            ["rbf", "linear", "poly", "sigmoid"],
            key=f"{prefix}model_svm_kernel",
        )
        c_value = st.number_input(
            "C", min_value=0.01, value=1.0, step=0.1, key=f"{prefix}model_svm_c"
        )
        class_weight_label = st.selectbox(
            "class_weight",
            ["balanced", "None"],
            key=f"{prefix}model_svm_class_weight",
            help=(
                "Peso de clases para SVM. Valores: balanced o None. Default: "
                "balanced. Usar None en datos raros suele reducir sensibilidad "
                "a accidentes."
            ),
        )
        params = {
            "kernel": kernel,
            "C": float(c_value),
            "class_weight": None if class_weight_label == "None" else class_weight_label,
            "probability": False,
        }
    return params


def _apply_model_params_to_prefix(
    *,
    model_choice: str,
    prefix: str,
    params: Dict[str, object],
) -> None:
    if model_choice in {"Random Forest", "Balanced Random Forest"}:
        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth")
        st.session_state[f"{prefix}model_rf_n_estimators"] = max(50, n_estimators)
        st.session_state[f"{prefix}model_rf_max_depth"] = int(max_depth or 0)
    elif model_choice == "XGBoost":
        n_estimators = int(params.get("n_estimators", 300))
        max_depth = int(params.get("max_depth", 6))
        learning_rate = float(params.get("learning_rate", 0.1))
        subsample = float(params.get("subsample", 1.0))
        colsample = float(params.get("colsample_bytree", 1.0))
        reg_alpha = float(params.get("reg_alpha", 0.0))
        reg_lambda = float(params.get("reg_lambda", 1.0))
        gamma = float(params.get("gamma", 0.0))
        st.session_state[f"{prefix}model_xgb_n_estimators"] = max(50, n_estimators)
        st.session_state[f"{prefix}model_xgb_max_depth"] = max(2, max_depth)
        st.session_state[f"{prefix}model_xgb_learning_rate"] = max(0.01, learning_rate)
        st.session_state[f"{prefix}model_xgb_subsample"] = max(0.5, subsample)
        st.session_state[f"{prefix}model_xgb_colsample"] = max(0.5, colsample)
        st.session_state[f"{prefix}model_xgb_reg_alpha"] = max(0.0, reg_alpha)
        st.session_state[f"{prefix}model_xgb_reg_lambda"] = max(0.0, reg_lambda)
        st.session_state[f"{prefix}model_xgb_gamma"] = max(0.0, gamma)
    elif model_choice == "Neural Network":
        hidden_dim = int(params.get("hidden_dim", 256))
        num_layers = int(params.get("num_layers", 2))
        dropout = float(params.get("dropout", 0.2))
        learning_rate = float(params.get("learning_rate", 0.001))
        weight_decay = float(params.get("weight_decay", 1e-5))
        batch_size = int(params.get("batch_size", 1024))
        # epochs no se optimiza en Optuna: se conserva el valor maximo fijado
        # en los widgets de Modelos (early stopping decide cuando cortar).
        st.session_state[f"{prefix}model_nn_hidden_dim"] = max(32, hidden_dim)
        st.session_state[f"{prefix}model_nn_num_layers"] = max(1, num_layers)
        st.session_state[f"{prefix}model_nn_dropout"] = max(0.0, min(0.5, dropout))
        st.session_state[f"{prefix}model_nn_learning_rate"] = max(0.0001, learning_rate)
        st.session_state[f"{prefix}model_nn_weight_decay"] = max(0.0, weight_decay)
        st.session_state[f"{prefix}model_nn_batch_size"] = batch_size
    elif model_choice == "SVM":
        kernel_value = params.get("kernel", "rbf")
        if kernel_value not in {"rbf", "linear", "poly", "sigmoid"}:
            kernel_value = "rbf"
        c_value = float(params.get("C", 1.0))
        st.session_state[f"{prefix}model_svm_kernel"] = str(kernel_value)
        st.session_state[f"{prefix}model_svm_c"] = max(0.01, c_value)


def _apply_optuna_model_params_to_state(
    *,
    model_choice: str,
    balance_mode: str,
    calibration_method: str,
    base_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> Optional[str]:
    """Aplica los best_model_params de Optuna a los selectores de params por grupo.

    Para cada grupo (Base, Cluster, Base + Cluster), busca la corrida de Optuna
    que coincide con las columnas del grupo, el model_choice y la variante
    sin/con SMOTE. Si la encuentra, sobreescribe los widgets del prefijo
    correspondiente (base_/cluster_only_/cluster_). El match y la aplicacion
    ocurren por grupo de forma independiente.
    """
    features_path = st.session_state.get("flow_features_path")
    features_source = st.session_state.get("flow_features_source")
    feature_key = _feature_selection_key(
        features_path,
        features_source,
        features_df,
    )

    selected_features = st.session_state.get("selected_features")
    numeric_cols = _get_feature_cols(base_df)
    cluster_cols_set = set(_get_cluster_cols(base_df))
    if selected_features is None:
        cols_all = list(numeric_cols)
    else:
        cols_all = [col for col in selected_features if col in numeric_cols]
    cols_base = [col for col in cols_all if col not in cluster_cols_set]
    cols_cluster_only = [col for col in cols_all if col in cluster_cols_set]

    store = st.session_state.get("optuna_results_store") or {}

    group_defs = [
        ("base", "Base", "base_", cols_base),
        ("cluster_only", "Cluster", "cluster_only_", cols_cluster_only),
        ("base_cluster", "Base + Cluster", "cluster_", cols_all),
    ]

    signatures_map: Dict[str, str] = dict(
        st.session_state.get("optuna_model_params_applied_signatures") or {}
    )

    applied_labels: List[str] = []
    already_labels: List[str] = []
    missing_labels: List[str] = []

    for group_key, group_label, prefix, cols in group_defs:
        if group_key == "cluster_only" and not cols_cluster_only:
            signatures_map.pop(prefix, None)
            continue
        if group_key == "base_cluster" and set(cols_all) == set(cols_base):
            signatures_map.pop(prefix, None)
            continue
        if not cols:
            signatures_map.pop(prefix, None)
            continue

        key = _optuna_result_key(feature_key, cols)
        entry = store.get(key)
        if not isinstance(entry, dict):
            missing_labels.append(group_label)
            signatures_map.pop(prefix, None)
            continue
        # Respeta el opt-in del usuario: si no aceptó fallback de
        # calibración, sólo aplicar params de Optuna cuando la corrida
        # optimizó exactamente la calibración elegida en Modelos.
        allow_calibration_fallback = bool(
            st.session_state.get("allow_optuna_calibration_fallback", False)
        )
        match = _get_optuna_model_result_variant_match(
            entry.get("results"),
            model_choice=model_choice,
            balance_mode=balance_mode,
            calibration_method=calibration_method,
            allow_any_calibration_within_mode=allow_calibration_fallback,
        )
        result = match.get("result") if isinstance(match, dict) else None
        if not isinstance(result, dict):
            missing_labels.append(group_label)
            signatures_map.pop(prefix, None)
            continue
        best_model_params = result.get("best_model_params")
        if not isinstance(best_model_params, dict) or not best_model_params:
            missing_labels.append(group_label)
            signatures_map.pop(prefix, None)
            continue

        try:
            params_sig = json.dumps(best_model_params, sort_keys=True)
        except TypeError:
            params_sig = str(best_model_params)
        resolved_calibration_method = (
            str(match.get("resolved_calibration_method"))
            if isinstance(match, dict) and match.get("resolved_calibration_method")
            else _normalize_calibration_method(calibration_method)
        )
        full_sig = (
            f"{key}|{model_choice}|{balance_mode}|"
            f"{resolved_calibration_method}|{params_sig}"
        )
        if signatures_map.get(prefix) == full_sig:
            already_labels.append(group_label)
            continue

        _apply_model_params_to_prefix(
            model_choice=model_choice,
            prefix=prefix,
            params=best_model_params,
        )
        signatures_map[prefix] = full_sig
        applied_label = group_label
        if isinstance(match, dict) and bool(match.get("used_fallback")):
            applied_label = (
                f"{group_label} ({_calibration_method_label(resolved_calibration_method)})"
            )
        applied_labels.append(applied_label)

    st.session_state["optuna_model_params_applied_signatures"] = signatures_map

    if not applied_labels and not already_labels:
        if missing_labels:
            return (
                f"Optuna sin resultados para {model_choice} "
                f"({_optuna_balance_mode_label(balance_mode)} | "
                f"{_calibration_method_label(calibration_method)}) en: "
                + ", ".join(missing_labels)
                + "."
            )
        return None

    parts: List[str] = []
    if applied_labels:
        parts.append("aplicados: " + ", ".join(applied_labels))
    if already_labels:
        parts.append("ya cargados: " + ", ".join(already_labels))
    if missing_labels:
        parts.append("sin match: " + ", ".join(missing_labels))
    return (
        "Parametros Optuna "
        f"({_optuna_balance_mode_label(balance_mode)} | "
        f"{_calibration_method_label(calibration_method)}) — "
        + " | ".join(parts)
    )


def _render_model_tab() -> None:
    st.subheader("Modelos de prediccion")
    _render_state_diagnostics("modelos")

    accidents_df = st.session_state.get("accidents_df")
    features_df = st.session_state.get("flow_features_df")
    cluster_features_df = st.session_state.get("cluster_features_df")

    if accidents_df is None or accidents_df.empty:
        st.info("Cargue accidentes en la pestana Eventos.")
        return
    if features_df is None or features_df.empty:
        st.info("Calcule variables de flujo en la pestana Feature engineering.")
        return

    base_df = add_accident_target(features_df, accidents_df)
    if base_df.empty:
        st.warning("No se pudo preparar el dataset base.")
        return

    balanced_df = st.session_state.get("balanced_base_df")
    if balanced_df is None:
        st.warning("No hay dataset base balanceado en memoria.")
    
    cluster_choice = st.session_state.get("cluster_choice", "(sin clusters)")
    cluster_cols_in_features = (
        isinstance(features_df, pd.DataFrame)
        and not features_df.empty
        and bool(_get_cluster_cols(features_df))
    )
    balanced_cluster_only_df_chk = st.session_state.get("balanced_cluster_only_df")
    balanced_cluster_df_chk = st.session_state.get("balanced_cluster_df")
    cluster_cols_in_balanced = (
        balanced_cluster_only_df_chk is not None
        and bool(_get_cluster_cols(balanced_cluster_only_df_chk))
    ) or (
        balanced_cluster_df_chk is not None
        and bool(_get_cluster_cols(balanced_cluster_df_chk))
    )
    has_cluster_features = (
        isinstance(cluster_features_df, pd.DataFrame)
        and not cluster_features_df.empty
    ) or cluster_cols_in_features
    has_cluster_available = has_cluster_features or cluster_cols_in_balanced
    
    threshold_objective_options = {
        "Recall@N alertas/día": "recall_at_alerts_per_day",
        "FAR": "far",
        "Balanced F1": "balanced_f1",
        "F1": "f1",
        "MCC": "mcc",
        "Costo operacional": "operational_cost",
    }
    current_threshold_objective = _option_value_from_state(
        threshold_objective_options,
        "model_threshold_objective",
        default_label="Recall@N alertas/día",
    )
    model_threshold_visibility = _threshold_field_visibility_for_objective(
        current_threshold_objective
    )

    test_size = float(st.session_state.get("test_size", 0.2))
    st.caption(f"Test size: {test_size:.2f}")

    far_target = float(
        _render_conditional_slider(
            "FAR target",
            visible=model_threshold_visibility["far_target"],
            min_value=0.0,
            max_value=0.5,
            value=float(st.session_state.get("far_target", 0.2)),
            step=0.01,
            key="far_target",
            help=(
                "Falsa alarma máxima aceptada cuando el objetivo de threshold es FAR. "
                "Rango: 0.00 a 0.50. Default: 0.20. No es una probabilidad calibrada; "
                "controla tasa de falsos positivos en validación."
            ),
        )
    )
    val_size = st.slider(
        "Validation size",
        min_value=0.05,
        max_value=0.4,
        value=float(st.session_state.get("val_size", 0.2)),
        step=0.05,
        key="val_size",
        help=(
            "Fracción temporal de train+val reservada para seleccionar threshold. "
            "Rango: 0.05 a 0.40. Default: 0.20. Una validación demasiado pequeña "
            "vuelve inestable el threshold en eventos raros."
        ),
    )

    protocol_options = {
        "Conservador": "conservative",
        "Robusto": "robust",
    }
    selected_protocol_labels = st.multiselect(
        "Protocolos de evaluación",
        list(protocol_options.keys()),
        default=list(protocol_options.keys()),
        key="model_threshold_protocols",
        help=(
            "Permite entrenar/evaluar Conservador, Robusto o ambos. Conservador "
            "elige threshold en validación y evalúa ese mismo modelo en test. "
            "Robusto usa folds temporales OOF dentro de train+val. Elegir ambos "
            "duplica aproximadamente el tiempo de entrenamiento."
        ),
    )
    threshold_protocols = [
        protocol_options[label]
        for label in selected_protocol_labels
        if label in protocol_options
    ] or ["conservative"]

    col_thr_a, col_thr_b, col_thr_c = st.columns(3)
    with col_thr_a:
        threshold_objective_label = st.selectbox(
            "Objetivo de threshold",
            list(threshold_objective_options.keys()),
            index=0,
            key="model_threshold_objective",
            help=(
                "Métrica usada para escoger el umbral operativo. Valores: FAR, "
                "Balanced F1, F1, MCC, Recall@N alertas/día o Costo operacional. "
                "Default: Recall@N alertas/día. Usar solo F1 puede ignorar carga "
                "operacional de falsas alarmas."
            ),
        )
    with col_thr_b:
        alerts_per_day = float(
            _render_conditional_number_input(
                "Alertas máximas por día",
                visible=model_threshold_visibility["alerts_per_day"],
                min_value=0.1,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="model_alerts_per_day",
                help=(
                    "Presupuesto diario de alertas para Recall@N y métricas "
                    "operacionales. Rango práctico: 0.1 a 50. Default: 5.0. "
                    "Valores muy bajos pueden ocultar eventos detectables."
                ),
            )
        )
    with col_thr_c:
        calibration_options = _calibration_method_options()
        calibration_labels = [label for label, _ in calibration_options]
        calibration_map = {
            label: key for label, key in calibration_options
        }
        calibration_label = st.selectbox(
            "Calibración",
            calibration_labels,
            index=0,
            key="model_calibration_method",
            help=(
                "Transforma scores antes de elegir threshold. Valores: Platt "
                "scaling (sigmoid), Isotonic o Sin calibración. Default: "
                "Platt scaling (sigmoid). Isotonic puede sobreajustar con "
                "pocos positivos."
            ),
        )
        calibration_method = calibration_map[calibration_label]
    col_cost_a, col_cost_b, col_cost_c = st.columns(3)
    with col_cost_a:
        fn_cost = float(
            _render_conditional_number_input(
                "Costo FN",
                visible=model_threshold_visibility["fn_cost"],
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="model_fn_cost",
                help=(
                    "Costo de no alertar un accidente real. Default: 10.0. Subirlo "
                    "favorece recall, pero puede aumentar falsas alarmas."
                ),
            )
        )
    with col_cost_b:
        fp_cost = float(
            _render_conditional_number_input(
                "Costo FP",
                visible=model_threshold_visibility["fp_cost"],
                min_value=0.0,
                value=1.0,
                step=0.5,
                key="model_fp_cost",
                help=(
                    "Costo de una falsa alarma. Default: 1.0. Subirlo reduce alertas, "
                    "pero puede bajar recall."
                ),
            )
        )
    with col_cost_c:
        robust_folds = st.number_input(
            "Folds robustos",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="model_robust_folds",
            help=(
                "Número de folds temporales OOF para el protocolo robusto. "
                "Rango: 2 a 10. Default: 3. Más folds cuestan más y pueden fallar "
                "si cada ventana tiene pocos accidentes."
            ),
        )
    threshold_objective = threshold_objective_options.get(
        threshold_objective_label,
        "recall_at_alerts_per_day",
    )
    balance_strategy_options = {
        "Sin balance interno": "none",
        "Class weight / scale_pos_weight": "class_weight",
        "SMOTE interno": "smote",
    }
    balance_strategy_label = st.selectbox(
        "Estrategia de desbalance",
        list(balance_strategy_options.keys()),
        index=1,
        key="model_balance_strategy",
        help=(
            "Define cómo manejar el desbalance dentro de los folds de entrenamiento. "
            "Valores: sin balance, class_weight/scale_pos_weight o SMOTE interno. "
            "Default: class_weight/scale_pos_weight. SMOTE se aplica solo a train, "
            "nunca a validación ni test."
        ),
    )
    balance_strategy = balance_strategy_options.get(balance_strategy_label, "class_weight")
    use_balanced_current = bool(
        st.session_state.get(
            "use_balanced_base_toggle",
            bool(
                st.session_state.get("use_balanced_base", False)
                or st.session_state.get("use_balanced_cluster", False)
            ),
        )
    )

    model_choice = st.selectbox(
        "Modelo",
        ["Random Forest", "Balanced Random Forest", "XGBoost", "SVM", "Neural Network"],
        key="model_choice",
        help=(
            "Modelo a entrenar en la pestaña Modelos. Balanced Random Forest "
            "requiere imbalanced-learn y compara submuestreo balanceado frente a RF. "
            "Neural Network usa un MLP con PyTorch (StandardScaler + early stopping)."
        ),
    )
    optuna_balance_mode = _optuna_model_tab_balance_mode(
        balance_strategy=balance_strategy,
        use_balanced=use_balanced_current,
    )

    _apply_feature_source_for_model_tab(
        model_choice=model_choice,
        balance_mode=optuna_balance_mode,
        calibration_method=calibration_method,
        base_df=base_df,
        features_df=features_df,
    )
    _render_selected_features_info()

    random_state = st.number_input(
        "random_state", min_value=0, value=42, step=1, key="model_random_state"
    )
    model_n_jobs: Optional[int] = None
    if model_choice in {"Random Forest", "Balanced Random Forest"}:
        model_n_jobs = _render_model_n_jobs_input(
            "Jobs paralelos RF/ranking",
            key="model_rf_parallel_jobs",
            default=min(10, _max_optuna_parallel_jobs()),
        )
    elif model_choice == "XGBoost":
        model_n_jobs = _render_model_n_jobs_input(
            "Jobs paralelos XGBoost",
            key="model_xgb_parallel_jobs",
            default=1,
            shared_key="global_xgb_parallel_jobs",
        )
    elif model_choice == "Neural Network":
        st.caption(
            "Neural Network entrena en dispositivo (MPS/CUDA/CPU); "
            "no requiere n_jobs. Se paraleliza a través de Optuna."
        )
    elif model_choice == "SVM":
        st.caption(
            "SVM no expone `n_jobs`; en Comparación controlada se paraleliza "
            "a través de Optuna."
        )

    optuna_status = _apply_optuna_model_params_to_state(
        model_choice=model_choice,
        balance_mode=optuna_balance_mode,
        calibration_method=calibration_method,
        base_df=base_df,
        features_df=features_df,
    )
    if optuna_status:
        st.caption(optuna_status)

    param_tabs = st.tabs(
        ["Parámetros Base", "Parámetros Cluster", "Parámetros Base + Cluster"]
    )
    with param_tabs[0]:
        model_params_base = _render_model_params_ui(model_choice, "base_")
    with param_tabs[1]:
        model_params_cluster_only = _render_model_params_ui(
            model_choice,
            "cluster_only_",
        )
    with param_tabs[2]:
        model_params_cluster = _render_model_params_ui(model_choice, "cluster_")
    if model_n_jobs is not None:
        model_params_base["n_jobs"] = int(model_n_jobs)
        model_params_cluster_only["n_jobs"] = int(model_n_jobs)
        model_params_cluster["n_jobs"] = int(model_n_jobs)
        st.caption(
            f"`n_jobs` efectivo para {model_choice}: {int(model_n_jobs)} "
            "(se aplicará a Base, Cluster y Base + Cluster)."
        )

    st.markdown(
        "El selector permite entrenar con el dataset balanceado "
        "(con SMOTE y split train/test guardado) en vez del dataset original. "
        "Si no se selecciona, el entrenamiento usa el dataset original y hace el split "
        "con el test size indicado."
    )

    use_balanced = st.checkbox(
        "Usar dataset balanceado",
        value=bool(
            st.session_state.get("use_balanced_base", False)
            or st.session_state.get("use_balanced_cluster", False)
        ),
        key="use_balanced_base_toggle",
    )
    st.session_state["use_balanced_base"] = use_balanced
    st.session_state["use_balanced_cluster"] = use_balanced
    if use_balanced and balance_strategy == "smote":
        st.warning(
            "El dataset balanceado legacy ya contiene SMOTE en train. "
            "Para evitar doble SMOTE, esta corrida usará 'Sin balance interno' "
            "sobre ese split prebalanceado."
        )

    def _primary_protocol_result(
        results_by_protocol: Dict[str, Dict[str, object]]
    ) -> Dict[str, object]:
        if "robust" in results_by_protocol:
            return results_by_protocol["robust"]
        if "conservative" in results_by_protocol:
            return results_by_protocol["conservative"]
        return next(iter(results_by_protocol.values()))

    def _train_selected_protocols(
        *,
        use_split: bool,
        df: Optional[pd.DataFrame] = None,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        feature_cols: List[str],
        model_params: Dict[str, object],
    ) -> Dict[str, Dict[str, object]]:
        by_protocol: Dict[str, Dict[str, object]] = {}
        effective_balance_strategy = (
            "none" if (use_balanced and use_split) else balance_strategy
        )
        for protocol in threshold_protocols:
            common_kwargs = {
                "val_size": float(val_size),
                "far_target": float(far_target),
                "random_state": int(random_state),
                "threshold_protocol": protocol,
                "threshold_objective": threshold_objective,
                "calibration_method": str(calibration_method),
                "alerts_per_day": float(alerts_per_day),
                "fn_cost": float(fn_cost),
                "fp_cost": float(fp_cost),
                "robust_folds": int(robust_folds),
                "balance_strategy": effective_balance_strategy,
            }
            if use_split:
                if train_df is None or test_df is None:
                    raise ValueError("Split train/test requerido.")
                by_protocol[protocol] = _train_model_on_split(
                    train_df,
                    test_df,
                    feature_cols,
                    model_choice,
                    model_params,
                    **common_kwargs,
                )
            else:
                if df is None:
                    raise ValueError("Dataset requerido.")
                by_protocol[protocol] = _train_model(
                    df,
                    feature_cols,
                    model_choice,
                    model_params,
                    test_size=float(test_size),
                    **common_kwargs,
                )
        return by_protocol

    def _resolve_model_feature_group_cols(
        df: pd.DataFrame,
        *,
        feature_group: str,
        label: str,
    ) -> List[str]:
        override_key = _MODEL_FEATURE_GROUP_OVERRIDE_KEYS.get(feature_group)
        override = (
            st.session_state.get(override_key) if override_key else None
        )
        if override is not None:
            selected_features: Optional[List[str]] = list(override)
        else:
            selected_features = st.session_state.get("selected_features")
        feature_cols, missing = _resolve_feature_group_cols(
            df,
            selected_features,
            feature_group=feature_group,
        )
        if selected_features is not None and missing:
            st.warning(
                f"Variables seleccionadas no estan en el dataset para {label}: "
                + ", ".join(missing)
            )
        if selected_features is not None and not feature_cols:
            raise ValueError(
                f"Seleccione al menos una variable del grupo {label} en Feature selection."
            )
        if not feature_cols:
            raise ValueError(f"No hay variables numericas para {label}.")
        return list(feature_cols)

    def _train_feature_group_protocols(
        *,
        label: str,
        feature_group: str,
        model_params: Dict[str, object],
        use_split: bool,
        df: Optional[pd.DataFrame] = None,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object], List[str]]:
        feature_source_df = train_df if use_split else df
        if feature_source_df is None:
            raise ValueError(f"Dataset requerido para {label}.")
        feature_cols = _resolve_model_feature_group_cols(
            feature_source_df,
            feature_group=feature_group,
            label=label,
        )
        results_by_protocol = _train_selected_protocols(
            use_split=use_split,
            df=df,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            model_params=model_params,
        )
        return (
            results_by_protocol,
            _primary_protocol_result(results_by_protocol),
            feature_cols,
        )

    if st.button("Entrenar modelos"):
        use_balanced = bool(st.session_state.get("use_balanced_base", False))
        balanced_df = st.session_state.get("balanced_base_df")
        if balanced_df is None:
            balanced_df = st.session_state.get("balanced_cluster_only_df")
        if balanced_df is None:
            balanced_df = st.session_state.get("balanced_cluster_df")
        cluster_cols_in_balanced = bool(
            _get_cluster_cols(st.session_state.get("balanced_cluster_only_df") if st.session_state.get("balanced_cluster_only_df") is not None else pd.DataFrame())
        ) or bool(
            _get_cluster_cols(st.session_state.get("balanced_cluster_df") if st.session_state.get("balanced_cluster_df") is not None else pd.DataFrame())
        )
        has_cluster = has_cluster_features or cluster_cols_in_balanced
        base_feature_cols_used: List[str] = []
        cluster_only_feature_cols_used: Optional[List[str]] = None
        cluster_feature_cols_used: Optional[List[str]] = None
        total_steps = 2 + (2 if has_cluster else 0)
        progress = _StreamlitProgress(total=total_steps)
        with st.spinner("Entrenando modelos..."):
            progress.set_description("Preparando datasets")
            progress.update(1)

            base_result: Optional[Dict[str, object]] = None
            base_results_by_protocol: Dict[str, Dict[str, object]] = {}
            if use_balanced and balanced_df is None:
                st.warning("No hay dataset balanceado. Usando original.")

            if use_balanced and st.session_state.get("balanced_base_df") is not None:
                split = _split_balanced_dataset(st.session_state["balanced_base_df"])
                if split is None:
                    st.warning(
                        "El dataset base balanceado no tiene split train/test valido. "
                        "Usando dataset original."
                    )
                else:
                    train_df, test_df = split
                    st.caption(
                        f"Base (train/test): {len(train_df):,} / {len(test_df):,}"
                    )
                    try:
                        progress.set_description("Entrenando modelo Base")
                        (
                            base_results_by_protocol,
                            base_result,
                            base_feature_cols_used,
                        ) = _train_feature_group_protocols(
                            label="Base",
                            feature_group="base",
                            model_params=model_params_base,
                            use_split=True,
                            train_df=train_df,
                            test_df=test_df,
                        )
                    except Exception as exc:
                        progress.close()
                        st.error(f"No se pudo entrenar el modelo base: {exc}")
                        return

            if base_result is None:
                pos_count = int(base_df["target"].sum())
                st.caption(
                    f"Filas: {len(base_df):,} | Accidentes: {pos_count:,}"
                )
                if pos_count == 0:
                    progress.close()
                    st.warning("No se encontraron accidentes alineados con las variables.")
                    return
                try:
                    progress.set_description("Entrenando modelo Base")
                    (
                        base_results_by_protocol,
                        base_result,
                        base_feature_cols_used,
                    ) = _train_feature_group_protocols(
                        label="Base",
                        feature_group="base",
                        model_params=model_params_base,
                        use_split=False,
                        df=base_df,
                    )
                except Exception as exc:
                    progress.close()
                    st.error(f"No se pudo entrenar el modelo base: {exc}")
                    return

            progress.update(1)
            results = {
                ("Base", protocol): result["metrics"]
                for protocol, result in base_results_by_protocol.items()
            }

            cluster_only_result: Optional[Dict[str, object]] = None
            cluster_only_results_by_protocol: Dict[str, Dict[str, object]] = {}
            cluster_result: Optional[Dict[str, object]] = None
            cluster_results_by_protocol: Dict[str, Dict[str, object]] = {}
            if has_cluster:
                balanced_cluster_only_df = st.session_state.get(
                    "balanced_cluster_only_df"
                )
                balanced_cluster_df = st.session_state.get("balanced_cluster_df")

                if use_balanced and balanced_cluster_only_df is None and balanced_cluster_df is None:
                    st.warning(
                        "No hay dataset balanceado para Cluster/Base + Cluster "
                        "(quizas no se genero). Usando original."
                    )

                # Dataset balanceado para modelo Cluster-only (aligned con Optuna "Cluster")
                cluster_only_split: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
                if use_balanced and balanced_cluster_only_df is not None:
                    split_co = _split_balanced_dataset(balanced_cluster_only_df)
                    if split_co is None:
                        st.warning(
                            "El dataset balanceado Cluster no tiene split valido. "
                            "Usando dataset original para Cluster."
                        )
                    else:
                        train_co, test_co = split_co
                        if not _get_cluster_cols(train_co):
                            st.warning(
                                "El dataset balanceado Cluster no incluye variables de cluster. "
                                "Usando dataset original para Cluster."
                            )
                        else:
                            cluster_only_split = (train_co, test_co)

                # Dataset balanceado para modelo Base + Cluster (aligned con Optuna "Base + Cluster")
                cluster_split: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
                if use_balanced and balanced_cluster_df is not None:
                    split = _split_balanced_dataset(balanced_cluster_df)
                    if split is None:
                        st.warning(
                            "El dataset balanceado Base + Cluster no tiene split valido. "
                            "Usando dataset original."
                        )
                    else:
                        train_df, test_df = split
                        if not _get_cluster_cols(train_df):
                            st.warning(
                                "El dataset balanceado Base + Cluster no incluye variables de cluster. "
                                "Usando dataset original."
                            )
                        else:
                            cluster_split = (train_df, test_df)

                # Fallback: construir cluster_train_df desde el original si alguno falta
                cluster_train_df: Optional[pd.DataFrame] = None
                if cluster_only_split is None or cluster_split is None:
                    cluster_train_df = _build_cluster_dataset(
                        base_df,
                        cluster_features_df=cluster_features_df,
                    )

                # Entrenamiento modelo Cluster-only
                if cluster_only_split is not None:
                    train_co, test_co = cluster_only_split
                    st.caption(
                        f"Cluster (train/test): {len(train_co):,} / {len(test_co):,}"
                    )
                    try:
                        progress.set_description("Entrenando modelo Cluster")
                        (
                            cluster_only_results_by_protocol,
                            cluster_only_result,
                            cluster_only_feature_cols_used,
                        ) = _train_feature_group_protocols(
                            label="Cluster",
                            feature_group="cluster",
                            model_params=model_params_cluster_only,
                            use_split=True,
                            train_df=train_co,
                            test_df=test_co,
                        )
                    except Exception as exc:
                        progress.close()
                        st.error(f"No se pudo entrenar el modelo Cluster: {exc}")
                        return
                    progress.update(1)
                elif cluster_train_df is not None:
                    try:
                        progress.set_description("Entrenando modelo Cluster")
                        (
                            cluster_only_results_by_protocol,
                            cluster_only_result,
                            cluster_only_feature_cols_used,
                        ) = _train_feature_group_protocols(
                            label="Cluster",
                            feature_group="cluster",
                            model_params=model_params_cluster_only,
                            use_split=False,
                            df=cluster_train_df,
                        )
                    except Exception as exc:
                        progress.close()
                        st.error(f"No se pudo entrenar el modelo Cluster: {exc}")
                        return
                    progress.update(1)

                # Entrenamiento modelo Base + Cluster
                if cluster_split is not None:
                    train_df, test_df = cluster_split
                    st.caption(
                        f"Base + Cluster (train/test): {len(train_df):,} / {len(test_df):,}"
                    )
                    try:
                        progress.set_description("Entrenando modelo Base + Cluster")
                        (
                            cluster_results_by_protocol,
                            cluster_result,
                            cluster_feature_cols_used,
                        ) = _train_feature_group_protocols(
                            label="Base + Cluster",
                            feature_group="base_cluster",
                            model_params=model_params_cluster,
                            use_split=True,
                            train_df=train_df,
                            test_df=test_df,
                        )
                    except Exception as exc:
                        progress.close()
                        st.error(
                            f"No se pudo entrenar el modelo Base + Cluster: {exc}"
                        )
                        return
                    progress.update(1)
                elif cluster_train_df is not None:
                    try:
                        progress.set_description("Entrenando modelo Base + Cluster")
                        (
                            cluster_results_by_protocol,
                            cluster_result,
                            cluster_feature_cols_used,
                        ) = _train_feature_group_protocols(
                            label="Base + Cluster",
                            feature_group="base_cluster",
                            model_params=model_params_cluster,
                            use_split=False,
                            df=cluster_train_df,
                        )
                    except Exception as exc:
                        progress.close()
                        st.error(
                            f"No se pudo entrenar el modelo Base + Cluster: {exc}"
                        )
                        return
                    progress.update(1)

                if cluster_only_result is not None:
                    for protocol, result in cluster_only_results_by_protocol.items():
                        results[("Cluster", protocol)] = result["metrics"]
                if cluster_result is not None:
                    for protocol, result in cluster_results_by_protocol.items():
                        results[("Base + Cluster", protocol)] = result["metrics"]

            metrics_df = pd.DataFrame(results).T
            if isinstance(metrics_df.index, pd.MultiIndex):
                metrics_df.index = metrics_df.index.set_names(
                    ["feature_set", "threshold_protocol"]
            )
            st.subheader("Resultados")
            st.dataframe(metrics_df, width="stretch")
            if cluster_only_results_by_protocol or cluster_results_by_protocol:
                delta_rows: Dict[Tuple[str, str], pd.Series] = {}
                for protocol in threshold_protocols:
                    base_key = ("Base", protocol)
                    if base_key not in metrics_df.index:
                        continue
                    base_numeric = pd.to_numeric(
                        metrics_df.loc[base_key],
                        errors="coerce",
                    )
                    for feature_label in ["Cluster", "Base + Cluster"]:
                        compare_key = (feature_label, protocol)
                        if compare_key not in metrics_df.index:
                            continue
                        compare_numeric = pd.to_numeric(
                            metrics_df.loc[compare_key],
                            errors="coerce",
                        )
                        delta_rows[(feature_label, protocol)] = (
                            compare_numeric - base_numeric
                        )
                delta_df = pd.DataFrame(delta_rows).T
                if isinstance(delta_df.index, pd.MultiIndex):
                    delta_df.index = delta_df.index.set_names(
                        ["feature_set", "threshold_protocol"]
                    )
                st.subheader("Delta vs Base")
                st.dataframe(delta_df, width="stretch")
            st.subheader("Matriz de confusion")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.caption("Base (protocolo principal)")
                base_cm = base_result.get("confusion_matrix")
                if base_cm is not None:
                    base_cm_df = pd.DataFrame(
                        base_cm,
                        index=["Actual 0", "Actual 1"],
                        columns=["Pred 0", "Pred 1"],
                    )
                    st.dataframe(base_cm_df, width="stretch")
            with col_b:
                if cluster_only_result is not None:
                    st.caption("Cluster (protocolo principal)")
                    cluster_only_cm = cluster_only_result.get("confusion_matrix")
                    if cluster_only_cm is not None:
                        cluster_only_cm_df = pd.DataFrame(
                            cluster_only_cm,
                            index=["Actual 0", "Actual 1"],
                            columns=["Pred 0", "Pred 1"],
                        )
                        st.dataframe(cluster_only_cm_df, width="stretch")
            with col_c:
                if cluster_result is not None:
                    st.caption("Base + Cluster (protocolo principal)")
                    cluster_cm = cluster_result.get("confusion_matrix")
                    if cluster_cm is not None:
                        cluster_cm_df = pd.DataFrame(
                            cluster_cm,
                            index=["Actual 0", "Actual 1"],
                            columns=["Pred 0", "Pred 1"],
                        )
                        st.dataframe(cluster_cm_df, width="stretch")
            if len(threshold_protocols) > 1:
                with st.expander("Matrices por protocolo", expanded=False):
                    for label, protocol_results in [
                        ("Base", base_results_by_protocol),
                        ("Cluster", cluster_only_results_by_protocol),
                        ("Base + Cluster", cluster_results_by_protocol),
                    ]:
                        for protocol, result in protocol_results.items():
                            st.caption(f"{label} | {protocol}")
                            matrix = result.get("confusion_matrix")
                            if matrix is not None:
                                st.dataframe(
                                    pd.DataFrame(
                                        matrix,
                                        index=["Actual 0", "Actual 1"],
                                        columns=["Pred 0", "Pred 1"],
                                    ),
                                    width="stretch",
                                )
            history_entry: Optional[Dict[str, object]] = None
            try:
                history_entry = _record_experiment_history(
                    base_df=base_df,
                    features_df=features_df,
                    balanced_df=balanced_df,
                    base_feature_cols=base_feature_cols_used,
                    base_result=base_result,
                    cluster_only_feature_cols=cluster_only_feature_cols_used,
                    cluster_only_result=cluster_only_result,
                    cluster_feature_cols=cluster_feature_cols_used,
                    cluster_result=cluster_result,
                    model_choice=model_choice,
                    model_params_base=model_params_base,
                    model_params_cluster_only=model_params_cluster_only,
                    model_params_cluster=model_params_cluster,
                    random_state=int(random_state),
                    test_size=float(test_size),
                    val_size=float(val_size),
                    far_target=float(far_target),
                    use_balanced=bool(use_balanced),
                    protocol_results={
                        "Base": base_results_by_protocol,
                        "Cluster": cluster_only_results_by_protocol,
                        "Base + Cluster": cluster_results_by_protocol,
                    },
                    threshold_protocols=list(threshold_protocols),
                    threshold_objective=threshold_objective,
                    calibration_method=str(calibration_method),
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                    robust_folds=int(robust_folds),
                    balance_strategy=balance_strategy,
                )
                st.caption("Historial actualizado y modelos guardados.")
            except Exception as exc:
                st.warning(f"No se pudo guardar en History: {exc}")
            if history_entry is not None and cluster_result is not None:
                _render_base_cluster_xai_block(
                    history_entry,
                    key_prefix=f"model_run_{history_entry.get('run_id', 'latest')}",
                    default_visible=True,
                )
            progress.close()


def _render_history_tab() -> None:
    st.subheader("History")
    entries = _load_history_entries()
    if not entries:
        st.info("No hay historial disponible.")
        return
    entries_sorted = sorted(
        entries, key=lambda item: str(item.get("timestamp", ""))
    )
    def _feature_file_label(entry: Dict[str, object]) -> str:
        features = entry.get("features", {})
        if not isinstance(features, dict):
            return "(sin archivo)"
        features_path = features.get("features_path")
        if features_path:
            try:
                return Path(str(features_path)).name
            except Exception:
                return str(features_path)
        features_source = features.get("features_source")
        if features_source:
            return f"(sin archivo) {features_source}"
        return "(sin archivo)"

    feature_labels = sorted({_feature_file_label(entry) for entry in entries_sorted})
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_choice = st.selectbox(
            "Filtrar por archivo de features",
            options=["Todos"] + feature_labels,
            key="history_features_filter",
        )
    
    # --- TRAMO FILTER LOGIC ---
    def _get_tramo_label(entry: Dict[str, object]) -> str:
        tramo = entry.get("dataset", {}).get("tramo", {})
        if not isinstance(tramo, dict):
            return "Toda la autopista"
        label = tramo.get("label")
        if label:
            return str(label)
        # Fallback if label is missing but parts exist
        eje = tramo.get("eje")
        calzada = tramo.get("calzada")
        p_start = tramo.get("portico_inicio")
        p_end = tramo.get("portico_fin")
        if eje and calzada and p_start and p_end:
             return f"{eje} | {calzada} | {p_start} -> {p_end}"
        return "Toda la autopista"

    tramo_labels = sorted({_get_tramo_label(entry) for entry in entries_sorted})
    with col_f2:
        tramo_choice = st.selectbox(
            "Filtrar por tramo",
            options=["Todos"] + tramo_labels,
            key="history_tramo_filter",
        )
    # --------------------------

    if filter_choice != "Todos":
        entries_sorted = [
            entry
            for entry in entries_sorted
            if _feature_file_label(entry) == filter_choice
        ]
    
    if tramo_choice != "Todos":
        entries_sorted = [
            entry
            for entry in entries_sorted
            if _get_tramo_label(entry) == tramo_choice
        ]

    if not entries_sorted:
        st.info("No hay historial para el filtro seleccionado.")
        return
    st.caption(f"Entradas: {len(entries_sorted)}")
    for idx, entry in enumerate(entries_sorted, start=1):
        timestamp = entry.get("timestamp", "-")
        models = entry.get("models", {})
        model_name = "-"
        if isinstance(models, dict):
            base = models.get("Base")
            if isinstance(base, dict):
                model_name = base.get("model_name", "-")
        title = f"{idx}. {timestamp} | {model_name}"
        with st.expander(title, expanded=False):
            st.caption(f"run_id: {entry.get('run_id', '-')}")
            run_id = entry.get("run_id")
            if st.button(
                "Eliminar registro",
                key=f"history_delete_acc_{run_id or idx}",
            ):
                if _delete_history_entry(run_id):
                    st.success("Registro eliminado.")
                    st.rerun()
                else:
                    st.warning("No se pudo eliminar el registro.")

            dataset = entry.get("dataset", {})
            st.markdown("**Dataset**")
            if dataset:
                st.json(dataset)

            training = entry.get("training", {})
            st.markdown("**Entrenamiento**")
            if training:
                st.json(training)

            features = entry.get("features", {})
            st.markdown("**Features calculadas**")
            if features:
                st.json(features)

            feature_sel = entry.get("feature_selection", {})
            st.markdown("**Feature selection**")
            if feature_sel:
                selected = feature_sel.get("selected_features", [])
                st.caption(
                    f"Seleccionadas: {len(selected) if isinstance(selected, list) else 0}"
                )
                if isinstance(selected, list) and selected:
                    st.dataframe(
                        pd.DataFrame({"variable": selected}),
                        width="stretch",
                    )
                importance_top = feature_sel.get("importance_top", [])
                if isinstance(importance_top, list) and importance_top:
                    st.caption("Importancia (top 25)")
                    st.dataframe(
                        pd.DataFrame(importance_top), width="stretch"
                    )
                importance_csv = feature_sel.get("importance_csv")
                if importance_csv:
                    st.caption(f"CSV importancia: {importance_csv}")

            optuna = entry.get("optuna", {})
            st.markdown("**Optuna**")
            if optuna:
                if isinstance(optuna, dict) and (
                    "base" in optuna or "base_cluster" in optuna
                ):
                    base_optuna = optuna.get("base")
                    if isinstance(base_optuna, dict) and base_optuna:
                        st.caption("Base")
                        st.json(base_optuna)
                    cluster_optuna = optuna.get("base_cluster")
                    if isinstance(cluster_optuna, dict) and cluster_optuna:
                        st.caption("Base + Cluster")
                        st.json(cluster_optuna)
                else:
                    st.json(optuna)

            balance = entry.get("balance", {})
            st.markdown("**Balance**")
            if balance:
                if isinstance(balance, dict) and (
                    "base" in balance or "base_cluster" in balance
                ):
                    for label, key in (
                        ("Base", "base"),
                        ("Base + Cluster", "base_cluster"),
                    ):
                        item = balance.get(key)
                        if not isinstance(item, dict):
                            continue
                        params = item.get("params")
                        stats = item.get("stats", {})
                        if params or stats:
                            st.caption(label)
                        if params:
                            st.json(params)
                        if isinstance(stats, dict):
                            for split_label, records in stats.items():
                                st.caption(f"{label} | Distribucion: {split_label}")
                                if isinstance(records, list) and records:
                                    st.dataframe(
                                        pd.DataFrame(records), width="stretch"
                                    )
                else:
                    balance_params = balance.get("params")
                    if balance_params:
                        st.json(balance_params)
                    stats = balance.get("stats", {})
                    if isinstance(stats, dict):
                        for label, records in stats.items():
                            st.caption(f"Distribucion: {label}")
                            if isinstance(records, list) and records:
                                st.dataframe(
                                    pd.DataFrame(records), width="stretch"
                                )

            if isinstance(models, dict) and models:
                st.markdown("**Modelos y resultados**")
                metrics_table = {}
                for name, model_entry in models.items():
                    if isinstance(model_entry, dict):
                        metrics_table[name] = model_entry.get("metrics", {})
                if metrics_table:
                    st.dataframe(
                        pd.DataFrame(metrics_table).T, width="stretch"
                    )
                for name, model_entry in models.items():
                    if not isinstance(model_entry, dict):
                        continue
                    st.caption(f"{name} | modelo: {model_entry.get('model_name')}")
                    model_path = model_entry.get("model_path")
                    if model_path:
                        st.caption(f"Archivo modelo: {model_path}")
                    model_params = model_entry.get("model_params")
                    if model_params:
                        st.json(model_params)
                    split_info = model_entry.get("split_info")
                    if split_info:
                        st.json(split_info)
                    feature_cols = model_entry.get("feature_cols")
                    if isinstance(feature_cols, list) and feature_cols:
                        st.caption(
                            f"Variables usadas: {len(feature_cols)}"
                        )
                        st.dataframe(
                            pd.DataFrame({"variable": feature_cols}),
                            width="stretch",
                        )
                    cm = model_entry.get("confusion_matrix")
                    if isinstance(cm, list) and cm:
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual 0", "Actual 1"],
                            columns=["Pred 0", "Pred 1"],
                        )
                        st.caption("Matriz de confusion")
                        st.dataframe(cm_df, width="stretch")
            _render_base_cluster_xai_block(
                entry,
                key_prefix=f"history_xai_{run_id or idx}",
                default_visible=False,
            )



def _segment_columns_from_features(
    df: pd.DataFrame,
) -> Optional[Tuple[str, str]]:
    candidates = [
        ("portico_last", "portico_next"),
        ("portico_inicio", "portico_fin"),
    ]
    for last_col, next_col in candidates:
        if last_col in df.columns and next_col in df.columns:
            return last_col, next_col
    return None


def _best_accident_window(
    times: np.ndarray,
    *,
    window: pd.Timedelta,
    min_time: pd.Timestamp,
    max_time: pd.Timestamp,
) -> Optional[Dict[str, object]]:
    if times is None or len(times) == 0:
        return None
    times = pd.to_datetime(times, errors="coerce")
    times = times[(times >= min_time) & (times <= max_time)]
    if len(times) == 0:
        return None
    times = np.sort(times)
    max_start = max_time - window
    if max_start < min_time:
        return None

    right = 0
    best_count = 0
    best_start: Optional[pd.Timestamp] = None
    for left in range(len(times)):
        start = times[left]
        if start < min_time:
            continue
        if start > max_start:
            break
        while right < len(times) and times[right] <= start + window:
            right += 1
        count = right - left
        if count > best_count:
            best_count = count
            best_start = pd.Timestamp(start)

    if best_start is None:
        return None
    return {
        "window_start": best_start,
        "window_end": best_start + window,
        "accidents_window": int(best_count),
    }


def _render_find_samples_sizes_experiment() -> None:
    st.subheader("Find samples sizes")
    st.caption(
        "Busca el tramo y ventana temporal con mayor densidad de accidentes, "
        "usa ese periodo como dataset total para train/val/test."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No hay archivos de eventos (accidents) en Datos.")
        return
    event_names = [p.name for p in event_files]
    selected_event = st.selectbox(
        "Archivo de Eventos", event_names, key="exp_samples_event_file"
    )

    feature_files = _list_flow_feature_files()
    if not feature_files:
        st.warning("No hay archivos de features en Resultados.")
        return
    feature_names = [p.name for p in feature_files]
    selected_features = st.selectbox(
        "Archivo de Features (Flow + Cluster)",
        feature_names,
        key="exp_samples_feature_file",
    )

    selected_features_path = next(
        (p for p in feature_files if p.name == selected_features), None
    )
    max_window_default = 180
    if selected_features_path:
        max_window_days = _get_feature_max_window_days(selected_features_path)
        if max_window_days is not None:
            max_window_default = max_window_days
    if st.session_state.get("exp_samples_feature_file_prev") != selected_features:
        st.session_state["exp_samples_window_max"] = int(max_window_default)
        st.session_state["exp_samples_feature_file_prev"] = selected_features

    st.markdown("**Busqueda de ventana temporal**")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        min_window_days = st.number_input(
            "Ventana min (dias)",
            min_value=1,
            value=180,
            step=1,
            key="exp_samples_window_min",
        )
    with col_w2:
        if "exp_samples_window_max" in st.session_state:
            max_window_days = st.number_input(
                "Ventana max (dias)",
                min_value=1,
                step=1,
                key="exp_samples_window_max",
            )
        else:
            max_window_days = st.number_input(
                "Ventana max (dias)",
                min_value=1,
                value=int(max_window_default),
                step=1,
                key="exp_samples_window_max",
            )
    with col_w3:
        step_window_days = st.number_input(
            "Paso (dias)",
            min_value=1,
            value=30,
            step=1,
            key="exp_samples_window_step",
        )

    st.markdown("**Segmente de Autopista**")
    selected_event_path = next(
        (p for p in event_files if p.name == selected_event), None
    )
    accidents_df_for_tramo = None
    if selected_event_path:
        accidents_df_for_tramo = _load_accidents_for_event(selected_event_path)

    # Pre-read allowed porticos if possible or just pass None
    allowed_porticos = None
    if selected_features_path:
        allowed_porticos = _load_porticos_from_feature_file(selected_features_path)

    tramo_tuple = _build_tramo_selector(
        accidents_df_for_tramo,
        date_start=None,
        date_end=None,
        allowed_porticos=allowed_porticos,
        key="exp_samples_tramo_choice",
    )
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        # Disable max_segments if a specific tramo is selected
        tramo_selected = tramo_tuple is not None
        max_segments = st.number_input(
            "Max segmentos a evaluar (0 = todos)",
            min_value=0,
            value=25,
            step=1,
            key="exp_samples_max_segments",
            disabled=tramo_selected,
            help="Deshabilitado si se selecciona un tramo especifico."
        )
    with col_s2:
        min_accidents_window = st.number_input(
            "Min accidentes por ventana",
            min_value=1,
            value=1,
            step=1,
            key="exp_samples_min_accidents_window",
        )
    with col_s3:
        top_show = st.number_input(
            "Top resultados a mostrar",
            min_value=1,
            value=10,
            step=1,
            key="exp_samples_top_show",
        )

    metric_choice = st.selectbox(
        "Criterio de seleccion",
        ["Accidentes por dia", "Accidentes totales"],
        key="exp_samples_metric_choice",
    )
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        eval_top_n = st.number_input(
            "Evaluar top candidatos (entrenar modelos)",
            min_value=1,
            value=1,
            step=1,
            key="exp_samples_eval_top_n",
        )
    with col_eval2:
        objective_options = _optuna_objective_options(
            [
                "f1",
                "roc_auc",
                "accuracy",
                "recall",
                "precision",
                "fnr",
                "far_sens",
                "mcc",
                "brier_score",
            ]
        )
        objective_label = st.selectbox(
            "Metrica objetivo (mejor mix)",
            list(objective_options.keys()),
            key="exp_samples_objective_metric",
        )
        objective_cfg = objective_options.get(
            objective_label, {"key": "f1", "direction": "maximize"}
        )
        objective_key = objective_cfg["key"]
        objective_direction = objective_cfg["direction"]
    
    use_cluster_features = st.checkbox(
        "Incluir variables de cluster (si existen)",
        value=True,
        key="exp_samples_use_cluster",
    )
    
    st.markdown("**Feature selection (Iteracion K)**")
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        min_k_val = st.number_input("Min K", 1, 1000, 1, key="exp_samp_min_k")
    with col_k2:
        max_k_val = st.number_input("Max K", 1, 1000, 8, key="exp_samp_max_k")
    with col_k3:
        step_k_val = st.number_input("Step K", 1, 100, 5, key="exp_samp_step_k")

    st.markdown("**Configuracion del modelo**")
    model_choice = st.selectbox(
        "Modelo para Experimento",
        ["Random Forest", "XGBoost", "SVM"],
        index=1,
        key="exp_samples_model_choice",
    )
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        n_trials = st.number_input(
            "Optuna Trials por paso",
            min_value=5,
            value=30,
            step=5,
            key="exp_samples_n_trials",
        )
    with col_n2:
        timeout = st.number_input(
            "Optuna Timeout (seg) por paso",
            min_value=10,
            value=3600,
            step=10,
            key="exp_samples_timeout",
        )
    optuna_n_jobs = _render_optuna_n_jobs_input(
        "Optuna jobs paralelos",
        key="exp_samples_optuna_n_jobs",
        default=1,
    )

    far_target = 0.2
    threshold_strategy = "optuna"
    threshold_strategy_label = "Optimizar threshold"
    with st.expander("Configuracion avanzada (parametros y rangos)"):
        st.markdown("**Split de datos (sobre ventana)**")
        c_split1, c_split2 = st.columns(2)
        with c_split1:
            val_size = st.slider(
                "Validation Size (relativo)",
                0.1,
                0.9,
                0.2,
                0.05,
                key="exp_samples_val_size",
            )
        with c_split2:
            test_size = st.slider(
                "Test Size (relativo)",
                0.1,
                0.9,
                0.2,
                0.05,
                key="exp_samples_test_size",
            )
        st.markdown("**Calibracion de umbral**")
        threshold_options = {
            "Optimizar threshold": "optuna",
            "Calibrar por FAR": "far",
        }
        threshold_strategy = _option_value_from_state(
            threshold_options,
            "exp_samples_threshold_strategy",
            default_label="Calibrar por FAR",
        )
        threshold_visibility = _threshold_field_visibility_for_strategy(
            threshold_strategy
        )
        far_target = float(
            _render_conditional_slider(
                "FAR target",
                visible=threshold_visibility["far_target"],
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="exp_samples_far_target",
            )
        )
        threshold_labels = list(threshold_options.keys())
        threshold_strategy_label = st.selectbox(
            "Estrategia de umbral",
            threshold_labels,
            index=threshold_labels.index("Calibrar por FAR"),
            key="exp_samples_threshold_strategy",
        )
        threshold_strategy = threshold_options[threshold_strategy_label]
        calibration_methods = _calibration_method_multiselect(
            "Calibración",
            key="exp_samples_calibration_methods",
            default_methods=["sigmoid", "isotonic"],
        )

        st.markdown("**Rango SMOTE**")
        c_smote1, c_smote2 = st.columns(2)
        with c_smote1:
            smote_k_min = st.number_input(
                "K Neighbors Min",
                1,
                20,
                1,
                key="exp_samples_smote_k_min",
            )
            smote_k_max = st.number_input(
                "K Neighbors Max",
                1,
                20,
                10,
                key="exp_samples_smote_k_max",
            )
        with c_smote2:
            smote_str_min = st.slider(
                "Sampling Strategy Min",
                0.1,
                1.0,
                0.1,
                0.1,
                key="exp_samples_smote_str_min",
            )
            smote_str_max = st.slider(
                "Sampling Strategy Max",
                0.1,
                1.0,
                1.0,
                0.1,
                key="exp_samples_smote_str_max",
            )

        st.markdown(f"**Rangos para {model_choice}**")
        model_ranges = {}
        if model_choice == "Random Forest":
            c_rf1, c_rf2 = st.columns(2)
            with c_rf1:
                rf_ne_min = st.number_input(
                    "N Estimators Min",
                    10,
                    1000,
                    50,
                    step=10,
                    key="exp_samples_rf_ne_min",
                )
                rf_ne_max = st.number_input(
                    "N Estimators Max",
                    10,
                    1000,
                    300,
                    step=10,
                    key="exp_samples_rf_ne_max",
                )
            with c_rf2:
                rf_md_min = st.number_input(
                    "Max Depth Min",
                    1,
                    50,
                    3,
                    key="exp_samples_rf_md_min",
                )
                rf_md_max = st.number_input(
                    "Max Depth Max",
                    1,
                    50,
                    15,
                    key="exp_samples_rf_md_max",
                )
            model_ranges = {
                "n_estimators": {"min": rf_ne_min, "max": rf_ne_max},
                "max_depth": {"min": rf_md_min, "max": rf_md_max},
            }
        elif model_choice == "XGBoost":
            c_xgb1, c_xgb2 = st.columns(2)
            with c_xgb1:
                xgb_ne_min = st.number_input(
                    "N Estimators Min",
                    10,
                    1000,
                    50,
                    step=10,
                    key="exp_samples_xgb_ne_min",
                )
                xgb_ne_max = st.number_input(
                    "N Estimators Max",
                    10,
                    1000,
                    300,
                    step=10,
                    key="exp_samples_xgb_ne_max",
                )
                xgb_lr_min = st.number_input(
                    "Learning Rate Min",
                    0.001,
                    1.0,
                    0.01,
                    format="%.3f",
                    key="exp_samples_xgb_lr_min",
                )
                xgb_lr_max = st.number_input(
                    "Learning Rate Max",
                    0.001,
                    1.0,
                    0.3,
                    format="%.3f",
                    key="exp_samples_xgb_lr_max",
                )
            with c_xgb2:
                xgb_md_min = st.number_input(
                    "Max Depth Min",
                    1,
                    50,
                    3,
                    key="exp_samples_xgb_md_min",
                )
                xgb_md_max = st.number_input(
                    "Max Depth Max",
                    1,
                    50,
                    15,
                    key="exp_samples_xgb_md_max",
                )
                xgb_sub_min = st.slider(
                    "Subsample Min",
                    0.1,
                    1.0,
                    0.5,
                    0.1,
                    key="exp_samples_xgb_sub_min",
                )
                xgb_sub_max = st.slider(
                    "Subsample Max",
                    0.1,
                    1.0,
                    1.0,
                    0.1,
                    key="exp_samples_xgb_sub_max",
                )
                xgb_col_min = st.slider(
                    "Colsample ByTree Min",
                    0.1,
                    1.0,
                    0.5,
                    0.1,
                    key="exp_samples_xgb_col_min",
                )
                xgb_col_max = st.slider(
                    "Colsample ByTree Max",
                    0.1,
                    1.0,
                    1.0,
                    0.1,
                    key="exp_samples_xgb_col_max",
                )
            model_ranges = {
                "n_estimators": {"min": xgb_ne_min, "max": xgb_ne_max},
                "max_depth": {"min": xgb_md_min, "max": xgb_md_max},
                "learning_rate": {"min": xgb_lr_min, "max": xgb_lr_max},
                "subsample": {"min": xgb_sub_min, "max": xgb_sub_max},
                "colsample_bytree": {"min": xgb_col_min, "max": xgb_col_max},
            }
        elif model_choice == "SVM":
            c_svm1, c_svm2 = st.columns(2)
            with c_svm1:
                svm_c_min = st.number_input(
                    "C Min",
                    0.01,
                    1000.0,
                    0.1,
                    format="%.2f",
                    key="exp_samples_svm_c_min",
                )
            with c_svm2:
                svm_c_max = st.number_input(
                    "C Max",
                    0.01,
                    1000.0,
                    50.0,
                    format="%.2f",
                    key="exp_samples_svm_c_max",
                )
            model_ranges = {"C": {"min": svm_c_min, "max": svm_c_max}}

    if st.button("Buscar ventana y entrenar", key="exp_samples_run"):
        if min_window_days > max_window_days:
            st.error("La ventana minima no puede ser mayor que la maxima.")
            return
        if not calibration_methods:
            st.error("Seleccione al menos un calibrador.")
            return
        if int(min_k_val) > int(max_k_val):
            st.error("Min K no puede ser mayor que Max K.")
            return
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_db_path = _init_experiment_db(
            "Find samples sizes",
            {
                "run_id": run_id,
                "dataset_name": selected_event,
                "features_name": selected_features,
                "model_choice": model_choice,
                "objective_label": objective_label,
                "objective_metric": objective_key,
                "objective_direction": objective_direction,
                "far_target": float(far_target),
                "threshold_strategy": threshold_strategy,
                "threshold_strategy_label": threshold_strategy_label,
                "calibration_methods": list(calibration_methods),
                "val_size": float(val_size),
                "test_size": float(test_size),
                "window_min_days": int(min_window_days),
                "window_max_days": int(max_window_days),
                "window_step_days": int(step_window_days),
                "max_segments": 1 if tramo_tuple else int(max_segments),
                "eval_top_n": int(eval_top_n),
                "use_cluster_features": bool(use_cluster_features),
                "min_k": int(min_k_val),
                "max_k": int(max_k_val),
                "step_k": int(step_k_val),
            },
        )
        if exp_db_path:
            st.caption(f"DB live: {exp_db_path}")

        accidents_path = next(p for p in event_files if p.name == selected_event)
        features_path = next(p for p in feature_files if p.name == selected_features)

        try:
            raw_accidents_df = read_csv_with_progress(str(accidents_path))
        except Exception as exc:
            st.error(f"Error cargando accidentes: {exc}")
            return

        try:
            porticos_df = load_porticos()
            if porticos_df is None or porticos_df.empty:
                st.error("No se pudieron cargar los porticos (Porticos.csv).")
                return
        except Exception as exc:
            st.error(f"Error cargando porticos: {exc}")
            return

        try:
            accidents_df, excluded = process_accidentes_df(
                raw_accidents_df, porticos_df, return_excluded=True
            )
            if accidents_df.empty:
                st.warning(
                    "No quedaron accidentes validos tras el procesamiento."
                )
                return
            st.success(
                f"Accidentes procesados: {len(accidents_df)} (Excluidos: {len(excluded)})"
            )
        except Exception as exc:
            st.error(f"Error procesando accidentes: {exc}")
            return

        if str(features_path).endswith(".duckdb"):
            if duckdb:
                con = duckdb.connect(str(features_path), read_only=True)
                tables = con.execute("SHOW TABLES").fetchall()
                if tables:
                    table_name = tables[0][0]
                    features_df = con.execute(
                        f"SELECT * FROM {table_name}"
                    ).df()
                else:
                    st.error("Empty DuckDB")
                    con.close()
                    return
                con.close()
            else:
                st.error("DuckDB not installed")
                return
        else:
            features_df = read_csv_with_progress(str(features_path))

        if features_df is None or features_df.empty:
            st.error("El archivo de features esta vacio.")
            return
        if "interval_start" not in features_df.columns:
            st.error("Las variables no tienen interval_start.")
            return

        segment_cols = _segment_columns_from_features(features_df)
        if not segment_cols:
            st.error(
                "El archivo de features no contiene columnas de tramo "
                "(portico_last/portico_next o portico_inicio/portico_fin)."
            )
            return
        seg_last_col, seg_next_col = segment_cols

        features_df = features_df.copy()
        if (seg_last_col, seg_next_col) != ("portico_last", "portico_next"):
            features_df = features_df.rename(
                columns={seg_last_col: "portico_last", seg_next_col: "portico_next"}
            )
        features_df["interval_start"] = pd.to_datetime(
            features_df["interval_start"], errors="coerce"
        )
        features_df["portico_last"] = _normalize_portico_series(
            features_df["portico_last"]
        )
        features_df["portico_next"] = _normalize_portico_series(
            features_df["portico_next"]
        )
        features_df = features_df.dropna(
            subset=["interval_start", "portico_last", "portico_next"]
        )
        if features_df.empty:
            st.error("No hay datos de features para los tramos.")
            return
        available_feature_cols = _get_feature_cols(features_df)
        if not use_cluster_features:
            cluster_cols = set(_get_cluster_cols(features_df))
            available_feature_cols = [
                c for c in available_feature_cols if c not in cluster_cols
            ]
        if not available_feature_cols:
            st.error("No hay variables numericas disponibles para entrenar.")
            return
        if int(min_k_val) > len(available_feature_cols):
            st.error(
                "Min K es mayor que la cantidad de variables disponibles "
                f"({len(available_feature_cols)})."
            )
            return

        acc_seg = accidents_df.copy()
        acc_seg["accidente_time"] = pd.to_datetime(
            acc_seg["accidente_time"], errors="coerce"
        )
        acc_seg["portico_last"] = _normalize_portico_series(
            acc_seg["ultimo_portico"]
        )
        acc_seg["portico_next"] = _normalize_portico_series(
            acc_seg["proximo_portico"]
        )
        acc_seg = acc_seg.dropna(
            subset=["accidente_time", "portico_last", "portico_next"]
        )
        if acc_seg.empty:
            st.warning("No hay accidentes con tramo asignado.")
            return

        segments_df = features_df[
            ["portico_last", "portico_next"]
        ].drop_duplicates()
        ranges_df = (
            features_df.groupby(["portico_last", "portico_next"])[
                "interval_start"
            ]
            .agg(feature_min="min", feature_max="max")
            .reset_index()
        )
        counts_df = (
            acc_seg.groupby(["portico_last", "portico_next"])
            .size()
            .reset_index(name="accidents_total")
        )
        segments_df = segments_df.merge(
            ranges_df, on=["portico_last", "portico_next"], how="left"
        ).merge(
            counts_df, on=["portico_last", "portico_next"], how="left"
        )
        # Apply specific Tramo filtering
        if tramo_tuple:
             eje, calzada, p_start, p_end = tramo_tuple
             p_start_n = _normalize_portico_code(p_start)
             p_end_n = _normalize_portico_code(p_end)
             segments_df = segments_df[
                 (segments_df["portico_last"] == p_start_n) &
                 (segments_df["portico_next"] == p_end_n)
             ]
             if segments_df.empty:
                 st.error(f"El tramo seleccionado {p_start}->{p_end} no existe en las features.")
                 return

        segments_df["accidents_total"] = (
            segments_df["accidents_total"].fillna(0).astype(int)
        )
        segments_df = segments_df[segments_df["accidents_total"] > 0]
        if segments_df.empty:
            st.warning("No hay segmentos con accidentes.")
            return

        try:
            seg_meta = get_portico_segments(porticos_df)
            if seg_meta is not None and not seg_meta.empty:
                seg_meta = seg_meta.copy()
                seg_meta["portico_last"] = _normalize_portico_series(
                    seg_meta["portico_last"]
                )
                seg_meta["portico_next"] = _normalize_portico_series(
                    seg_meta["portico_next"]
                )
                segments_df = segments_df.merge(
                    seg_meta[["eje", "calzada", "portico_last", "portico_next"]],
                    on=["portico_last", "portico_next"],
                    how="left",
                )
        except Exception:
            pass

        if not tramo_tuple and max_segments and max_segments > 0:
            segments_df = segments_df.sort_values(
                "accidents_total", ascending=False
            ).head(int(max_segments))
        segments_df = segments_df.reset_index(drop=True)

        window_days = list(
            range(
                int(min_window_days),
                int(max_window_days) + 1,
                int(step_window_days),
            )
        )
        if not window_days:
            window_days = [int(min_window_days)]

        acc_times = {
            key: np.sort(group.to_numpy())
            for key, group in acc_seg.groupby(
                ["portico_last", "portico_next"]
            )["accidente_time"]
        }

        candidates: List[Dict[str, object]] = []
        progress_bar = st.progress(0, text="Buscando ventanas...")
        total_segments = len(segments_df)
        for idx, row in enumerate(segments_df.itertuples(index=False), start=1):
            key = (row.portico_last, row.portico_next)
            times = acc_times.get(key)
            if times is None or len(times) == 0:
                continue
            if pd.isna(row.feature_min) or pd.isna(row.feature_max):
                continue
            min_time = pd.Timestamp(row.feature_min)
            max_time = pd.Timestamp(row.feature_max)
            for window_len in window_days:
                window = pd.Timedelta(days=int(window_len))
                best = _best_accident_window(
                    times,
                    window=window,
                    min_time=min_time,
                    max_time=max_time,
                )
                if not best:
                    continue
                if best["accidents_window"] < int(min_accidents_window):
                    continue
                candidates.append(
                    {
                        "portico_last": row.portico_last,
                        "portico_next": row.portico_next,
                        "eje": getattr(row, "eje", None),
                        "calzada": getattr(row, "calzada", None),
                        "feature_min": min_time,
                        "feature_max": max_time,
                        "window_days": int(window_len),
                        "window_start": best["window_start"],
                        "window_end": best["window_end"],
                        "accidents_window": best["accidents_window"],
                        "accidents_per_day": best["accidents_window"]
                        / max(1, int(window_len)),
                        "accidents_total": int(row.accidents_total),
                    }
                )
            progress_bar.progress(
                int(idx / total_segments * 100),
                text=f"Buscando ventanas... {idx}/{total_segments}",
            )
        progress_bar.empty()

        if not candidates:
            st.warning("No se encontraron ventanas candidatas.")
            return

        candidates_df = pd.DataFrame(candidates)
        if metric_choice == "Accidentes por dia":
            sort_cols = ["accidents_per_day", "accidents_window"]
        else:
            sort_cols = ["accidents_window", "accidents_per_day"]
        candidates_df = candidates_df.sort_values(
            sort_cols, ascending=False
        ).reset_index(drop=True)

        st.markdown("**Top candidatos**")
        st.dataframe(candidates_df.head(int(top_show)), width="stretch")

        val_ratio = float(val_size)
        test_ratio = float(test_size)
        if val_ratio <= 0 or test_ratio <= 0:
            st.error("Validation/Test deben ser mayores que 0.")
            return
        if val_ratio + test_ratio >= 1:
            st.error("Validation + Test debe ser menor que 1.")
            return

        search_space = {
            "smote": {
                "k_neighbors": {"min": smote_k_min, "max": smote_k_max},
                "sampling_strategy": {
                    "min": smote_str_min,
                    "max": smote_str_max,
                },
            },
            "model": model_ranges,
        }

        def _build_candidate_payload(
            row: pd.Series, *, rank: int
        ) -> Dict[str, object]:
            return {
                "experiment": "Find samples sizes",
                "type": "Find samples sizes",
                "candidate_rank": int(rank),
                "objective_metric": objective_key,
                "objective_label": objective_label,
                "objective_direction": objective_direction,
                "run_id": run_id,
                "dataset_name": selected_event,
                "features_name": selected_features,
                "segment_portico_last": row["portico_last"],
                "segment_portico_next": row["portico_next"],
                "segment_eje": row.get("eje"),
                "segment_calzada": row.get("calzada"),
                "window_days": int(row["window_days"]),
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "accidents_window": int(row["accidents_window"]),
                "accidents_per_day": float(row["accidents_per_day"]),
                "accidents_total_segment": int(row["accidents_total"]),
                "feature_min": row.get("feature_min"),
                "feature_max": row.get("feature_max"),
                "model_choice": model_choice,
                "threshold_strategy": threshold_strategy,
                "calibration_methods": list(calibration_methods),
                "n_trials": int(n_trials),
                "timeout": int(timeout),
                "optuna_n_jobs": int(optuna_n_jobs),
                "far_target": float(far_target),
                "window_train_ratio": float(1 - val_ratio - test_ratio),
                "window_val_ratio": float(val_ratio),
                "window_test_ratio": float(test_ratio),
                "use_cluster_features": bool(use_cluster_features),
                "min_k": int(min_k_val),
                "max_k": int(max_k_val),
                "step_k": int(step_k_val),
                "search_space_config": json.dumps(search_space),
            }

        def _evaluate_candidate(
            row: pd.Series, *, rank: int
        ) -> Tuple[List[Dict[str, object]], List[Optional[object]]]:
            # Updated signature to match what we implemented earlier (List return)
            payload = _build_candidate_payload(row, rank=rank)
            seg_mask = (
                (features_df["portico_last"] == row["portico_last"])
                & (features_df["portico_next"] == row["portico_next"])
            )
            if "eje" in features_df.columns and pd.notna(row.get("eje")):
                seg_mask &= features_df["eje"] == row["eje"]
            if "calzada" in features_df.columns and pd.notna(
                row.get("calzada")
            ):
                seg_mask &= features_df["calzada"] == row["calzada"]

            segment_features = features_df.loc[seg_mask].copy()
            if segment_features.empty:
                payload["error"] = "No hay features para el tramo."
                return [payload], [None]

            segment_accidents = acc_seg.loc[
                (acc_seg["portico_last"] == row["portico_last"])
                & (acc_seg["portico_next"] == row["portico_next"])
            ].copy()

            segment_base_df = add_accident_target(
                segment_features, segment_accidents
            )
            if segment_base_df.empty:
                payload["error"] = "Dataset vacio tras merge."
                return [payload], [None]

            window_mask = (
                (segment_base_df["interval_start"] >= row["window_start"])
                & (segment_base_df["interval_start"] <= row["window_end"])
            )
            window_df = segment_base_df.loc[window_mask].copy()
            if window_df.empty:
                payload["error"] = "No hay datos dentro de la ventana."
                return [payload], [None]

            try:
                train_df, holdout_df = _temporal_train_test_split(
                    window_df, test_size=float(val_ratio + test_ratio)
                )
                holdout_test_ratio = test_ratio / (val_ratio + test_ratio)
                val_df, test_df = _temporal_train_test_split(
                    holdout_df, test_size=float(holdout_test_ratio)
                )
            except Exception as exc:
                payload["error"] = f"Split temporal fallo: {exc}"
                return [payload], [None]

            if train_df["target"].nunique() < 2:
                payload["error"] = "Train solo tiene una clase."
                return [payload], [None]
            if val_df["target"].nunique() < 2:
                payload["error"] = "Val solo tiene una clase."
                return [payload], [None]
            if test_df["target"].nunique() < 2:
                payload["error"] = "Test solo tiene una clase."
                return [payload], [None]

            all_feature_cols = _get_feature_cols(segment_base_df)
            if not use_cluster_features:
                cluster_cols = set(_get_cluster_cols(segment_base_df))
                all_feature_cols = [
                    c for c in all_feature_cols if c not in cluster_cols
                ]
            if not all_feature_cols:
                payload["error"] = "No hay variables numericas para entrenar."
                return [payload], [None]

            runner = ExperimentsRunner()
            selected_feature_cols = all_feature_cols
            
            # Feature Selection (Importance) - Calculate once
            # Always calculate importance for K selection
            try:
                importance_df = runner.calculate_feature_importance(
                    segment_base_df, all_feature_cols
                )
                ordered = importance_df["variable"].tolist()
            except Exception as exc:
                payload["error"] = f"Feature selection fallo: {exc}"
                return [payload], [None]

            # K Iteration Loop
            k_results: List[Dict[str, object]] = []
            k_models: List[Optional[object]] = []
            
            # Determine K loop range
            min_k = int(min_k_val)
            max_k = int(max_k_val)
            step_k = int(step_k_val)
            total_features = len(ordered)

            if total_features < min_k:
                payload["error"] = (
                    "Min K es mayor que la cantidad de variables disponibles "
                    f"({total_features})."
                )
                return [payload], [None]

            eff_max_k = min(max_k, total_features)
            k_values = list(range(min_k, eff_max_k + 1, step_k))
            if k_values and k_values[-1] != eff_max_k:
                k_values.append(eff_max_k)
            if not k_values:
                payload["error"] = "No hay valores de K validos para entrenar."
                return [payload], [None]

            for calibration_method in calibration_methods:
                for k_curr in k_values:
                    p_k = payload.copy()
                    p_k["k"] = int(k_curr)
                    p_k["k_features"] = int(k_curr)
                    p_k["calibration_method"] = str(calibration_method)

                    curr_features = ordered[:k_curr]
                    if not curr_features:
                        continue

                    try:
                        result = runner.run_optimization_loop(
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df,
                            feature_cols=curr_features,
                            model_choice=model_choice,
                            n_trials=int(n_trials),
                            timeout=int(timeout),
                            optuna_n_jobs=int(optuna_n_jobs),
                            far_target=float(far_target),
                            search_space_config=search_space,
                            objective_key=objective_key,
                            objective_direction=objective_direction,
                            threshold_strategy=threshold_strategy,
                            calibration_method=str(calibration_method),
                            return_model=True,
                        )
                    except Exception as exc:
                        p_k["error"] = f"Error en entrenamiento K={k_curr}: {exc}"
                        k_results.append(p_k)
                        k_models.append(None)
                        continue

                    model_obj = result.pop("model", None)
                    p_k.update(result)
                    k_results.append(p_k)
                    k_models.append(model_obj)

            if not k_results:
                 payload["error"] = "No se generaron resultados para ningun K."
                 return [payload], [None]
                 
            return k_results, k_models

        eval_limit = max(1, int(eval_top_n))
        eval_limit = min(eval_limit, len(candidates_df))
        eval_candidates = candidates_df.head(eval_limit)
        progress_eval = st.progress(0, text="Evaluando candidatos...")
        eval_results: List[Dict[str, object]] = []
        eval_models: List[Optional[object]] = []
        for idx, (_, row) in enumerate(eval_candidates.iterrows(), start=1):
            payloads_k, models_k = _evaluate_candidate(row, rank=idx)
            eval_results.extend(payloads_k)
            eval_models.extend(models_k)
            for p in payloads_k:
                _append_experiment_result(exp_db_path, p)
            progress_eval.progress(
                int(idx / eval_limit * 100),
                text=f"Evaluando candidatos... {idx}/{eval_limit}",
            )
        progress_eval.empty()

        res_df = pd.DataFrame(eval_results)
        metric_key = objective_key
        metric_direction = (
            "min" if objective_direction == "minimize" else "max"
        )
        if metric_key == "far_sens":
            if {"far", "sensitivity"}.issubset(res_df.columns):
                res_df = res_df.copy()
                res_df["far_sens"] = (
                    res_df["far"] - (res_df["sensitivity"] * 1e-3)
                )
            else:
                st.warning(
                    "No se encontro FAR/Sensibilidad para calcular la metrica."
                )
        if metric_key not in res_df.columns:
            st.warning("No hay métricas disponibles para seleccionar un óptimo.")
            best_candidates = pd.DataFrame()
        else:
            best_candidates = res_df.copy()
            if "error" in best_candidates.columns:
                best_candidates = best_candidates[best_candidates["error"].isna()]
            best_candidates = best_candidates.dropna(subset=[metric_key])
        best_row = None
        best_rank = None
        best_idx = None
        if best_candidates.empty:
            st.warning(
                "No se pudo seleccionar un mejor candidato por la metrica objetivo."
            )
        else:
            if metric_direction == "min":
                best_row = best_candidates.loc[
                    best_candidates[metric_key].idxmin()
                ]
            else:
                best_row = best_candidates.loc[
                    best_candidates[metric_key].idxmax()
                ]
            best_rank = int(best_row.get("candidate_rank", 0))
            best_idx = best_row.name

        res_df["is_best"] = False
        if best_idx is not None:
             try:
                res_df.loc[best_idx, "is_best"] = True
             except KeyError:
                pass

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = None
        if best_idx is not None and 0 <= best_idx < len(eval_models):
            best_model = eval_models[best_idx]
            if best_model is not None:
                try:
                    import joblib  # type: ignore
                    model_path = str(
                        RESULTS_DIR / f"find_samples_sizes_model_{stamp}.joblib"
                    )
                    joblib.dump(best_model, model_path)
                except Exception as exc:
                    st.warning(f"No se pudo guardar el modelo: {exc}")

        if model_path:
            try:
                res_df.loc[best_idx, "model_path"] = model_path
            except KeyError:
                pass
            if best_row is not None:
                best_row = best_row.copy()
                best_row["model_path"] = model_path

        st.subheader("Resultados")
        st.dataframe(res_df, width="stretch")

        if best_row is not None:
            best_k = best_row.get("k")
            if best_k is None or pd.isna(best_k):
                best_k = best_row.get("k_features", "?")
            st.success(
                "Mejor mix segun "
                f"{objective_label}: "
                f"{best_row['segment_portico_last']} -> {best_row['segment_portico_next']} | "
                f"{best_row['window_start']} a {best_row['window_end']} | "
                f"K={best_k}"
            )
            if model_path:
                st.caption(f"Modelo guardado: {model_path}")
            cm = best_row.get("confusion_matrix")
            if isinstance(cm, list) and cm:
                cm_data = cm
                if len(cm) == 4 and not isinstance(cm[0], (list, tuple)):
                    tn, fp, fn, tp = cm
                    cm_data = [[tn, fp], [fn, tp]]
                cm_df = pd.DataFrame(
                    cm_data,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
                st.caption("Matriz de confusion (mejor mix)")
                st.dataframe(cm_df, width="stretch")
            _append_experiment_best(exp_db_path, dict(best_row))

        res_path = RESULTS_DIR / f"find_samples_sizes_results_{stamp}.csv"
        res_df.to_csv(res_path, index=False)
        cand_path = RESULTS_DIR / f"find_samples_sizes_candidates_{stamp}.csv"
        candidates_df.to_csv(cand_path, index=False)
        st.success(f"Resultados guardados en {res_path}")
        st.caption(f"Candidatos guardados en {cand_path}")


def _render_best_highway_section_experiment() -> None:
    st.subheader("Best highway section")
    st.caption(
        "Recorre todos los tramos con datos, aplica seleccion de features, "
        "Optuna, SMOTE y entrenamiento para Base y Base + Cluster."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No hay archivos de eventos (accidents) en Datos.")
        return
    event_names = [p.name for p in event_files]
    selected_event = st.selectbox(
        "Archivo de Eventos", event_names, key="exp_best_section_event_file"
    )

    feature_files = _list_flow_feature_files()
    if not feature_files:
        st.warning("No hay archivos de features en Resultados.")
        return
    feature_names = [p.name for p in feature_files]
    selected_features = st.selectbox(
        "Archivo de Features (Flow + Cluster)",
        feature_names,
        key="exp_best_section_feature_file",
    )

    objective_options = _optuna_objective_options(
        [
            "f1",
            "roc_auc",
            "accuracy",
            "recall",
            "precision",
            "fnr",
            "far_sens",
            "mcc",
            "brier_score",
        ]
    )
    objective_label = st.selectbox(
        "Metrica objetivo (mejor mix)",
        list(objective_options.keys()),
        key="exp_best_section_objective_metric",
    )
    objective_cfg = objective_options.get(
        objective_label, {"key": "f1", "direction": "maximize"}
    )
    objective_key = objective_cfg["key"]
    objective_direction = objective_cfg["direction"]

    st.markdown("**Feature selection**")
    feature_top_n = st.number_input(
        "Numero de variables mas importantes",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        key="exp_best_section_feature_top_n",
    )

    st.markdown("**Configuracion del modelo**")
    model_choice = st.selectbox(
        "Modelo para Experimento",
        ["Random Forest", "XGBoost", "SVM"],
        key="exp_best_section_model_choice",
    )

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        n_trials = st.number_input(
            "Optuna Trials por tramo",
            min_value=5,
            value=30,
            step=5,
            key="exp_best_section_n_trials",
        )
    with col_n2:
        timeout = st.number_input(
            "Optuna Timeout (seg) por tramo",
            min_value=10,
            value=3600,
            step=10,
            key="exp_best_section_timeout",
        )
    optuna_n_jobs = _render_optuna_n_jobs_input(
        "Optuna jobs paralelos",
        key="exp_best_section_optuna_n_jobs",
        default=1,
    )

    far_target = 0.2
    threshold_strategy = "optuna"
    threshold_strategy_label = "Optimizar threshold"
    with st.expander("Configuracion avanzada (parametros y rangos)"):
        st.markdown("**Split de datos**")
        c_split1, c_split2 = st.columns(2)
        with c_split1:
            val_size = st.slider(
                "Validation Size (sobre train)",
                0.1,
                0.9,
                0.2,
                0.05,
                key="exp_best_section_val_size",
            )
        with c_split2:
            test_size = st.slider(
                "Test Size (sobre total)",
                0.1,
                0.9,
                0.2,
                0.05,
                key="exp_best_section_test_size",
            )
        st.markdown("**Calibracion de umbral**")
        threshold_options = {
            "Optimizar threshold": "optuna",
            "Calibrar por FAR": "far",
        }
        threshold_strategy = _option_value_from_state(
            threshold_options,
            "exp_best_section_threshold_strategy",
            default_label="Optimizar threshold",
        )
        threshold_visibility = _threshold_field_visibility_for_strategy(
            threshold_strategy
        )
        far_target = float(
            _render_conditional_slider(
                "FAR target",
                visible=threshold_visibility["far_target"],
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="exp_best_section_far_target",
            )
        )
        threshold_strategy_label = st.selectbox(
            "Estrategia de umbral",
            list(threshold_options.keys()),
            key="exp_best_section_threshold_strategy",
        )
        threshold_strategy = threshold_options[threshold_strategy_label]
        calibration_methods = _calibration_method_multiselect(
            "Calibración",
            key="exp_best_section_calibration_methods",
            default_methods=["sigmoid", "isotonic"],
        )

        st.markdown("**Rango SMOTE**")
        c_smote1, c_smote2 = st.columns(2)
        with c_smote1:
            smote_k_min = st.number_input(
                "K Neighbors Min",
                1,
                20,
                1,
                key="exp_best_section_smote_k_min",
            )
            smote_k_max = st.number_input(
                "K Neighbors Max",
                1,
                20,
                10,
                key="exp_best_section_smote_k_max",
            )
        with c_smote2:
            smote_str_min = st.slider(
                "Sampling Strategy Min",
                0.1,
                1.0,
                0.1,
                0.1,
                key="exp_best_section_smote_str_min",
            )
            smote_str_max = st.slider(
                "Sampling Strategy Max",
                0.1,
                1.0,
                1.0,
                0.1,
                key="exp_best_section_smote_str_max",
            )

        st.markdown(f"**Rangos para {model_choice}**")
        model_ranges = {}
        if model_choice == "Random Forest":
            c_rf1, c_rf2 = st.columns(2)
            with c_rf1:
                rf_ne_min = st.number_input(
                    "N Estimators Min",
                    10,
                    1000,
                    50,
                    step=10,
                    key="exp_best_section_rf_ne_min",
                )
                rf_ne_max = st.number_input(
                    "N Estimators Max",
                    10,
                    1000,
                    300,
                    step=10,
                    key="exp_best_section_rf_ne_max",
                )
            with c_rf2:
                rf_md_min = st.number_input(
                    "Max Depth Min",
                    1,
                    50,
                    3,
                    key="exp_best_section_rf_md_min",
                )
                rf_md_max = st.number_input(
                    "Max Depth Max",
                    1,
                    50,
                    15,
                    key="exp_best_section_rf_md_max",
                )
            model_ranges = {
                "n_estimators": {"min": rf_ne_min, "max": rf_ne_max},
                "max_depth": {"min": rf_md_min, "max": rf_md_max},
            }
        elif model_choice == "XGBoost":
            c_xgb1, c_xgb2 = st.columns(2)
            with c_xgb1:
                xgb_ne_min = st.number_input(
                    "N Estimators Min",
                    10,
                    1000,
                    50,
                    step=10,
                    key="exp_best_section_xgb_ne_min",
                )
                xgb_ne_max = st.number_input(
                    "N Estimators Max",
                    10,
                    1000,
                    300,
                    step=10,
                    key="exp_best_section_xgb_ne_max",
                )
                xgb_lr_min = st.number_input(
                    "Learning Rate Min",
                    0.001,
                    1.0,
                    0.01,
                    format="%.3f",
                    key="exp_best_section_xgb_lr_min",
                )
                xgb_lr_max = st.number_input(
                    "Learning Rate Max",
                    0.001,
                    1.0,
                    0.3,
                    format="%.3f",
                    key="exp_best_section_xgb_lr_max",
                )
            with c_xgb2:
                xgb_md_min = st.number_input(
                    "Max Depth Min",
                    1,
                    50,
                    3,
                    key="exp_best_section_xgb_md_min",
                )
                xgb_md_max = st.number_input(
                    "Max Depth Max",
                    1,
                    50,
                    15,
                    key="exp_best_section_xgb_md_max",
                )
                xgb_sub_min = st.slider(
                    "Subsample Min",
                    0.1,
                    1.0,
                    0.5,
                    0.1,
                    key="exp_best_section_xgb_sub_min",
                )
                xgb_sub_max = st.slider(
                    "Subsample Max",
                    0.1,
                    1.0,
                    1.0,
                    0.1,
                    key="exp_best_section_xgb_sub_max",
                )
                xgb_col_min = st.slider(
                    "Colsample ByTree Min",
                    0.1,
                    1.0,
                    0.5,
                    0.1,
                    key="exp_best_section_xgb_col_min",
                )
                xgb_col_max = st.slider(
                    "Colsample ByTree Max",
                    0.1,
                    1.0,
                    1.0,
                    0.1,
                    key="exp_best_section_xgb_col_max",
                )
            model_ranges = {
                "n_estimators": {"min": xgb_ne_min, "max": xgb_ne_max},
                "max_depth": {"min": xgb_md_min, "max": xgb_md_max},
                "learning_rate": {"min": xgb_lr_min, "max": xgb_lr_max},
                "subsample": {"min": xgb_sub_min, "max": xgb_sub_max},
                "colsample_bytree": {"min": xgb_col_min, "max": xgb_col_max},
            }
        elif model_choice == "SVM":
            c_svm1, c_svm2 = st.columns(2)
            with c_svm1:
                svm_c_min = st.number_input(
                    "C Min",
                    0.01,
                    1000.0,
                    0.1,
                    format="%.2f",
                    key="exp_best_section_svm_c_min",
                )
            with c_svm2:
                svm_c_max = st.number_input(
                    "C Max",
                    0.01,
                    1000.0,
                    50.0,
                    format="%.2f",
                    key="exp_best_section_svm_c_max",
                )
            model_ranges = {"C": {"min": svm_c_min, "max": svm_c_max}}

    if st.button("Iniciar experimento", key="exp_best_section_run"):
        if not calibration_methods:
            st.error("Seleccione al menos un calibrador.")
            return
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_db_path = _init_experiment_db(
            "Best highway section",
            {
                "run_id": run_id,
                "dataset_name": selected_event,
                "features_name": selected_features,
                "model_choice": model_choice,
                "objective_label": objective_label,
                "objective_metric": objective_key,
                "objective_direction": objective_direction,
                "feature_selection_top_n": int(feature_top_n),
                "far_target": float(far_target),
                "threshold_strategy": threshold_strategy,
                "threshold_strategy_label": threshold_strategy_label,
                "calibration_methods": list(calibration_methods),
                "val_size": float(val_size),
                "test_size": float(test_size),
            },
        )
        if exp_db_path:
            st.caption(f"DB live: {exp_db_path}")

        accidents_path = next(p for p in event_files if p.name == selected_event)
        features_path = next(p for p in feature_files if p.name == selected_features)

        try:
            raw_accidents_df = read_csv_with_progress(str(accidents_path))
        except Exception as exc:
            st.error(f"Error cargando accidentes: {exc}")
            return

        try:
            porticos_df = load_porticos()
            if porticos_df is None or porticos_df.empty:
                st.error("No se pudieron cargar los porticos (Porticos.csv).")
                return
        except Exception as exc:
            st.error(f"Error cargando porticos: {exc}")
            return

        try:
            accidents_df, excluded = process_accidentes_df(
                raw_accidents_df, porticos_df, return_excluded=True
            )
            if accidents_df.empty:
                st.warning(
                    "No quedaron accidentes validos tras el procesamiento."
                )
                return
            st.success(
                f"Accidentes procesados: {len(accidents_df)} (Excluidos: {len(excluded)})"
            )
        except Exception as exc:
            st.error(f"Error procesando accidentes: {exc}")
            return

        if features_path.suffix.lower() != ".duckdb":
            st.error("El archivo de features debe ser .duckdb.")
            return
        if duckdb is None:
            st.error("duckdb no esta instalado.")
            return

        con = None
        try:
            con = duckdb.connect(str(features_path), read_only=True)
            table_rows = con.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in table_rows]
            table_name = _pick_duckdb_table(tables, ["flow_features", "features"])
            if not table_name:
                st.error("La base de datos de features esta vacia.")
                return
            table_ref = _duckdb_quote_identifier(table_name)
            cols_info = con.execute(f"DESCRIBE {table_ref}").fetchall()
            columns = {row[0] for row in cols_info}
            segment_cols = None
            if {"portico_last", "portico_next"}.issubset(columns):
                segment_cols = ("portico_last", "portico_next")
            elif {"portico_inicio", "portico_fin"}.issubset(columns):
                segment_cols = ("portico_inicio", "portico_fin")
            if not segment_cols:
                st.error(
                    "El archivo de features no contiene columnas de tramo "
                    "(portico_last/portico_next o portico_inicio/portico_fin)."
                )
                return

            last_col, next_col = segment_cols
            last_ref = _duckdb_quote_identifier(last_col)
            next_ref = _duckdb_quote_identifier(next_col)
            segments_df = con.execute(
                f"SELECT DISTINCT {last_ref} AS portico_last, {next_ref} AS portico_next "
                f"FROM {table_ref} "
                f"WHERE {last_ref} IS NOT NULL AND {next_ref} IS NOT NULL"
            ).df()
        except Exception as exc:
            st.error(f"Error leyendo features: {exc}")
            return
        finally:
            if con is not None:
                con.close()

        if segments_df is None or segments_df.empty:
            st.warning("No se encontraron tramos en el archivo de features.")
            return

        segments_df = segments_df.copy()
        segments_df["portico_last_raw"] = segments_df["portico_last"].astype(str).str.strip()
        segments_df["portico_next_raw"] = segments_df["portico_next"].astype(str).str.strip()
        segments_df["portico_last"] = _normalize_portico_series(
            segments_df["portico_last_raw"]
        )
        segments_df["portico_next"] = _normalize_portico_series(
            segments_df["portico_next_raw"]
        )
        segments_df = segments_df.dropna(subset=["portico_last", "portico_next"])
        if segments_df.empty:
            st.warning("No hay tramos validos en el archivo de features.")
            return

        try:
            seg_meta = get_portico_segments(porticos_df)
            if seg_meta is not None and not seg_meta.empty:
                seg_meta = seg_meta.copy()
                seg_meta["portico_last"] = _normalize_portico_series(
                    seg_meta["portico_last"]
                )
                seg_meta["portico_next"] = _normalize_portico_series(
                    seg_meta["portico_next"]
                )
                segments_df = segments_df.merge(
                    seg_meta[["eje", "calzada", "portico_last", "portico_next"]],
                    on=["portico_last", "portico_next"],
                    how="left",
                )
        except Exception:
            pass

        acc_seg = accidents_df.copy()
        acc_seg["portico_last"] = _normalize_portico_series(
            acc_seg["ultimo_portico"]
        )
        acc_seg["portico_next"] = _normalize_portico_series(
            acc_seg["proximo_portico"]
        )
        acc_seg = acc_seg.dropna(
            subset=["portico_last", "portico_next", "accidente_time"]
        )
        acc_groups = {
            key: group.copy()
            for key, group in acc_seg.groupby(["portico_last", "portico_next"])
        }

        cluster_cols_available = _get_cluster_cols(
            pd.DataFrame(columns=list(columns))
        )
        has_cluster_available = bool(cluster_cols_available)

        search_space = {
            "smote": {
                "k_neighbors": {"min": smote_k_min, "max": smote_k_max},
                "sampling_strategy": {
                    "min": smote_str_min,
                    "max": smote_str_max,
                },
            },
            "model": model_ranges,
        }

        runner = ExperimentsRunner()
        results: List[Dict[str, object]] = []
        total_segments = len(segments_df)
        progress_bar = st.progress(0, text="Procesando tramos...")
        con = None
        table_ref = _duckdb_quote_identifier(table_name)
        seg_columns = set(columns)

        try:
            con = duckdb.connect(str(features_path), read_only=True)

            for idx, row in enumerate(segments_df.itertuples(index=False), start=1):
                seg_last = getattr(row, "portico_last", None)
                seg_next = getattr(row, "portico_next", None)
                seg_last_raw = getattr(row, "portico_last_raw", seg_last)
                seg_next_raw = getattr(row, "portico_next_raw", seg_next)
                eje = getattr(row, "eje", None)
                calzada = getattr(row, "calzada", None)

                payload_common = {
                    "experiment": "Best highway section",
                    "type": "Base",
                    "run_id": run_id,
                    "dataset_name": selected_event,
                    "features_name": selected_features,
                    "segment_portico_last": seg_last,
                    "segment_portico_next": seg_next,
                    "segment_eje": eje,
                    "segment_calzada": calzada,
                    "segment_index": int(idx),
                    "objective_metric": objective_key,
                    "objective_label": objective_label,
                    "model_choice": model_choice,
                    "n_trials": int(n_trials),
                    "timeout": int(timeout),
                    "optuna_n_jobs": int(optuna_n_jobs),
                    "far_target": float(far_target),
                    "threshold_strategy": threshold_strategy,
                    "threshold_strategy_label": threshold_strategy_label,
                    "calibration_methods": list(calibration_methods),
                    "val_size": float(val_size),
                    "test_size": float(test_size),
                    "search_space_config": json.dumps(search_space),
                    "feature_selection_top_n": int(feature_top_n),
                }

                progress_bar.progress(
                    int(idx / total_segments * 100),
                    text=f"Procesando tramo {idx}/{total_segments}",
                )

                accidents_segment = acc_groups.get((seg_last, seg_next))
                if accidents_segment is None or accidents_segment.empty:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "No hay accidentes en el tramo."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = "No hay accidentes en el tramo."
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                tramo_tuple = (eje, calzada, seg_last_raw, seg_next_raw)
                clauses, params, filter_ok = _build_tramo_duckdb_filters(
                    tramo_tuple, seg_columns
                )
                if not filter_ok:
                    payload_base = dict(payload_common)
                    payload_base["error"] = (
                        "No se pudo filtrar el tramo en el archivo de features."
                    )
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue
                try:
                    query = f"SELECT * FROM {table_ref}"
                    if clauses:
                        query += " WHERE " + " AND ".join(clauses)
                    segment_features = con.execute(query, params).df()
                except Exception as exc:
                    payload_base = dict(payload_common)
                    payload_base["error"] = (
                        f"Error cargando features del tramo: {exc}"
                    )
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if segment_features is None or segment_features.empty:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "No hay features para el tramo."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if segment_cols != ("portico_last", "portico_next"):
                    segment_features = segment_features.rename(
                        columns={
                            segment_cols[0]: "portico_last",
                            segment_cols[1]: "portico_next",
                        }
                    )

                if "interval_start" not in segment_features.columns:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "Las features no tienen interval_start."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                segment_features = segment_features.copy()
                segment_features["interval_start"] = pd.to_datetime(
                    segment_features["interval_start"], errors="coerce"
                )

                segment_base_df = add_accident_target(
                    segment_features, accidents_segment
                )
                if segment_base_df.empty:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "Dataset vacio tras merge."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if test_size <= 0 or test_size >= 1:
                    st.error("Test size debe estar entre 0 y 1.")
                    progress_bar.empty()
                    return
                if val_size <= 0 or val_size >= 1:
                    st.error("Validation size debe estar entre 0 y 1.")
                    progress_bar.empty()
                    return
                val_ratio = float(val_size)

                try:
                    train_df, test_df = _temporal_train_test_split(
                        segment_base_df, test_size=float(test_size)
                    )
                    train_opt_df, val_df = _temporal_train_test_split(
                        train_df, test_size=float(val_ratio)
                    )
                except Exception as exc:
                    payload_base = dict(payload_common)
                    payload_base["error"] = f"Split fallo: {exc}"
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if (
                    train_df.empty
                    or val_df.empty
                    or test_df.empty
                    or train_df["target"].nunique() < 2
                    or val_df["target"].nunique() < 2
                    or test_df["target"].nunique() < 2
                ):
                    payload_base = dict(payload_common)
                    payload_base["error"] = "Split sin clases suficientes."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                all_feature_cols = _get_feature_cols(segment_base_df)
                cluster_cols = _get_cluster_cols(segment_base_df)
                base_cols = [c for c in all_feature_cols if c not in cluster_cols]

                cluster_set = set(cluster_cols)
                combined_ordered: List[str] = []
                base_ordered_from_combined: List[str] = []
                combined_selected_cols: List[str] = []
                combined_top_n = 0
                cluster_in_top_n = 0
                importance_error = None
                if not all_feature_cols:
                    importance_error = "No hay variables numericas para entrenar."
                else:
                    try:
                        combined_importance_df = (
                            runner.calculate_feature_importance(
                                segment_base_df, all_feature_cols
                            )
                        )
                        combined_ordered = combined_importance_df[
                            "variable"
                        ].tolist()
                        base_ordered_from_combined = [
                            col for col in combined_ordered if col in base_cols
                        ]
                        combined_top_n = max(
                            1,
                            min(int(feature_top_n), len(combined_ordered)),
                        )
                        combined_selected_cols = combined_ordered[:combined_top_n]
                        cluster_in_top_n = sum(
                            1
                            for col in combined_selected_cols
                            if col in cluster_set
                        )
                        if not combined_selected_cols:
                            importance_error = (
                                "No hay variables numericas para entrenar."
                            )
                    except Exception as exc:
                        importance_error = f"Feature selection fallo: {exc}"

                def _run_dataset(
                    *,
                    dataset_type: str,
                    candidate_cols: List[str],
                    selected_cols_override: Optional[List[str]] = None,
                ) -> List[Dict[str, object]]:
                    payloads: List[Dict[str, object]] = []
                    for calibration_method in calibration_methods:
                        payload = dict(payload_common)
                        payload["type"] = dataset_type
                        payload["feature_selection_total"] = int(len(candidate_cols))
                        payload["calibration_method"] = str(calibration_method)
                        if not candidate_cols:
                            payload["error"] = "No hay variables numericas para entrenar."
                            payloads.append(payload)
                            continue
                        if not selected_cols_override:
                            payload["error"] = (
                                "No hay ranking de importancia para seleccionar "
                                "variables."
                            )
                            payloads.append(payload)
                            continue
                        selected_cols = [
                            col
                            for col in selected_cols_override
                            if col in candidate_cols
                        ]
                        if not selected_cols:
                            payload["error"] = (
                                "No hay variables numericas para entrenar."
                            )
                            payloads.append(payload)
                            continue

                        payload["feature_selection_selected"] = int(len(selected_cols))

                        try:
                            result = runner.run_optimization_loop(
                                train_df=train_opt_df,
                                val_df=val_df,
                                test_df=test_df,
                                feature_cols=selected_cols,
                                model_choice=model_choice,
                                n_trials=int(n_trials),
                                timeout=int(timeout),
                                optuna_n_jobs=int(optuna_n_jobs),
                                far_target=float(far_target),
                                search_space_config=search_space,
                                objective_key=objective_key,
                                objective_direction=objective_direction,
                                threshold_strategy=threshold_strategy,
                                calibration_method=str(calibration_method),
                            )
                            payload.update(result)
                        except Exception as exc:
                            payload["error"] = f"Error en Optuna: {exc}"
                        payloads.append(payload)
                    return payloads

                if importance_error:
                    payload_base = dict(payload_common)
                    payload_base["type"] = "Base"
                    payload_base["feature_selection_total"] = int(len(base_cols))
                    payload_base["error"] = importance_error
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["feature_selection_total"] = int(
                            len(all_feature_cols)
                        )
                        payload_cluster["error"] = importance_error
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                base_target_n = combined_top_n - cluster_in_top_n
                base_target_n = min(base_target_n, len(base_ordered_from_combined))

                if base_target_n <= 0:
                    payload_base = dict(payload_common)
                    payload_base["type"] = "Base"
                    payload_base["feature_selection_total"] = int(len(base_cols))
                    payload_base["error"] = (
                        "K total sin variables base disponibles."
                    )
                else:
                    base_selected_cols = base_ordered_from_combined[:base_target_n]
                    payload_base_list = _run_dataset(
                        dataset_type="Base",
                        candidate_cols=base_cols,
                        selected_cols_override=base_selected_cols,
                    )
                    payload_base = None
                if payload_base is not None:
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                else:
                    for payload_item in payload_base_list:
                        results.append(payload_item)
                        _append_experiment_result(exp_db_path, payload_item)

                if cluster_cols:
                    payload_cluster_list = _run_dataset(
                        dataset_type="Base + Cluster",
                        candidate_cols=all_feature_cols,
                        selected_cols_override=combined_selected_cols,
                    )
                    for payload_item in payload_cluster_list:
                        results.append(payload_item)
                        _append_experiment_result(exp_db_path, payload_item)
                elif has_cluster_available:
                    payload_cluster = dict(payload_common)
                    payload_cluster["type"] = "Base + Cluster"
                    payload_cluster["error"] = (
                        "No hay columnas de cluster en el dataset."
                    )
                    results.append(payload_cluster)
                    _append_experiment_result(exp_db_path, payload_cluster)

        finally:
            if con is not None:
                con.close()

        progress_bar.empty()

        if not results:
            st.warning("No se generaron resultados.")
            return

        res_df = pd.DataFrame(results)
        metric_key = objective_key
        metric_direction = (
            "min" if objective_direction == "minimize" else "max"
        )
        if metric_key == "far_sens":
            if {"far", "sensitivity"}.issubset(res_df.columns):
                res_df = res_df.copy()
                res_df["far_sens"] = (
                    res_df["far"] - (res_df["sensitivity"] * 1e-3)
                )
            else:
                st.warning(
                    "No se encontro FAR/Sensibilidad para calcular la metrica."
                )
        valid_df = res_df.copy()
        if "error" in valid_df.columns:
            valid_df = valid_df[valid_df["error"].isna()]
        if metric_key in valid_df.columns:
            valid_df = valid_df.dropna(subset=[metric_key])
        best_row = None
        if not valid_df.empty and metric_key in valid_df.columns:
            if metric_direction == "min":
                best_row = valid_df.loc[valid_df[metric_key].idxmin()]
            else:
                best_row = valid_df.loc[valid_df[metric_key].idxmax()]

        res_df["is_best"] = False
        if best_row is not None:
            res_df.loc[best_row.name, "is_best"] = True

        res_df["is_best_segment"] = False
        if {
            "segment_portico_last",
            "segment_portico_next",
            metric_key,
        }.issubset(res_df.columns):
            seg_valid = res_df.copy()
            if "error" in seg_valid.columns:
                seg_valid = seg_valid[seg_valid["error"].isna()]
            seg_valid = seg_valid.dropna(subset=[metric_key])
            if not seg_valid.empty:
                group_cols = ["segment_portico_last", "segment_portico_next"]
                if metric_direction == "min":
                    best_idx = seg_valid.groupby(group_cols)[metric_key].idxmin()
                else:
                    best_idx = seg_valid.groupby(group_cols)[metric_key].idxmax()
                res_df.loc[best_idx, "is_best_segment"] = True

        if "type" in res_df.columns and metric_key in res_df.columns:
            for dtype, group in res_df.groupby("type"):
                group_ok = group.copy()
                if "error" in group_ok.columns:
                    group_ok = group_ok[group_ok["error"].isna()]
                group_ok = group_ok.dropna(subset=[metric_key])
                if group_ok.empty:
                    continue
                if metric_direction == "min":
                    best_idx = group_ok[metric_key].idxmin()
                else:
                    best_idx = group_ok[metric_key].idxmax()
                res_df.loc[best_idx, "is_best_type"] = True

        st.subheader("Resultados")
        st.dataframe(res_df, width="stretch")

        if best_row is not None:
            st.success(
                "Mejor mix segun "
                f"{objective_label}: "
                f"{best_row.get('segment_portico_last', '?')} -> {best_row.get('segment_portico_next', '?')} "
                f"({best_row.get('type', '-')})"
            )
            best_payload = dict(best_row)
            _append_experiment_best(exp_db_path, best_payload)

            cm = best_row.get("confusion_matrix")
            if isinstance(cm, list) and cm:
                cm_data = cm
                if len(cm) == 4 and not isinstance(cm[0], (list, tuple)):
                    tn, fp, fn, tp = cm
                    cm_data = [[tn, fp], [fn, tp]]
                cm_df = pd.DataFrame(
                    cm_data,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
                st.caption("Matriz de confusion (mejor mix)")
                st.dataframe(cm_df, width="stretch")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_path = RESULTS_DIR / f"best_highway_section_results_{stamp}.csv"
        res_df.to_csv(res_path, index=False)
        st.success(f"Resultados guardados en {res_path}")


def _default_controlled_comparison_search_space() -> Dict[str, object]:
    return {
        "smote": {
            "k_neighbors": {"min": 1, "max": 15, "step": 1},
            "sampling_strategy": {"min": 0.001, "max": 0.1, "step": 0.005},
        },
        "rf": {
            "n_estimators": {"min": 50, "max": 300, "step": 25},
            "max_depth": {"min": 3, "max": 15, "step": 1},
            "min_samples_split": {"min": 2, "max": 10, "step": 1},
            "min_samples_leaf": {"min": 1, "max": 5, "step": 1},
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced"],
        },
        "svm": {
            "C": {"min": 0.1, "max": 10.0, "step": 0.5},
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "degree": {"min": 2, "max": 5, "step": 1},
            "coef0": {"min": 0.0, "max": 1.0, "step": 0.2},
            "class_weight": [None, "balanced"],
        },
        "xgb": {
            "n_estimators": {"min": 50, "max": 300, "step": 25},
            "max_depth": {"min": 3, "max": 10, "step": 1},
            "learning_rate": {
                "choices": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
            },
            "subsample": {"min": 0.6, "max": 1.0, "step": 0.1},
            "colsample_bytree": {"min": 0.6, "max": 1.0, "step": 0.1},
            "min_child_weight": {"choices": [0.1, 0.3, 1.0, 3.0, 10.0]},
            "reg_alpha": {"choices": [0.0, 0.01, 0.1, 1.0, 5.0]},
            "reg_lambda": {"choices": [0.5, 1.0, 2.0, 5.0, 10.0]},
            "gamma": {"choices": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]},
            "scale_pos_weight_multipliers": [0.5, 1.0, 2.0, 5.0, 10.0],
            "max_delta_step": [0.0, 1.0],
        },
        "balanced_rf": {
            "replacement": [False],
        },
    }


def _enrich_best_section_controlled_frame(
    frame: object,
    *,
    run_id: str,
    dataset_name: str,
    features_name: str,
    segment_index: int,
    segment_info: Dict[str, object],
    checkpoint_run_dir: Optional[object] = None,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    enriched = frame.copy()
    enriched["experiment"] = "Best highway section"
    enriched["protocol_family"] = "Controlled comparison"
    enriched["sweep_run_id"] = run_id
    enriched["dataset_name"] = dataset_name
    enriched["features_name"] = features_name
    enriched["segment_index"] = int(segment_index)
    enriched["segment_eje"] = segment_info.get("eje")
    enriched["segment_calzada"] = segment_info.get("calzada")
    enriched["segment_portico_last"] = segment_info.get("portico_inicio")
    enriched["segment_portico_next"] = segment_info.get("portico_fin")
    enriched["segment_label"] = segment_info.get("segment_label")
    enriched["segment_info"] = json.dumps(
        segment_info,
        ensure_ascii=True,
        default=_json_default,
    )
    if checkpoint_run_dir:
        enriched["checkpoint_run_dir"] = str(checkpoint_run_dir)
    return enriched


def _render_best_highway_section_controlled_experiment() -> None:
    st.subheader("Best highway section")
    st.caption(
        "Barre tramos usando el mismo protocolo de Comparación controlada: "
        "split temporal congelado por tramo, ranking solo en train, Base/Cluster/Base + Cluster, "
        "sin/con SMOTE y protocolos Conservador/Robusto."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No hay archivos de eventos (accidents) en Datos.")
        return
    feature_files = _list_flow_feature_files()
    if not feature_files:
        st.warning("No hay archivos de features en Resultados.")
        return

    event_names = [p.name for p in event_files]
    feature_names = [p.name for p in feature_files]
    selected_event = st.selectbox(
        "Archivo de Eventos",
        event_names,
        key="exp_best_section_controlled_event_file",
    )
    selected_features = st.selectbox(
        "Archivo de Features",
        feature_names,
        key="exp_best_section_controlled_feature_file",
    )
    selected_event_path = next(
        (p for p in event_files if p.name == selected_event),
        None,
    )
    selected_features_path = next(
        (p for p in feature_files if p.name == selected_features),
        None,
    )
    if selected_event_path is None or selected_features_path is None:
        st.error("No se pudieron resolver los archivos seleccionados.")
        return

    dataset_date_start, dataset_date_end, dataset_date_valid = (
        _render_controlled_feature_date_range_inputs(
            selected_features_path,
            key_prefix="exp_best_section_controlled",
        )
    )
    if not dataset_date_valid:
        return

    try:
        schema_df = _inspect_controlled_feature_schema(selected_features_path)
    except Exception as exc:
        st.error(f"No se pudo inspeccionar el archivo de features: {exc}")
        return

    all_schema_cols = _get_feature_cols(schema_df)
    cluster_schema_cols = _get_cluster_cols(schema_df)
    base_schema_cols = [
        col for col in all_schema_cols if col not in cluster_schema_cols
    ]
    if not cluster_schema_cols:
        st.error(
            "El archivo seleccionado no contiene variables de cluster. "
            "Este barrido usa el estándar de Comparación controlada y requiere "
            "Base, Cluster y Base + Cluster."
        )
        return
    if not base_schema_cols:
        st.error("No hay variables Base disponibles para comparar.")
        return
    max_available_features = max(1, len(all_schema_cols))

    objective_options = _controlled_objective_options()
    threshold_objective_options = {
        "Recall@N alertas/día": "recall_at_alerts_per_day",
        "FAR": "far",
        "Balanced F1": "balanced_f1",
        "F1": "f1",
        "MCC": "mcc",
        "Costo operacional": "operational_cost",
    }
    protocol_options = {
        "Conservador": "conservative",
        "Robusto": "robust",
    }

    st.markdown("**Configuración general**")
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
            key="exp_best_section_controlled_random_state",
        )
    with cfg2:
        n_trials = st.number_input(
            "Optuna trials",
            min_value=1,
            value=30,
            step=1,
            key="exp_best_section_controlled_n_trials",
        )
    with cfg3:
        timeout = st.number_input(
            "Optuna timeout (seg)",
            min_value=1,
            value=3600,
            step=10,
            key="exp_best_section_controlled_timeout",
        )
    with cfg4:
        objective_labels = list(objective_options.keys())
        objective_label = st.selectbox(
            "Métrica Optuna/ranking",
            objective_labels,
            index=objective_labels.index("Balanced F1"),
            key="exp_best_section_controlled_objective_metric",
        )
    objective_metric = objective_options.get(objective_label, "balanced_f1")

    selected_models = st.multiselect(
        "Modelos a comparar",
        list(CONTROLLED_COMPARISON_MODELS),
        default=list(CONTROLLED_COMPARISON_MODELS),
        key="exp_best_section_controlled_selected_models",
    )
    selected_protocol_labels = st.multiselect(
        "Protocolos de evaluación",
        list(protocol_options.keys()),
        default=list(protocol_options.keys()),
        key="exp_best_section_controlled_threshold_protocols",
    )
    threshold_protocols = [
        protocol_options[label]
        for label in selected_protocol_labels
        if label in protocol_options
    ] or ["conservative"]

    st.markdown("**Threshold operacional**")
    thr1, thr2, thr3, thr4 = st.columns(4)
    with thr1:
        threshold_objective_label = st.selectbox(
            "Objetivo de threshold",
            list(threshold_objective_options.keys()),
            index=0,
            key="exp_best_section_controlled_threshold_objective",
        )
    threshold_objective = threshold_objective_options.get(
        threshold_objective_label,
        "recall_at_alerts_per_day",
    )
    threshold_visibility = _threshold_field_visibility_for_objective(
        threshold_objective
    )
    with thr2:
        alerts_per_day = float(
            _render_conditional_number_input(
                "Alertas máximas por día",
                visible=threshold_visibility["alerts_per_day"],
                min_value=0.1,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="exp_best_section_controlled_alerts_per_day",
            )
        )
    with thr3:
        far_target = float(
            _render_conditional_slider(
                "FAR target",
                visible=threshold_visibility["far_target"],
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="exp_best_section_controlled_far_target",
            )
        )
    with thr4:
        calibration_methods = _calibration_method_multiselect(
            "Calibración",
            key="exp_best_section_controlled_calibration_method",
            default_methods=["sigmoid", "isotonic"],
        )

    cost1, cost2, cost3 = st.columns(3)
    with cost1:
        fn_cost = float(
            _render_conditional_number_input(
                "Costo FN",
                visible=threshold_visibility["fn_cost"],
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="exp_best_section_controlled_fn_cost",
            )
        )
    with cost2:
        fp_cost = float(
            _render_conditional_number_input(
                "Costo FP",
                visible=threshold_visibility["fp_cost"],
                min_value=0.0,
                value=1.0,
                step=0.5,
                key="exp_best_section_controlled_fp_cost",
            )
        )
    with cost3:
        robust_folds = st.number_input(
            "Folds robustos",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="exp_best_section_controlled_robust_folds",
        )

    st.markdown("**Split, K y paralelización**")
    split1, split2, split3, split4 = st.columns(4)
    with split1:
        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="exp_best_section_controlled_test_size",
        )
    with split2:
        val_size = st.slider(
            "Validation size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="exp_best_section_controlled_val_size",
        )
    with split3:
        parallel_jobs = st.number_input(
            "Jobs paralelos RF/ranking",
            min_value=1,
            max_value=_max_optuna_parallel_jobs(),
            value=min(10, _max_optuna_parallel_jobs()),
            step=1,
            key="exp_best_section_controlled_parallel_jobs",
        )
    with split4:
        xgb_parallel_jobs = _render_model_n_jobs_input(
            "Jobs paralelos XGBoost",
            key="exp_best_section_controlled_xgb_parallel_jobs",
            default=1,
            shared_key="global_xgb_parallel_jobs",
        )
    optuna_n_jobs = _render_optuna_n_jobs_input(
        "Optuna jobs paralelos",
        key="exp_best_section_controlled_optuna_n_jobs",
        default=5,
    )

    k_state_file_key = "exp_best_section_controlled_k_feature_file"
    if st.session_state.get(k_state_file_key) != str(selected_features_path):
        st.session_state["exp_best_section_controlled_k_min"] = min(
            10,
            max_available_features,
        )
        st.session_state["exp_best_section_controlled_k_max"] = max_available_features
        st.session_state["exp_best_section_controlled_k_step"] = min(
            5,
            max_available_features,
        )
        st.session_state[k_state_file_key] = str(selected_features_path)

    k1, k2, k3 = st.columns(3)
    with k1:
        k_min = st.number_input(
            "K mínimo",
            min_value=1,
            max_value=max_available_features,
            value=min(10, max_available_features),
            step=1,
            key="exp_best_section_controlled_k_min",
        )
    with k2:
        k_max = st.number_input(
            "K máximo",
            min_value=1,
            max_value=max_available_features,
            value=max_available_features,
            step=1,
            key="exp_best_section_controlled_k_max",
        )
    with k3:
        k_step = st.number_input(
            "Paso K",
            min_value=1,
            max_value=max_available_features,
            value=min(5, max_available_features),
            step=1,
            key="exp_best_section_controlled_k_step",
        )

    reuse_checkpoints = st.checkbox(
        "Reutilizar checkpoints compatibles por tramo",
        value=True,
        key="exp_best_section_controlled_reuse_checkpoints",
        help=(
            "Si ya existe una corrida compatible por tramo, se carga o reanuda. "
            "Desactívalo para recalcular todo desde cero."
        ),
    )
    search_space = _default_controlled_comparison_search_space()
    with st.expander("Rangos equivalentes a Comparación controlada", expanded=False):
        st.json(search_space)

    k_grid_by_set = {
        "Base": _k_grid_values(
            k_min=int(k_min),
            k_max=int(k_max),
            k_step=int(k_step),
            feature_count=len(base_schema_cols),
        ),
        "Cluster": _k_grid_values(
            k_min=int(k_min),
            k_max=int(k_max),
            k_step=int(k_step),
            feature_count=len(cluster_schema_cols),
        ),
        "Base + Cluster": _k_grid_values(
            k_min=int(k_min),
            k_max=int(k_max),
            k_step=int(k_step),
            feature_count=len(all_schema_cols),
        ),
    }
    st.caption(
        "Variables detectadas: "
        f"Base={len(base_schema_cols)} | Cluster={len(cluster_schema_cols)} | "
        f"Base + Cluster={len(all_schema_cols)}. "
        "Grilla K: "
        f"Base={k_grid_by_set['Base']} | "
        f"Cluster={k_grid_by_set['Cluster']} | "
        f"Base + Cluster={k_grid_by_set['Base + Cluster']}."
    )

    if st.button(
        "Iniciar barrido controlado por tramo",
        key="exp_best_section_controlled_run",
    ):
        if not selected_models:
            st.error("Seleccione al menos un modelo para ejecutar el barrido.")
            return
        if not threshold_protocols:
            st.error("Seleccione al menos un protocolo de evaluación.")
            return
        if not calibration_methods:
            st.error("Seleccione al menos un calibrador.")
            return
        if int(k_min) > int(k_max):
            st.error("K mínimo no puede ser mayor que K máximo.")
            return

        accidents_df_for_tramo = _load_accidents_for_event(selected_event_path)
        if accidents_df_for_tramo is None or accidents_df_for_tramo.empty:
            st.error("No se pudieron cargar accidentes procesados para el evento seleccionado.")
            return

        try:
            con = duckdb.connect(str(selected_features_path), read_only=True)
            table_rows = con.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in table_rows]
            table_name = _pick_duckdb_table(
                tables,
                ["flow_features", "features", "cluster_features"],
            )
            if not table_name:
                st.error("La base de datos de features está vacía.")
                return
            table_ref = _duckdb_quote_identifier(table_name)
            cols_info = con.execute(f"DESCRIBE {table_ref}").fetchall()
            columns = {row[0] for row in cols_info}
            if {"portico_last", "portico_next"}.issubset(columns):
                segment_cols = ("portico_last", "portico_next")
            elif {"portico_inicio", "portico_fin"}.issubset(columns):
                segment_cols = ("portico_inicio", "portico_fin")
            else:
                st.error(
                    "El archivo de features no contiene columnas de tramo "
                    "(portico_last/portico_next o portico_inicio/portico_fin)."
                )
                return
            select_parts = [
                f"{_duckdb_quote_identifier(segment_cols[0])} AS portico_last",
                f"{_duckdb_quote_identifier(segment_cols[1])} AS portico_next",
            ]
            if "eje" in columns:
                select_parts.append(f"{_duckdb_quote_identifier('eje')} AS eje")
            if "calzada" in columns:
                select_parts.append(f"{_duckdb_quote_identifier('calzada')} AS calzada")
            clauses = [
                f"{_duckdb_quote_identifier(segment_cols[0])} IS NOT NULL",
                f"{_duckdb_quote_identifier(segment_cols[1])} IS NOT NULL",
            ]
            params: List[object] = []
            if "interval_start" in columns:
                interval_ref = _duckdb_quote_identifier("interval_start")
                if dataset_date_start is not None:
                    clauses.append(f"TRY_CAST({interval_ref} AS TIMESTAMP) >= ?")
                    params.append(pd.Timestamp(dataset_date_start))
                if dataset_date_end is not None:
                    clauses.append(f"TRY_CAST({interval_ref} AS TIMESTAMP) <= ?")
                    params.append(pd.Timestamp(dataset_date_end))
            segments_df = con.execute(
                "SELECT DISTINCT "
                + ", ".join(select_parts)
                + f" FROM {table_ref} WHERE "
                + " AND ".join(clauses),
                params,
            ).df()
        except Exception as exc:
            st.error(f"Error leyendo tramos desde features: {exc}")
            return
        finally:
            try:
                con.close()
            except Exception:
                pass

        if segments_df.empty:
            st.warning("No se encontraron tramos con datos en el rango seleccionado.")
            return
        segments_df = segments_df.copy()
        segments_df["portico_last_raw"] = segments_df["portico_last"].astype(str).str.strip()
        segments_df["portico_next_raw"] = segments_df["portico_next"].astype(str).str.strip()
        segments_df["portico_last_norm"] = _normalize_portico_series(
            segments_df["portico_last_raw"]
        )
        segments_df["portico_next_norm"] = _normalize_portico_series(
            segments_df["portico_next_raw"]
        )
        segments_df = segments_df.dropna(
            subset=["portico_last_norm", "portico_next_norm"]
        )
        if segments_df.empty:
            st.warning("No hay tramos válidos en el archivo de features.")
            return
        if not {"eje", "calzada"}.issubset(segments_df.columns):
            try:
                porticos_df = load_porticos()
                seg_meta = get_portico_segments(porticos_df)
                if seg_meta is not None and not seg_meta.empty:
                    seg_meta = seg_meta.copy()
                    seg_meta["portico_last_norm"] = _normalize_portico_series(
                        seg_meta["portico_last"]
                    )
                    seg_meta["portico_next_norm"] = _normalize_portico_series(
                        seg_meta["portico_next"]
                    )
                    segments_df = segments_df.merge(
                        seg_meta[
                            [
                                "eje",
                                "calzada",
                                "portico_last_norm",
                                "portico_next_norm",
                            ]
                        ],
                        on=["portico_last_norm", "portico_next_norm"],
                        how="left",
                    )
            except Exception:
                pass
        segments_df = segments_df.drop_duplicates(
            subset=[
                col
                for col in ["eje", "calzada", "portico_last_raw", "portico_next_raw"]
                if col in segments_df.columns
            ]
        ).reset_index(drop=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_meta = {
            "run_id": run_id,
            "protocol_family": "Controlled comparison",
            "dataset_name": selected_event,
            "features_name": selected_features,
            "dataset_date_start": (
                None
                if dataset_date_start is None
                else str(pd.Timestamp(dataset_date_start))
            ),
            "dataset_date_end": (
                None
                if dataset_date_end is None
                else str(pd.Timestamp(dataset_date_end))
            ),
            "random_state": int(random_state),
            "objective_metric": objective_metric,
            "objective_label": objective_label,
            "threshold_protocols": list(threshold_protocols),
            "threshold_objective": threshold_objective,
            "threshold_objective_label": threshold_objective_label,
            "calibration_methods": list(calibration_methods),
            "far_target": float(far_target),
            "alerts_per_day": float(alerts_per_day),
            "fn_cost": float(fn_cost),
            "fp_cost": float(fp_cost),
            "robust_folds": int(robust_folds),
            "selected_models": list(selected_models),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "k_step": int(k_step),
            "k_grid_by_set": k_grid_by_set,
            "n_trials": int(n_trials),
            "timeout": int(timeout),
            "optuna_n_jobs": int(optuna_n_jobs),
            "parallel_jobs": int(parallel_jobs),
            "xgb_parallel_jobs": int(xgb_parallel_jobs),
            "search_space_config": search_space,
            "segment_count": int(len(segments_df)),
            "reuse_checkpoints": bool(reuse_checkpoints),
        }
        exp_db_path = _init_experiment_db("Best highway section", exp_meta)
        if exp_db_path:
            st.caption(f"DB live: {exp_db_path}")

        runner = ExperimentsRunner(random_state=int(random_state))
        checkpoint_root = RESULTS_DIR / "best_highway_section_controlled_runs"
        total_segments = int(len(segments_df))
        progress_bar = st.progress(0, text="Preparando barrido controlado por tramo...")
        summary_frames: List[pd.DataFrame] = []
        curve_frames: List[pd.DataFrame] = []
        grid_frames: List[pd.DataFrame] = []
        error_records: List[Dict[str, object]] = []

        for idx, row in enumerate(segments_df.itertuples(index=False), start=1):
            seg_last_raw = str(getattr(row, "portico_last_raw", "")).strip()
            seg_next_raw = str(getattr(row, "portico_next_raw", "")).strip()
            eje = getattr(row, "eje", None)
            calzada = getattr(row, "calzada", None)
            segment_label = (
                f"{'' if pd.isna(eje) else str(eje)} | "
                f"{'' if pd.isna(calzada) else str(calzada)} | "
                f"{seg_last_raw} -> {seg_next_raw}"
            ).strip(" |")
            segment_info = {
                "eje": None if pd.isna(eje) else str(eje),
                "calzada": None if pd.isna(calzada) else str(calzada),
                "portico_inicio": seg_last_raw,
                "portico_fin": seg_next_raw,
                "segment_label": segment_label,
            }
            tramo_tuple = (
                segment_info["eje"],
                segment_info["calzada"],
                seg_last_raw,
                seg_next_raw,
            )

            progress_bar.progress(
                int(((idx - 1) / max(1, total_segments)) * 100),
                text=f"Preparando tramo {idx}/{total_segments}: {segment_label}",
            )

            try:
                base_df = _prepare_controlled_comparison_base_df(
                    accidents_df_for_tramo=accidents_df_for_tramo,
                    selected_features_path=selected_features_path,
                    tramo_tuple=tramo_tuple,
                    date_start=dataset_date_start,
                    date_end=dataset_date_end,
                )
            except Exception as exc:
                error_payload = {
                    "experiment": "Best highway section",
                    "protocol_family": "Controlled comparison",
                    "sweep_run_id": run_id,
                    "dataset_name": selected_event,
                    "features_name": selected_features,
                    "segment_index": int(idx),
                    "segment_eje": segment_info["eje"],
                    "segment_calzada": segment_info["calzada"],
                    "segment_portico_last": seg_last_raw,
                    "segment_portico_next": seg_next_raw,
                    "segment_label": segment_label,
                    "objective_metric": objective_metric,
                    "objective_label": objective_label,
                    "status": "failed",
                    "error": str(exc),
                }
                error_records.append(error_payload)
                grid_frames.append(pd.DataFrame([error_payload]))
                _append_experiment_result(exp_db_path, error_payload)
                continue

            def _progress_callback(value: int, message: str) -> None:
                bounded = max(0, min(100, int(value)))
                overall = int((((idx - 1) + bounded / 100.0) / max(1, total_segments)) * 100)
                progress_bar.progress(
                    max(0, min(100, overall)),
                    text=f"Tramo {idx}/{total_segments}: {message}",
                )

            def _result_callback(payload: Dict[str, object]) -> None:
                payload_df = _enrich_best_section_controlled_frame(
                    pd.DataFrame([payload]),
                    run_id=run_id,
                    dataset_name=selected_event,
                    features_name=selected_features,
                    segment_index=idx,
                    segment_info=segment_info,
                )
                if not payload_df.empty:
                    _append_experiment_result(
                        exp_db_path,
                        payload_df.iloc[0].to_dict(),
                    )

            for calibration_method in calibration_methods:
                try:
                    payload = runner.run_controlled_comparison(
                        base_df,
                        event_path=selected_event_path,
                        features_path=selected_features_path,
                        segment_info=segment_info,
                        dataset_date_start=dataset_date_start,
                        dataset_date_end=dataset_date_end,
                        objective_metric=objective_metric,
                        threshold_protocols=list(threshold_protocols),
                        threshold_objective=threshold_objective,
                        calibration_method=str(calibration_method),
                        far_target=float(far_target),
                        alerts_per_day=float(alerts_per_day),
                        fn_cost=float(fn_cost),
                        fp_cost=float(fp_cost),
                        robust_folds=int(robust_folds),
                        test_size=float(test_size),
                        val_size=float(val_size),
                        k_min=int(k_min),
                        k_max=int(k_max),
                        k_step=int(k_step),
                        n_trials=int(n_trials),
                        timeout=int(timeout),
                        optuna_n_jobs=int(optuna_n_jobs),
                        parallel_jobs=int(parallel_jobs),
                        xgb_parallel_jobs=int(xgb_parallel_jobs),
                        selected_models=list(selected_models),
                        search_space_config=search_space,
                        progress_callback=_progress_callback,
                        result_callback=_result_callback,
                        checkpoint_root=checkpoint_root,
                        auto_resume=bool(reuse_checkpoints)
                        and len(calibration_methods) == 1,
                        start_fresh=not bool(reuse_checkpoints),
                    )
                except Exception as exc:
                    error_payload = {
                        "experiment": "Best highway section",
                        "protocol_family": "Controlled comparison",
                        "sweep_run_id": run_id,
                        "dataset_name": selected_event,
                        "features_name": selected_features,
                        "segment_index": int(idx),
                        "segment_eje": segment_info["eje"],
                        "segment_calzada": segment_info["calzada"],
                        "segment_portico_last": seg_last_raw,
                        "segment_portico_next": seg_next_raw,
                        "segment_label": segment_label,
                        "objective_metric": objective_metric,
                        "objective_label": objective_label,
                        "calibration_method": str(calibration_method),
                        "status": "failed",
                        "error": str(exc),
                    }
                    error_records.append(error_payload)
                    grid_frames.append(pd.DataFrame([error_payload]))
                    _append_experiment_result(exp_db_path, error_payload)
                    continue

                checkpoint_run_dir = payload.get("checkpoint_run_dir")
                summary_frames.append(
                    _enrich_best_section_controlled_frame(
                        payload.get("best_summary_df"),
                        run_id=run_id,
                        dataset_name=selected_event,
                        features_name=selected_features,
                        segment_index=idx,
                        segment_info=segment_info,
                        checkpoint_run_dir=checkpoint_run_dir,
                    )
                )
                curve_frames.append(
                    _enrich_best_section_controlled_frame(
                        payload.get("curves_df"),
                        run_id=run_id,
                        dataset_name=selected_event,
                        features_name=selected_features,
                        segment_index=idx,
                        segment_info=segment_info,
                        checkpoint_run_dir=checkpoint_run_dir,
                    )
                )
                grid_frames.append(
                    _enrich_best_section_controlled_frame(
                        payload.get("grid_results_df"),
                        run_id=run_id,
                        dataset_name=selected_event,
                        features_name=selected_features,
                        segment_index=idx,
                        segment_info=segment_info,
                        checkpoint_run_dir=checkpoint_run_dir,
                    )
                )

        progress_bar.empty()

        summary_df = (
            pd.concat([frame for frame in summary_frames if not frame.empty], ignore_index=True)
            if any(not frame.empty for frame in summary_frames)
            else pd.DataFrame()
        )
        curves_df = (
            pd.concat([frame for frame in curve_frames if not frame.empty], ignore_index=True)
            if any(not frame.empty for frame in curve_frames)
            else pd.DataFrame()
        )
        grid_results_df = (
            pd.concat([frame for frame in grid_frames if not frame.empty], ignore_index=True)
            if any(not frame.empty for frame in grid_frames)
            else pd.DataFrame()
        )

        best_row = None
        if not summary_df.empty and "val_objective_score" in summary_df.columns:
            summary_df = summary_df.copy()
            summary_df["val_objective_score"] = pd.to_numeric(
                summary_df["val_objective_score"],
                errors="coerce",
            )
            valid_summary = summary_df.copy()
            if "status" in valid_summary.columns:
                valid_summary = valid_summary[
                    valid_summary["status"].astype(str).str.lower() == "completed"
                ]
            valid_summary = valid_summary.dropna(subset=["val_objective_score"])
            summary_df["is_best"] = False
            summary_df["is_best_segment"] = False
            if not valid_summary.empty:
                best_idx = valid_summary["val_objective_score"].idxmax()
                summary_df.loc[best_idx, "is_best"] = True
                best_row = summary_df.loc[best_idx]
                if "segment_label" in valid_summary.columns:
                    segment_best_idx = valid_summary.groupby(
                        "segment_label",
                        dropna=False,
                    )["val_objective_score"].idxmax()
                    summary_df.loc[segment_best_idx, "is_best_segment"] = True
                rank_df = valid_summary[["val_objective_score"]].copy()
                rank_df["rank_global"] = (
                    rank_df["val_objective_score"]
                    .rank(method="first", ascending=False)
                    .astype(int)
                )
                summary_df.loc[rank_df.index, "rank_global"] = rank_df["rank_global"]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = RESULTS_DIR / f"best_highway_section_controlled_summary_{stamp}.csv"
        curves_path = RESULTS_DIR / f"best_highway_section_controlled_curves_{stamp}.csv"
        detail_path = RESULTS_DIR / f"best_highway_section_controlled_grid_{stamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        curves_df.to_csv(curves_path, index=False)
        grid_results_df.to_csv(detail_path, index=False)

        st.session_state["exp_best_section_controlled_last_results"] = {
            "summary_path": str(summary_path),
            "curves_path": str(curves_path),
            "detail_path": str(detail_path),
            "summary_df": summary_df,
            "curves_df": curves_df,
            "grid_results_df": grid_results_df,
            "run_id": run_id,
        }

        if error_records:
            st.warning(
                f"{len(error_records)} tramo(s) no pudieron evaluarse; quedaron registrados en la tabla completa."
            )
        if best_row is not None:
            best_payload = dict(best_row)
            _append_experiment_best(exp_db_path, best_payload)
            st.success(
                "Mejor tramo por validación "
                f"{objective_label}: {best_payload.get('segment_label', '?')} | "
                f"{best_payload.get('model_name', '-')} | "
                f"{best_payload.get('feature_set', '-')} | "
                f"{best_payload.get('balance_mode', '-')} | "
                f"{best_payload.get('threshold_protocol', '-')} | "
                f"K={best_payload.get('k_optimo', '?')}"
            )
        elif summary_df.empty:
            st.warning("No se generaron resultados completados.")

        st.caption(f"Resumen guardado: {summary_path}")
        st.caption(f"Curvas guardadas: {curves_path}")
        st.caption(f"Detalle guardado: {detail_path}")

        if not summary_df.empty or not grid_results_df.empty:
            if "is_best_segment" in summary_df.columns:
                section_rank = summary_df[
                    summary_df["is_best_segment"].astype(str).str.lower().isin(
                        {"true", "1"}
                    )
                ].copy()
                if not section_rank.empty:
                    section_rank = section_rank.sort_values(
                        "val_objective_score",
                        ascending=False,
                    )
                    st.markdown("**Mejor combinación por tramo**")
                    preferred_cols = [
                        "rank_global",
                        "segment_label",
                        "model_name",
                        "feature_set",
                        "balance_mode",
                        "threshold_protocol",
                        "k_optimo",
                        "val_objective_score",
                        "test_objective_score",
                        "best_test_balanced_f1",
                        "best_test_pr_auc",
                        "best_test_false_alarms_per_day",
                    ]
                    st.dataframe(
                        section_rank[
                            [col for col in preferred_cols if col in section_rank.columns]
                        ],
                        width="stretch",
                    )
            _render_controlled_comparison_results_panel(
                summary_df,
                curves_df,
                grid_results_df=grid_results_df,
                key_prefix=f"best_section_controlled_{stamp}",
            )


def _render_best_highway_section_k_experiment() -> None:
    st.subheader("Best mix Highway section & K")
    st.caption(
        "Recorre todos los tramos con datos, aplica seleccion de features, "
        "Optuna, SMOTE y entrenamiento para Base y Base + Cluster, "
        "iterando sobre distintos K."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No hay archivos de eventos (accidents) en Datos.")
        return
    event_names = [p.name for p in event_files]
    selected_event = st.selectbox(
        "Archivo de Eventos", event_names, key="exp_best_section_k_event_file"
    )

    feature_files = _list_flow_feature_files()
    if not feature_files:
        st.warning("No hay archivos de features en Resultados.")
        return
    feature_names = [p.name for p in feature_files]
    selected_features = st.selectbox(
        "Archivo de Features (Flow + Cluster)",
        feature_names,
        key="exp_best_section_k_feature_file",
    )

    objective_options = _optuna_objective_options(
        [
            "f1",
            "roc_auc",
            "accuracy",
            "recall",
            "precision",
            "fnr",
            "far_sens",
            "mcc",
            "brier_score",
        ]
    )
    objective_label = st.selectbox(
        "Metrica objetivo (mejor mix)",
        list(objective_options.keys()),
        key="exp_best_section_k_objective_metric",
    )
    objective_cfg = objective_options.get(
        objective_label, {"key": "f1", "direction": "maximize"}
    )
    objective_key = objective_cfg["key"]
    objective_direction = objective_cfg["direction"]

    st.markdown("**Feature selection (K)**")
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        k_min = st.number_input(
            "K Min",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            key="exp_best_section_k_min",
        )
    with col_k2:
        k_max = st.number_input(
            "K Max",
            min_value=1,
            max_value=200,
            value=50,
            step=1,
            key="exp_best_section_k_max",
        )
    with col_k3:
        k_step = st.number_input(
            "Paso K",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="exp_best_section_k_step",
        )

    st.markdown("**Configuracion del modelo**")
    model_choice = st.selectbox(
        "Modelo para Experimento",
        ["Random Forest", "XGBoost", "SVM"],
        key="exp_best_section_k_model_choice",
    )

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        n_trials = st.number_input(
            "Optuna Trials por tramo",
            min_value=5,
            value=30,
            step=5,
            key="exp_best_section_k_n_trials",
        )
    with col_n2:
        timeout = st.number_input(
            "Optuna Timeout (seg) por tramo",
            min_value=10,
            value=3600,
            step=10,
            key="exp_best_section_k_timeout",
        )
    optuna_n_jobs = _render_optuna_n_jobs_input(
        "Optuna jobs paralelos",
        key="exp_best_section_k_optuna_n_jobs",
        default=1,
    )

    far_target = 0.2
    threshold_strategy = "optuna"
    threshold_strategy_label = "Optimizar threshold"
    with st.expander("Configuracion avanzada (parametros y rangos)"):
        st.markdown("**Split de datos**")
        c_split1, c_split2 = st.columns(2)
        with c_split1:
            val_size = st.slider(
                "Validation Size (sobre train)",
                0.1,
                0.9,
                0.2,
                0.05,
                key="exp_best_section_k_val_size",
            )
        with c_split2:
            test_size = st.slider(
                "Test Size (sobre total)",
                0.1,
                0.9,
                0.2,
                0.05,
                key="exp_best_section_k_test_size",
            )
        st.markdown("**Calibracion de umbral**")
        threshold_options = {
            "Optimizar threshold": "optuna",
            "Calibrar por FAR": "far",
        }
        threshold_strategy = _option_value_from_state(
            threshold_options,
            "exp_best_section_k_threshold_strategy",
            default_label="Optimizar threshold",
        )
        threshold_visibility = _threshold_field_visibility_for_strategy(
            threshold_strategy
        )
        far_target = float(
            _render_conditional_slider(
                "FAR target",
                visible=threshold_visibility["far_target"],
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="exp_best_section_k_far_target",
            )
        )
        threshold_strategy_label = st.selectbox(
            "Estrategia de umbral",
            list(threshold_options.keys()),
            key="exp_best_section_k_threshold_strategy",
        )
        threshold_strategy = threshold_options[threshold_strategy_label]
        calibration_methods = _calibration_method_multiselect(
            "Calibración",
            key="exp_best_section_k_calibration_methods",
            default_methods=["sigmoid", "isotonic"],
        )

        st.markdown("**Rango SMOTE**")
        c_smote1, c_smote2 = st.columns(2)
        with c_smote1:
            smote_k_min = st.number_input(
                "K Neighbors Min",
                1,
                20,
                1,
                key="exp_best_section_k_smote_k_min",
            )
            smote_k_max = st.number_input(
                "K Neighbors Max",
                1,
                20,
                10,
                key="exp_best_section_k_smote_k_max",
            )
        with c_smote2:
            smote_str_min = st.slider(
                "Sampling Strategy Min",
                0.1,
                1.0,
                0.1,
                0.1,
                key="exp_best_section_k_smote_str_min",
            )
            smote_str_max = st.slider(
                "Sampling Strategy Max",
                0.1,
                1.0,
                1.0,
                0.1,
                key="exp_best_section_k_smote_str_max",
            )

        st.markdown(f"**Rangos para {model_choice}**")
        model_ranges = {}
        if model_choice == "Random Forest":
            c_rf1, c_rf2 = st.columns(2)
            with c_rf1:
                rf_ne_min = st.number_input(
                    "N Estimators Min",
                    10,
                    1000,
                    50,
                    step=10,
                    key="exp_best_section_k_rf_ne_min",
                )
                rf_ne_max = st.number_input(
                    "N Estimators Max",
                    10,
                    1000,
                    300,
                    step=10,
                    key="exp_best_section_k_rf_ne_max",
                )
            with c_rf2:
                rf_md_min = st.number_input(
                    "Max Depth Min",
                    1,
                    50,
                    3,
                    key="exp_best_section_k_rf_md_min",
                )
                rf_md_max = st.number_input(
                    "Max Depth Max",
                    1,
                    50,
                    15,
                    key="exp_best_section_k_rf_md_max",
                )
            model_ranges = {
                "n_estimators": {"min": rf_ne_min, "max": rf_ne_max},
                "max_depth": {"min": rf_md_min, "max": rf_md_max},
            }
        elif model_choice == "XGBoost":
            c_xgb1, c_xgb2 = st.columns(2)
            with c_xgb1:
                xgb_ne_min = st.number_input(
                    "N Estimators Min",
                    10,
                    1000,
                    50,
                    step=10,
                    key="exp_best_section_k_xgb_ne_min",
                )
                xgb_ne_max = st.number_input(
                    "N Estimators Max",
                    10,
                    1000,
                    300,
                    step=10,
                    key="exp_best_section_k_xgb_ne_max",
                )
                xgb_lr_min = st.number_input(
                    "Learning Rate Min",
                    0.001,
                    1.0,
                    0.01,
                    format="%.3f",
                    key="exp_best_section_k_xgb_lr_min",
                )
                xgb_lr_max = st.number_input(
                    "Learning Rate Max",
                    0.001,
                    1.0,
                    0.3,
                    format="%.3f",
                    key="exp_best_section_k_xgb_lr_max",
                )
            with c_xgb2:
                xgb_md_min = st.number_input(
                    "Max Depth Min",
                    1,
                    50,
                    3,
                    key="exp_best_section_k_xgb_md_min",
                )
                xgb_md_max = st.number_input(
                    "Max Depth Max",
                    1,
                    50,
                    15,
                    key="exp_best_section_k_xgb_md_max",
                )
                xgb_sub_min = st.slider(
                    "Subsample Min",
                    0.1,
                    1.0,
                    0.5,
                    0.1,
                    key="exp_best_section_k_xgb_sub_min",
                )
                xgb_sub_max = st.slider(
                    "Subsample Max",
                    0.1,
                    1.0,
                    1.0,
                    0.1,
                    key="exp_best_section_k_xgb_sub_max",
                )
                xgb_col_min = st.slider(
                    "Colsample ByTree Min",
                    0.1,
                    1.0,
                    0.5,
                    0.1,
                    key="exp_best_section_k_xgb_col_min",
                )
                xgb_col_max = st.slider(
                    "Colsample ByTree Max",
                    0.1,
                    1.0,
                    1.0,
                    0.1,
                    key="exp_best_section_k_xgb_col_max",
                )
            model_ranges = {
                "n_estimators": {"min": xgb_ne_min, "max": xgb_ne_max},
                "max_depth": {"min": xgb_md_min, "max": xgb_md_max},
                "learning_rate": {"min": xgb_lr_min, "max": xgb_lr_max},
                "subsample": {"min": xgb_sub_min, "max": xgb_sub_max},
                "colsample_bytree": {"min": xgb_col_min, "max": xgb_col_max},
            }
        elif model_choice == "SVM":
            c_svm1, c_svm2 = st.columns(2)
            with c_svm1:
                svm_c_min = st.number_input(
                    "C Min",
                    0.01,
                    1000.0,
                    0.1,
                    format="%.2f",
                    key="exp_best_section_k_svm_c_min",
                )
            with c_svm2:
                svm_c_max = st.number_input(
                    "C Max",
                    0.01,
                    1000.0,
                    50.0,
                    format="%.2f",
                    key="exp_best_section_k_svm_c_max",
                )
            model_ranges = {"C": {"min": svm_c_min, "max": svm_c_max}}

    if st.button("Iniciar experimento", key="exp_best_section_k_run"):
        if not calibration_methods:
            st.error("Seleccione al menos un calibrador.")
            return
        if int(k_min) > int(k_max):
            st.error("K Min no puede ser mayor que K Max.")
            return
        if int(k_step) <= 0:
            st.error("Paso K debe ser mayor que 0.")
            return

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_db_path = _init_experiment_db(
            "Best mix Highway section & K",
            {
                "run_id": run_id,
                "dataset_name": selected_event,
                "features_name": selected_features,
                "model_choice": model_choice,
                "objective_label": objective_label,
                "objective_metric": objective_key,
                "objective_direction": objective_direction,
                "k_min": int(k_min),
                "k_max": int(k_max),
                "k_step": int(k_step),
                "far_target": float(far_target),
                "threshold_strategy": threshold_strategy,
                "threshold_strategy_label": threshold_strategy_label,
                "calibration_methods": list(calibration_methods),
                "val_size": float(val_size),
                "test_size": float(test_size),
            },
        )
        if exp_db_path:
            st.caption(f"DB live: {exp_db_path}")

        accidents_path = next(p for p in event_files if p.name == selected_event)
        features_path = next(p for p in feature_files if p.name == selected_features)

        try:
            raw_accidents_df = read_csv_with_progress(str(accidents_path))
        except Exception as exc:
            st.error(f"Error cargando accidentes: {exc}")
            return

        try:
            porticos_df = load_porticos()
            if porticos_df is None or porticos_df.empty:
                st.error("No se pudieron cargar los porticos (Porticos.csv).")
                return
        except Exception as exc:
            st.error(f"Error cargando porticos: {exc}")
            return

        try:
            accidents_df, excluded = process_accidentes_df(
                raw_accidents_df, porticos_df, return_excluded=True
            )
            if accidents_df.empty:
                st.warning(
                    "No quedaron accidentes validos tras el procesamiento."
                )
                return
            st.success(
                f"Accidentes procesados: {len(accidents_df)} (Excluidos: {len(excluded)})"
            )
        except Exception as exc:
            st.error(f"Error procesando accidentes: {exc}")
            return

        if features_path.suffix.lower() != ".duckdb":
            st.error("El archivo de features debe ser .duckdb.")
            return
        if duckdb is None:
            st.error("duckdb no esta instalado.")
            return

        con = None
        try:
            con = duckdb.connect(str(features_path), read_only=True)
            table_rows = con.execute("SHOW TABLES").fetchall()
            tables = [row[0] for row in table_rows]
            table_name = _pick_duckdb_table(tables, ["flow_features", "features"])
            if not table_name:
                st.error("La base de datos de features esta vacia.")
                return
            table_ref = _duckdb_quote_identifier(table_name)
            cols_info = con.execute(f"DESCRIBE {table_ref}").fetchall()
            columns = {row[0] for row in cols_info}
            segment_cols = None
            if {"portico_last", "portico_next"}.issubset(columns):
                segment_cols = ("portico_last", "portico_next")
            elif {"portico_inicio", "portico_fin"}.issubset(columns):
                segment_cols = ("portico_inicio", "portico_fin")
            if not segment_cols:
                st.error(
                    "El archivo de features no contiene columnas de tramo "
                    "(portico_last/portico_next o portico_inicio/portico_fin)."
                )
                return

            last_col, next_col = segment_cols
            last_ref = _duckdb_quote_identifier(last_col)
            next_ref = _duckdb_quote_identifier(next_col)
            segments_df = con.execute(
                f"SELECT DISTINCT {last_ref} AS portico_last, {next_ref} AS portico_next "
                f"FROM {table_ref} "
                f"WHERE {last_ref} IS NOT NULL AND {next_ref} IS NOT NULL"
            ).df()
        except Exception as exc:
            st.error(f"Error leyendo features: {exc}")
            return
        finally:
            if con is not None:
                con.close()

        if segments_df is None or segments_df.empty:
            st.warning("No se encontraron tramos en el archivo de features.")
            return

        segments_df = segments_df.copy()
        segments_df["portico_last_raw"] = segments_df["portico_last"].astype(str).str.strip()
        segments_df["portico_next_raw"] = segments_df["portico_next"].astype(str).str.strip()
        segments_df["portico_last"] = _normalize_portico_series(
            segments_df["portico_last_raw"]
        )
        segments_df["portico_next"] = _normalize_portico_series(
            segments_df["portico_next_raw"]
        )
        segments_df = segments_df.dropna(subset=["portico_last", "portico_next"])
        if segments_df.empty:
            st.warning("No hay tramos validos en el archivo de features.")
            return

        try:
            seg_meta = get_portico_segments(porticos_df)
            if seg_meta is not None and not seg_meta.empty:
                seg_meta = seg_meta.copy()
                seg_meta["portico_last"] = _normalize_portico_series(
                    seg_meta["portico_last"]
                )
                seg_meta["portico_next"] = _normalize_portico_series(
                    seg_meta["portico_next"]
                )
                segments_df = segments_df.merge(
                    seg_meta[["eje", "calzada", "portico_last", "portico_next"]],
                    on=["portico_last", "portico_next"],
                    how="left",
                )
        except Exception:
            pass

        acc_seg = accidents_df.copy()
        acc_seg["portico_last"] = _normalize_portico_series(
            acc_seg["ultimo_portico"]
        )
        acc_seg["portico_next"] = _normalize_portico_series(
            acc_seg["proximo_portico"]
        )
        acc_seg = acc_seg.dropna(
            subset=["portico_last", "portico_next", "accidente_time"]
        )
        acc_groups = {
            key: group.copy()
            for key, group in acc_seg.groupby(["portico_last", "portico_next"])
        }

        cluster_cols_available = _get_cluster_cols(
            pd.DataFrame(columns=list(columns))
        )
        has_cluster_available = bool(cluster_cols_available)

        search_space = {
            "smote": {
                "k_neighbors": {"min": smote_k_min, "max": smote_k_max},
                "sampling_strategy": {
                    "min": smote_str_min,
                    "max": smote_str_max,
                },
            },
            "model": model_ranges,
        }

        runner = ExperimentsRunner()
        results: List[Dict[str, object]] = []
        total_segments = len(segments_df)
        progress_bar = st.progress(0, text="Procesando tramos...")
        con = None
        table_ref = _duckdb_quote_identifier(table_name)
        seg_columns = set(columns)

        try:
            con = duckdb.connect(str(features_path), read_only=True)

            for idx, row in enumerate(segments_df.itertuples(index=False), start=1):
                seg_last = getattr(row, "portico_last", None)
                seg_next = getattr(row, "portico_next", None)
                seg_last_raw = getattr(row, "portico_last_raw", seg_last)
                seg_next_raw = getattr(row, "portico_next_raw", seg_next)
                eje = getattr(row, "eje", None)
                calzada = getattr(row, "calzada", None)

                payload_common = {
                    "experiment": "Best mix Highway section & K",
                    "type": "Base",
                    "run_id": run_id,
                    "dataset_name": selected_event,
                    "features_name": selected_features,
                    "segment_portico_last": seg_last,
                    "segment_portico_next": seg_next,
                    "segment_eje": eje,
                    "segment_calzada": calzada,
                    "segment_index": int(idx),
                    "objective_metric": objective_key,
                    "objective_label": objective_label,
                    "model_choice": model_choice,
                    "n_trials": int(n_trials),
                    "timeout": int(timeout),
                    "optuna_n_jobs": int(optuna_n_jobs),
                    "far_target": float(far_target),
                    "threshold_strategy": threshold_strategy,
                    "threshold_strategy_label": threshold_strategy_label,
                    "calibration_methods": list(calibration_methods),
                    "val_size": float(val_size),
                    "test_size": float(test_size),
                    "search_space_config": json.dumps(search_space),
                    "k_min": int(k_min),
                    "k_max": int(k_max),
                    "k_step": int(k_step),
                }

                progress_bar.progress(
                    int(idx / total_segments * 100),
                    text=f"Procesando tramo {idx}/{total_segments}",
                )

                accidents_segment = acc_groups.get((seg_last, seg_next))
                if accidents_segment is None or accidents_segment.empty:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "No hay accidentes en el tramo."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = "No hay accidentes en el tramo."
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                tramo_tuple = (eje, calzada, seg_last_raw, seg_next_raw)
                clauses, params, filter_ok = _build_tramo_duckdb_filters(
                    tramo_tuple, seg_columns
                )
                if not filter_ok:
                    payload_base = dict(payload_common)
                    payload_base["error"] = (
                        "No se pudo filtrar el tramo en el archivo de features."
                    )
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue
                try:
                    query = f"SELECT * FROM {table_ref}"
                    if clauses:
                        query += " WHERE " + " AND ".join(clauses)
                    segment_features = con.execute(query, params).df()
                except Exception as exc:
                    payload_base = dict(payload_common)
                    payload_base["error"] = (
                        f"Error cargando features del tramo: {exc}"
                    )
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if segment_features is None or segment_features.empty:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "No hay features para el tramo."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if segment_cols != ("portico_last", "portico_next"):
                    segment_features = segment_features.rename(
                        columns={
                            segment_cols[0]: "portico_last",
                            segment_cols[1]: "portico_next",
                        }
                    )

                if "interval_start" not in segment_features.columns:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "Las features no tienen interval_start."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                segment_features = segment_features.copy()
                segment_features["interval_start"] = pd.to_datetime(
                    segment_features["interval_start"], errors="coerce"
                )

                segment_base_df = add_accident_target(
                    segment_features, accidents_segment
                )
                if segment_base_df.empty:
                    payload_base = dict(payload_common)
                    payload_base["error"] = "Dataset vacio tras merge."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if test_size <= 0 or test_size >= 1:
                    st.error("Test size debe estar entre 0 y 1.")
                    progress_bar.empty()
                    return
                if val_size <= 0 or val_size >= 1:
                    st.error("Validation size debe estar entre 0 y 1.")
                    progress_bar.empty()
                    return
                val_ratio = float(val_size)

                try:
                    train_df, test_df = _temporal_train_test_split(
                        segment_base_df, test_size=float(test_size)
                    )
                    train_opt_df, val_df = _temporal_train_test_split(
                        train_df, test_size=float(val_ratio)
                    )
                except Exception as exc:
                    payload_base = dict(payload_common)
                    payload_base["error"] = f"Split fallo: {exc}"
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                if (
                    train_df.empty
                    or val_df.empty
                    or test_df.empty
                    or train_df["target"].nunique() < 2
                    or val_df["target"].nunique() < 2
                    or test_df["target"].nunique() < 2
                ):
                    payload_base = dict(payload_common)
                    payload_base["error"] = "Split sin clases suficientes."
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = payload_base["error"]
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                all_feature_cols = _get_feature_cols(segment_base_df)
                cluster_cols = _get_cluster_cols(segment_base_df)
                base_cols = [c for c in all_feature_cols if c not in cluster_cols]

                cluster_set = set(cluster_cols)
                combined_ordered: List[str] = []
                base_ordered_from_combined: List[str] = []
                importance_error = None
                if not all_feature_cols:
                    importance_error = "No hay variables numericas para entrenar."
                else:
                    try:
                        combined_importance_df = (
                            runner.calculate_feature_importance(
                                segment_base_df, all_feature_cols
                            )
                        )
                        combined_ordered = combined_importance_df[
                            "variable"
                        ].tolist()
                        base_ordered_from_combined = [
                            col for col in combined_ordered if col in base_cols
                        ]
                        if not combined_ordered:
                            importance_error = (
                                "No hay variables numericas para entrenar."
                            )
                    except Exception as exc:
                        importance_error = f"Feature selection fallo: {exc}"

                def _run_dataset(
                    payload_seed: Dict[str, object],
                    *,
                    dataset_type: str,
                    candidate_cols: List[str],
                    selected_cols_override: Optional[List[str]] = None,
                ) -> List[Dict[str, object]]:
                    payloads: List[Dict[str, object]] = []
                    for calibration_method in calibration_methods:
                        payload = dict(payload_seed)
                        payload["type"] = dataset_type
                        payload["feature_selection_total"] = int(len(candidate_cols))
                        payload["calibration_method"] = str(calibration_method)
                        if not candidate_cols:
                            payload["error"] = "No hay variables numericas para entrenar."
                            payloads.append(payload)
                            continue
                        if not selected_cols_override:
                            payload["error"] = (
                                "No hay ranking de importancia para seleccionar "
                                "variables."
                            )
                            payloads.append(payload)
                            continue
                        selected_cols = [
                            col
                            for col in selected_cols_override
                            if col in candidate_cols
                        ]
                        if not selected_cols:
                            payload["error"] = (
                                "No hay variables numericas para entrenar."
                            )
                            payloads.append(payload)
                            continue

                        payload["feature_selection_selected"] = int(len(selected_cols))

                        try:
                            result = runner.run_optimization_loop(
                                train_df=train_opt_df,
                                val_df=val_df,
                                test_df=test_df,
                                feature_cols=selected_cols,
                                model_choice=model_choice,
                                n_trials=int(n_trials),
                                timeout=int(timeout),
                                optuna_n_jobs=int(optuna_n_jobs),
                                far_target=float(far_target),
                                search_space_config=search_space,
                                objective_key=objective_key,
                                objective_direction=objective_direction,
                                threshold_strategy=threshold_strategy,
                                calibration_method=str(calibration_method),
                            )
                            payload.update(result)
                        except Exception as exc:
                            payload["error"] = f"Error en Optuna: {exc}"
                        payloads.append(payload)
                    return payloads

                if importance_error:
                    payload_base = dict(payload_common)
                    payload_base["type"] = "Base"
                    payload_base["feature_selection_total"] = int(len(base_cols))
                    payload_base["error"] = importance_error
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["feature_selection_total"] = int(
                            len(all_feature_cols)
                        )
                        payload_cluster["error"] = importance_error
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                total_available = len(combined_ordered)
                k_values = list(
                    range(int(k_min), int(k_max) + 1, int(k_step))
                )
                if k_values and k_values[-1] != int(k_max):
                    k_values.append(int(k_max))
                if not k_values:
                    k_values = [int(k_min)]
                k_values = [
                    k for k in k_values if k > 0 and k <= total_available
                ]
                if not k_values:
                    err_msg = (
                        "Rango K excede numero de variables disponibles."
                    )
                    payload_base = dict(payload_common)
                    payload_base["type"] = "Base"
                    payload_base["error"] = err_msg
                    results.append(payload_base)
                    _append_experiment_result(exp_db_path, payload_base)
                    if has_cluster_available:
                        payload_cluster = dict(payload_common)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = err_msg
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)
                    continue

                for k_val in k_values:
                    combined_selected_cols = combined_ordered[:k_val]
                    cluster_in_top_n = sum(
                        1
                        for col in combined_selected_cols
                        if col in cluster_set
                    )
                    base_target_n = k_val - cluster_in_top_n
                    base_target_n = min(
                        base_target_n, len(base_ordered_from_combined)
                    )

                    payload_common_k = dict(payload_common)
                    payload_common_k["k"] = int(k_val)

                    if base_target_n <= 0:
                        payload_base = dict(payload_common_k)
                        payload_base["type"] = "Base"
                        payload_base["feature_selection_total"] = int(len(base_cols))
                        payload_base["error"] = (
                            "K total sin variables base disponibles."
                        )
                    else:
                        base_selected_cols = base_ordered_from_combined[:base_target_n]
                        payload_base_list = _run_dataset(
                            payload_common_k,
                            dataset_type="Base",
                            candidate_cols=base_cols,
                            selected_cols_override=base_selected_cols,
                        )
                        payload_base = None
                    if payload_base is not None:
                        results.append(payload_base)
                        _append_experiment_result(exp_db_path, payload_base)
                    else:
                        for payload_item in payload_base_list:
                            results.append(payload_item)
                            _append_experiment_result(exp_db_path, payload_item)

                    if cluster_cols:
                        payload_cluster_list = _run_dataset(
                            payload_common_k,
                            dataset_type="Base + Cluster",
                            candidate_cols=all_feature_cols,
                            selected_cols_override=combined_selected_cols,
                        )
                        for payload_item in payload_cluster_list:
                            results.append(payload_item)
                            _append_experiment_result(exp_db_path, payload_item)
                    elif has_cluster_available:
                        payload_cluster = dict(payload_common_k)
                        payload_cluster["type"] = "Base + Cluster"
                        payload_cluster["error"] = (
                            "No hay columnas de cluster en el dataset."
                        )
                        results.append(payload_cluster)
                        _append_experiment_result(exp_db_path, payload_cluster)

        finally:
            if con is not None:
                con.close()

        progress_bar.empty()

        if not results:
            st.warning("No se generaron resultados.")
            return

        res_df = pd.DataFrame(results)
        metric_key = objective_key
        metric_direction = (
            "min" if objective_direction == "minimize" else "max"
        )
        if metric_key == "far_sens":
            if {"far", "sensitivity"}.issubset(res_df.columns):
                res_df = res_df.copy()
                res_df["far_sens"] = (
                    res_df["far"] - (res_df["sensitivity"] * 1e-3)
                )
            else:
                st.warning(
                    "No se encontro FAR/Sensibilidad para calcular la metrica."
                )
        valid_df = res_df.copy()
        if "error" in valid_df.columns:
            valid_df = valid_df[valid_df["error"].isna()]
        if metric_key in valid_df.columns:
            valid_df = valid_df.dropna(subset=[metric_key])
        best_row = None
        if not valid_df.empty and metric_key in valid_df.columns:
            if metric_direction == "min":
                best_row = valid_df.loc[valid_df[metric_key].idxmin()]
            else:
                best_row = valid_df.loc[valid_df[metric_key].idxmax()]

        res_df["is_best"] = False
        if best_row is not None:
            res_df.loc[best_row.name, "is_best"] = True

        if "type" in res_df.columns and metric_key in res_df.columns:
            for dtype, group in res_df.groupby("type"):
                group_ok = group.copy()
                if "error" in group_ok.columns:
                    group_ok = group_ok[group_ok["error"].isna()]
                group_ok = group_ok.dropna(subset=[metric_key])
                if group_ok.empty:
                    continue
                if metric_direction == "min":
                    best_idx = group_ok[metric_key].idxmin()
                else:
                    best_idx = group_ok[metric_key].idxmax()
                res_df.loc[best_idx, "is_best_type"] = True

        st.subheader("Resultados")
        st.dataframe(res_df, width="stretch")

        if best_row is not None:
            st.success(
                "Mejor mix segun "
                f"{objective_label}: "
                f"{best_row.get('segment_portico_last', '?')} -> {best_row.get('segment_portico_next', '?')} "
                f"| K={best_row.get('k', '?')} "
                f"({best_row.get('type', '-')})"
            )
            best_payload = dict(best_row)
            _append_experiment_best(exp_db_path, best_payload)

            cm = best_row.get("confusion_matrix")
            if isinstance(cm, list) and cm:
                cm_data = cm
                if len(cm) == 4 and not isinstance(cm[0], (list, tuple)):
                    tn, fp, fn, tp = cm
                    cm_data = [[tn, fp], [fn, tp]]
                cm_df = pd.DataFrame(
                    cm_data,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
                st.caption("Matriz de confusion (mejor mix)")
                st.dataframe(cm_df, width="stretch")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_path = RESULTS_DIR / f"best_highway_section_k_results_{stamp}.csv"
        res_df.to_csv(res_path, index=False)
        st.success(f"Resultados guardados en {res_path}")


def _render_controlled_comparison_experiment() -> None:
    _apply_pending_controlled_job_config()
    st.subheader("Comparación controlada")
    st.caption(
        "Compara los modelos seleccionados entre Random Forest, SVM y XGBoost "
        "sobre Base, Cluster y Base + Cluster, con y sin SMOTE, usando un "
        "único split temporal congelado."
    )

    event_files = _list_event_files()
    if not event_files:
        st.warning("No hay archivos de eventos (accidents) en Datos.")
        return
    feature_files = _list_flow_feature_files()
    if not feature_files:
        st.warning("No hay archivos de features en Resultados.")
        return

    event_names = [p.name for p in event_files]
    feature_names = [p.name for p in feature_files]
    selected_event = st.selectbox(
        "Archivo de Eventos",
        event_names,
        key="exp_controlled_event_file",
    )
    selected_features = st.selectbox(
        "Archivo de Features",
        feature_names,
        key="exp_controlled_feature_file",
    )

    selected_event_path = next(
        (p for p in event_files if p.name == selected_event),
        None,
    )
    selected_features_path = next(
        (p for p in feature_files if p.name == selected_features),
        None,
    )
    if selected_event_path is None or selected_features_path is None:
        st.error("No se pudieron resolver los archivos seleccionados.")
        return

    dataset_date_start, dataset_date_end, dataset_date_valid = (
        _render_controlled_feature_date_range_inputs(selected_features_path)
    )
    if not dataset_date_valid:
        return

    schema_df = pd.DataFrame()
    schema_error = None
    try:
        schema_df = _inspect_controlled_feature_schema(selected_features_path)
    except Exception as exc:
        schema_error = str(exc)

    if schema_error:
        st.error(f"No se pudo inspeccionar el archivo de features: {schema_error}")
        return

    all_schema_cols = _get_feature_cols(schema_df)
    cluster_schema_cols = _get_cluster_cols(schema_df)
    base_schema_cols = [
        col for col in all_schema_cols if col not in cluster_schema_cols
    ]
    if not cluster_schema_cols:
        st.error(
            "El archivo seleccionado no contiene variables de cluster. "
            "La comparación controlada requiere Base, Cluster y Base + Cluster."
        )
        return
    max_available_features = max(1, len(all_schema_cols))

    accidents_df_for_tramo = _load_accidents_for_event(selected_event_path)
    allowed_porticos = _load_porticos_from_feature_file(selected_features_path)
    tramo_tuple = _build_tramo_selector(
        accidents_df_for_tramo,
        date_start=dataset_date_start,
        date_end=dataset_date_end,
        allowed_porticos=allowed_porticos,
        key="exp_controlled_tramo_choice",
    )
    if not tramo_tuple:
        st.info("Seleccione un tramo específico para ejecutar la comparación.")
        return

    eje, calzada, p_start, p_end = tramo_tuple
    segment_info = {
        "eje": eje,
        "calzada": calzada,
        "portico_inicio": p_start,
        "portico_fin": p_end,
        "segment_label": f"{eje} | {calzada} | {p_start} -> {p_end}",
    }
    protocol_mode_label = st.radio(
        "Protocolo experimental",
        [
            "Comparación controlada",
            "Modelos por K",
            "Ablación cruzada con tuning congelado",
        ],
        horizontal=True,
        key="exp_controlled_protocol_mode",
        help=(
            "Comparación controlada calcula un ranking independiente por Base, "
            "Cluster y Base + Cluster. Modelos por K calcula un único ranking "
            "global como el tab Feature selection y barre K global. Ablación "
            "cruzada con tuning congelado tunea Base y Base + Cluster, cruza "
            "sus hiperparámetros modelo+SMOTE sobre ambos targets con el mismo K "
            "y recalibra threshold por target; úselo para aislar tuning, no para "
            "comparar Cluster-only."
        ),
    )
    use_modelos_k_protocol = protocol_mode_label == "Modelos por K"
    use_frozen_tuning_ablation = (
        protocol_mode_label == "Ablación cruzada con tuning congelado"
    )
    objective_options = _controlled_objective_options()
    threshold_objective_options = {
        "Recall@N alertas/día": "recall_at_alerts_per_day",
        "FAR": "far",
        "Balanced F1": "balanced_f1",
        "F1": "f1",
        "MCC": "mcc",
        "Costo operacional": "operational_cost",
    }
    protocol_options = {
        "Conservador": "conservative",
        "Robusto": "robust",
    }

    st.markdown("**Configuración general**")
    col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
    with col_cfg1:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1,
            key="exp_controlled_random_state",
        )
    with col_cfg2:
        n_trials = st.number_input(
            "Optuna trials",
            min_value=1,
            value=30,
            step=1,
            key="exp_controlled_n_trials",
        )
    with col_cfg3:
        timeout = st.number_input(
            "Optuna timeout (seg)",
            min_value=1,
            value=3600,
            step=10,
            key="exp_controlled_timeout",
        )
    with col_cfg4:
        objective_labels = list(objective_options.keys())
        objective_label = st.selectbox(
            "Métrica Optuna/ranking",
            objective_labels,
            index=objective_labels.index("Balanced F1"),
            key="exp_controlled_objective_metric",
            help=(
                "Define la métrica que Optuna maximiza para elegir hiperparámetros. "
                "Valores permitidos: PR-AUC, ROC-AUC, Balanced F1, F1, MCC, "
                "Brier, Recall@N alertas/día o Costo operacional. Default: Balanced F1. "
                "Usar F1 con eventos raros puede sobreajustar el threshold de validación."
            ),
        )
    objective_metric = objective_options.get(objective_label, "pr_auc")
    selected_models = st.multiselect(
        "Modelos a comparar",
        list(CONTROLLED_COMPARISON_MODELS),
        default=list(CONTROLLED_COMPARISON_MODELS),
        key="exp_controlled_selected_models",
        help=(
            "Selecciona el subconjunto de modelos que quieres ejecutar en la "
            "comparación controlada. Incluye Balanced Random Forest si quieres "
            "comparar submuestreo balanceado; si falta imbalanced-learn, esa "
            "combinación fallará y quedará registrada."
        ),
    )
    if not selected_models:
        st.warning("Seleccione al menos un modelo para la comparación controlada.")

    selected_protocol_labels = st.multiselect(
        "Protocolos de evaluación",
        list(protocol_options.keys()),
        default=list(protocol_options.keys()),
        key="exp_controlled_threshold_protocols",
        help=(
            "Permite elegir Conservador, Robusto o ambos. Conservador ajusta el "
            "threshold en la validación temporal y evalúa el mismo modelo en test. "
            "Robusto usa folds temporales OOF dentro de train+val y luego evalúa "
            "test una sola vez. Elegir ambos duplica aproximadamente el tiempo de "
            "ejecución y no debe mezclarse con resultados v2 antiguos."
        ),
    )
    threshold_protocols = [
        protocol_options[label]
        for label in selected_protocol_labels
        if label in protocol_options
    ] or ["conservative"]

    feature_selection_n_estimators = 200
    feature_selection_max_depth: Optional[int] = None
    feature_selection_n_jobs = -1
    if use_modelos_k_protocol:
        st.markdown("**Ranking global estilo Feature selection**")
        rank_col1, rank_col2, rank_col3 = st.columns(3)
        with rank_col1:
            feature_selection_n_estimators = int(
                st.number_input(
                    "FS n_estimators",
                    min_value=50,
                    value=200,
                    step=50,
                    key="exp_controlled_fs_n_estimators",
                    help=(
                        "Mismo control que Feature selection. Se calcula una sola "
                        "vez por corrida y luego se barre K global."
                    ),
                )
            )
        with rank_col2:
            feature_selection_max_depth_raw = int(
                st.number_input(
                    "FS max_depth (0 = sin limite)",
                    min_value=0,
                    value=0,
                    step=1,
                    key="exp_controlled_fs_max_depth",
                    help="Mismo significado que en el tab Feature selection.",
                )
            )
            feature_selection_max_depth = (
                None
                if feature_selection_max_depth_raw == 0
                else int(feature_selection_max_depth_raw)
            )
        with rank_col3:
            feature_selection_n_jobs = -1
            st.metric("FS random_state", int(random_state))
        st.caption(
            "Este modo usa RandomForestClassifier con class_weight='balanced', "
            "criterion='gini' y n_jobs=-1, igual que Feature selection. Optuna "
            "se ejecuta para cada K global y cada combinación."
        )

    st.markdown("**Threshold operacional**")
    col_thr1, col_thr2, col_thr3, col_thr4 = st.columns(4)
    with col_thr1:
        threshold_objective_label = st.selectbox(
            "Objetivo de threshold",
            list(threshold_objective_options.keys()),
            index=0,
            key="exp_controlled_threshold_objective",
            help=(
                "Define cómo se escoge el umbral operativo después de entrenar. "
                "Valores permitidos: FAR, Balanced F1, F1, MCC, Recall@N "
                "alertas/día o Costo operacional. Default: Recall@N alertas/día. "
                "No uses métricas puramente de fila si la operación exige limitar "
                "alertas diarias."
            ),
        )
    threshold_objective = threshold_objective_options.get(
        threshold_objective_label,
        "recall_at_alerts_per_day",
    )
    threshold_visibility = _threshold_field_visibility_for_objective(
        threshold_objective
    )
    with col_thr2:
        controlled_alerts_per_day = float(
            _render_conditional_number_input(
                "Alertas máximas por día",
                visible=threshold_visibility["alerts_per_day"],
                min_value=0.1,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="exp_controlled_alerts_per_day",
                help=(
                    "Presupuesto diario de alertas para Recall@N. Rango práctico: "
                    "0.1 a 50. Default: 5.0. Baja este valor si necesitas menos "
                    "revisiones humanas; un valor demasiado bajo puede esconder "
                    "accidentes detectables."
                ),
            )
        )
    with col_thr3:
        controlled_far_target = float(
            _render_conditional_slider(
                "FAR target",
                visible=threshold_visibility["far_target"],
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="exp_controlled_far_target",
                help=(
                    "Falsa alarma máxima aceptada cuando el objetivo de threshold es "
                    "FAR. Rango: 0.00 a 0.50. Default: 0.20. No lo interpretes como "
                    "probabilidad calibrada; controla tasa de falsos positivos en "
                    "validación."
                ),
            )
        )
    with col_thr4:
        calibration_methods = _calibration_method_multiselect(
            "Calibración",
            key="exp_controlled_calibration_method",
            default_methods=["sigmoid", "isotonic"],
            help=(
                "Transforma scores antes de elegir threshold. Default: comparar "
                "Platt scaling (sigmoid) e Isotonic. Sin calibración queda "
                "disponible para comparación directa."
            ),
        )
    col_cost1, col_cost2, col_cost3 = st.columns(3)
    with col_cost1:
        controlled_fn_cost = float(
            _render_conditional_number_input(
                "Costo FN",
                visible=threshold_visibility["fn_cost"],
                min_value=0.0,
                value=10.0,
                step=1.0,
                key="exp_controlled_fn_cost",
                help=(
                    "Costo unitario de no alertar un accidente real. Default: 10.0. "
                    "Aumentarlo empuja thresholds más sensibles; valores extremos "
                    "pueden producir demasiadas falsas alarmas."
                ),
            )
        )
    with col_cost2:
        controlled_fp_cost = float(
            _render_conditional_number_input(
                "Costo FP",
                visible=threshold_visibility["fp_cost"],
                min_value=0.0,
                value=1.0,
                step=0.5,
                key="exp_controlled_fp_cost",
                help=(
                    "Costo unitario de una falsa alarma. Default: 1.0. Aumentarlo "
                    "endurece el threshold; valores altos pueden bajar mucho el recall."
                ),
            )
        )
    with col_cost3:
        robust_folds = st.number_input(
            "Folds robustos",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="exp_controlled_robust_folds",
            help=(
                "Número de folds temporales OOF usados por el protocolo robusto. "
                "Rango: 2 a 10. Default: 3. Más folds aumentan costo y pueden "
                "fallar si cada ventana queda con muy pocos accidentes."
            ),
        )
    threshold_objective = threshold_objective_options.get(
        threshold_objective_label,
        "recall_at_alerts_per_day",
    )

    col_split1, col_split2, col_split3, col_split4 = st.columns(4)
    with col_split1:
        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="exp_controlled_test_size",
        )
    with col_split2:
        val_size = st.slider(
            "Validation size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="exp_controlled_val_size",
        )
    max_parallel_input_jobs = _max_optuna_parallel_jobs()
    with col_split3:
        parallel_jobs = st.number_input(
            "Jobs paralelos RF/ranking",
            min_value=1,
            max_value=max_parallel_input_jobs,
            value=10,
            step=1,
            key="exp_controlled_parallel_jobs",
        )
    with col_split4:
        xgb_parallel_jobs = _render_model_n_jobs_input(
            "Jobs paralelos XGBoost",
            key="exp_controlled_xgb_parallel_jobs",
            default=1,
            shared_key="global_xgb_parallel_jobs",
        )

    optuna_n_jobs = _render_optuna_n_jobs_input(
        "Optuna jobs paralelos",
        key="exp_controlled_optuna_n_jobs",
        default=5,
    )
    st.caption(
        "La UI es la fuente de verdad: Random Forest usa `Jobs paralelos RF/ranking`, "
        "XGBoost usa `Jobs paralelos XGBoost`, y Optuna usa sólo sus jobs de trials."
    )

    k_state_file_key = "exp_controlled_k_feature_file"
    if st.session_state.get(k_state_file_key) != str(selected_features_path):
        st.session_state["exp_controlled_k_min"] = min(10, max_available_features)
        st.session_state["exp_controlled_k_max"] = max_available_features
        st.session_state["exp_controlled_k_step"] = min(5, max_available_features)
        st.session_state[k_state_file_key] = str(selected_features_path)
    else:
        for state_key, fallback in [
            ("exp_controlled_k_min", min(10, max_available_features)),
            ("exp_controlled_k_max", max_available_features),
            ("exp_controlled_k_step", min(5, max_available_features)),
        ]:
            try:
                current_value = int(st.session_state.get(state_key, fallback))
            except Exception:
                current_value = fallback
            st.session_state[state_key] = max(
                1,
                min(current_value, max_available_features),
            )

    st.markdown("**Grilla de K**")
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        k_min = st.number_input(
            "K mínimo",
            min_value=1,
            max_value=max_available_features,
            value=min(10, max_available_features),
            step=1,
            key="exp_controlled_k_min",
            help=(
                "Cantidad mínima de features a evaluar por conjunto. "
                f"Default: {min(10, max_available_features)}; rango 1 a "
                f"{max_available_features}. Si se elige demasiado bajo, los "
                "modelos pueden quedar subespecificados."
            ),
        )
    with col_k2:
        k_max = st.number_input(
            "K máximo",
            min_value=1,
            max_value=max_available_features,
            value=max_available_features,
            step=1,
            key="exp_controlled_k_max",
            help=(
                "Cantidad máxima de features disponibles en el archivo seleccionado. "
                f"Default y máximo: {max_available_features}. El runner lo recorta "
                "por Base, Cluster y Base + Cluster según las columnas reales."
            ),
        )
    with col_k3:
        k_step = st.number_input(
            "Paso K",
            min_value=1,
            max_value=max_available_features,
            value=min(5, max_available_features),
            step=1,
            key="exp_controlled_k_step",
            help=(
                "Incremento entre valores consecutivos de K. Default: "
                f"{min(5, max_available_features)}. Pasos muy grandes pueden saltar "
                "el mejor tamaño de feature set."
            ),
        )

    with st.expander("Rangos de optimización", expanded=False):
        st.markdown("**SMOTE**")
        col_smote1, col_smote2, col_smote3 = st.columns(3)
        with col_smote1:
            smote_k_min = st.number_input(
                "SMOTE k min",
                min_value=1,
                max_value=50,
                value=1,
                step=1,
                key="exp_controlled_smote_k_min",
            )
            smote_k_max = st.number_input(
                "SMOTE k max",
                min_value=1,
                max_value=50,
                value=15,
                step=1,
                key="exp_controlled_smote_k_max",
            )
            smote_k_step = st.number_input(
                "SMOTE k step",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key="exp_controlled_smote_k_step",
            )
        with col_smote2:
            smote_str_min = st.number_input(
                "Sampling strategy min",
                min_value=0.001,
                max_value=1.0,
                value=0.001,
                step=0.005,
                format="%.3f",
                key="exp_controlled_smote_str_min",
                help=(
                    "Razón mínima objetivo minoritaria/mayoritaria que SMOTE puede probar. "
                    "Default: 0.001; rango 0.001 a 1.0. Valores bajo la razón actual "
                    "se descartan internamente, y valores altos pueden crear demasiados sintéticos."
                ),
            )
            smote_str_max = st.number_input(
                "Sampling strategy max",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.005,
                format="%.3f",
                key="exp_controlled_smote_str_max",
                help=(
                    "Razón máxima objetivo minoritaria/mayoritaria para SMOTE. "
                    "Default: 0.100; rango 0.001 a 1.0. Subirlo aumenta recall potencial "
                    "pero también riesgo de sobre-muestreo y falsas alarmas."
                ),
            )
            smote_str_step = st.number_input(
                "Sampling strategy step",
                min_value=0.001,
                max_value=1.0,
                value=0.005,
                step=0.001,
                format="%.3f",
                key="exp_controlled_smote_str_step",
                help=(
                    "Incremento entre razones de SMOTE en la grilla. Default: 0.005. "
                    "Un paso demasiado fino aumenta trials equivalentes; uno muy grueso puede "
                    "omitir una zona estable de balance."
                ),
            )

        st.markdown("**Random Forest**")
        col_rf1, col_rf2, col_rf3 = st.columns(3)
        with col_rf1:
            rf_ne_min = st.number_input(
                "RF n_estimators min",
                min_value=10,
                max_value=2000,
                value=50,
                step=10,
                key="exp_controlled_rf_ne_min",
            )
            rf_ne_max = st.number_input(
                "RF n_estimators max",
                min_value=10,
                max_value=2000,
                value=300,
                step=10,
                key="exp_controlled_rf_ne_max",
            )
            rf_ne_step = st.number_input(
                "RF n_estimators step",
                min_value=1,
                max_value=500,
                value=25,
                step=1,
                key="exp_controlled_rf_ne_step",
            )
        with col_rf2:
            rf_depth_min = st.number_input(
                "RF max_depth min",
                min_value=1,
                max_value=100,
                value=3,
                step=1,
                key="exp_controlled_rf_depth_min",
            )
            rf_depth_max = st.number_input(
                "RF max_depth max",
                min_value=1,
                max_value=100,
                value=15,
                step=1,
                key="exp_controlled_rf_depth_max",
            )
            rf_depth_step = st.number_input(
                "RF max_depth step",
                min_value=1,
                max_value=25,
                value=1,
                step=1,
                key="exp_controlled_rf_depth_step",
            )
        with col_rf3:
            rf_split_min = st.number_input(
                "RF min_samples_split min",
                min_value=2,
                max_value=50,
                value=2,
                step=1,
                key="exp_controlled_rf_split_min",
            )
            rf_split_max = st.number_input(
                "RF min_samples_split max",
                min_value=2,
                max_value=50,
                value=10,
                step=1,
                key="exp_controlled_rf_split_max",
            )
            rf_split_step = st.number_input(
                "RF min_samples_split step",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key="exp_controlled_rf_split_step",
            )
        col_rf4, col_rf5 = st.columns(2)
        with col_rf4:
            rf_leaf_min = st.number_input(
                "RF min_samples_leaf min",
                min_value=1,
                max_value=50,
                value=1,
                step=1,
                key="exp_controlled_rf_leaf_min",
            )
            rf_leaf_max = st.number_input(
                "RF min_samples_leaf max",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
                key="exp_controlled_rf_leaf_max",
            )
            rf_leaf_step = st.number_input(
                "RF min_samples_leaf step",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key="exp_controlled_rf_leaf_step",
            )
        with col_rf5:
            rf_max_features_labels = st.multiselect(
                "RF max_features",
                ["sqrt", "log2", "None"],
                default=["sqrt", "log2", "None"],
                key="exp_controlled_rf_max_features",
            )

        st.markdown("**SVM**")
        col_svm1, col_svm2, col_svm3 = st.columns(3)
        with col_svm1:
            svm_c_min = st.number_input(
                "SVM C min",
                min_value=0.01,
                max_value=1000.0,
                value=0.1,
                step=0.1,
                format="%.2f",
                key="exp_controlled_svm_c_min",
            )
            svm_c_max = st.number_input(
                "SVM C max",
                min_value=0.01,
                max_value=1000.0,
                value=10.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_svm_c_max",
            )
            svm_c_step = st.number_input(
                "SVM C step",
                min_value=0.01,
                max_value=100.0,
                value=0.5,
                step=0.01,
                format="%.2f",
                key="exp_controlled_svm_c_step",
            )
        with col_svm2:
            svm_degree_min = st.number_input(
                "SVM degree min",
                min_value=2,
                max_value=10,
                value=2,
                step=1,
                key="exp_controlled_svm_degree_min",
            )
            svm_degree_max = st.number_input(
                "SVM degree max",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                key="exp_controlled_svm_degree_max",
            )
            svm_degree_step = st.number_input(
                "SVM degree step",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="exp_controlled_svm_degree_step",
            )
        with col_svm3:
            svm_coef0_min = st.number_input(
                "SVM coef0 min",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_svm_coef0_min",
            )
            svm_coef0_max = st.number_input(
                "SVM coef0 max",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_svm_coef0_max",
            )
            svm_coef0_step = st.number_input(
                "SVM coef0 step",
                min_value=0.05,
                max_value=5.0,
                value=0.2,
                step=0.05,
                format="%.2f",
                key="exp_controlled_svm_coef0_step",
            )
        col_svm4, col_svm5 = st.columns(2)
        with col_svm4:
            svm_kernels = st.multiselect(
                "SVM kernels",
                ["rbf", "linear", "poly", "sigmoid"],
                default=["rbf", "linear"],
                key="exp_controlled_svm_kernels",
            )
        with col_svm5:
            svm_gamma_raw = st.text_input(
                "SVM gamma (lista separada por comas)",
                value="scale,auto",
                key="exp_controlled_svm_gamma",
            )

        st.markdown("**XGBoost**")
        col_xgb1, col_xgb2, col_xgb3 = st.columns(3)
        with col_xgb1:
            xgb_ne_min = st.number_input(
                "XGB n_estimators min",
                min_value=10,
                max_value=5000,
                value=50,
                step=10,
                key="exp_controlled_xgb_ne_min",
            )
            xgb_ne_max = st.number_input(
                "XGB n_estimators max",
                min_value=10,
                max_value=5000,
                value=300,
                step=10,
                key="exp_controlled_xgb_ne_max",
            )
            xgb_ne_step = st.number_input(
                "XGB n_estimators step",
                min_value=1,
                max_value=1000,
                value=25,
                step=1,
                key="exp_controlled_xgb_ne_step",
            )
            xgb_lr_min = st.number_input(
                "XGB learning_rate min",
                min_value=0.001,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.3f",
                key="exp_controlled_xgb_lr_min",
            )
            xgb_lr_max = st.number_input(
                "XGB learning_rate max",
                min_value=0.001,
                max_value=1.0,
                value=0.30,
                step=0.001,
                format="%.3f",
                key="exp_controlled_xgb_lr_max",
            )
            xgb_lr_step = st.number_input(
                "XGB learning_rate step",
                min_value=0.001,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.3f",
                key="exp_controlled_xgb_lr_step",
            )
        with col_xgb2:
            xgb_depth_min = st.number_input(
                "XGB max_depth min",
                min_value=1,
                max_value=100,
                value=3,
                step=1,
                key="exp_controlled_xgb_depth_min",
            )
            xgb_depth_max = st.number_input(
                "XGB max_depth max",
                min_value=1,
                max_value=100,
                value=15,
                step=1,
                key="exp_controlled_xgb_depth_max",
            )
            xgb_depth_step = st.number_input(
                "XGB max_depth step",
                min_value=1,
                max_value=25,
                value=1,
                step=1,
                key="exp_controlled_xgb_depth_step",
            )
            xgb_sub_min = st.number_input(
                "XGB subsample min",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_sub_min",
            )
            xgb_sub_max = st.number_input(
                "XGB subsample max",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_sub_max",
            )
            xgb_sub_step = st.number_input(
                "XGB subsample step",
                min_value=0.05,
                max_value=1.0,
                value=0.1,
                step=0.05,
                format="%.2f",
                key="exp_controlled_xgb_sub_step",
            )
            xgb_col_min = st.number_input(
                "XGB colsample_bytree min",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_col_min",
            )
            xgb_col_max = st.number_input(
                "XGB colsample_bytree max",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_col_max",
            )
            xgb_col_step = st.number_input(
                "XGB colsample_bytree step",
                min_value=0.05,
                max_value=1.0,
                value=0.1,
                step=0.05,
                format="%.2f",
                key="exp_controlled_xgb_col_step",
            )
        with col_xgb3:
            xgb_child_min = st.number_input(
                "XGB min_child_weight min",
                min_value=1.0,
                max_value=100.0,
                value=1.0,
                step=1.0,
                format="%.1f",
                key="exp_controlled_xgb_child_min",
            )
            xgb_child_max = st.number_input(
                "XGB min_child_weight max",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                format="%.1f",
                key="exp_controlled_xgb_child_max",
            )
            xgb_child_step = st.number_input(
                "XGB min_child_weight step",
                min_value=0.5,
                max_value=50.0,
                value=1.0,
                step=0.5,
                format="%.1f",
                key="exp_controlled_xgb_child_step",
            )
            xgb_alpha_min = st.number_input(
                "XGB reg_alpha min",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_alpha_min",
            )
            xgb_alpha_max = st.number_input(
                "XGB reg_alpha max",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_alpha_max",
            )
            xgb_alpha_step = st.number_input(
                "XGB reg_alpha step",
                min_value=0.05,
                max_value=10.0,
                value=0.1,
                step=0.05,
                format="%.2f",
                key="exp_controlled_xgb_alpha_step",
            )
            xgb_lambda_min = st.number_input(
                "XGB reg_lambda min",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_lambda_min",
            )
            xgb_lambda_max = st.number_input(
                "XGB reg_lambda max",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_lambda_max",
            )
            xgb_lambda_step = st.number_input(
                "XGB reg_lambda step",
                min_value=0.05,
                max_value=20.0,
                value=0.1,
                step=0.05,
                format="%.2f",
                key="exp_controlled_xgb_lambda_step",
            )
            xgb_gamma_min = st.number_input(
                "XGB gamma min",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_gamma_min",
            )
            xgb_gamma_max = st.number_input(
                "XGB gamma max",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                key="exp_controlled_xgb_gamma_max",
            )
            xgb_gamma_step = st.number_input(
                "XGB gamma step",
                min_value=0.05,
                max_value=10.0,
                value=0.1,
                step=0.05,
                format="%.2f",
                key="exp_controlled_xgb_gamma_step",
            )

    rf_max_features = [
        None if str(value) == "None" else value
        for value in rf_max_features_labels
    ] or ["sqrt"]
    svm_gamma_values: List[object] = []
    for chunk in str(svm_gamma_raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            svm_gamma_values.append(float(token))
        except ValueError:
            svm_gamma_values.append(token)
    if not svm_gamma_values:
        svm_gamma_values = ["scale"]
    if not svm_kernels:
        svm_kernels = ["rbf", "linear"]

    search_space = {
        "smote": {
            "k_neighbors": {
                "min": int(smote_k_min),
                "max": int(smote_k_max),
                "step": int(smote_k_step),
            },
            "sampling_strategy": {
                "min": float(smote_str_min),
                "max": float(smote_str_max),
                "step": float(smote_str_step),
            },
        },
        "rf": {
            "n_estimators": {
                "min": int(rf_ne_min),
                "max": int(rf_ne_max),
                "step": int(rf_ne_step),
            },
            "max_depth": {
                "min": int(rf_depth_min),
                "max": int(rf_depth_max),
                "step": int(rf_depth_step),
            },
            "min_samples_split": {
                "min": int(rf_split_min),
                "max": int(rf_split_max),
                "step": int(rf_split_step),
            },
            "min_samples_leaf": {
                "min": int(rf_leaf_min),
                "max": int(rf_leaf_max),
                "step": int(rf_leaf_step),
            },
            "max_features": rf_max_features,
            "class_weight": [None, "balanced"],
        },
        "svm": {
            "C": {
                "min": float(svm_c_min),
                "max": float(svm_c_max),
                "step": float(svm_c_step),
            },
            "kernel": list(svm_kernels),
            "gamma": list(svm_gamma_values),
            "degree": {
                "min": int(svm_degree_min),
                "max": int(svm_degree_max),
                "step": int(svm_degree_step),
            },
            "coef0": {
                "min": float(svm_coef0_min),
                "max": float(svm_coef0_max),
                "step": float(svm_coef0_step),
            },
            "class_weight": [None, "balanced"],
        },
        "xgb": {
            "n_estimators": {
                "min": int(xgb_ne_min),
                "max": int(xgb_ne_max),
                "step": int(xgb_ne_step),
            },
            "max_depth": {
                "min": int(xgb_depth_min),
                "max": int(xgb_depth_max),
                "step": int(xgb_depth_step),
            },
            "learning_rate": {
                "min": float(xgb_lr_min),
                "max": float(xgb_lr_max),
                "step": float(xgb_lr_step),
            },
            "subsample": {
                "min": float(xgb_sub_min),
                "max": float(xgb_sub_max),
                "step": float(xgb_sub_step),
            },
            "colsample_bytree": {
                "min": float(xgb_col_min),
                "max": float(xgb_col_max),
                "step": float(xgb_col_step),
            },
            "min_child_weight": {
                "min": float(xgb_child_min),
                "max": float(xgb_child_max),
                "step": float(xgb_child_step),
            },
            "reg_alpha": {
                "min": float(xgb_alpha_min),
                "max": float(xgb_alpha_max),
                "step": float(xgb_alpha_step),
            },
            "reg_lambda": {
                "min": float(xgb_lambda_min),
                "max": float(xgb_lambda_max),
                "step": float(xgb_lambda_step),
            },
            "gamma": {
                "min": float(xgb_gamma_min),
                "max": float(xgb_gamma_max),
                "step": float(xgb_gamma_step),
            },
            "scale_pos_weight_multipliers": [0.5, 1.0, 2.0, 5.0, 10.0],
            "max_delta_step": [0.0, 1.0],
        },
        "balanced_rf": {
            "replacement": [False],
        },
    }

    checkpoint_mode = "Start fresh"
    checkpoint_preview: Optional[Dict[str, object]] = None
    checkpoint_root = RESULTS_DIR / "controlled_comparison_runs"

    st.caption(
        "Variables detectadas: "
        f"Base={len(base_schema_cols)} | "
        f"Cluster={len(cluster_schema_cols)} | "
        f"Base + Cluster={len(all_schema_cols)}"
    )
    if use_frozen_tuning_ablation:
        protocol_family = FROZEN_TUNING_ABLATION_PROTOCOL_FAMILY
    elif use_modelos_k_protocol:
        protocol_family = "modelos_por_k"
    else:
        protocol_family = "controlled_comparison"

    if use_modelos_k_protocol:
        k_grid_global = _k_grid_values(
            k_min=int(k_min),
            k_max=int(k_max),
            k_step=int(k_step),
            feature_count=len(all_schema_cols),
        )
        k_grid_by_set = {
            "Base": list(k_grid_global),
            "Cluster": list(k_grid_global),
            "Base + Cluster": list(k_grid_global),
        }
        st.caption(
            "Grilla efectiva de K global: "
            f"{k_grid_global}. Cada familia usa las variables de su grupo que "
            "aparezcan dentro del Top-K global."
        )
    elif use_frozen_tuning_ablation:
        k_grid_global = []
        common_k_grid = _k_grid_values(
            k_min=int(k_min),
            k_max=int(k_max),
            k_step=int(k_step),
            feature_count=len(base_schema_cols),
        )
        k_grid_by_set = {
            "Base": list(common_k_grid),
            "Base + Cluster": list(common_k_grid),
        }
        st.caption(
            "Grilla común de K para Base↔Base+Cluster: "
            f"{common_k_grid}. El máximo se limita por Base para que "
            "Base + Cluster no gane sólo por tener más columnas."
        )
        st.info(
            "Ablación cruzada: tuning congelado = modelo+SMOTE; threshold "
            "recalibrado por target. Cluster-only queda fuera de esta matriz."
        )
    else:
        k_grid_global = []
        k_grid_by_set = {
            "Base": _k_grid_values(
                k_min=int(k_min),
                k_max=int(k_max),
                k_step=int(k_step),
                feature_count=len(base_schema_cols),
            ),
            "Cluster": _k_grid_values(
                k_min=int(k_min),
                k_max=int(k_max),
                k_step=int(k_step),
                feature_count=len(cluster_schema_cols),
            ),
            "Base + Cluster": _k_grid_values(
                k_min=int(k_min),
                k_max=int(k_max),
                k_step=int(k_step),
                feature_count=len(all_schema_cols),
            ),
        }
        st.caption(
            "Grilla efectiva de K: "
            f"Base={k_grid_by_set['Base']} | "
            f"Cluster={k_grid_by_set['Cluster']} | "
            f"Base + Cluster={k_grid_by_set['Base + Cluster']}"
        )
    protocol_preview = {
        "protocol_family": protocol_family,
        "split_mode": "Temporal",
        "metric": objective_metric,
        "objective_metric": objective_metric,
        "objective_label": objective_label,
        "optuna_objective_metric": objective_metric,
        "optuna_objective_label": objective_label,
        "threshold_protocols": list(threshold_protocols),
        "threshold_objective": threshold_objective,
        "threshold_objective_label": threshold_objective_label,
        "calibration_methods": list(calibration_methods),
        "far_target": float(controlled_far_target),
        "alerts_per_day": float(controlled_alerts_per_day),
        "fn_cost": float(controlled_fn_cost),
        "fp_cost": float(controlled_fp_cost),
        "robust_folds": int(robust_folds),
        "test_only_final": True,
        "models": list(selected_models),
        "feature_sets": (
            list(FROZEN_TUNING_ABLATION_FEATURE_SETS)
            if use_frozen_tuning_ablation
            else ["Base", "Cluster", "Base + Cluster"]
        ),
        "balance_modes": ["none", "smote"],
        "test_size": float(test_size),
        "val_size": float(val_size),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "k_step": int(k_step),
        "k_grid_by_set": k_grid_by_set,
        "k_grid_global": list(k_grid_global),
        "feature_ranking_mode": (
            "feature_selection_global"
            if use_modelos_k_protocol
            else "controlled"
        ),
        "ranking_protocol": (
            "feature_selection_tab"
            if use_modelos_k_protocol
            else "controlled_train_only_per_feature_set"
        ),
        "feature_selection_params": {
            "n_estimators": int(feature_selection_n_estimators),
            "max_depth": feature_selection_max_depth,
            "random_state": int(random_state),
            "class_weight": "balanced",
            "criterion": "gini",
            "n_jobs": int(feature_selection_n_jobs),
        },
        "n_trials": int(n_trials),
        "timeout": int(timeout),
        "optuna_n_jobs": int(optuna_n_jobs),
        "parallel_jobs": int(parallel_jobs),
        "xgb_parallel_jobs": int(xgb_parallel_jobs),
        "search_space_config": search_space,
        "segment_info": segment_info,
        "event_path": str(selected_event_path),
        "features_path": str(selected_features_path),
        "dataset_date_start": (
            None if dataset_date_start is None else str(pd.Timestamp(dataset_date_start))
        ),
        "dataset_date_end": (
            None if dataset_date_end is None else str(pd.Timestamp(dataset_date_end))
        ),
    }
    if use_frozen_tuning_ablation:
        protocol_preview["ablation_config"] = dict(FROZEN_TUNING_ABLATION_CONFIG)
    if len(calibration_methods) == 1:
        protocol_preview["calibration_method"] = calibration_methods[0]
        checkpoint_context = build_controlled_comparison_context(
            event_path=selected_event_path,
            features_path=selected_features_path,
            segment_info=segment_info,
            protocol=protocol_preview,
        )
        checkpoint_preview = preview_controlled_comparison_checkpoint(
            checkpoint_context,
            checkpoint_root=checkpoint_root,
        )
    else:
        checkpoint_preview = {}
        st.caption(
            "El preview/reuso de checkpoint se resolverá por calibrador al ejecutar "
            "la corrida múltiple."
        )
    if checkpoint_preview.get("checkpoint_available"):
        if checkpoint_preview.get("can_resume"):
            st.info(
                "Se encontró un checkpoint compatible en progreso. "
                f"Run ID: {checkpoint_preview.get('run_id')} | "
                f"Paso actual: {checkpoint_preview.get('current_step_id') or '-'} | "
                f"Progreso: {checkpoint_preview.get('completed_steps')}/"
                f"{checkpoint_preview.get('total_steps')}"
            )
            checkpoint_mode = st.radio(
                "Checkpoint compatible",
                ["Resume checkpoint", "Start fresh"],
                horizontal=True,
                key="exp_controlled_checkpoint_mode",
            )
        elif checkpoint_preview.get("can_load_completed"):
            st.info(
                "Ya existe una corrida compatible completada. "
                f"Run ID: {checkpoint_preview.get('run_id')} | "
                f"Actualizado: {checkpoint_preview.get('updated_at')}"
            )
            checkpoint_mode = st.radio(
                "Checkpoint compatible",
                ["Load checkpoint result", "Start fresh"],
                horizontal=True,
                key="exp_controlled_checkpoint_mode_completed",
            )

    _render_controlled_comparison_memory_estimator(
        accidents_df_for_tramo=accidents_df_for_tramo,
        selected_event_path=selected_event_path,
        selected_features_path=selected_features_path,
        tramo_tuple=tramo_tuple,
        segment_info=segment_info,
        dataset_date_start=dataset_date_start,
        dataset_date_end=dataset_date_end,
        test_size=float(test_size),
        val_size=float(val_size),
        k_min=int(k_min),
        k_max=int(k_max),
        k_step=int(k_step),
        xgb_parallel_jobs=int(xgb_parallel_jobs),
        selected_models=list(selected_models),
        search_space=search_space,
    )

    if st.button(
        (
            "Ejecutar Modelos por K"
            if use_modelos_k_protocol
            else (
                "Ejecutar ablación cruzada"
                if use_frozen_tuning_ablation
                else "Ejecutar comparación controlada"
            )
        ),
        key="exp_controlled_run",
    ):
        if int(k_min) > int(k_max):
            st.error("K mínimo no puede ser mayor que K máximo.")
            return
        if int(k_step) <= 0:
            st.error("Paso K debe ser mayor que 0.")
            return
        if float(test_size) <= 0 or float(test_size) >= 1:
            st.error("Test size debe estar entre 0 y 1.")
            return
        if float(val_size) <= 0 or float(val_size) >= 1:
            st.error("Validation size debe estar entre 0 y 1.")
            return
        if not selected_models:
            st.error("Seleccione al menos un modelo para ejecutar la comparación controlada.")
            return
        if not threshold_protocols:
            st.error("Seleccione al menos un protocolo de evaluación.")
            return
        if not calibration_methods:
            st.error("Seleccione al menos un calibrador.")
            return
        if int(rf_depth_min) > int(rf_depth_max):
            st.error("RF max_depth min no puede ser mayor que RF max_depth max.")
            return
        if not base_schema_cols or not cluster_schema_cols:
            st.error("La configuración actual no tiene conjuntos Base y Cluster válidos.")
            return

        try:
            base_df = _prepare_controlled_comparison_base_df(
                accidents_df_for_tramo=accidents_df_for_tramo,
                selected_features_path=selected_features_path,
                tramo_tuple=tramo_tuple,
                date_start=dataset_date_start,
                date_end=dataset_date_end,
            )
        except Exception as exc:
            st.error(f"No se pudo preparar el dataset del tramo: {exc}")
            return

        exp_meta = {
            "dataset_name": selected_event,
            "features_name": selected_features,
            "dataset_date_start": (
                None
                if dataset_date_start is None
                else str(pd.Timestamp(dataset_date_start))
            ),
            "dataset_date_end": (
                None
                if dataset_date_end is None
                else str(pd.Timestamp(dataset_date_end))
            ),
            "protocol_family": protocol_family,
            "feature_ranking_mode": (
                "feature_selection_global"
                if use_modelos_k_protocol
                else "controlled"
            ),
            "ranking_protocol": (
                "feature_selection_tab"
                if use_modelos_k_protocol
                else "controlled_train_only_per_feature_set"
            ),
            "feature_selection_params": {
                "n_estimators": int(feature_selection_n_estimators),
                "max_depth": feature_selection_max_depth,
                "random_state": int(random_state),
                "class_weight": "balanced",
                "criterion": "gini",
                "n_jobs": int(feature_selection_n_jobs),
            },
            "run_mode": checkpoint_mode,
            "random_state": int(random_state),
            "objective_metric": objective_metric,
            "objective_label": objective_label,
            "threshold_protocols": list(threshold_protocols),
            "threshold_objective": threshold_objective,
            "threshold_objective_label": threshold_objective_label,
            "calibration_methods": list(calibration_methods),
            "far_target": float(controlled_far_target),
            "alerts_per_day": float(controlled_alerts_per_day),
            "fn_cost": float(controlled_fn_cost),
            "fp_cost": float(controlled_fp_cost),
            "robust_folds": int(robust_folds),
            "n_trials": int(n_trials),
            "timeout": int(timeout),
            "optuna_n_jobs": int(optuna_n_jobs),
            "parallel_jobs": int(parallel_jobs),
            "xgb_parallel_jobs": int(xgb_parallel_jobs),
            "selected_models": list(selected_models),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "k_step": int(k_step),
            "segment_info": segment_info,
            "search_space_config": search_space,
        }
        if use_frozen_tuning_ablation:
            exp_meta["ablation_config"] = dict(FROZEN_TUNING_ABLATION_CONFIG)
        if checkpoint_mode != "Start fresh" and checkpoint_preview:
            exp_meta["checkpoint_run_dir"] = checkpoint_preview.get("run_dir")
        exp_db_path = _init_experiment_db("Controlled comparison", exp_meta)
        if exp_db_path:
            st.caption(f"DB live: {exp_db_path}")
            if checkpoint_mode != "Start fresh" and checkpoint_preview:
                seeded_rows = _seed_controlled_comparison_live_db(
                    exp_db_path,
                    checkpoint_run_dir=checkpoint_preview.get("run_dir"),
                    dataset_name=selected_event,
                    features_name=selected_features,
                    segment_info=segment_info,
                )
                if seeded_rows > 0:
                    st.caption(
                        f"DB live inicializada con {seeded_rows} resultados del checkpoint."
                    )

        progress_bar = st.progress(0, text="Preparando comparación controlada...")

        def _progress_callback(value: int, message: str) -> None:
            progress_bar.progress(
                max(0, min(100, int(value))),
                text=str(message),
            )

        def _result_callback(payload: Dict[str, object]) -> None:
            payload_out = dict(payload)
            payload_out["experiment"] = "Controlled comparison"
            payload_out["dataset_name"] = selected_event
            payload_out["features_name"] = selected_features
            payload_out["dataset_date_start"] = (
                None if dataset_date_start is None else str(pd.Timestamp(dataset_date_start))
            )
            payload_out["dataset_date_end"] = (
                None if dataset_date_end is None else str(pd.Timestamp(dataset_date_end))
            )
            payload_out["segment_info"] = segment_info
            _append_experiment_result(exp_db_path, payload_out)

        runner = ExperimentsRunner(random_state=int(random_state))
        try:
            payloads: List[Dict[str, object]] = []
            for calibration_method in calibration_methods:
                payload = runner.run_controlled_comparison(
                    base_df,
                    event_path=selected_event_path,
                    features_path=selected_features_path,
                    segment_info=segment_info,
                    dataset_date_start=dataset_date_start,
                    dataset_date_end=dataset_date_end,
                    objective_metric=objective_metric,
                    threshold_protocols=list(threshold_protocols),
                    threshold_objective=threshold_objective,
                    calibration_method=calibration_method,
                    far_target=float(controlled_far_target),
                    alerts_per_day=float(controlled_alerts_per_day),
                    fn_cost=float(controlled_fn_cost),
                    fp_cost=float(controlled_fp_cost),
                    robust_folds=int(robust_folds),
                    feature_ranking_mode=(
                        "feature_selection_global"
                        if use_modelos_k_protocol
                        else "controlled"
                    ),
                    experimental_protocol=(
                        FROZEN_TUNING_ABLATION_PROTOCOL_FAMILY
                        if use_frozen_tuning_ablation
                        else None
                    ),
                    feature_selection_n_estimators=int(feature_selection_n_estimators),
                    feature_selection_max_depth=feature_selection_max_depth,
                    feature_selection_n_jobs=int(feature_selection_n_jobs),
                    test_size=float(test_size),
                    val_size=float(val_size),
                    k_min=int(k_min),
                    k_max=int(k_max),
                    k_step=int(k_step),
                    n_trials=int(n_trials),
                    timeout=int(timeout),
                    optuna_n_jobs=int(optuna_n_jobs),
                    parallel_jobs=int(parallel_jobs),
                    xgb_parallel_jobs=int(xgb_parallel_jobs),
                    selected_models=list(selected_models),
                    search_space_config=search_space,
                    progress_callback=_progress_callback,
                    result_callback=_result_callback,
                    checkpoint_root=checkpoint_root,
                    auto_resume=(
                        len(calibration_methods) == 1
                        and checkpoint_mode in {
                            "Resume checkpoint",
                            "Load checkpoint result",
                        }
                    ),
                    start_fresh=checkpoint_mode == "Start fresh",
                )
                payloads.append(payload)
        except Exception as exc:
            progress_bar.empty()
            st.error(f"No se pudo ejecutar la comparación controlada: {exc}")
            return

        progress_bar.empty()
        payload = payloads[-1] if payloads else {}
        summary_df = pd.concat(
            [
                frame
                for frame in [
                    item.get("best_summary_df")
                    for item in payloads
                    if isinstance(item, dict)
                ]
                if isinstance(frame, pd.DataFrame) and not frame.empty
            ],
            ignore_index=True,
        ) if payloads else pd.DataFrame()
        curves_df = pd.concat(
            [
                frame
                for frame in [
                    item.get("curves_df")
                    for item in payloads
                    if isinstance(item, dict)
                ]
                if isinstance(frame, pd.DataFrame) and not frame.empty
            ],
            ignore_index=True,
        ) if payloads else pd.DataFrame()
        grid_results_df = pd.concat(
            [
                frame
                for frame in [
                    item.get("grid_results_df")
                    for item in payloads
                    if isinstance(item, dict)
                ]
                if isinstance(frame, pd.DataFrame) and not frame.empty
            ],
            ignore_index=True,
        ) if payloads else pd.DataFrame()
        ablation_deltas_df = pd.concat(
            [
                frame
                for frame in [
                    item.get("ablation_deltas_df")
                    for item in payloads
                    if isinstance(item, dict)
                ]
                if isinstance(frame, pd.DataFrame) and not frame.empty
            ],
            ignore_index=True,
        ) if payloads else pd.DataFrame()
        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pd.DataFrame()
        if not isinstance(curves_df, pd.DataFrame):
            curves_df = pd.DataFrame()
        if not isinstance(grid_results_df, pd.DataFrame):
            grid_results_df = pd.DataFrame()
        if not isinstance(ablation_deltas_df, pd.DataFrame):
            ablation_deltas_df = pd.DataFrame()

        for record in summary_df.to_dict(orient="records"):
            record_out = dict(record)
            record_out["dataset_name"] = selected_event
            record_out["features_name"] = selected_features
            record_out["dataset_date_start"] = (
                None if dataset_date_start is None else str(pd.Timestamp(dataset_date_start))
            )
            record_out["dataset_date_end"] = (
                None if dataset_date_end is None else str(pd.Timestamp(dataset_date_end))
            )
            record_out["segment_info"] = segment_info
            _append_experiment_best(exp_db_path, record_out)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = RESULTS_DIR / f"controlled_comparison_summary_{stamp}.csv"
        curves_path = RESULTS_DIR / f"controlled_comparison_curves_{stamp}.csv"
        detail_path = RESULTS_DIR / f"controlled_comparison_grid_{stamp}.csv"
        ablation_deltas_path = (
            RESULTS_DIR / f"controlled_comparison_ablation_deltas_{stamp}.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        curves_df.to_csv(curves_path, index=False)
        grid_results_df.to_csv(detail_path, index=False)
        if not ablation_deltas_df.empty:
            ablation_deltas_df.to_csv(ablation_deltas_path, index=False)

        last_results = {
            "run_id": payload.get("run_id"),
            "checkpoint_run_dir": payload.get("checkpoint_run_dir"),
            "loaded_from_checkpoint": bool(payload.get("loaded_from_checkpoint")),
            "auto_resumed": bool(payload.get("auto_resumed")),
            "summary_name": summary_path.name,
            "curves_name": curves_path.name,
            "detail_name": detail_path.name,
            "summary_path": str(summary_path),
            "curves_path": str(curves_path),
            "detail_path": str(detail_path),
        }
        if not ablation_deltas_df.empty:
            last_results.update(
                {
                    "ablation_deltas_name": ablation_deltas_path.name,
                    "ablation_deltas_path": str(ablation_deltas_path),
                }
            )
        st.session_state["exp_controlled_last_results"] = last_results

    _render_controlled_comparison_current_result(
        checkpoint_root=checkpoint_root,
    )

    _render_controlled_comparison_protocol_description(
        objective_label=objective_label,
        selected_models=list(selected_models),
        threshold_protocols=list(threshold_protocols),
        threshold_objective_label=threshold_objective_label,
        calibration_methods=list(calibration_methods),
        alerts_per_day=float(controlled_alerts_per_day),
        fn_cost=float(controlled_fn_cost),
        fp_cost=float(controlled_fp_cost),
        robust_folds=int(robust_folds),
        test_size=float(test_size),
        val_size=float(val_size),
        k_min=int(k_min),
        k_max=int(k_max),
        k_step=int(k_step),
        n_trials=int(n_trials),
        timeout=int(timeout),
        optuna_n_jobs=int(optuna_n_jobs),
        parallel_jobs=int(parallel_jobs),
        xgb_parallel_jobs=int(xgb_parallel_jobs),
    )


def _render_experiments_tab() -> None:
    st.header("Experimentos")

    tab_new, tab_past, tab_import = st.tabs(["Ejecutar Nuevo", "Resultados Anteriores", "Importar Experimento"])

    # --- TAB: Importar Experimento ---
    with tab_import:
        st.subheader("Importar Experimento (ZIP)")
        uploaded_zip = st.file_uploader("Subir archivo ZIP de experimento", type=["zip"], key="exp_import_zip")
        if uploaded_zip:
            if st.button("Importar y Extraer"):
                try:
                    import zipfile
                    with zipfile.ZipFile(uploaded_zip, "r") as z:
                        z.extractall(RESULTS_DIR)
                    st.success(f"Experimentos importados exitosamente en {RESULTS_DIR}")
                    # Clear cache to refresh file lists
                    st.cache_data.clear() 
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importando ZIP: {e}")

    # --- TAB: Resultados Anteriores ---
    with tab_past:
        st.subheader("Visualización y Exportación")
        past_files = _list_experiment_result_files()
        if past_files:
            past_options = {
                _experiment_result_option_label(path): path for path in past_files
            }
            sel_past = st.selectbox(
                "Seleccionar archivo de resultados previos",
                list(past_options.keys()),
                key="history_exp_select",
            )
            
            if sel_past:
                path = past_options[sel_past]
                
                # --- Export Logic ---
                timestamp = _experiment_result_timestamp(path)
                
                col_view, col_export = st.columns([0.8, 0.2])
                with col_export:
                    if timestamp:
                        related_files = _experiment_result_related_files(
                            path,
                            timestamp,
                        )
                        if related_files:
                            try:
                                import zipfile
                                import io
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                    for f in related_files:
                                        zf.write(
                                            f,
                                            arcname=_experiment_export_arcname(f),
                                        )
                                zip_buffer.seek(0)
                                st.download_button(
                                    label="Exportar Experimento (ZIP)",
                                    data=zip_buffer,
                                    file_name=_experiment_export_zip_name(
                                        path,
                                        timestamp,
                                    ),
                                    mime="application/zip"
                                )
                            except Exception as e:
                                st.error(f"Error creando ZIP: {e}")
                        else:
                            st.warning("No se encontraron archivos relacionados.")
                    else:
                        st.caption("No se pudo identificar timestamp para exportar.")

                try:
                    past_df = pd.read_csv(path)
                    is_calibration_sweep = _is_calibration_sweep_result_file(
                        path,
                        past_df,
                    )
                    is_controlled_comparison = False
                    is_find_samples = False
                    is_best_section = False
                    is_best_section_k = False
                    if "experiment" in past_df.columns:
                        is_controlled_comparison = past_df["experiment"].astype(str).str.contains(
                            "controlled comparison", case=False, na=False
                        ).any()
                        is_find_samples = past_df["experiment"].astype(str).str.contains(
                            "find samples", case=False, na=False
                        ).any()
                        is_best_section = past_df["experiment"].astype(str).str.contains(
                            "best highway section", case=False, na=False
                        ).any()
                        is_best_section_k = past_df["experiment"].astype(str).str.contains(
                            "best mix highway section", case=False, na=False
                        ).any()
                        if not is_best_section_k:
                            is_best_section_k = past_df["experiment"].astype(str).str.contains(
                                "highway section & k", case=False, na=False
                            ).any()
                    if not is_find_samples and "type" in past_df.columns:
                        is_find_samples = past_df["type"].astype(str).str.contains(
                            "find samples", case=False, na=False
                        ).any()
                    if not is_controlled_comparison and "protocol_family" in past_df.columns:
                        is_controlled_comparison = past_df[
                            "protocol_family"
                        ].astype(str).str.contains(
                            "controlled comparison",
                            case=False,
                            na=False,
                        ).any()
                    if not is_best_section and "type" in past_df.columns:
                        is_best_section = past_df["type"].astype(str).str.contains(
                            "best highway section", case=False, na=False
                        ).any()
                    if not is_find_samples and path.name.startswith(
                        "find_samples_sizes_results_"
                    ):
                        is_find_samples = True
                    if not is_best_section and path.name.startswith(
                        "best_highway_section_results_"
                    ):
                        is_best_section = True
                    if not is_best_section_k and path.name.startswith(
                        "best_highway_section_k_results_"
                    ):
                        is_best_section_k = True
                    if not is_controlled_comparison and path.name.startswith(
                        "controlled_comparison_summary_"
                    ):
                        is_controlled_comparison = True
                    if not is_controlled_comparison and path.name.startswith(
                        "best_highway_section_controlled_summary_"
                    ):
                        is_controlled_comparison = True

                    if is_calibration_sweep:
                        st.caption(
                            "Experimento detectado: Calibración score + threshold"
                        )
                        _render_calibration_sweep_results(
                            _calibration_sweep_result_state_from_path(
                                path,
                                past_df,
                            ),
                            key_prefix=(
                                f"history_calibration_{timestamp or path.stem}"
                            ),
                        )
                    elif is_controlled_comparison:
                        controlled_prefix = "controlled_comparison"
                        controlled_label = "Controlled comparison"
                        if path.name.startswith(
                            "best_highway_section_controlled_summary_"
                        ):
                            controlled_prefix = "best_highway_section_controlled"
                            controlled_label = (
                                "Best highway section · protocolo controlado"
                            )
                        st.caption(f"Experimento detectado: {controlled_label}")
                        curves_df = pd.DataFrame()
                        detail_df = pd.DataFrame()
                        ablation_deltas_df = pd.DataFrame()
                        if timestamp:
                            curves_path = (
                                RESULTS_DIR
                                / f"{controlled_prefix}_curves_{timestamp}.csv"
                            )
                            detail_path = (
                                RESULTS_DIR
                                / f"{controlled_prefix}_grid_{timestamp}.csv"
                            )
                            ablation_deltas_path = (
                                RESULTS_DIR
                                / f"{controlled_prefix}_ablation_deltas_{timestamp}.csv"
                            )
                            if curves_path.exists():
                                try:
                                    curves_df = pd.read_csv(curves_path)
                                except Exception:
                                    curves_df = pd.DataFrame()
                            if detail_path.exists():
                                try:
                                    detail_df = pd.read_csv(detail_path)
                                except Exception:
                                    detail_df = pd.DataFrame()
                            if ablation_deltas_path.exists():
                                try:
                                    ablation_deltas_df = pd.read_csv(
                                        ablation_deltas_path
                                    )
                                except Exception:
                                    ablation_deltas_df = pd.DataFrame()
                        _render_controlled_comparison_results_panel(
                            past_df,
                            curves_df,
                            grid_results_df=detail_df,
                            ablation_deltas_df=ablation_deltas_df,
                            key_prefix=f"history_controlled_{timestamp or 'na'}",
                        )
                    elif is_find_samples:
                        st.caption("Experimento detectado: Find samples sizes")
                        plot_df = past_df.copy()
                        if "far_sens" not in plot_df.columns and {
                            "far",
                            "sensitivity",
                        }.issubset(plot_df.columns):
                            plot_df["far_sens"] = (
                                plot_df["far"]
                                - (plot_df["sensitivity"] * 1e-3)
                            )
                        metric_options = {
                            "best_f1": "F1",
                            "accuracy": "Accuracy",
                            "recall": "Recall",
                            "precision": "Precision",
                            "roc_auc": "ROC-AUC",
                            "mcc": "MCC",
                            "brier_score": "Brier (menor es mejor)",
                            "fnr": "FNR (menor es mejor)",
                            "far_sens": "FAR - Sensibilidad (menor es mejor)",
                        }
                        available_metrics = {
                            k: v
                            for k, v in metric_options.items()
                            if k in plot_df.columns
                        }
                        if not available_metrics:
                            st.info("No hay métricas disponibles para graficar.")
                            st.dataframe(past_df, width="stretch")
                        else:
                            metric_labels = list(available_metrics.values())
                            selected_metric_label = st.selectbox(
                                "Métrica a graficar",
                                metric_labels,
                                key="history_find_samples_metric",
                            )
                            metric_key = next(
                                k
                                for k, v in available_metrics.items()
                                if v == selected_metric_label
                            )
                            if "error" in plot_df.columns:
                                plot_df = plot_df[
                                    plot_df["error"].isna()
                                    | (plot_df["error"] == "")
                                ]

                            best_row = None
                            if (
                                "is_best" in plot_df.columns
                                and plot_df["is_best"].astype(str).str.lower().isin(
                                    {"true", "1", "yes"}
                                ).any()
                            ):
                                best_mask = plot_df["is_best"].astype(str).str.lower().isin(
                                    {"true", "1", "yes"}
                                )
                                best_row = plot_df.loc[
                                    best_mask
                                ].iloc[0]
                            elif not plot_df.empty and metric_key in plot_df.columns:
                                if metric_key in {"fnr", "far_sens", "brier_score"}:
                                    best_row = plot_df.loc[plot_df[metric_key].idxmin()]
                                else:
                                    best_row = plot_df.loc[plot_df[metric_key].idxmax()]

                            if best_row is not None:
                                st.markdown("**Resultado óptimo**")
                                objective_label = best_row.get("objective_label")
                                if objective_label:
                                    st.caption(f"Objetivo: {objective_label}")
                                st.caption(
                                    f"{best_row.get('segment_portico_last', '?')} -> {best_row.get('segment_portico_next', '?')} "
                                    f"| {best_row.get('window_start', '?')} a {best_row.get('window_end', '?')}"
                                )
                                metrics_cols = [
                                    "best_f1",
                                    "accuracy",
                                    "recall",
                                    "precision",
                                    "roc_auc",
                                    "fnr",
                                    "far_sens",
                                ]
                                metrics_payload = {
                                    key: best_row.get(key)
                                    for key in metrics_cols
                                    if key in best_row
                                }
                                if metrics_payload:
                                    st.json(metrics_payload)
                                model_path = best_row.get("model_path")
                                if model_path and isinstance(model_path, str):
                                    st.caption(f"Modelo: {model_path}")
                                cm = best_row.get("confusion_matrix")
                                if cm:
                                    try:
                                        import ast
                                        if isinstance(cm, str):
                                            cm = ast.literal_eval(cm)
                                        if isinstance(cm, list) and len(cm) == 4:
                                            tn, fp, fn, tp = cm
                                            cm = [[tn, fp], [fn, tp]]
                                        if isinstance(cm, list) and len(cm) == 2:
                                            cm_df = pd.DataFrame(
                                                cm,
                                                index=["Actual 0", "Actual 1"],
                                                columns=["Pred 0", "Pred 1"],
                                            )
                                            st.caption("Matriz de confusion")
                                            st.dataframe(cm_df, width="stretch")
                                    except Exception:
                                        st.text(f"CM Raw: {cm}")

                            tab_viz, tab_data = st.tabs(["Gráfico", "Datos"])
                            with tab_viz:
                                if (
                                    "candidate_rank" in plot_df.columns
                                    and metric_key in plot_df.columns
                                ):
                                    try:
                                        import altair as alt
                                        chart = alt.Chart(plot_df).mark_line(point=True).encode(
                                            x=alt.X(
                                                "candidate_rank:O",
                                                axis=alt.Axis(title="Candidato"),
                                            ),
                                            y=alt.Y(
                                                metric_key,
                                                axis=alt.Axis(title=available_metrics[metric_key]),
                                            ),
                                            color=alt.Color(
                                                "window_days:O",
                                                title="Ventana (dias)",
                                            ),
                                            tooltip=[
                                                "candidate_rank",
                                                "window_days",
                                                "accidents_per_day",
                                                metric_key,
                                                "segment_portico_last",
                                                "segment_portico_next",
                                            ],
                                        ).interactive()
                                        st.altair_chart(chart, width="stretch")
                                    except ImportError:
                                        st.warning("Altair no instalado.")
                                else:
                                    st.info("No hay columnas suficientes para graficar.")
                                if {
                                    "window_days",
                                    "accidents_per_day",
                                }.issubset(plot_df.columns):
                                    try:
                                        import altair as alt
                                        scatter = (
                                            alt.Chart(plot_df)
                                            .mark_circle(size=70, opacity=0.7)
                                            .encode(
                                                x=alt.X(
                                                    "window_days:Q",
                                                    axis=alt.Axis(title="Ventana (dias)"),
                                                ),
                                                y=alt.Y(
                                                    "accidents_per_day:Q",
                                                    axis=alt.Axis(title="Accidentes por dia"),
                                                ),
                                                color=alt.Color(
                                                    "segment_portico_last:N",
                                                    title="Portico inicio",
                                                ),
                                                tooltip=[
                                                    "window_days",
                                                    "accidents_per_day",
                                                    "segment_portico_last",
                                                    "segment_portico_next",
                                                ],
                                            )
                                        ).interactive()
                                        st.altair_chart(scatter, width="stretch")
                                    except ImportError:
                                        pass
                            with tab_data:
                                st.dataframe(past_df, width="stretch")
                    elif is_best_section_k:
                        st.caption("Experimento detectado: Best mix Highway section & K")
                        plot_df = past_df.copy()
                        if "far_sens" not in plot_df.columns and {
                            "far",
                            "sensitivity",
                        }.issubset(plot_df.columns):
                            plot_df["far_sens"] = (
                                plot_df["far"]
                                - (plot_df["sensitivity"] * 1e-3)
                            )
                        metric_options = {
                            "best_f1": "F1",
                            "accuracy": "Accuracy",
                            "recall": "Recall",
                            "precision": "Precision",
                            "roc_auc": "ROC-AUC",
                            "mcc": "MCC",
                            "brier_score": "Brier (menor es mejor)",
                            "fnr": "FNR (menor es mejor)",
                            "far_sens": "FAR - Sensibilidad (menor es mejor)",
                        }
                        available_metrics = {
                            k: v
                            for k, v in metric_options.items()
                            if k in plot_df.columns
                        }
                        if not available_metrics:
                            st.info("No hay métricas disponibles para graficar.")
                            st.dataframe(past_df, width="stretch")
                        else:
                            dataset_types = []
                            if "type" in plot_df.columns:
                                dataset_types = sorted(
                                    [
                                        t
                                        for t in plot_df["type"]
                                        .dropna()
                                        .unique()
                                        .tolist()
                                        if t
                                    ]
                                )
                            selected_type = "Todos"
                            if dataset_types:
                                selected_type = st.selectbox(
                                    "Dataset",
                                    ["Todos"] + dataset_types,
                                    key="history_best_section_k_type",
                                )
                            metric_labels = list(available_metrics.values())
                            selected_metric_label = st.selectbox(
                                "Métrica a graficar",
                                metric_labels,
                                key="history_best_section_k_metric",
                            )
                            metric_key = next(
                                k
                                for k, v in available_metrics.items()
                                if v == selected_metric_label
                            )

                            if "error" in plot_df.columns:
                                plot_df = plot_df[
                                    plot_df["error"].isna()
                                    | (plot_df["error"] == "")
                                ]
                            if selected_type != "Todos" and "type" in plot_df.columns:
                                plot_df = plot_df[plot_df["type"] == selected_type]
                            if metric_key in plot_df.columns:
                                plot_df = plot_df.dropna(subset=[metric_key])

                            if "k" in plot_df.columns:
                                plot_df = plot_df.copy()
                                plot_df["k"] = pd.to_numeric(
                                    plot_df["k"], errors="coerce"
                                )
                                plot_df = plot_df.dropna(subset=["k"])
                            else:
                                st.info("No hay columna 'k' para graficar.")
                                st.dataframe(past_df, width="stretch")
                                plot_df = pd.DataFrame()

                            best_row = None
                            if not plot_df.empty and metric_key in plot_df.columns:
                                if metric_key in {"fnr", "far_sens", "brier_score"}:
                                    best_row = plot_df.loc[plot_df[metric_key].idxmin()]
                                else:
                                    best_row = plot_df.loc[plot_df[metric_key].idxmax()]

                            if best_row is not None:
                                st.markdown("**Resultado óptimo**")
                                objective_label = best_row.get("objective_label")
                                if objective_label:
                                    st.caption(f"Objetivo: {objective_label}")
                                st.caption(
                                    f"{best_row.get('segment_portico_last', '?')} -> {best_row.get('segment_portico_next', '?')} "
                                    f"| K={best_row.get('k', '?')} "
                                    f"| {best_row.get('type', '-')}"
                                )
                                metrics_cols = [
                                    "best_f1",
                                    "accuracy",
                                    "recall",
                                    "precision",
                                    "roc_auc",
                                    "fnr",
                                    "far_sens",
                                ]
                                metrics_payload = {
                                    key: best_row.get(key)
                                    for key in metrics_cols
                                    if key in best_row
                                }
                                if metrics_payload:
                                    st.json(metrics_payload)
                                cm = best_row.get("confusion_matrix")
                                if cm:
                                    try:
                                        import ast
                                        if isinstance(cm, str):
                                            cm = ast.literal_eval(cm)
                                        if isinstance(cm, list) and len(cm) == 4:
                                            tn, fp, fn, tp = cm
                                            cm = [[tn, fp], [fn, tp]]
                                        if isinstance(cm, list) and len(cm) == 2:
                                            cm_df = pd.DataFrame(
                                                cm,
                                                index=["Actual 0", "Actual 1"],
                                                columns=["Pred 0", "Pred 1"],
                                            )
                                            st.caption("Matriz de confusion")
                                            st.dataframe(cm_df, width="stretch")
                                    except Exception:
                                        st.text(f"CM Raw: {cm}")

                            if {
                                "segment_portico_last",
                                "segment_portico_next",
                                metric_key,
                            }.issubset(plot_df.columns):
                                if plot_df.empty:
                                    st.info("No hay datos para graficar.")
                                    st.dataframe(past_df, width="stretch")
                                else:
                                    plot_df = plot_df.copy()
                                    plot_df["section_label"] = (
                                        plot_df["segment_portico_last"].astype(str)
                                        + " -> "
                                        + plot_df["segment_portico_next"].astype(str)
                                    )

                                    group_cols = [
                                        "segment_portico_last",
                                        "segment_portico_next",
                                    ]
                                    if metric_key in {"fnr", "far_sens", "brier_score"}:
                                        best_idx = plot_df.groupby(group_cols)[metric_key].idxmin()
                                    else:
                                        best_idx = plot_df.groupby(group_cols)[metric_key].idxmax()
                                    best_sections = plot_df.loc[best_idx].copy()

                                    tab_viz, tab_data = st.tabs(
                                        ["Gráfico", "Datos"]
                                    )
                                    with tab_viz:
                                        try:
                                            import altair as alt
                                            color_enc = alt.value("#1f77b4")
                                            if "type" in best_sections.columns:
                                                color_enc = alt.Color(
                                                    "type:N", title="Dataset"
                                                )
                                            chart = (
                                                alt.Chart(best_sections)
                                                .mark_bar()
                                                .encode(
                                                    x=alt.X(
                                                        "section_label:N",
                                                        sort="-y",
                                                        axis=alt.Axis(
                                                            title="Tramo",
                                                            labelAngle=-35,
                                                        ),
                                                    ),
                                                    y=alt.Y(
                                                        metric_key,
                                                        axis=alt.Axis(
                                                            title=available_metrics[metric_key]
                                                        ),
                                                    ),
                                                    color=color_enc,
                                                    tooltip=[
                                                        "section_label",
                                                        "k",
                                                        metric_key,
                                                        "type",
                                                    ],
                                                )
                                            ).interactive()
                                            st.altair_chart(chart, width="stretch")
                                        except ImportError:
                                            st.warning("Altair no instalado.")

                                        section_labels = sorted(
                                            best_sections["section_label"]
                                            .dropna()
                                            .unique()
                                            .tolist()
                                        )
                                        if section_labels:
                                            selected_section = st.selectbox(
                                                "Seleccionar tramo",
                                                section_labels,
                                                key="history_best_section_k_section",
                                            )
                                            section_df = plot_df[
                                                plot_df["section_label"] == selected_section
                                            ]
                                            if not section_df.empty:
                                                try:
                                                    import altair as alt
                                                    color_enc = alt.value("#1f77b4")
                                                    if "type" in section_df.columns:
                                                        color_enc = alt.Color(
                                                            "type:N", title="Dataset"
                                                        )
                                                    line = (
                                                        alt.Chart(section_df)
                                                        .mark_line(point=True)
                                                        .encode(
                                                            x=alt.X(
                                                                "k:Q",
                                                                axis=alt.Axis(title="K"),
                                                            ),
                                                            y=alt.Y(
                                                                metric_key,
                                                                axis=alt.Axis(
                                                                    title=available_metrics[metric_key]
                                                                ),
                                                            ),
                                                            color=color_enc,
                                                            tooltip=[
                                                                "k",
                                                                metric_key,
                                                                "type",
                                                            ],
                                                        )
                                                    ).interactive()
                                                    st.altair_chart(line, width="stretch")
                                                except ImportError:
                                                    st.warning("Altair no instalado.")
                                    with tab_data:
                                        st.dataframe(past_df, width="stretch")
                            else:
                                st.dataframe(past_df, width="stretch")
                    elif is_best_section:
                        st.caption("Experimento detectado: Best highway section")
                        plot_df = past_df.copy()
                        if "far_sens" not in plot_df.columns and {
                            "far",
                            "sensitivity",
                        }.issubset(plot_df.columns):
                            plot_df["far_sens"] = (
                                plot_df["far"]
                                - (plot_df["sensitivity"] * 1e-3)
                            )
                        metric_options = {
                            "best_f1": "F1",
                            "accuracy": "Accuracy",
                            "recall": "Recall",
                            "precision": "Precision",
                            "roc_auc": "ROC-AUC",
                            "mcc": "MCC",
                            "brier_score": "Brier (menor es mejor)",
                            "fnr": "FNR (menor es mejor)",
                            "far_sens": "FAR - Sensibilidad (menor es mejor)",
                        }
                        available_metrics = {
                            k: v
                            for k, v in metric_options.items()
                            if k in plot_df.columns
                        }
                        if not available_metrics:
                            st.info("No hay métricas disponibles para graficar.")
                            st.dataframe(past_df, width="stretch")
                        else:
                            dataset_types = []
                            if "type" in plot_df.columns:
                                dataset_types = sorted(
                                    [
                                        t
                                        for t in plot_df["type"]
                                        .dropna()
                                        .unique()
                                        .tolist()
                                        if t
                                    ]
                                )
                            selected_type = "Todos"
                            if dataset_types:
                                selected_type = st.selectbox(
                                    "Dataset",
                                    ["Todos"] + dataset_types,
                                    key="history_best_section_type",
                                )
                            metric_labels = list(available_metrics.values())
                            selected_metric_label = st.selectbox(
                                "Métrica a graficar",
                                metric_labels,
                                key="history_best_section_metric",
                            )
                            metric_key = next(
                                k
                                for k, v in available_metrics.items()
                                if v == selected_metric_label
                            )
                            if "error" in plot_df.columns:
                                plot_df = plot_df[
                                    plot_df["error"].isna()
                                    | (plot_df["error"] == "")
                                ]
                            if selected_type != "Todos" and "type" in plot_df.columns:
                                plot_df = plot_df[plot_df["type"] == selected_type]

                            best_row = None
                            if not plot_df.empty and metric_key in plot_df.columns:
                                if metric_key in {"fnr", "far_sens", "brier_score"}:
                                    best_row = plot_df.loc[plot_df[metric_key].idxmin()]
                                else:
                                    best_row = plot_df.loc[plot_df[metric_key].idxmax()]

                            if best_row is not None:
                                st.markdown("**Resultado óptimo**")
                                objective_label = best_row.get("objective_label")
                                if objective_label:
                                    st.caption(f"Objetivo: {objective_label}")
                                st.caption(
                                    f"{best_row.get('segment_portico_last', '?')} -> {best_row.get('segment_portico_next', '?')} "
                                    f"| {best_row.get('type', '-')}"
                                )
                                metrics_cols = [
                                    "best_f1",
                                    "accuracy",
                                    "recall",
                                    "precision",
                                    "roc_auc",
                                    "fnr",
                                    "far_sens",
                                ]
                                metrics_payload = {
                                    key: best_row.get(key)
                                    for key in metrics_cols
                                    if key in best_row
                                }
                                if metrics_payload:
                                    st.json(metrics_payload)
                                cm = best_row.get("confusion_matrix")
                                if cm:
                                    try:
                                        import ast
                                        if isinstance(cm, str):
                                            cm = ast.literal_eval(cm)
                                        if isinstance(cm, list) and len(cm) == 4:
                                            tn, fp, fn, tp = cm
                                            cm = [[tn, fp], [fn, tp]]
                                        if isinstance(cm, list) and len(cm) == 2:
                                            cm_df = pd.DataFrame(
                                                cm,
                                                index=["Actual 0", "Actual 1"],
                                                columns=["Pred 0", "Pred 1"],
                                            )
                                            st.caption("Matriz de confusion")
                                            st.dataframe(cm_df, width="stretch")
                                    except Exception:
                                        st.text(f"CM Raw: {cm}")

                            st.dataframe(past_df, width="stretch")
                    else:
                        # Metrics Summary (Max F1 per type)
                        if "best_f1" in past_df.columns and "type" in past_df.columns:
                            st.caption("Mejor F1 por estrategia:")
                            best_by_type = past_df.loc[past_df.groupby("type")["best_f1"].idxmax()]
                            # Display simple metrics
                            if not best_by_type.empty:
                                cols = st.columns(len(best_by_type))
                                for i, (idx, row) in enumerate(best_by_type.iterrows()):
                                    with cols[i]:
                                        delta_label = ""
                                        if "k" in best_by_type.columns:
                                            delta_label = f"k={row['k']}"
                                        st.metric(
                                            label=row["type"],
                                            value=f"{row['best_f1']:.4f}",
                                            delta=delta_label,
                                        )

                        tab_viz, tab_data = st.tabs(["Gráfico", "Datos"])
                        
                        with tab_viz:
                            if "k" in past_df.columns:
                                # Metric Selector
                                metric_options = {
                                    "best_f1": "Best F1 Score",
                                    "accuracy": "Accuracy",
                                    "recall": "Recall (Sens)",
                                    "precision": "Precision",
                                    "roc_auc": "ROC-AUC",
                                    "mcc": "MCC",
                                    "brier_score": "Brier",
                                    "fnr": "FNR",
                                }
                                # Filter only available columns
                                available_metrics = {k: v for k, v in metric_options.items() if k in past_df.columns}
                                if not available_metrics:
                                    available_metrics = {"best_f1": "Best F1 Score"} if "best_f1" in past_df.columns else {}
                                
                                selected_metric_key = "best_f1"
                                if available_metrics:
                                    col_sel, _ = st.columns([0.3, 0.7])
                                    with col_sel:
                                        selected_metric_label = st.selectbox(
                                            "Métrica a graficar",
                                            options=list(available_metrics.values()),
                                            index=0
                                        )
                                        # Reverse lookup key
                                        selected_metric_key = next(k for k, v in available_metrics.items() if v == selected_metric_label)

                                if selected_metric_key in past_df.columns and "type" in past_df.columns:
                                    try:
                                        import altair as alt
                                        
                                        # Calculate min and max for Y scale padding
                                        y_min = past_df[selected_metric_key].min()
                                        y_max = past_df[selected_metric_key].max()
                                        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
                                        
                                        chart = alt.Chart(past_df).mark_line(point=True).encode(
                                             x=alt.X('k', axis=alt.Axis(title='Top K Features')),
                                             y=alt.Y(selected_metric_key, scale=alt.Scale(domain=[max(0, y_min - padding), min(1, y_max + padding)]), axis=alt.Axis(title=available_metrics[selected_metric_key])),
                                             color='type',
                                             tooltip=['k', selected_metric_key, 'type', 'n_features']
                                        ).interactive()
                                        
                                        st.altair_chart(chart, width="stretch")
                                    except ImportError:
                                        st.warning("Altair no instalado.")
                                else:
                                    st.info(f"Columnas insuficientes para graficar (requiere k, type, {selected_metric_key}).")
                            else:
                                st.info("No hay columna 'k' para graficar.")

                        with tab_data:
                            if "k" in past_df.columns:
                                sorted_ks = sorted(past_df["k"].unique())
                                if sorted_ks:
                                    selected_k = st.select_slider(
                                        "Número de Features (K)", 
                                        options=sorted_ks,
                                        value=sorted_ks[-1]
                                    )
                                    
                                    subset = past_df[past_df["k"] == selected_k]
                                    
                                    # Use columns to show models side-by-side
                                    if not subset.empty:
                                        cols = st.columns(len(subset))
                                        for i, (idx, row) in enumerate(subset.iterrows()):
                                            with cols[i]:
                                                with st.container(border=True):
                                                    st.subheader(f"{row.get('type', 'Unknown')}")
                                                    st.caption(f"F1: {row.get('best_f1', 0):.4f}")
                                                    
                                                    st.markdown("**Métricas Detalladas**")
                                                    c_m1, c_m2 = st.columns(2)
                                                    with c_m1:
                                                        st.write(f"- **F1 Score:** {row.get('best_f1', 0):.4f}")
                                                        st.write(f"- **Accuracy:** {row.get('accuracy', 0):.4f}")
                                                        st.write(f"- **ROC-AUC:** {row.get('roc_auc', 0):.4f}")
                                                    with c_m2:
                                                        st.write(f"- **Recall:** {row.get('recall', 0):.4f}")
                                                        st.write(f"- **Precision:** {row.get('precision', 0):.4f}")
                                                        st.write(f"- **FNR:** {row.get('fnr', 0):.4f}")
                                                    
                                                    st.caption(f"Eval Threshold: {row.get('threshold', 0.5):.4f}")

                                                    cm = row.get("confusion_matrix")
                                                    if cm:
                                                        try:
                                                            import ast
                                                            if isinstance(cm, str):
                                                                cm = ast.literal_eval(cm)
                                                            if isinstance(cm, list) and len(cm) == 4:
                                                                st.markdown("**Matriz de Confusión**")
                                                                tn, fp, fn, tp = cm
                                                                cm_df = pd.DataFrame(
                                                                    [[tn, fp], [fn, tp]],
                                                                    columns=["Pred 0", "Pred 1"],
                                                                    index=["Real 0", "Real 1"]
                                                                )
                                                                st.dataframe(cm_df, width="stretch")
                                                        except:
                                                            st.text(f"CM Raw: {cm}")

                                                    st.markdown("**Info Dataset**")
                                                    d_rows = row.get("dataset_rows")
                                                    if d_rows:
                                                         if isinstance(d_rows, str):
                                                             try:
                                                                 import ast
                                                                 d_rows = ast.literal_eval(d_rows)
                                                             except:
                                                                 pass
                                                         if isinstance(d_rows, dict):
                                                             st.caption(f"Train: {d_rows.get('train')} | Val: {d_rows.get('val')} | Test: {d_rows.get('test')}")
                                                    
                                                    st.caption(f"Eventos: {row.get('dataset_name', '?')}")
                                                    st.caption(f"Features: {row.get('features_name', '?')}")
                                                    
                                                    st.markdown("**Mejores Hiperparámetros**")
                                                    params = row.get("best_params")
                                                    if isinstance(params, str):
                                                        try:
                                                            import ast
                                                            params_dict = ast.literal_eval(params)
                                                            st.json(params_dict)
                                                        except:
                                                            st.text(params)
                                                    elif isinstance(params, dict):
                                                        st.json(params)
                                                    else:
                                                        st.write(params)
                                                
                                                    features_list = row.get("feature_cols")
                                                    if features_list:
                                                        st.markdown("**Variables Utilizadas:**")
                                                        if isinstance(features_list, str):
                                                            try:
                                                                import ast
                                                                f_list = ast.literal_eval(features_list)
                                                                st.caption(", ".join(f_list))
                                                            except:
                                                                st.caption(features_list)
                                                        else:
                                                             st.caption(str(features_list))
                                else:
                                    st.dataframe(past_df, width="stretch")
                            else:
                                st.dataframe(past_df, width="stretch")

                except Exception as e:
                    st.error(f"Error cargando archivo: {e}")
        else:
            st.info("No hay experimentos previos guardados.")

    # --- TAB: Ejecutar Nuevo ---
    with tab_new:
        st.subheader("Configuracion de Experimento")
        exp_kind = st.radio(
            "Tipo de experimento",
            [
                "Features sampler",
                "Find samples sizes",
                "Best highway section",
                "Best mix Highway section & K",
                "Calibración score + threshold",
                "Comparación controlada",
            ],
            key="exp_kind_choice",
        )
        if exp_kind == "Find samples sizes":
            _render_find_samples_sizes_experiment()
            return
        if exp_kind == "Best highway section":
            _render_best_highway_section_controlled_experiment()
            return
        if exp_kind == "Best mix Highway section & K":
            _render_best_highway_section_k_experiment()
            return
        if exp_kind == "Calibración score + threshold":
            _render_calibration_sweep_experiment()
            return
        if exp_kind == "Comparación controlada":
            _render_controlled_comparison_experiment()
            return

        st.subheader("Features sampler")
        
        # 1. Select Event File
        event_files = _list_event_files()
        if not event_files:
            st.warning("No hay archivos de eventos (accidents) en Datos.")
            return
        event_names = [p.name for p in event_files]
        selected_event = st.selectbox("Archivo de Eventos", event_names, key="exp_event_file")
        
        # 2. Select Features File (Includes both Flow and Cluster variables)
        feature_files = _list_flow_feature_files()
        if not feature_files:
            st.warning("No hay archivos de features en Resultados.")
            return
        feature_names = [p.name for p in feature_files]
        selected_features = st.selectbox("Archivo de Features (Flow + Cluster)", feature_names, key="exp_feature_file")

        selected_features_path = next(
            (p for p in feature_files if p.name == selected_features),
            None,
        )
        allowed_porticos: Optional[set[str]] = None
        if selected_features_path is not None:
            allowed_porticos = _load_porticos_from_feature_file(
                selected_features_path
            )
            if allowed_porticos is None:
                st.warning(
                    "No se pudo leer porticos del archivo para filtrar tramos."
                )
        accidents_df_for_tramo = st.session_state.get("accidents_df")
        tramo_tuple = _build_tramo_selector(
            accidents_df_for_tramo,
            date_start=None,
            date_end=None,
            allowed_porticos=allowed_porticos,
            key="exp_features_sampler_tramo_choice",
        )
        tramo_info = None
        if tramo_tuple:
            eje, calzada, p_start, p_end = tramo_tuple
            tramo_info = {
                "eje": eje,
                "calzada": calzada,
                "portico_inicio": p_start,
                "portico_fin": p_end,
            }
        
        # Model Selection
        model_choice = st.selectbox(
            "Modelo para Experimento",
            ["Random Forest", "XGBoost", "SVM"],
            key="exp_model_choice"
        )

        objective_options = _optuna_objective_options(
            [
                "f1",
                "roc_auc",
                "accuracy",
                "recall",
                "precision",
                "fnr",
                "far_sens",
                "mcc",
                "brier_score",
            ]
        )
        objective_label = st.selectbox(
            "Metrica objetivo (Optuna/SMOTE)",
            list(objective_options.keys()),
            key="exp_features_objective_metric",
        )
        objective_cfg = objective_options.get(
            objective_label, {"key": "f1", "direction": "maximize"}
        )
        objective_key = objective_cfg["key"]
        objective_direction = objective_cfg["direction"]
        objective_verb = (
            "minimiza" if objective_direction == "minimize" else "optimiza"
        )
        st.caption(
            f"Optuna {objective_verb} {objective_label} en el set de validacion."
        )
        
        # Settings
        col1, col2 = st.columns(2)
        with col1:
            n_trials = st.number_input("Optuna Trials por paso", min_value=5, value=30, step=5, key="exp_n_trials")
        with col2:
            timeout = st.number_input("Optuna Timeout (seg) por paso", min_value=10, value=3600, step=10, key="exp_timeout")
        optuna_n_jobs = _render_optuna_n_jobs_input(
            "Optuna jobs paralelos",
            key="exp_optuna_n_jobs",
            default=1,
        )
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            max_k_limit = st.number_input(
                "Max K Features Limit",
                min_value=5,
                value=50,
                step=5,
                key="exp_max_k_limit",
            )
        with col_k2:
            step_size = st.number_input(
                "Paso K",
                min_value=1,
                value=5,
                step=1,
                key="exp_step_size",
            )

        # Advanced Configuration
        far_target = 0.2
        threshold_strategy = "optuna"
        threshold_strategy_label = "Optimizar threshold"
        with st.expander("Configuración Avanzada (Parámetros y Rangos)"):
            st.markdown("**Split de Datos**")
            c_split1, c_split2 = st.columns(2)
            with c_split1:
                val_size = st.slider("Validation Size (vs Train)", 0.1, 0.5, 0.2, 0.05, key="exp_val_size")
            with c_split2:
                test_size = st.slider("Test Size (Global)", 0.1, 0.5, 0.2, 0.05, key="exp_test_size")
            st.markdown("**Calibración de umbral**")
            threshold_options = {
                "Optimizar threshold": "optuna",
                "Calibrar por FAR": "far",
            }
            threshold_strategy = _option_value_from_state(
                threshold_options,
                "exp_features_threshold_strategy",
                default_label="Optimizar threshold",
            )
            threshold_visibility = _threshold_field_visibility_for_strategy(
                threshold_strategy
            )
            far_target = float(
                _render_conditional_slider(
                    "FAR target",
                    visible=threshold_visibility["far_target"],
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.01,
                    key="exp_far_target",
                )
            )
            threshold_strategy_label = st.selectbox(
                "Estrategia de umbral",
                list(threshold_options.keys()),
                key="exp_features_threshold_strategy",
            )
            threshold_strategy = threshold_options[threshold_strategy_label]
            calibration_methods = _calibration_method_multiselect(
                "Calibración",
                key="exp_features_calibration_methods",
                default_methods=["sigmoid", "isotonic"],
            )

                
            st.markdown("**Rango SMOTE**")
            c_smote1, c_smote2 = st.columns(2)
            with c_smote1:
                smote_k_min = st.number_input("K Neighbors Min", 1, 20, 1, key="exp_smote_k_min")
                smote_k_max = st.number_input("K Neighbors Max", 1, 20, 10, key="exp_smote_k_max")
            with c_smote2:
                smote_str_min = st.slider("Sampling Strategy Min", 0.1, 1.0, 0.1, 0.1, key="exp_smote_str_min")
                smote_str_max = st.slider("Sampling Strategy Max", 0.1, 1.0, 1.0, 0.1, key="exp_smote_str_max")
            
            # Model Specific Params
            st.markdown(f"**Rangos para {model_choice}**")
            model_ranges = {}
            
            if model_choice == "Random Forest":
                c_rf1, c_rf2 = st.columns(2)
                with c_rf1:
                    rf_ne_min = st.number_input("N Estimators Min", 10, 1000, 50, step=10, key="exp_rf_ne_min")
                    rf_ne_max = st.number_input("N Estimators Max", 10, 1000, 300, step=10, key="exp_rf_ne_max")
                with c_rf2:
                    rf_md_min = st.number_input("Max Depth Min", 1, 50, 3, key="exp_rf_md_min")
                    rf_md_max = st.number_input("Max Depth Max", 1, 50, 15, key="exp_rf_md_max")
                
                model_ranges = {
                    "n_estimators": {"min": rf_ne_min, "max": rf_ne_max},
                    "max_depth": {"min": rf_md_min, "max": rf_md_max}
                }

            elif model_choice == "XGBoost":
                c_xgb1, c_xgb2 = st.columns(2)
                with c_xgb1:
                    xgb_ne_min = st.number_input("N Estimators Min", 10, 1000, 50, step=10, key="exp_xgb_ne_min")
                    xgb_ne_max = st.number_input("N Estimators Max", 10, 1000, 300, step=10, key="exp_xgb_ne_max")
                    xgb_lr_min = st.number_input("Learning Rate Min", 0.001, 1.0, 0.01, format="%.3f", key="exp_xgb_lr_min")
                    xgb_lr_max = st.number_input("Learning Rate Max", 0.001, 1.0, 0.3, format="%.3f", key="exp_xgb_lr_max")
                with c_xgb2:
                    xgb_md_min = st.number_input("Max Depth Min", 1, 50, 3, key="exp_xgb_md_min")
                    xgb_md_max = st.number_input("Max Depth Max", 1, 50, 15, key="exp_xgb_md_max")
                    xgb_sub_min = st.slider("Subsample Min", 0.1, 1.0, 0.5, 0.1, key="exp_xgb_sub_min")
                    xgb_sub_max = st.slider("Subsample Max", 0.1, 1.0, 1.0, 0.1, key="exp_xgb_sub_max")
                    xgb_col_min = st.slider("Colsample ByTree Min", 0.1, 1.0, 0.5, 0.1, key="exp_xgb_col_min")
                    xgb_col_max = st.slider("Colsample ByTree Max", 0.1, 1.0, 1.0, 0.1, key="exp_xgb_col_max")
                    
                model_ranges = {
                    "n_estimators": {"min": xgb_ne_min, "max": xgb_ne_max},
                    "max_depth": {"min": xgb_md_min, "max": xgb_md_max},
                    "learning_rate": {"min": xgb_lr_min, "max": xgb_lr_max},
                    "subsample": {"min": xgb_sub_min, "max": xgb_sub_max},
                    "colsample_bytree": {"min": xgb_col_min, "max": xgb_col_max},
                }

            elif model_choice == "SVM":
                c_svm1, c_svm2 = st.columns(2)
                with c_svm1:
                     svm_c_min = st.number_input("C Min", 0.01, 1000.0, 0.1, format="%.2f", key="exp_svm_c_min")
                with c_svm2:
                     svm_c_max = st.number_input("C Max", 0.01, 1000.0, 50.0, format="%.2f", key="exp_svm_c_max")
                
                model_ranges = {
                     "C": {"min": svm_c_min, "max": svm_c_max}
                }

        if st.button("Iniciar Experimento"):
            if not calibration_methods:
                st.error("Seleccione al menos un calibrador.")
                return
            # Load Data
            try:
                accidents_path = next(p for p in event_files if p.name == selected_event)
                features_path = selected_features_path or next(
                    p for p in feature_files if p.name == selected_features
                )
                
                # Load using robust reader (handles sep and encoding)
                raw_accidents_df = read_csv_with_progress(str(accidents_path))
                
                # Load Porticos for processing
                try:
                    porticos_df = load_porticos()
                    if porticos_df is None or porticos_df.empty:
                        st.error("No se pudieron cargar los porticos (Porticos.csv).")
                        return
                except Exception as e:
                    st.error(f"Error cargando porticos: {e}")
                    return

                # Process Accidents (calculate ultimo_portico, accidente_time, etc.)
                try:
                    accidents_df, excluded = process_accidentes_df(
                        raw_accidents_df, porticos_df, return_excluded=True
                    )
                    if accidents_df.empty:
                        st.warning("No quedaron accidentes validos tras el procesamiento (verificar porticos/nombres).")
                        return
                    st.success(f"Accidentes procesados: {len(accidents_df)} (Excluidos: {len(excluded)})")
                except Exception as e:
                    st.error(f"Error procesando accidentes: {e}")
                    return
                
                # Handle DuckDB or CSV for features
                if str(features_path).endswith(".duckdb"):
                     if duckdb:
                        con = duckdb.connect(str(features_path), read_only=True)
                        # Assuming table name is first table
                        tables = con.execute("SHOW TABLES").fetchall()
                        if tables:
                            table_name = tables[0][0]
                            table_ref = _duckdb_quote_identifier(table_name)
                            query = f"SELECT * FROM {table_ref}"
                            params: List[object] = []
                            if tramo_tuple:
                                cols_info = con.execute(
                                    f"DESCRIBE {table_ref}"
                                ).fetchall()
                                columns = {row[0] for row in cols_info}
                                clauses, params, filter_ok = _build_tramo_duckdb_filters(
                                    tramo_tuple, columns
                                )
                                if not filter_ok:
                                    st.warning(
                                        "El archivo no contiene columnas para filtrar por tramo "
                                        "(se buscaron: portico, portico_last/portico_next, "
                                        "portico_inicio/portico_fin, ultimo_portico)."
                                    )
                                    con.close()
                                    return
                                if clauses:
                                    query += " WHERE " + " AND ".join(clauses)
                            features_df = con.execute(query, params).df()
                        else:
                            st.error("Empty DuckDB")
                            con.close()
                            return
                        con.close()
                     else:
                        st.error("DuckDB not installed")
                        return
                else:
                    features_df = read_csv_with_progress(str(features_path))
                    if tramo_tuple:
                        features_df, filter_ok = _apply_tramo_filter_df(
                            features_df, tramo_tuple
                        )
                        if not filter_ok:
                            st.warning(
                                "El archivo no contiene columnas para filtrar por tramo "
                                "(se buscaron: portico, portico_last/portico_next, "
                                "portico_inicio/portico_fin, ultimo_portico)."
                            )
                            return

                if features_df is None or features_df.empty:
                    if tramo_tuple:
                        st.warning(
                            "No se encontraron variables para el tramo seleccionado."
                        )
                    else:
                        st.error("El archivo de features esta vacio.")
                    return
                    
                # Merge to create Base DF
                # Note: add_accident_target handles merging features with accidents
                base_df = add_accident_target(features_df, accidents_df)
                if base_df.empty:
                    st.error("Dataset vacio tras merge.")
                    return

                # Identify Column Groups
                # 1. Cluster Columns
                cluster_cols = _get_cluster_cols(base_df)
                # 2. All Numeric Columns (Feature Candidates)
                all_feature_cols = _get_feature_cols(base_df)
                # 3. Base (Flow) Columns = All - Cluster
                base_cols = [c for c in all_feature_cols if c not in cluster_cols]
                
                if not cluster_cols:
                    st.warning("No se detectaron columnas de cluster en el archivo.")
                
                # Define search space from inputs
                search_space = {
                    "smote": {
                        "k_neighbors": {"min": smote_k_min, "max": smote_k_max},
                        "sampling_strategy": {"min": smote_str_min, "max": smote_str_max}
                    },
                    "model": model_ranges
                }
                
                # Prepare Runner
                runner = ExperimentsRunner()
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                exp_meta = {
                    "run_id": run_id,
                    "dataset_name": selected_event,
                    "features_name": selected_features,
                    "model_choice": model_choice,
                    "objective_label": objective_label,
                    "objective_metric": objective_key,
                    "objective_direction": objective_direction,
                    "far_target": float(far_target),
                    "threshold_strategy": threshold_strategy,
                    "threshold_strategy_label": threshold_strategy_label,
                    "calibration_methods": list(calibration_methods),
                    "test_size": float(test_size),
                    "val_size": float(val_size),
                    "optuna_n_jobs": int(optuna_n_jobs),
                    "max_k_limit": int(max_k_limit),
                    "step_size": int(step_size),
                }
                if tramo_info:
                    exp_meta["tramo"] = tramo_info
                exp_db_path = _init_experiment_db("Features sampler", exp_meta)
                if exp_db_path:
                    st.caption(f"DB live: {exp_db_path}")

                def _db_callback(payload: Dict[str, object]) -> None:
                    payload = dict(payload)
                    payload["experiment"] = "Features sampler"
                    payload["run_id"] = run_id
                    payload["model_choice"] = model_choice
                    if tramo_info:
                        payload["tramo"] = tramo_info
                    _append_experiment_result(exp_db_path, payload)
                
                # 1. Feature Importance (Full dataset)
                with st.spinner("Calculando importancia de variables (dataset completo)..."):
                    if not base_cols:
                        st.error("No hay columnas de flujo base encontradas.")
                        return
                    imp_full = runner.calculate_feature_importance(
                        base_df, all_feature_cols
                    )
                    combined_ordered = imp_full["variable"].tolist()
                    base_ordered = [
                        col for col in combined_ordered if col in base_cols
                    ]
                st.success(
                    "Importancia calculada "
                    f"({len(combined_ordered)} variables totales, "
                    f"{len(base_ordered)} base)."
                )
                combined_ordered_for_run = combined_ordered if cluster_cols else []
                
                # 3. Run Loop
                progress_bar = st.progress(0, text="Iniciando experimentos...")
                total_ordered = combined_ordered_for_run or base_ordered
                k_limit = min(len(total_ordered), int(max_k_limit))
                start_k = min(int(step_size), k_limit) if k_limit else 0
                st.info(
                    "Iniciando loop de experimentos "
                    f"(K={start_k}..{k_limit}, paso={int(step_size)})..."
                )

                results: List[Dict[str, object]] = []
                for calibration_method in calibration_methods:
                    results.extend(
                        runner.run_iterative_experiment(
                            base_df=base_df,
                            base_features_ordered=base_ordered,
                            cluster_features=combined_ordered_for_run,
                            model_choice=model_choice,
                            n_trials=int(n_trials),
                            timeout=int(timeout),
                            optuna_n_jobs=int(optuna_n_jobs),
                            far_target=float(far_target),
                            search_space_config=search_space,
                            step_size=int(step_size),
                            test_size=float(test_size),
                            val_size=float(val_size),
                            objective_key=objective_key,
                            objective_direction=objective_direction,
                            objective_label=objective_label,
                            cluster_feature_names=cluster_cols,
                            threshold_strategy=threshold_strategy,
                            calibration_method=str(calibration_method),
                            progress_bar=progress_bar,
                            dataset_name=selected_event,
                            features_name=selected_features,
                            max_k_limit=int(max_k_limit),
                            result_callback=_db_callback,
                        )
                    )
                
                # Results
                if results:
                    res_df = pd.DataFrame(results)
                    st.dataframe(res_df, width="stretch")
                    
                    # Save
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    res_path = RESULTS_DIR / f"experiments_results_{stamp}.csv"
                    res_df.to_csv(res_path, index=False)
                    st.success(f"Resultados guardados en {res_path}")
                    
                    # Plot
                    try:
                         import altair as alt
                         plot_df = res_df.copy()
                         plot_metric_key = objective_key
                         plot_metric_label = objective_label
                         if plot_metric_key == "far_sens":
                             if {"far", "sensitivity"}.issubset(plot_df.columns):
                                 plot_df["far_sens"] = (
                                     plot_df["far"]
                                     - (plot_df["sensitivity"] * 1e-3)
                                 )
                             else:
                                 plot_metric_key = "best_f1"
                                 plot_metric_label = "F1"
                         if plot_metric_key not in plot_df.columns:
                             plot_metric_key = "best_f1"
                             plot_metric_label = "F1"
                         chart = alt.Chart(plot_df).mark_line(point=True).encode(
                             x=alt.X("k", axis=alt.Axis(title="Top K Features")),
                             y=alt.Y(
                                 plot_metric_key,
                                 axis=alt.Axis(title=plot_metric_label),
                             ),
                             color="type",
                             tooltip=["k", plot_metric_key, "type"],
                         ).interactive()
                         
                         chart = chart.properties(width=700)
                         st.altair_chart(chart)
                    except ImportError:
                         st.warning("Altair no instalado para graficos.")
                else:
                    st.warning("No se generaron resultados.")
                
            except Exception as e:
                st.error(f"Error en experimento: {e}")
def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    _init_state()
    if set_page_config:
        st.set_page_config(page_title="Cluster/Accident", layout="wide")
    st.title("Crash prediction by Drivers Behavior")



    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)

    tabs = st.tabs(
        [
            "Eventos",
            "Feature engineering",
            "Match",
            "Feature selection",
            "Optuna",
            "Balance",
            "Modelos",
            "History",
            "Experiments",
        ]
    )
    with tabs[0]:
        _render_event_tab()
    with tabs[2]:
        _render_match_tab()
    with tabs[1]:
        _render_variables_tab()
    with tabs[3]:
        _render_feature_selection_tab()
    with tabs[4]:
        _render_optuna_tab()
    with tabs[5]:
        _render_balance_tab()
    with tabs[6]:
        _render_model_tab()
    with tabs[7]:
        _render_history_tab()
    with tabs[8]:
        _render_experiments_tab()


if __name__ == "__main__":
    main()
