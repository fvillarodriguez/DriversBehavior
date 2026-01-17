#!/usr/bin/env python3
"""
Streamlit app to run clustering workflows with unified tabs.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, time as dt_time
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "Resultados"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import (  # noqa: E402
    FlowColumns,
    FlowSampleSelection,
    get_flow_db_summary,
    load_flujos,
)
from clustering import (  # noqa: E402
    Clusterization,
    TTC_MAX_BY_PORTICO,
    _build_batch_ranges,
    _build_cluster_feature_db_path,
    _choose_feature_columns,
    _clusterize_in_batches,
    _order_feature_columns,
    _prepare_cluster_features,
    _scale_cluster_features,
    build_cluster_descriptive,
    build_cluster_summary,
    compute_gmm_metrics,
    compute_kmeans_metrics,
    list_cluster_feature_db_paths,
    load_cluster_features_duckdb,
    save_cluster_descriptive,
    save_cluster_features,
    save_cluster_features_duckdb,
    save_cluster_labels,
    save_cluster_metrics,
    save_cluster_summary,
    split_frequent_drivers,
    assign_clusters_kmeans,
    assign_clusters_gmm,
)

REQUIRED_FEATURE_COLS = {
    "plate",
    "total_passes",
    "avg_speed_kmh",
    "avg_relative_speed",
    "avg_headway_s",
    "conflict_rate",
    "lane_prop_1",
    "lane_prop_2",
    "lane_change_rate",
}
RUN_LOG_PATH = RESULTS_DIR / "cluster_run_log.jsonl"
COLOR_MAP_PATH = RESULTS_DIR / "cluster_color_map.csv"
CLUSTER_LABEL_PATTERN = re.compile(
    r"^cluster_(?P<method>kmeans|gmm|hdbscan)(?:_k(?P<k>\d+))?(?:.*)?\.csv$"
)
SUMMARY_PATTERN = re.compile(
    r"^cluster_summary(?:_(?P<method>kmeans|gmm|hdbscan))?(?:_k(?P<k>\d+))?(?:.*)?\.csv$"
)


class StreamlitProgress:
    def __init__(
        self,
        total: int,
        label: str = "",
        container: Optional[object] = None,
    ) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self._container = container or st
        self._label = self._container.empty()
        self._bar = self._container.progress(0)
        if label:
            self._label.text(label)

    def set_description(self, text: str) -> None:
        self._label.text(str(text))

    def update(self, n: int = 1) -> None:
        self.current += n
        fraction = min(self.current / self.total, 1.0)
        self._bar.progress(int(fraction * 100))

    def close(self) -> None:
        self._bar.progress(100)

    def reset(self, total: int, label: Optional[str] = None) -> None:
        self.total = max(1, int(total))
        self.current = 0
        if label:
            self._label.text(label)
        self._bar.progress(0)


def _init_state() -> None:
    st.session_state.setdefault("features_df", None)
    st.session_state.setdefault("features_source", None)
    st.session_state.setdefault("features_path", None)
    st.session_state.setdefault("metrics_df", None)
    st.session_state.setdefault("metrics_method", None)
    st.session_state.setdefault("metrics_feature_cols", None)
    st.session_state.setdefault("metrics_rows", None)
    st.session_state.setdefault("metrics_x_scaled", None)
    st.session_state.setdefault("metrics_path", None)
    st.session_state.setdefault("metrics_params", None)


def _reset_metrics_state() -> None:
    st.session_state["metrics_df"] = None
    st.session_state["metrics_method"] = None
    st.session_state["metrics_feature_cols"] = None
    st.session_state["metrics_rows"] = None
    st.session_state["metrics_x_scaled"] = None
    st.session_state["metrics_path"] = None
    st.session_state["metrics_params"] = None


def _store_features(df: pd.DataFrame, source: str, path: Optional[Path]) -> None:
    st.session_state["features_df"] = df
    st.session_state["features_source"] = source
    st.session_state["features_path"] = str(path) if path else None
    _reset_metrics_state()


def _clear_features() -> None:
    st.session_state["features_df"] = None
    st.session_state["features_source"] = None
    st.session_state["features_path"] = None
    _reset_metrics_state()


def _run_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{stamp}"


def _write_run_log(entry: dict) -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _feature_file_info() -> tuple[Optional[str], Optional[str]]:
    feature_path = st.session_state.get("features_path")
    if not feature_path:
        return None, None
    path = Path(feature_path)
    return str(path), path.name


def _save_metrics_snapshot(
    metrics_df: pd.DataFrame,
    method: str,
    k_min: int,
    k_max: int,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = RESULTS_DIR / f"cluster_metrics_{method}_k{k_min}-{k_max}_{stamp}.csv"
    metrics_df.to_csv(path, index=False)
    return path


def _save_gmm_comparison_metrics(
    metrics_df: pd.DataFrame, ks: list[int]
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ks_label = "-".join(str(k) for k in ks)
    path = RESULTS_DIR / f"cluster_metrics_gmm_compare_k{ks_label}_{stamp}.csv"
    metrics_df.to_csv(path, index=False)
    return path


def _save_gmm_comparison_quality(
    quality_map: dict[int, pd.DataFrame], ks: list[int]
) -> Optional[Path]:
    rows = []
    for k in ks:
        quality_df = quality_map.get(k)
        if quality_df is None or quality_df.empty:
            continue
        df_copy = quality_df.copy()
        df_copy.insert(0, "k", int(k))
        rows.append(df_copy)
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ks_label = "-".join(str(k) for k in ks)
    path = RESULTS_DIR / f"cluster_quality_gmm_compare_k{ks_label}_{stamp}.csv"
    combined.to_csv(path, index=False)
    return path


def _save_k_optimo_results(
    results_df: pd.DataFrame,
    method: str,
    k_min: int,
    k_max: int,
    k_step: int,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = RESULTS_DIR / (
        f"cluster_k_optimo_{method}_k{k_min}-{k_max}_s{k_step}_{stamp}.csv"
    )
    results_df.to_csv(path, index=False)
    return path


def _log_metrics_event(
    method: str,
    feature_cols: List[str],
    rows: int,
    metrics_path: Path,
    params: dict,
) -> None:
    feature_path, feature_file = _feature_file_info()
    entry = {
        "run_id": _run_id("metrics"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "metrics",
        "method": method,
        "feature_cols": list(feature_cols),
        "rows": int(rows),
        "metrics_path": str(metrics_path),
        "metrics_file": metrics_path.name,
        "metrics_params": params,
        "feature_source": st.session_state.get("features_source"),
        "feature_path": feature_path,
        "feature_file": feature_file,
    }
    _write_run_log(entry)


def _log_gmm_comparison_run(
    *,
    feature_cols: List[str],
    rows: int,
    feature_path: Path,
    metrics_path: Path,
    metrics_params: dict,
    quality_path: Optional[Path] = None,
    train_params: Optional[dict] = None,
    train_distribution: Optional[dict] = None,
) -> None:
    entry = {
        "run_id": _run_id("gmm_compare"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "gmm_comparison",
        "method": "gmm",
        "feature_cols": list(feature_cols),
        "rows": int(rows),
        "metrics_path": str(metrics_path),
        "metrics_file": metrics_path.name,
        "metrics_params": metrics_params,
        "feature_source": "duckdb",
        "feature_path": str(feature_path),
        "feature_file": feature_path.name,
    }
    if quality_path is not None:
        entry["quality_path"] = str(quality_path)
        entry["quality_file"] = quality_path.name
    if train_params:
        entry["train_params"] = train_params
    if train_distribution:
        entry["distribution"] = train_distribution
    _write_run_log(entry)


def _log_k_optimo_run(
    *,
    method: str,
    feature_cols: List[str],
    rows: int,
    results_path: Path,
    params: dict,
    train_params: Optional[dict] = None,
    train_distribution: Optional[dict] = None,
) -> None:
    feature_path, feature_file = _feature_file_info()
    entry = {
        "run_id": _run_id("k_optimo"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "k_optimo",
        "method": method,
        "feature_cols": list(feature_cols),
        "rows": int(rows),
        "metrics_path": str(results_path),
        "metrics_file": results_path.name,
        "metrics_params": params,
        "feature_source": st.session_state.get("features_source"),
        "feature_path": feature_path,
        "feature_file": feature_file,
    }
    if train_params:
        entry["train_params"] = train_params
    if train_distribution:
        entry["distribution"] = train_distribution
    _write_run_log(entry)


def _log_cluster_run(
    *,
    method: str,
    feature_cols: List[str],
    rows: int,
    labels_path: Path,
    summary_path: Path,
    descriptive_path: Path,
    metrics_path: Optional[Path],
    metrics_params: Optional[dict],
    extra_params: Optional[dict] = None,
    train_params: Optional[dict] = None,
    train_distribution: Optional[dict] = None,
) -> None:
    feature_path, feature_file = _feature_file_info()
    entry = {
        "run_id": _run_id("cluster"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "clustering",
        "method": method,
        "feature_cols": list(feature_cols),
        "rows": int(rows),
        "labels_path": str(labels_path),
        "labels_file": labels_path.name,
        "summary_path": str(summary_path),
        "summary_file": summary_path.name,
        "descriptive_path": str(descriptive_path),
        "descriptive_file": descriptive_path.name,
        "feature_source": st.session_state.get("features_source"),
        "feature_path": feature_path,
        "feature_file": feature_file,
    }
    if metrics_path is not None:
        entry["metrics_path"] = str(metrics_path)
        entry["metrics_file"] = metrics_path.name
    if metrics_params is not None:
        entry["metrics_params"] = metrics_params
    if extra_params:
        entry["params"] = extra_params
    if train_params:
        entry["train_params"] = train_params
    if train_distribution:
        entry["distribution"] = train_distribution
    _write_run_log(entry)


def _log_frequent_definition(
    *,
    min_passes: int,
    min_days: int,
    min_months: int,
    total_rows: int,
    frequent_rows: int,
    rare_rows: int,
) -> None:
    feature_path, feature_file = _feature_file_info()
    frequent_share = frequent_rows / total_rows if total_rows else 0.0
    rare_share = rare_rows / total_rows if total_rows else 0.0
    entry = {
        "run_id": _run_id("freq_def"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": "frequent_definition",
        "method": "frequent_definition",
        "rows": int(total_rows),
        "feature_source": st.session_state.get("features_source"),
        "feature_path": feature_path,
        "feature_file": feature_file,
        "params": {
            "min_total_passes": int(min_passes),
            "min_days_active": int(min_days),
            "min_months_active": int(min_months),
        },
        "distribution": {
            "frequent_rows": int(frequent_rows),
            "rare_rows": int(rare_rows),
            "frequent_share": frequent_share,
            "rare_share": rare_share,
        },
    }
    _write_run_log(entry)


def _get_features() -> Optional[pd.DataFrame]:
    df = st.session_state.get("features_df")
    if isinstance(df, pd.DataFrame):
        return df
    return None


def _validate_features(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = sorted(REQUIRED_FEATURE_COLS - set(df.columns))
    return not missing, missing


def _render_flow_summary() -> Optional[object]:
    try:
        summary = get_flow_db_summary()
    except ImportError as exc:
        st.error(str(exc))
        return None
    if summary.row_count == 0:
        st.warning("La base de flujos esta vacia. Importe datos primero.")
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


def _estimate_batch_ranges(
    summary,
    batch_mode: str,
    date_start: Optional[pd.Timestamp],
    date_end: Optional[pd.Timestamp],
    monthly_weighting: bool,
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
    split_months = batch_mode == "week" and monthly_weighting
    return _build_batch_ranges(
        range_start, range_end, batch_mode, split_months=split_months
    )


def _build_sample_mode_selector() -> str:
    return st.radio(
        "Muestreo",
        ["Todo", "Rango de fechas", "Porcentaje"],
        horizontal=True,
    )


def _build_sample_inputs(
    summary,
    mode: str,
) -> Tuple[FlowSampleSelection, bool, bool]:
    row_limit = None
    date_start = None
    date_end = None
    range_valid = True

    if mode == "Rango de fechas":
        default_start, default_end = _date_defaults(summary)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Fecha inicio", value=default_start)
        with col2:
            end_date = st.date_input("Fecha fin", value=default_end)
        use_time = st.checkbox("Usar horas en el rango", value=False)
        if use_time:
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.time_input("Hora inicio", value=dt_time(0, 0))
            with col2:
                end_time = st.time_input("Hora fin", value=dt_time(23, 59))
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
            percent = st.slider("Porcentaje", min_value=1, max_value=100, value=10)
            row_limit = max(1, int(summary.row_count * (percent / 100.0)))
            st.caption(f"Se consultaran {row_limit:,} filas.")

    sample = FlowSampleSelection(
        date_start=date_start,
        date_end=date_end,
        row_limit=row_limit,
    )
    return sample, mode == "Porcentaje" and row_limit is not None, range_valid


def _render_feature_loader() -> Optional[pd.DataFrame]:
    #st.subheader("Variables para clustering")

    features_df = _get_features()
    has_memory = features_df is not None and not features_df.empty
    source_key = "feature_source"
    options = ["Cargar existentes", "Calcular nuevas"]
    if has_memory:
        options.append("En memoria")
    alias_map = {
        "Cargadas en memoria": "En memoria",
        "Variables en memoria": "En memoria",
        "Usar variables existentes": "Cargar existentes",
    }
    pending_source = st.session_state.pop("feature_source_request", None)
    if pending_source in alias_map:
        pending_source = alias_map[pending_source]
    if pending_source in options:
        st.session_state[source_key] = pending_source
    if source_key in st.session_state:
        current = st.session_state[source_key]
        if current in alias_map:
            st.session_state[source_key] = alias_map[current]
        if st.session_state[source_key] not in options:
            st.session_state[source_key] = "En memoria" if has_memory else options[0]
    if source_key not in st.session_state:
        st.session_state[source_key] = "En memoria" if has_memory else options[0]

    notice = st.session_state.pop("features_notice", None)
    if notice:
        st.success(notice)
    source = st.radio(
        "Fuente",
        options,
        horizontal=True,
        key=source_key,
    )

    if source == "En memoria":
        if not has_memory:
            st.info("No hay variables cargadas en memoria.")
            return _get_features()
        if st.button("Limpiar variables en memoria"):
            _clear_features()
            st.session_state["feature_source_request"] = options[0]
            st.session_state["features_notice"] = "Datos en memoria limpiados."
            st.rerun()
        _render_feature_preview(features_df)
        return _get_features()

    if source == "Cargar existentes":
        db_paths = list_cluster_feature_db_paths()
        if not db_paths:
            st.warning("No se encontraron archivos cluster_features*.duckdb en Resultados.")
            return _get_features()
        names = [path.name for path in db_paths]
        selected = st.selectbox("Archivo de variables", names)
        if st.button("Cargar variables"):
            path = RESULTS_DIR / selected
            try:
                df = load_cluster_features_duckdb(path)
            except ImportError as exc:
                st.error(str(exc))
                return _get_features()
            if df.empty:
                st.warning("El archivo existe pero no tiene datos.")
                return _get_features()
            valid, missing = _validate_features(df)
            if not valid:
                st.error(
                    "Faltan columnas requeridas: "
                    + ", ".join(missing)
                )
                return _get_features()
            _store_features(df, "duckdb", path)
            st.session_state["feature_source_request"] = "En memoria"
            st.session_state["features_notice"] = (
                f"Variables cargadas: {len(df):,} filas."
            )
            st.rerun()
        return _get_features()

    summary = _render_flow_summary()
    if summary is None or summary.row_count == 0:
        return _get_features()

    mode = _build_sample_mode_selector()
    
    use_batches = st.checkbox(
        "Procesar por lotes (mes/semana)",
        value=False,
    )

    with st.form("cluster_features_form"):
        sample, percent_mode, range_valid = _build_sample_inputs(summary, mode)
        
        if percent_mode and use_batches:
             st.warning("El muestreo por porcentaje no ignora la opcion de lotes (no compatible).")

        batch_mode = "month"
        monthly_weighting = False
        if use_batches:
            batch_mode = st.radio("Modo de lotes", ["month", "week"], horizontal=True)
            monthly_weighting = st.checkbox(
                "Ponderar variables por mes antes de consolidar",
                value=False,
            )
            if batch_mode == "week" and monthly_weighting:
                st.caption(
                    "Las semanas se consolidan por mes antes de generar la base final."
                )
        else:
            monthly_weighting = st.checkbox(
                "Ponderar variables por mes antes de consolidar",
                value=False,
            )

        run_calculation = st.form_submit_button("Calcular variables", disabled=not range_valid)

    if run_calculation:
        fc = FlowColumns()
        progress_container = st.container()
        batch_db_path = None
        if use_batches:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_suffix = f"batch_{batch_mode}_{stamp}"
            batch_db_path = _build_cluster_feature_db_path(batch_suffix)
        with st.spinner("Calculando variables..."):
            if use_batches:
                ranges = _estimate_batch_ranges(
                    summary,
                    batch_mode,
                    sample.date_start,
                    sample.date_end,
                    monthly_weighting,
                )
                with progress_container:
                    batch_progress = StreamlitProgress(
                        total=max(1, len(ranges)),
                        label="Procesando lotes",
                    )
                try:
                    features_df, batch_paths = _clusterize_in_batches(
                        fc,
                        TTC_MAX_BY_PORTICO,
                        batch_mode,
                        monthly_weighting,
                        date_start=sample.date_start,
                        date_end=sample.date_end,
                        batch_db_path=batch_db_path,
                        progress=batch_progress,
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    return _get_features()
                finally:
                    batch_progress.close()
                if batch_paths:
                    st.caption(f"Lotes generados: {len(batch_paths)} archivos.")
            else:
                with progress_container:
                    load_progress = StreamlitProgress(
                        total=4,
                        label="Cargando flujos",
                    )
                try:
                    flujos_df = load_flujos(sample=sample, progress=load_progress)
                finally:
                    load_progress.close()
                if flujos_df is None or flujos_df.empty:
                    st.warning("No se pudieron cargar flujos para calcular variables.")
                    return _get_features()

                with progress_container:
                    calc_progress = StreamlitProgress(
                        total=5,
                        label="Calculando variables",
                    )
                    headway_progress = StreamlitProgress(
                        total=1,
                        label="Headway/TTC por portico-carril",
                    )
                try:
                    features_df = Clusterization(
                        flujos_df,
                        fc,
                        monthly_weighting=monthly_weighting,
                        progress=calc_progress,
                        group_progress=headway_progress,
                    )
                finally:
                    calc_progress.close()
                    headway_progress.close()
        if features_df is None or features_df.empty:
            st.warning("No se encontraron registros validos.")
            return _get_features()
        source_label = "computed (DB)" if batch_db_path else "computed"
        _store_features(features_df, source_label, batch_db_path)
        st.session_state["feature_source_request"] = "En memoria"
        st.session_state["features_notice"] = (
            f"Variables calculadas: {len(features_df):,} filas."
            + (f" Guardadas en {batch_db_path.name}." if batch_db_path else "")
        )
        st.rerun()

    return _get_features()


def _render_feature_preview(features_df: pd.DataFrame) -> None:
    if features_df is None or features_df.empty:
        st.info("No hay variables cargadas.")
        return
    st.caption(
        f"Fuente: {st.session_state.get('features_source') or '-'} | "
        f"Filas: {len(features_df):,} | Columnas: {len(features_df.columns)}"
    )
    if st.session_state.get("features_path"):
        st.caption(f"Archivo: {st.session_state.get('features_path')}")
    preview_rows = st.slider("Filas de vista previa", 5, 100, 20, step=5)
    st.dataframe(features_df.head(preview_rows), width="stretch")

    st.subheader("Guardar variables en DuckDB")
    suffix = st.text_input(
        "Sufijo del archivo (opcional)",
        help="Se guardara como cluster_features(<sufijo>).duckdb",
    )
    if st.button("Guardar en DuckDB"):
        try:
            db_path = _build_cluster_feature_db_path(suffix)
            saved = save_cluster_features_duckdb(features_df, db_path=db_path)
        except ImportError as exc:
            st.error(str(exc))
            return
        st.session_state["features_path"] = str(saved)
        if st.session_state.get("features_source") is None:
            st.session_state["features_source"] = "duckdb"
        st.success(f"Guardado en: {saved}")


def _feature_selection(
    features_df: pd.DataFrame,
    *,
    key: Optional[str] = None,
    label: str = "Variables para clustering",
) -> List[str]:
    numeric_cols = (
        features_df.select_dtypes(include=["number"]).columns.tolist()
        if features_df is not None
        else []
    )
    numeric_cols = [col for col in numeric_cols if col != "cluster_label"]
    numeric_cols = _order_feature_columns(numeric_cols)
    default_cols = _choose_feature_columns(features_df) if features_df is not None else []
    default_cols = [col for col in default_cols if col in numeric_cols]

    if not numeric_cols:
        st.warning("No hay columnas numericas para clustering.")
        return []

    selected = st.multiselect(
        label,
        numeric_cols,
        default=default_cols,
        key=key,
    )
    return selected


def _prepare_cluster_data(
    features_df: pd.DataFrame, feature_cols: List[str]
) -> Optional[pd.DataFrame]:
    if features_df is None or features_df.empty:
        st.warning("No hay variables cargadas.")
        return None
    if not feature_cols:
        st.warning("Seleccione al menos una variable.")
        return None
    cluster_df = _prepare_cluster_features(features_df, feature_cols)
    dropped = len(features_df) - len(cluster_df)
    if dropped:
        st.info(f"Se descartaron {dropped:,} filas por valores invalidos.")
    if cluster_df.empty:
        st.warning("No hay datos validos despues de filtrar.")
        return None
    return cluster_df


def _store_metrics(
    method: str,
    feature_cols: List[str],
    rows: int,
    metrics_df: pd.DataFrame,
    x_scaled,
    metrics_path: Optional[Path],
    metrics_params: Optional[dict],
) -> None:
    st.session_state["metrics_method"] = method
    st.session_state["metrics_feature_cols"] = list(feature_cols)
    st.session_state["metrics_rows"] = rows
    st.session_state["metrics_df"] = metrics_df
    st.session_state["metrics_x_scaled"] = x_scaled
    st.session_state["metrics_path"] = str(metrics_path) if metrics_path else None
    st.session_state["metrics_params"] = metrics_params


def _get_cached_metrics(
    method: str, feature_cols: List[str], rows: int
) -> Tuple[Optional[pd.DataFrame], Optional[object], Optional[Path], Optional[dict]]:
    if (
        st.session_state.get("metrics_method") == method
        and st.session_state.get("metrics_feature_cols") == list(feature_cols)
        and st.session_state.get("metrics_rows") == rows
    ):
        metrics_path = st.session_state.get("metrics_path")
        if metrics_path:
            metrics_path = Path(metrics_path)
        return (
            st.session_state.get("metrics_df"),
            st.session_state.get("metrics_x_scaled"),
            metrics_path,
            st.session_state.get("metrics_params"),
        )
    return None, None, None, None


def _render_export_inputs(
    features_df: pd.DataFrame, metrics_df: Optional[pd.DataFrame]
) -> None:
    if metrics_df is not None and st.button("Exportar metricas a CSV"):
        path = save_cluster_metrics(metrics_df)
        st.success(f"Metricas guardadas en: {path}")


def _render_kmeans(
    features_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    # 1. Retrieve Frequent/Rare split settings
    min_days = st.session_state.get("freq_min_days", 1)
    min_months = st.session_state.get("freq_min_months", 1)
    min_passes = st.session_state.get("freq_min_passes", 20)
    
    # 2. Split Data
    # We need to split based on original features (metadata), then align with cluster_df
    # which has only feature columns and dropped NaNs.
    freq_full, rare_full = split_frequent_drivers(
        features_df,
        min_total_passes=min_passes,
        min_days_active=min_days,
        min_months_active=min_months,
    )
    
    # Filter cluster_df (aligned by index)
    cluster_freq = cluster_df.loc[cluster_df.index.intersection(freq_full.index)]
    cluster_rare = cluster_df.loc[cluster_df.index.intersection(rare_full.index)]

    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    total_train_rows = len(cluster_df)
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_train_rows if total_train_rows else 0.0
        ),
        "rare_share": (
            len(cluster_rare) / total_train_rows if total_train_rows else 0.0
        ),
    }

    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    total_train_rows = len(cluster_df)
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_train_rows if total_train_rows else 0.0
        ),
        "rare_share": (
            len(cluster_rare) / total_train_rows if total_train_rows else 0.0
        ),
    }

    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    total_train_rows = len(cluster_df)
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_train_rows if total_train_rows else 0.0
        ),
        "rare_share": (
            len(cluster_rare) / total_train_rows if total_train_rows else 0.0
        ),
    }

    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    total_train_rows = len(cluster_df)
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_train_rows if total_train_rows else 0.0
        ),
        "rare_share": (
            len(cluster_rare) / total_train_rows if total_train_rows else 0.0
        ),
    }
    
    st.caption(
        f"Datos validos para clustering: {len(cluster_df):,} "
        f"(Frecuentes: {len(cluster_freq):,}, Raros: {len(cluster_rare):,})"
    )

    max_k_allowed = len(cluster_freq) - 1
    if max_k_allowed < 2:
        st.warning("No hay suficientes muestras 'Frecuentes' para entrenar K-means (minimo 3).")
        return

    use_minibatch = st.checkbox("Usar MiniBatchKMeans para metricas", value=True)
    
    col_k, col_conf = st.columns(2)
    with col_k:
        k_min = st.number_input("K minimo", min_value=2, max_value=max_k_allowed, value=2)
        k_max_default = min(5, max_k_allowed)
        k_max = st.number_input(
            "K maximo",
            min_value=int(k_min),
            max_value=max_k_allowed,
            value=int(k_max_default),
        )
    with col_conf:
        confidence_pct = st.slider(
            "Umbral de Confianza (Percentil de Distancia)", 
            min_value=50, max_value=100, value=95,
            help="Si un conductor infrecuente está más lejos de su centroide que este percentil de los frecuentes, se marca como Desconocido (-1)."
        )

    metrics_df, x_scaled, metrics_path, metrics_params = _get_cached_metrics(
        "kmeans", feature_cols, len(cluster_freq)
    )
    
    # Calculate metrics on FREQUENT only
    if st.button("Calcular metricas K-means (sobre Frecuentes)"):
        with st.spinner("Calculando metricas..."):
            try:
                metrics_df, _scaler, x_scaled = compute_kmeans_metrics(
                    cluster_freq,
                    feature_cols,
                    k_min=int(k_min),
                    k_max=int(k_max),
                    use_minibatch=use_minibatch,
                    show_progress=False,
                )
            except ImportError as exc:
                st.error(str(exc))
                return
        metrics_path = _save_metrics_snapshot(
            metrics_df, "kmeans", int(k_min), int(k_max)
        )
        metrics_params = {
            "k_min": int(k_min),
            "k_max": int(k_max),
            "use_minibatch": bool(use_minibatch),
            "subset": "frequent",
            "min_passes": min_passes,
        }
        _store_metrics(
            "kmeans",
            feature_cols,
            len(cluster_freq),
            metrics_df,
            x_scaled,
            metrics_path,
            metrics_params,
        )
        _log_metrics_event(
            "kmeans",
            feature_cols,
            len(cluster_freq),
            metrics_path,
            metrics_params,
        )

    if metrics_df is not None and not metrics_df.empty:
        st.subheader("Metricas K-means")
        st.dataframe(metrics_df, width="stretch")
        best_sil = int(metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"])
        suggested_k = best_sil
    else:
        suggested_k = 2

    k_choice = st.number_input(
        "K para aplicar",
        min_value=2,
        max_value=max_k_allowed,
        value=int(suggested_k),
    )

    if st.button("Ejecutar K-means"):
        with st.spinner("Ejecutando K-means (Entrena Frecuentes -> Asigna Todos)..."):
            try:
                clustered_full, model, threshold_used = assign_clusters_kmeans(
                    frequent_df=cluster_freq,
                    rare_df=cluster_rare,
                    feature_cols=feature_cols,
                    k=int(k_choice),
                    confidence_threshold_percentile=float(confidence_pct),
                    random_state=42
                )
            except Exception as exc:
                st.error(f"Error en clustering: {exc}")
                return

        # Assign back metadata for saving (restore dropped columns if any, though here we just save cluster labels)
        # We need to map cluster labels back to features_df index or just save what we have.
        # clustered_full has feature_cols + 'cluster_label' + 'confidence_score' + 'is_rare'
        
        # Merge basic metadata like plate for the output
        if "plate" in features_df.columns:
            clustered_full = clustered_full.join(features_df[["plate"]], how="left", rsuffix="_meta")
            if "plate_meta" in clustered_full.columns:
                 clustered_full["plate"] = clustered_full["plate_meta"].fillna(clustered_full["plate"])
                 del clustered_full["plate_meta"]

        labels_path = save_cluster_labels(clustered_full, "kmeans", int(k_choice))
        
        # Filter Unknowns (-1) for summary stats? usually we want summary of defined clusters
        clustered_assigned = clustered_full[clustered_full["cluster_label"] != -1]
        
        summary_df = build_cluster_summary(clustered_assigned, feature_cols)
        descriptive_df = build_cluster_descriptive(clustered_assigned, feature_cols)
        summary_path = save_cluster_summary(summary_df, "kmeans", int(k_choice))
        descriptive_path = save_cluster_descriptive(
            descriptive_df, "kmeans", int(k_choice)
        )
        
        _log_cluster_run(
            method="kmeans",
            feature_cols=feature_cols,
            rows=len(clustered_full),
            labels_path=labels_path,
            summary_path=summary_path,
            descriptive_path=descriptive_path,
            metrics_path=metrics_path,
            metrics_params=metrics_params,
            extra_params={
                "k": int(k_choice), 
                "confidence_pct": confidence_pct, 
                "threshold_val": threshold_used
            },
            train_params=train_params,
            train_distribution=train_distribution,
        )
        st.success("Clustering completado.")
        
        # Coverage stats
        n_total = len(clustered_full)
        n_unknown = (clustered_full["cluster_label"] == -1).sum()
        n_assigned = n_total - n_unknown
        st.info(f"Asignados: {n_assigned:,} ({n_assigned/n_total:.1%}) | Desconocidos: {n_unknown:,} ({n_unknown/n_total:.1%})")
        st.caption(f"Umbral de distancia usado: {threshold_used:.4f}")
        
        st.caption(f"Etiquetas: {labels_path}")
        st.caption(f"Resumen: {summary_path}")
        st.subheader("Resumen por cluster")
        st.dataframe(summary_df, width="stretch")

    _render_export_inputs(features_df, metrics_df)


def _render_gmm(
    features_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    # 1. Retrieve Frequent/Rare split settings
    min_days = st.session_state.get("freq_min_days", 1)
    min_months = st.session_state.get("freq_min_months", 1)
    min_passes = st.session_state.get("freq_min_passes", 20)
    
    # 2. Split Data
    freq_full, rare_full = split_frequent_drivers(
        features_df,
        min_total_passes=min_passes,
        min_days_active=min_days,
        min_months_active=min_months,
    )
    
    cluster_freq = cluster_df.loc[cluster_df.index.intersection(freq_full.index)]
    cluster_rare = cluster_df.loc[cluster_df.index.intersection(rare_full.index)]

    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    total_train_rows = len(cluster_df)
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_train_rows if total_train_rows else 0.0
        ),
        "rare_share": (
            len(cluster_rare) / total_train_rows if total_train_rows else 0.0
        ),
    }
    
    st.caption(
        f"Datos validos para clustering: {len(cluster_df):,} "
        f"(Frecuentes: {len(cluster_freq):,}, Raros: {len(cluster_rare):,})"
    )

    max_k_allowed = len(cluster_freq) - 1
    if max_k_allowed < 2:
        st.warning("No hay suficientes muestras 'Frecuentes' para GMM (minimo 3 filas).")
        return

    col_k, col_conf = st.columns(2)
    with col_k:
        k_min = st.number_input("K minimo", min_value=2, max_value=max_k_allowed, value=2)
        k_max_default = min(5, max_k_allowed)
        k_max = st.number_input(
            "K maximo",
            min_value=int(k_min),
            max_value=max_k_allowed,
            value=int(k_max_default),
        )
    with col_conf:
        # Default probability 0.70
        confidence_proba = st.slider(
            "Umbral de Confianza (Probabilidad)", 
            min_value=0.0, max_value=1.0, value=0.70, step=0.05,
            help="Si la probabilidad máxima de pertenencia a un cluster es menor a este valor, se marca como Desconocido (-1)."
        )
    
    criterio = st.selectbox("Criterio sugerido", ["bic", "aic"], index=0)

    metrics_df, x_scaled, metrics_path, metrics_params = _get_cached_metrics(
        "gmm", feature_cols, len(cluster_freq)
    )
    if st.button("Calcular metricas GMM (sobre Frecuentes)"):
        with st.spinner("Calculando metricas..."):
            try:
                metrics_df, _scaler, x_scaled = compute_gmm_metrics(
                    cluster_freq,
                    feature_cols,
                    k_min=int(k_min),
                    k_max=int(k_max),
                    show_progress=False,
                )
            except ImportError as exc:
                st.error(str(exc))
                return
        metrics_path = _save_metrics_snapshot(
            metrics_df, "gmm", int(k_min), int(k_max)
        )
        metrics_params = {
            "k_min": int(k_min),
            "k_max": int(k_max),
            "covariance_type": "full",
            "max_iter": 200,
            "n_init": 3,
            "subset": "frequent",
            "min_passes": min_passes,
        }
        _store_metrics(
            "gmm",
            feature_cols,
            len(cluster_freq),
            metrics_df,
            x_scaled,
            metrics_path,
            metrics_params,
        )
        _log_metrics_event(
            "gmm",
            feature_cols,
            len(cluster_freq),
            metrics_path,
            metrics_params,
        )

    if metrics_df is not None and not metrics_df.empty:
        st.subheader("Metricas GMM")
        st.dataframe(metrics_df, width="stretch")
        best_bic = int(metrics_df.loc[metrics_df["bic"].idxmin(), "k"])
        best_aic = int(metrics_df.loc[metrics_df["aic"].idxmin(), "k"])
        best_k = best_aic if criterio == "aic" else best_bic
        st.caption(f"Sugerencia ({criterio.upper()}): k={best_k}")
        suggested_k = best_k
    else:
        suggested_k = 2

    k_choice = st.number_input(
        "K para aplicar",
        min_value=2,
        max_value=max_k_allowed,
        value=int(suggested_k),
    )

    if st.button("Ejecutar GMM"):
        with st.spinner("Ejecutando GMM (Entrena Frecuentes -> Asigna Todos)..."):
            try:
                clustered_full, model, threshold_used = assign_clusters_gmm(
                    frequent_df=cluster_freq,
                    rare_df=cluster_rare,
                    feature_cols=feature_cols,
                    k=int(k_choice),
                    confidence_threshold_proba=float(confidence_proba),
                    random_state=42
                )
            except Exception as exc:
                st.error(f"Error en clustering: {exc}")
                return

        # Merge metadata
        if "plate" in features_df.columns:
            clustered_full = clustered_full.join(features_df[["plate"]], how="left", rsuffix="_meta")
            if "plate_meta" in clustered_full.columns:
                 clustered_full["plate"] = clustered_full["plate_meta"].fillna(clustered_full["plate"])
                 del clustered_full["plate_meta"]

        labels_path = save_cluster_labels(clustered_full, "gmm", int(k_choice))
        
        # Summary for assigned only
        clustered_assigned = clustered_full[clustered_full["cluster_label"] != -1]
        
        summary_df = build_cluster_summary(clustered_assigned, feature_cols)
        descriptive_df = build_cluster_descriptive(clustered_assigned, feature_cols)
        summary_path = save_cluster_summary(summary_df, "gmm", int(k_choice))
        descriptive_path = save_cluster_descriptive(
            descriptive_df, "gmm", int(k_choice)
        )
        
        _log_cluster_run(
            method="gmm",
            feature_cols=feature_cols,
            rows=len(clustered_full),
            labels_path=labels_path,
            summary_path=summary_path,
            descriptive_path=descriptive_path,
            metrics_path=metrics_path,
            metrics_params=metrics_params,
            extra_params={
                "k": int(k_choice),
                "confidence_proba": confidence_proba,
            },
            train_params=train_params,
            train_distribution=train_distribution,
        )
        st.success("Clustering completado.")
        
        # Coverage stats
        n_total = len(clustered_full)
        n_unknown = (clustered_full["cluster_label"] == -1).sum()
        n_assigned = n_total - n_unknown
        st.info(f"Asignados: {n_assigned:,} ({n_assigned/n_total:.1%}) | Desconocidos: {n_unknown:,} ({n_unknown/n_total:.1%})")
        
        st.caption(f"Etiquetas: {labels_path}")
        st.caption(f"Resumen: {summary_path}")
        st.subheader("Resumen por cluster")
        st.dataframe(summary_df, width="stretch")

    _render_export_inputs(features_df, metrics_df)



def _render_hdbscan(
    features_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    min_days = st.session_state.get("freq_min_days", 1)
    min_months = st.session_state.get("freq_min_months", 1)
    min_passes = st.session_state.get("freq_min_passes", 20)

    freq_full, rare_full = split_frequent_drivers(
        features_df,
        min_total_passes=min_passes,
        min_days_active=min_days,
        min_months_active=min_months,
    )
    cluster_freq = cluster_df.loc[cluster_df.index.intersection(freq_full.index)]
    cluster_rare = cluster_df.loc[cluster_df.index.intersection(rare_full.index)]
    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    total_train_rows = len(cluster_df)
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_train_rows if total_train_rows else 0.0
        ),
        "rare_share": (
            len(cluster_rare) / total_train_rows if total_train_rows else 0.0
        ),
    }

    min_cluster_size = st.number_input(
        "min_cluster_size",
        min_value=2,
        value=15,
    )
    define_min_samples = st.checkbox("Definir min_samples", value=False)
    min_samples = None
    if define_min_samples:
        min_samples = st.number_input(
            "min_samples",
            min_value=1,
            value=int(min_cluster_size),
        )

    if st.button("Ejecutar HDBSCAN"):
        if len(cluster_df) < int(min_cluster_size):
            st.warning("No hay suficientes muestras para el min_cluster_size seleccionado.")
            return
        try:
            import hdbscan  # type: ignore
        except ImportError:
            st.error("hdbscan no esta instalado. Ejecute `pip install hdbscan`.")
            return

        try:
            x_scaled = _scale_cluster_features(cluster_df, feature_cols)
        except ImportError as exc:
            st.error(str(exc))
            return

        with st.spinner("Ejecutando HDBSCAN..."):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples) if min_samples is not None else None,
            )
            labels = clusterer.fit_predict(x_scaled)
            clustered = cluster_df.copy()
            clustered["cluster_label"] = labels

        labels_path = save_cluster_labels(clustered, "hdbscan")
        summary_df = build_cluster_summary(clustered, feature_cols)
        descriptive_df = build_cluster_descriptive(clustered, feature_cols)
        summary_path = save_cluster_summary(summary_df, "hdbscan")
        descriptive_path = save_cluster_descriptive(descriptive_df, "hdbscan")
        _log_cluster_run(
            method="hdbscan",
            feature_cols=feature_cols,
            rows=len(cluster_df),
            labels_path=labels_path,
            summary_path=summary_path,
            descriptive_path=descriptive_path,
            metrics_path=None,
            metrics_params=None,
            extra_params={
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples) if min_samples is not None else None,
            },
            train_params=train_params,
            train_distribution=train_distribution,
        )
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        st.success(
            f"HDBSCAN completado. Clusters: {n_clusters} | Ruido: {noise_count}"
        )
        st.caption(f"Etiquetas: {labels_path}")
        st.caption(f"Resumen: {summary_path}")
        st.caption(f"Descriptivo: {descriptive_path}")
        st.subheader("Resumen por cluster")
        st.dataframe(summary_df, width="stretch")

    _render_export_inputs(features_df, None)


def _render_clustering_section(features_df: pd.DataFrame) -> None:
    #st.subheader("Clusterizacion")
    with st.expander(
        "Definición de Conductores Frecuentes (Train Set)", expanded=False
    ):
        c1, c2, c3 = st.columns(3)
        min_passes = c1.number_input(
            "Mínimo de pasadas (total_passes)",
            min_value=0,
            value=int(st.session_state.get("freq_min_passes", 20)),
            step=1,
            key="cluster_train_min_passes",
        )
        min_days = c2.number_input(
            "Mínimo de días activos (n_days_active)",
            min_value=1,
            value=int(st.session_state.get("freq_min_days", 1)),
            step=1,
            key="cluster_train_min_days",
        )
        min_months = c3.number_input(
            "Mínimo de meses activos (n_months_active)",
            min_value=1,
            value=int(st.session_state.get("freq_min_months", 1)),
            step=1,
            key="cluster_train_min_months",
        )
        st.session_state["freq_min_passes"] = int(min_passes)
        st.session_state["freq_min_days"] = int(min_days)
        st.session_state["freq_min_months"] = int(min_months)

        if st.button("Verificar distribución Frecuentes vs Raros"):
            freq, rare = split_frequent_drivers(
                features_df,
                min_total_passes=int(min_passes),
                min_days_active=min_days,
                min_months_active=min_months,
            )
            st.info(
                f"Frecuentes: {len(freq):,} ({len(freq)/len(features_df):.1%}) | "
                f"Raros: {len(rare):,} ({len(rare)/len(features_df):.1%})"
            )
            _log_frequent_definition(
                min_passes=int(min_passes),
                min_days=int(min_days),
                min_months=int(min_months),
                total_rows=len(features_df),
                frequent_rows=len(freq),
                rare_rows=len(rare),
            )

    feature_cols = _feature_selection(features_df)
    cluster_df = _prepare_cluster_data(features_df, feature_cols)
    if cluster_df is None:
        return

    st.caption(f"Filas disponibles para clustering: {len(cluster_df):,}")
    method = st.selectbox("Metodo", ["kmeans", "gmm", "hdbscan"])
    if method == "kmeans":
        _render_kmeans(features_df, cluster_df, feature_cols)
    elif method == "gmm":
        _render_gmm(features_df, cluster_df, feature_cols)
    else:
        _render_hdbscan(features_df, cluster_df, feature_cols)


def _render_k_optimo_experiment(features_df: pd.DataFrame) -> None:
    st.header("K optimum")
    st.caption(
        "Entrena clustering con Frecuentes, asigna a Raros y "
        "busca el K con menor % de excluidos."
    )

    with st.expander(
        "Definición de Conductores Frecuentes (Train Set)", expanded=False
    ):
        c1, c2, c3 = st.columns(3)
        min_passes = c1.number_input(
            "Mínimo de pasadas (total_passes)",
            min_value=0,
            value=int(st.session_state.get("freq_min_passes", 20)),
            step=1,
            key="k_optimo_min_passes",
        )
        min_days = c2.number_input(
            "Mínimo de días activos (n_days_active)",
            min_value=1,
            value=int(st.session_state.get("freq_min_days", 1)),
            step=1,
            key="k_optimo_min_days",
        )
        min_months = c3.number_input(
            "Mínimo de meses activos (n_months_active)",
            min_value=1,
            value=int(st.session_state.get("freq_min_months", 1)),
            step=1,
            key="k_optimo_min_months",
        )
        st.session_state["freq_min_passes"] = int(min_passes)
        st.session_state["freq_min_days"] = int(min_days)
        st.session_state["freq_min_months"] = int(min_months)

        if st.button(
            "Verificar distribución Frecuentes vs Raros",
            key="k_optimo_check_distribution",
        ):
            freq, rare = split_frequent_drivers(
                features_df,
                min_total_passes=int(min_passes),
                min_days_active=min_days,
                min_months_active=min_months,
            )
            st.info(
                f"Frecuentes: {len(freq):,} ({len(freq)/len(features_df):.1%}) | "
                f"Raros: {len(rare):,} ({len(rare)/len(features_df):.1%})"
            )

    feature_cols = _feature_selection(
        features_df,
        key="k_optimo_feature_cols",
        label="Variables para clustering",
    )
    cluster_df = _prepare_cluster_data(features_df, feature_cols)
    if cluster_df is None:
        return

    freq_full, rare_full = split_frequent_drivers(
        features_df,
        min_total_passes=int(min_passes),
        min_days_active=min_days,
        min_months_active=min_months,
    )
    cluster_freq = cluster_df.loc[cluster_df.index.intersection(freq_full.index)]
    cluster_rare = cluster_df.loc[cluster_df.index.intersection(rare_full.index)]

    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    total_rows = len(cluster_df)
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_rows if total_rows else 0.0
        ),
        "rare_share": len(cluster_rare) / total_rows if total_rows else 0.0,
    }

    st.caption(
        f"Datos validos para clustering: {len(cluster_df):,} "
        f"(Frecuentes: {len(cluster_freq):,}, Raros: {len(cluster_rare):,})"
    )

    max_k_allowed = len(cluster_freq) - 1
    if max_k_allowed < 2:
        st.warning(
            "No hay suficientes muestras 'Frecuentes' para iterar K (minimo 3)."
        )
        return

    method = st.selectbox(
        "Metodo de clustering",
        ["kmeans", "gmm"],
        key="k_optimo_method",
    )
    confidence_pct = 95.0
    confidence_proba = 0.7
    if method == "kmeans":
        confidence_pct = st.slider(
            "Umbral de Confianza (Percentil de Distancia)",
            min_value=50,
            max_value=100,
            value=95,
            key="k_optimo_confidence_pct",
            help=(
                "Si un conductor infrecuente esta mas lejos que este percentil "
                "de los frecuentes, se marca como Desconocido (-1)."
            ),
        )
    else:
        confidence_proba = st.slider(
            "Umbral de Confianza (Probabilidad)",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key="k_optimo_confidence_proba",
            help=(
                "Si la probabilidad maxima de pertenencia es menor a este "
                "valor, se marca como Desconocido (-1)."
            ),
        )

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        k_min = st.number_input(
            "K minimo",
            min_value=2,
            max_value=max_k_allowed,
            value=2,
            key="k_optimo_k_min",
        )
    with col_k2:
        k_max_default = min(max_k_allowed, max(int(k_min), 10))
        k_max = st.number_input(
            "K maximo",
            min_value=int(k_min),
            max_value=max_k_allowed,
            value=int(k_max_default),
            key="k_optimo_k_max",
        )
    with col_k3:
        k_step = st.number_input(
            "Paso",
            min_value=1,
            max_value=max_k_allowed,
            value=1,
            key="k_optimo_k_step",
        )

    if st.button("Calcular K optimo", key="k_optimo_run"):
        if k_min > k_max:
            st.error("K minimo no puede ser mayor que K maximo.")
            return
        if k_step < 1:
            st.error("Paso debe ser mayor o igual a 1.")
            return

        ks = list(range(int(k_min), int(k_max) + 1, int(k_step)))
        if not ks:
            st.warning("No hay valores de K para evaluar.")
            return

        progress = st.progress(0, text="Calculando exclusiones...")
        results: list[dict] = []
        for idx, k in enumerate(ks, start=1):
            progress.progress(
                int(idx / len(ks) * 100),
                text=f"Procesando K={k} ({idx}/{len(ks)})",
            )
            try:
                if method == "kmeans":
                    clustered_full, _, threshold_used = assign_clusters_kmeans(
                        frequent_df=cluster_freq,
                        rare_df=cluster_rare,
                        feature_cols=feature_cols,
                        k=int(k),
                        confidence_threshold_percentile=float(confidence_pct),
                        random_state=42,
                    )
                else:
                    clustered_full, _, threshold_used = assign_clusters_gmm(
                        frequent_df=cluster_freq,
                        rare_df=cluster_rare,
                        feature_cols=feature_cols,
                        k=int(k),
                        confidence_threshold_proba=float(confidence_proba),
                        random_state=42,
                    )
            except Exception as exc:
                results.append({"k": int(k), "error": str(exc)})
                continue

            rare_total = int(len(cluster_rare))
            if rare_total:
                rare_mask = clustered_full.get("is_rare")
                rare_rows = (
                    clustered_full.loc[rare_mask]
                    if rare_mask is not None
                    else clustered_full.iloc[0:0]
                )
                excluded_count = int(
                    (rare_rows["cluster_label"] == -1).sum()
                )
            else:
                excluded_count = 0
            excluded_pct = (
                excluded_count / rare_total if rare_total else 0.0
            )
            results.append(
                {
                    "k": int(k),
                    "excluded_pct": float(excluded_pct),
                    "excluded_count": int(excluded_count),
                    "rare_total": int(rare_total),
                    "assigned_count": int(rare_total - excluded_count),
                    "threshold_used": float(threshold_used),
                }
            )

        progress.empty()

        if not results:
            st.warning("No se generaron resultados.")
            return

        results_df = pd.DataFrame(results)
        if "k" in results_df.columns:
            results_df = results_df.sort_values("k").reset_index(drop=True)
        st.subheader("Resultados")
        st.dataframe(results_df, width="stretch")

        valid_df = results_df.copy()
        if "error" in valid_df.columns:
            valid_df = valid_df[valid_df["error"].isna()]
        if "excluded_pct" in valid_df.columns:
            valid_df = valid_df.dropna(subset=["excluded_pct"])

        best_row = None
        if not valid_df.empty and "excluded_pct" in valid_df.columns:
            best_row = valid_df.sort_values(["excluded_pct", "k"]).iloc[0]
            st.success(
                "K optimo: "
                f"{int(best_row['k'])} "
                f"(Excluidos: {best_row['excluded_pct']:.1%})"
            )
        else:
            st.warning("No se pudo determinar K optimo.")

        params = {
            "k_min": int(k_min),
            "k_max": int(k_max),
            "k_step": int(k_step),
            "confidence_threshold_percentile": (
                float(confidence_pct) if method == "kmeans" else None
            ),
            "confidence_threshold_proba": (
                float(confidence_proba) if method == "gmm" else None
            ),
        }
        if best_row is not None:
            params["best_k"] = int(best_row["k"])
            params["best_excluded_pct"] = float(best_row["excluded_pct"])

        results_path = _save_k_optimo_results(
            results_df, method, int(k_min), int(k_max), int(k_step)
        )
        st.caption(f"Resultados guardados en: {results_path}")
        _log_k_optimo_run(
            method=method,
            feature_cols=feature_cols,
            rows=len(cluster_df),
            results_path=results_path,
            params=params,
            train_params=train_params,
            train_distribution=train_distribution,
        )

@st.cache_data(show_spinner=False)
def _explorer_load_features(path: Path) -> pd.DataFrame:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "duckdb no esta instalado. Ejecute `pip install duckdb`."
        ) from exc

    conn = duckdb.connect(str(path), read_only=True)
    try:
        info = conn.execute("PRAGMA table_info('cluster_features')").fetchall()
        if not info:
            return pd.DataFrame()
        return conn.execute("SELECT * FROM cluster_features").df()
    finally:
        conn.close()


def _explorer_list_feature_dbs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("cluster_features*.duckdb"))


def _render_feature_explorer_tab() -> None:
    st.header("Feature explorer")

    db_paths = _explorer_list_feature_dbs()
    if not db_paths:
        st.error("No se encontraron archivos cluster_features*.duckdb en Resultados.")
        return

    file_names = [path.name for path in db_paths]
    #st.subheader("Archivo de variables")
    selected_name = st.selectbox(
        "Archivos de variables", file_names, key="feature_explorer_file"
    )
    selected_path = RESULTS_DIR / selected_name
 
    try:
        df = _explorer_load_features(selected_path)
    except ImportError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - UI error path
        st.error(f"No se pudo cargar el archivo: {exc}")
        return

    if df.empty:
        st.warning("El archivo existe pero no contiene datos.")
        return

    total_rows = len(df)
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Filas", f"{total_rows:,}")
    info_col2.metric("Columnas", f"{len(df.columns)}")
    info_col3.metric("Archivo", selected_path.name)

    st.subheader("Configuracion")
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        default_sample = min(100_000, total_rows)
        max_sample = min(500_000, total_rows)
        min_sample = 1_000 if max_sample >= 1_000 else 1
        step = 1_000 if max_sample >= 1_000 else 1
        sample_size = st.slider(
            "Tamano de muestra",
            min_value=min_sample,
            max_value=max_sample,
            value=default_sample,
            step=step,
            key="feature_explorer_sample_size",
        )
        df_plot = df.sample(sample_size, random_state=42)
    with cfg_col2:
        preview_rows = st.slider(
            "Filas para vista previa",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="feature_explorer_preview_rows",
        )

    st.subheader("Vista previa de datos")
    st.dataframe(df.head(preview_rows), width="stretch")

    st.subheader("Estadisticas descriptivas (todas las variables)")
    try:
        desc = df.describe(include="all").transpose()
    except Exception as exc:  # pragma: no cover - defensive path
        st.error(f"No se pudieron calcular las estadisticas: {exc}")
    else:
        st.dataframe(desc, width="stretch")

    st.subheader("Graficos")
    numeric_cols = df_plot.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No hay columnas numericas para graficos.")
        return

    try:
        import plotly.express as px
    except ImportError:
        st.error("plotly no esta instalado. Ejecute `pip install plotly`.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        hist_col = st.selectbox(
            "Histograma", numeric_cols, index=0, key="feature_explorer_hist"
        )
        bins = st.slider(
            "Bins",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            key="feature_explorer_bins",
        )
    with col2:
        box_index = 1 if len(numeric_cols) > 1 else 0
        box_col = st.selectbox(
            "Boxplot", numeric_cols, index=box_index, key="feature_explorer_box"
        )
    with col3:
        x_col = st.selectbox(
            "Scatter X", numeric_cols, index=0, key="feature_explorer_scatter_x"
        )
        y_index = 1 if len(numeric_cols) > 1 else 0
        y_col = st.selectbox(
            "Scatter Y", numeric_cols, index=y_index, key="feature_explorer_scatter_y"
        )

    hist_fig = px.histogram(df_plot, x=hist_col, nbins=bins)
    st.plotly_chart(hist_fig, width="stretch")

    box_fig = px.box(df_plot, y=box_col, points="outliers")
    st.plotly_chart(box_fig, width="stretch")

    scatter_fig = px.scatter(df_plot, x=x_col, y=y_col, opacity=0.6)
    st.plotly_chart(scatter_fig, width="stretch")


@st.cache_data(show_spinner=False)
def _viz_load_cluster_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _viz_list_cluster_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    candidates = sorted(RESULTS_DIR.glob("cluster_*.csv"))
    return [path for path in candidates if CLUSTER_LABEL_PATTERN.match(path.name)]


def _viz_load_cluster_color_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if not {"cluster_label", "color"}.issubset(df.columns):
        return {}
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        if pd.isna(row["cluster_label"]) or pd.isna(row["color"]):
            continue
        mapping[str(row["cluster_label"])] = str(row["color"])
    return mapping


def _viz_save_cluster_color_map(path: Path, mapping: dict[str, str]) -> None:
    if not mapping:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"cluster_label": label, "color": mapping[label]}
        for label in sorted(mapping.keys())
    ]
    pd.DataFrame(data).to_csv(path, index=False)


def _load_run_log_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _delete_run_log_entry(run_id: Optional[str]) -> bool:
    if not run_id or not RUN_LOG_PATH.exists():
        return False
    try:
        lines = RUN_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    kept_lines: list[str] = []
    removed = False
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        if isinstance(entry, dict) and entry.get("run_id") == run_id:
            removed = True
            continue
        kept_lines.append(line)
    if not removed:
        return False
    RUN_LOG_PATH.write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""),
        encoding="utf-8",
    )
    return True


def _find_run_for_labels(entries: list[dict], labels_file: str) -> dict | None:
    matches = [
        entry
        for entry in entries
        if entry.get("event") == "clustering"
        and entry.get("labels_file") == labels_file
    ]
    if not matches:
        return None
    return matches[-1]


def _resolve_logged_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _render_cluster_visualization_tab() -> None:
    st.header("Cluster visualization")

    try:
        import plotly.express as px
    except ImportError:
        st.error("plotly no esta instalado. Ejecute `pip install plotly`.")
        return

    files = _viz_list_cluster_files()
    if not files:
        st.warning(
            "No se encontraron archivos cluster_*.csv de clustering en Resultados."
        )
        return

    file_names = [path.name for path in files]
    #st.subheader("Archivo de clusters")
    selected_name = st.selectbox(
        "Archivos de clusters", file_names, key="cluster_viz_file"
    )
    selected_path = RESULTS_DIR / selected_name

    if not CLUSTER_LABEL_PATTERN.match(selected_path.name):
        st.error("El nombre del archivo no coincide con el formato esperado.")
        return

    df = _viz_load_cluster_file(selected_path)
    if "cluster_label" not in df.columns:
        st.error("El archivo no contiene la columna cluster_label.")
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "cluster_label"]
    if len(numeric_cols) < 3:
        st.error("No hay suficientes variables numericas para la visualizacion 3D.")
        return

    run_info = _find_run_for_labels(
        _load_run_log_entries(RUN_LOG_PATH), selected_path.name
    )
    feature_cols_used: list[str] = []
    if run_info:
        cols = run_info.get("feature_cols")
        if isinstance(cols, list):
            feature_cols_used = [col for col in cols if col in numeric_cols]
    if not feature_cols_used:
        feature_cols_used = numeric_cols

    cluster_values = sorted(df["cluster_label"].dropna().unique().tolist())
    stored_color_map = _viz_load_cluster_color_map(COLOR_MAP_PATH)
    palette = px.colors.qualitative.Plotly
    default_color_map = {
        str(cluster): palette[i % len(palette)]
        for i, cluster in enumerate(cluster_values)
    }
    st.subheader("Configuracion")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Filas", f"{len(df):,}")
    info_col2.metric("Columnas", f"{len(df.columns)}")
    info_col3.metric("Archivo", selected_path.name)

    cluster_color_map: dict[str, str] = {}
    with st.expander("Colores de clusters", expanded=False):
        for cluster in cluster_values:
            key = str(cluster)
            default_color = stored_color_map.get(key, default_color_map[key])
            picked = st.color_picker(
                f"Cluster {cluster}",
                value=default_color,
                key=f"cluster_viz_color_{key}",
            )
            cluster_color_map[key] = picked
    if cluster_color_map and cluster_color_map != stored_color_map:
        merged_map = {**stored_color_map, **cluster_color_map}
        _viz_save_cluster_color_map(COLOR_MAP_PATH, merged_map)

    selected_clusters = st.multiselect(
        "Clusters",
        cluster_values,
        default=cluster_values,
        key="cluster_viz_clusters",
    )
    if not selected_clusters:
        st.warning("Seleccione al menos un cluster.")
        return

    df = df[df["cluster_label"].isin(selected_clusters)]
    total_rows = len(df)
    st.caption(f"Filas disponibles: {total_rows:,}")

    if total_rows == 0:
        st.warning("No hay datos para los clusters seleccionados.")
        return

    use_sample = st.checkbox(
        "Usar muestra para graficos",
        value=total_rows > 200_000,
        key="cluster_viz_use_sample",
    )
    if use_sample:
        default_sample = min(100_000, total_rows)
        max_sample = min(200_000, total_rows)
        sample_size = st.slider(
            "Tamano de muestra",
            min_value=1_000,
            max_value=max_sample,
            value=default_sample,
            step=1_000,
            key="cluster_viz_sample_size",
        )
        df_plot = df.sample(sample_size, random_state=42)
    else:
        df_plot = df

    df_plot = df_plot.copy()
    df_plot["cluster_label_str"] = df_plot["cluster_label"].astype(str)

    st.subheader("Resumen de clusters")
    if run_info:
        st.caption("Variables usadas: " + ", ".join(feature_cols_used))
        feature_file = run_info.get("feature_file")
        if feature_file:
            st.caption(f"Archivo de variables: {feature_file}")
        if run_info.get("method"):
            st.caption(f"Metodo: {run_info.get('method')}")
        if run_info.get("timestamp"):
            st.caption(f"Fecha: {run_info.get('timestamp')}")
        if run_info.get("metrics_path"):
            metrics_path = _resolve_logged_path(run_info["metrics_path"])
            if metrics_path.exists():
                with st.expander("Metricas del proceso", expanded=False):
                    try:
                        metrics_df = pd.read_csv(metrics_path)
                    except Exception as exc:
                        st.error(f"No se pudo leer metricas: {exc}")
                    else:
                        st.dataframe(metrics_df, width="stretch")
    else:
        st.caption(
            "Variables usadas: no registradas (mostrando todas las numericas disponibles)."
        )

    summary = (
        df.groupby("cluster_label", sort=True)[feature_cols_used]
        .mean()
        .reset_index()
    )
    sizes = df["cluster_label"].value_counts().sort_index()
    summary.insert(1, "cluster_size", summary["cluster_label"].map(sizes))
    st.dataframe(summary, width="stretch")

    st.subheader("3D scatter por cluster")
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x_col = st.selectbox(
            "Eje X", numeric_cols, index=0, key="cluster_viz_x"
        )
    with col_y:
        y_col = st.selectbox(
            "Eje Y",
            numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0,
            key="cluster_viz_y",
        )
    with col_z:
        z_col = st.selectbox(
            "Eje Z",
            numeric_cols,
            index=2 if len(numeric_cols) > 2 else 0,
            key="cluster_viz_z",
        )

    scatter_fig = px.scatter_3d(
        df_plot,
        x=x_col,
        y=y_col,
        z=z_col,
        color="cluster_label_str",
        color_discrete_map=cluster_color_map,
        opacity=0.7,
    )
    scatter_fig.update_traces(marker={"size": 2})
    st.plotly_chart(scatter_fig, width="stretch")

    st.subheader("2D scatter por cluster")
    col_x2, col_y2 = st.columns(2)
    with col_x2:
        x2_col = st.selectbox(
            "Eje X (2D)",
            numeric_cols,
            index=0,
            key="cluster_viz_x2",
        )
    with col_y2:
        y2_col = st.selectbox(
            "Eje Y (2D)",
            numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0,
            key="cluster_viz_y2",
        )

    scatter_2d = px.scatter(
        df_plot,
        x=x2_col,
        y=y2_col,
        color="cluster_label_str",
        color_discrete_map=cluster_color_map,
        opacity=0.7,
    )
    st.plotly_chart(scatter_2d, width="stretch")

    st.subheader("Distribucion de una variable")
    hist_col = st.selectbox(
        "Variable", numeric_cols, index=0, key="cluster_viz_hist_var"
    )
    bins = st.slider(
        "Bins",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        key="cluster_viz_bins",
    )
    hist_fig = px.histogram(
        df_plot,
        x=hist_col,
        nbins=bins,
        color="cluster_label_str",
        color_discrete_map=cluster_color_map,
    )
    st.plotly_chart(hist_fig, width="stretch")


def _char_list_summary_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    candidates = sorted(RESULTS_DIR.glob("cluster_summary*.csv"))
    return [path for path in candidates if SUMMARY_PATTERN.match(path.name)]


def _char_parse_summary_name(path: Path) -> tuple[str | None, int | None]:
    match = SUMMARY_PATTERN.match(path.name)
    if not match:
        return None, None
    method = (match.group("method") or "kmeans").lower()
    k_raw = match.group("k")
    if method in {"kmeans", "gmm"}:
        if not k_raw:
            return method, None
        try:
            return method, int(k_raw)
        except ValueError:
            return method, None
    if method == "hdbscan":
        return method, None
    return None, None


def _char_descriptive_filename(method: str | None, k_value: int | None) -> str | None:
    if method == "kmeans" and k_value is not None:
        return f"cluster_descriptive_k{k_value}.csv"
    if method == "gmm" and k_value is not None:
        return f"cluster_descriptive_gmm_k{k_value}.csv"
    if method == "hdbscan":
        return "cluster_descriptive_hdbscan.csv"
    return None


def _char_pick_label_column(df: pd.DataFrame) -> str | None:
    for name in ("cluster_label", "cluster", "label"):
        if name in df.columns:
            return name
    for col in df.columns:
        col_lower = str(col).lower()
        if "cluster" in col_lower and "size" not in col_lower:
            return str(col)
    return None


def _char_pick_size_column(df: pd.DataFrame) -> str | None:
    for name in ("cluster_size", "size", "count", "n"):
        if name in df.columns:
            return name
    for col in df.columns:
        col_lower = str(col).lower()
        if "size" in col_lower or "count" in col_lower:
            return str(col)
    return None


def _char_normalize_profile(profile: pd.DataFrame, method: str) -> pd.DataFrame:
    if profile.empty:
        return profile
    if method == "Z-score":
        means = profile.mean()
        stds = profile.std(ddof=0).replace(0, 1)
        return (profile - means) / stds
    if method == "Min-max":
        mins = profile.min()
        ranges = (profile.max() - mins).replace(0, 1)
        return (profile - mins) / ranges
    return profile


def _char_parse_k_list(text: str) -> list[int]:
    if not text:
        return []
    tokens = re.split(r"[,\s]+", text.strip())
    ks: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value >= 2:
            ks.append(value)
    return sorted(set(ks))


def _char_feature_selection(
    features_df: pd.DataFrame, key: str
) -> List[str]:
    numeric_cols = (
        features_df.select_dtypes(include=["number"]).columns.tolist()
        if features_df is not None
        else []
    )
    numeric_cols = [col for col in numeric_cols if col != "cluster_label"]
    numeric_cols = _order_feature_columns(numeric_cols)
    default_cols = (
        _choose_feature_columns(features_df)
        if features_df is not None
        else []
    )
    default_cols = [col for col in default_cols if col in numeric_cols]

    if not numeric_cols:
        st.warning("No hay columnas numericas disponibles.")
        return []

    return st.multiselect(
        "Variables para GMM",
        numeric_cols,
        default=default_cols,
        key=key,
    )


def _char_responsibility_stats(
    gmm,
    X_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resp = gmm.predict_proba(X_scaled)
    labels = resp.argmax(axis=1)
    max_prob = resp.max(axis=1)
    entropy = -(resp * np.log(resp + 1e-12)).sum(axis=1)
    return labels, max_prob, entropy


def _char_component_covariance(
    gmm, component: int, n_features: int
) -> np.ndarray:
    if gmm.covariance_type == "full":
        return gmm.covariances_[component]
    if gmm.covariance_type == "diag":
        return np.diag(gmm.covariances_[component])
    if gmm.covariance_type == "tied":
        return gmm.covariances_
    return np.eye(n_features) * gmm.covariances_[component]


def _char_quality_report(
    gmm,
    labels: np.ndarray,
    max_prob: np.ndarray,
    entropy: np.ndarray,
    low_conf_threshold: float,
    n_features: int,
) -> pd.DataFrame:
    rows = []
    total = max(1, len(labels))
    for k in range(gmm.n_components):
        mask = labels == k
        count = int(mask.sum())
        if count == 0:
            continue
        cov = _char_component_covariance(gmm, k, n_features)
        try:
            cond = float(np.linalg.cond(cov))
        except np.linalg.LinAlgError:
            cond = float("inf")
        rows.append(
            {
                "cluster": int(k),
                "n_hard": count,
                "hard_share": count / total,
                "weight": float(gmm.weights_[k]),
                "mean_max_prob": float(max_prob[mask].mean()),
                "mean_entropy": float(entropy[mask].mean()),
                "frac_low_conf": float(
                    (max_prob[mask] < low_conf_threshold).mean()
                ),
                "cov_condition": cond,
            }
        )
    if not rows:
        return pd.DataFrame()
    report = pd.DataFrame(rows).sort_values("n_hard", ascending=False)
    return report.reset_index(drop=True)


def _char_aligned_contingency(labels_a: np.ndarray, labels_b: np.ndarray):
    try:
        from sklearn.metrics.cluster import contingency_matrix
    except ImportError as exc:
        raise ImportError(
            "scikit-learn es requerido para la matriz de contingencia."
        ) from exc

    C = contingency_matrix(labels_a, labels_b)
    if C.size == 0:
        return C, list(range(C.shape[1])), {}

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        col_order = list(range(C.shape[1]))
        return C[:, col_order], col_order, {}

    cost = C.max() - C
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    col_order = [
        assignment[row] for row in range(C.shape[0]) if row in assignment
    ]
    col_order += [c for c in range(C.shape[1]) if c not in col_order]
    return C[:, col_order], col_order, assignment


def _char_sankey_data(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    prefix_a: str,
    prefix_b: str,
):
    try:
        from sklearn.metrics.cluster import contingency_matrix
    except ImportError as exc:
        raise ImportError(
            "scikit-learn es requerido para la matriz de contingencia."
        ) from exc

    C = contingency_matrix(labels_a, labels_b)
    labels = (
        [f"{prefix_a}-{i}" for i in range(C.shape[0])]
        + [f"{prefix_b}-{j}" for j in range(C.shape[1])]
    )
    sources = []
    targets = []
    values = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            value = int(C[i, j])
            if value == 0:
                continue
            sources.append(i)
            targets.append(C.shape[0] + j)
            values.append(value)
    return labels, sources, targets, values


def _find_run_for_summary(entries: list[dict], summary_file: str) -> dict | None:
    matches = [
        entry
        for entry in entries
        if entry.get("event") == "clustering"
        and entry.get("summary_file") == summary_file
    ]
    if not matches:
        return None
    return matches[-1]


def _render_gmm_comparison_loader() -> None:
    entries = _load_run_log_entries(RUN_LOG_PATH)
    gmm_entries = [
        entry
        for entry in entries
        if entry.get("event") == "gmm_comparison"
    ]
    gmm_entries = sorted(
        gmm_entries,
        key=lambda item: str(item.get("timestamp", "")),
        reverse=True,
    )

    metrics_path = None
    quality_path = None
    metrics_params = None
    feature_file = None
    rows_used = None

    if gmm_entries:
        labels = []
        for entry in gmm_entries:
            ts = entry.get("timestamp", "-")
            file_name = entry.get("feature_file", "-")
            params = entry.get("metrics_params") or {}
            ks = params.get("ks_used") or params.get("ks_requested")
            ks_label = ", ".join(str(k) for k in ks) if ks else "-"
            rows = entry.get("rows")
            label = f"{ts} | {file_name} | ks={ks_label}"
            if rows is not None:
                label = f"{label} | rows={rows}"
            labels.append(label)

        selected_idx = st.selectbox(
            "Comparaciones guardadas",
            list(range(len(gmm_entries))),
            format_func=lambda idx: labels[idx],
            key="char_gmm_saved_run",
        )
        selected_entry = gmm_entries[int(selected_idx)]
        metrics_path = selected_entry.get("metrics_path")
        quality_path = selected_entry.get("quality_path")
        metrics_params = selected_entry.get("metrics_params")
        feature_file = selected_entry.get("feature_file")
        rows_used = selected_entry.get("rows")
    else:
        metrics_files = sorted(
            RESULTS_DIR.glob("cluster_metrics_gmm_compare_*.csv")
        )
        if not metrics_files:
            st.info("No hay comparaciones guardadas para cargar.")
            return
        metrics_names = [path.name for path in metrics_files]
        selected_metrics = st.selectbox(
            "Archivo de metricas",
            metrics_names,
            key="char_gmm_saved_metrics",
        )
        metrics_path = str(RESULTS_DIR / selected_metrics)
        quality_files = sorted(
            RESULTS_DIR.glob("cluster_quality_gmm_compare_*.csv")
        )
        if quality_files:
            quality_names = [path.name for path in quality_files]
            selected_quality = st.selectbox(
                "Archivo de scorecard (opcional)",
                ["(ninguno)"] + quality_names,
                key="char_gmm_saved_quality",
            )
            if selected_quality != "(ninguno)":
                quality_path = str(RESULTS_DIR / selected_quality)

    if not metrics_path:
        st.warning("No se encontro el archivo de metricas.")
        return

    metrics_resolved = _resolve_logged_path(str(metrics_path))
    if not metrics_resolved.exists():
        st.error(f"No se encontro el archivo: {metrics_resolved}")
        return

    try:
        metrics_df = pd.read_csv(metrics_resolved)
    except Exception as exc:
        st.error(f"No se pudo leer metricas: {exc}")
        return

    st.subheader("Resultados guardados")
    if feature_file:
        st.caption(f"Archivo de variables: {feature_file}")
    if rows_used is not None:
        st.caption(f"Filas usadas: {rows_used:,}")
    st.caption(f"Metricas: {metrics_resolved}")

    if metrics_params:
        with st.expander("Parametros", expanded=False):
            st.json(metrics_params)

    train_params = None
    train_distribution = None
    if gmm_entries:
        train_params = selected_entry.get("train_params")
        train_distribution = selected_entry.get("distribution")
    if train_params or train_distribution:
        with st.expander("Train (frecuentes/raros)", expanded=False):
            if train_params:
                st.json(train_params)
            if train_distribution:
                st.json(train_distribution)

    if metrics_df.empty:
        st.info("No hay metricas disponibles.")
        return

    try:
        import plotly.express as px
    except ImportError:
        st.dataframe(metrics_df, width="stretch")
    else:
        st.dataframe(metrics_df, width="stretch")
        fig = px.line(
            metrics_df,
            x="k",
            y=["bic", "aic"],
            markers=True,
        )
        fig.update_layout(
            xaxis_title="K",
            yaxis_title="Score (menor es mejor)",
            legend_title_text="Metrica",
        )
        st.plotly_chart(fig, width="stretch")

    if quality_path:
        quality_resolved = _resolve_logged_path(str(quality_path))
        if quality_resolved.exists():
            st.caption(f"Scorecard: {quality_resolved}")
            try:
                quality_df = pd.read_csv(quality_resolved)
            except Exception as exc:
                st.error(f"No se pudo leer scorecard: {exc}")
            else:
                st.subheader("Scorecard")
                st.dataframe(quality_df, width="stretch")
        else:
            st.warning(f"No se encontro scorecard: {quality_resolved}")
    else:
        st.info(
            "No hay scorecard asociado. Ejecute un nuevo calculo para generar."
        )

    st.info(
        "Las vistas PCA, Cruce y Sankey requieren recalcular la comparacion."
    )


def _render_gmm_comparison_section() -> None:
    st.subheader("Comparacion GMM (K multiples)")

    mode = st.radio(
        "Modo",
        ["Calcular nuevos", "Cargar existentes"],
        horizontal=True,
        key="char_gmm_mode",
    )
    if mode == "Cargar existentes":
        _render_gmm_comparison_loader()
        return

    st.caption(
        "Los resultados se exportan automaticamente al finalizar el calculo."
    )

    db_paths = _explorer_list_feature_dbs()
    if not db_paths:
        st.warning(
            "No se encontraron archivos cluster_features*.duckdb en Resultados."
        )
        return

    file_names = [path.name for path in db_paths]
    selected_name = st.selectbox(
        "Archivo de variables",
        file_names,
        key="char_gmm_feature_file",
    )
    source_path = RESULTS_DIR / selected_name
    try:
        features_df = _explorer_load_features(source_path)
    except ImportError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"No se pudo cargar el archivo: {exc}")
        return

    if features_df is None or features_df.empty:
        st.warning("No hay datos disponibles para GMM.")
        return

    st.caption(
        f"Fuente: DuckDB | Filas: {len(features_df):,} | "
        f"Columnas: {len(features_df.columns)}"
    )
    st.caption(f"Archivo: {source_path}")

    feature_cols = _char_feature_selection(
        features_df, key="char_gmm_feature_cols"
    )
    if not feature_cols:
        return

    cluster_df = _prepare_cluster_features(features_df, feature_cols)
    dropped = len(features_df) - len(cluster_df)
    if dropped:
        st.info(f"Se descartaron {dropped:,} filas por valores invalidos.")
    if cluster_df.empty:
        st.warning("No hay datos validos despues de filtrar.")
        return

    with st.expander(
        "Filtros de Frecuentes (Train Set)", expanded=False
    ):
        c1, c2, c3 = st.columns(3)
        min_passes = c1.number_input(
            "Mínimo de pasadas (total_passes)",
            min_value=0,
            value=int(st.session_state.get("freq_min_passes", 20)),
            step=1,
            key="char_gmm_min_passes",
        )
        min_days = c2.number_input(
            "Mínimo de días activos (n_days_active)",
            min_value=1,
            value=int(st.session_state.get("freq_min_days", 1)),
            step=1,
            key="char_gmm_min_days",
        )
        min_months = c3.number_input(
            "Mínimo de meses activos (n_months_active)",
            min_value=1,
            value=int(st.session_state.get("freq_min_months", 1)),
            step=1,
            key="char_gmm_min_months",
        )
        st.session_state["freq_min_passes"] = int(min_passes)
        st.session_state["freq_min_days"] = int(min_days)
        st.session_state["freq_min_months"] = int(min_months)

    freq_full, rare_full = split_frequent_drivers(
        features_df,
        min_total_passes=int(min_passes),
        min_days_active=min_days,
        min_months_active=min_months,
    )
    cluster_freq = cluster_df.loc[cluster_df.index.intersection(freq_full.index)]
    cluster_rare = cluster_df.loc[cluster_df.index.intersection(rare_full.index)]
    total_valid_rows = len(cluster_df)
    train_params = {
        "min_total_passes": int(min_passes),
        "min_days_active": int(min_days),
        "min_months_active": int(min_months),
    }
    train_distribution = {
        "frequent_rows": int(len(cluster_freq)),
        "rare_rows": int(len(cluster_rare)),
        "frequent_share": (
            len(cluster_freq) / total_valid_rows if total_valid_rows else 0.0
        ),
        "rare_share": (
            len(cluster_rare) / total_valid_rows if total_valid_rows else 0.0
        ),
    }
    st.caption(
        f"Filas validas para GMM: {total_valid_rows:,} "
        f"(Frecuentes: {len(cluster_freq):,}, Raros: {len(cluster_rare):,})."
    )
    if len(cluster_freq) < 3:
        st.warning(
            "No hay suficientes muestras 'Frecuentes' para GMM (minimo 3)."
        )
        return

    cluster_df = cluster_freq
    total_rows = len(cluster_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        k_text = st.text_input(
            "Ks (separados por coma)",
            value="5,8,14",
            key="char_gmm_ks",
        )
    with col2:
        covariance_type = st.selectbox(
            "covariance_type",
            ["full", "diag", "tied", "spherical"],
            index=0,
            key="char_gmm_cov_type",
        )
    with col3:
        n_init = st.number_input(
            "n_init",
            min_value=1,
            value=10,
            step=1,
            key="char_gmm_n_init",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        reg_covar = st.number_input(
            "reg_covar",
            min_value=0.0,
            value=1e-6,
            step=1e-6,
            format="%.6f",
            key="char_gmm_reg_covar",
        )
    with col5:
        random_state = st.number_input(
            "random_state",
            min_value=0,
            value=42,
            step=1,
            key="char_gmm_random_state",
        )
    with col6:
        low_conf_threshold = st.slider(
            "Umbral baja confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="char_gmm_low_conf",
        )

    default_sample = min(200_000, total_rows)
    use_sample_default = total_rows > default_sample
    use_sample = st.checkbox(
        "Usar muestra",
        value=use_sample_default,
        key="char_gmm_use_sample",
    )
    sample_size = total_rows
    if use_sample:
        max_sample = min(500_000, total_rows)
        min_sample = 1_000 if max_sample >= 1_000 else 1
        step = 1_000 if max_sample >= 1_000 else 1
        sample_size = st.slider(
            "Tamano de muestra",
            min_value=min_sample,
            max_value=max_sample,
            value=min(default_sample, max_sample),
            step=step,
            key="char_gmm_sample_size",
        )
        cluster_df = cluster_df.sample(
            n=int(sample_size), random_state=int(random_state)
        )
        st.caption(f"Usando muestra: {len(cluster_df):,} filas.")

    ks = _char_parse_k_list(k_text)
    if not ks:
        st.warning("Ingrese al menos un K valido (>=2).")
        return

    current_params = {
        "source": "DuckDB",
        "source_path": str(source_path),
        "feature_cols": list(feature_cols),
        "rows": int(len(cluster_df)),
        "train_params": train_params,
        "ks": list(ks),
        "covariance_type": covariance_type,
        "n_init": int(n_init),
        "reg_covar": float(reg_covar),
        "random_state": int(random_state),
        "low_conf_threshold": float(low_conf_threshold),
    }

    run_btn = st.button("Ejecutar comparacion GMM", key="char_gmm_run")
    if run_btn:
        try:
            X_scaled, _scaler = _scale_cluster_features(
                cluster_df, feature_cols
            )
        except ImportError as exc:
            st.error(str(exc))
            return

        if max(ks) >= len(X_scaled):
            st.warning(
                "Algunos K son mayores o iguales al numero de muestras."
            )

        try:
            from sklearn.decomposition import PCA
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:
            st.error(
                "scikit-learn es requerido para comparar GMM. "
                "Instalelo con: pip install scikit-learn"
            )
            return

        with st.spinner("Entrenando GMMs y calculando metricas..."):
            metrics_rows = []
            labels_map = {}
            max_prob_map = {}
            entropy_map = {}
            quality_map = {}
            valid_ks = []
            skipped_ks = []

            pca_coords = None
            if len(feature_cols) >= 2:
                pca = PCA(n_components=2, random_state=int(random_state))
                pca_coords = pca.fit_transform(X_scaled)

            for k in ks:
                if k >= len(X_scaled):
                    skipped_ks.append(k)
                    continue
                gmm = GaussianMixture(
                    n_components=int(k),
                    covariance_type=covariance_type,
                    random_state=int(random_state),
                    n_init=int(n_init),
                    reg_covar=float(reg_covar),
                )
                gmm.fit(X_scaled)
                labels, max_prob, entropy = _char_responsibility_stats(
                    gmm, X_scaled
                )
                labels_map[k] = labels
                max_prob_map[k] = max_prob
                entropy_map[k] = entropy
                quality_map[k] = _char_quality_report(
                    gmm,
                    labels,
                    max_prob,
                    entropy,
                    float(low_conf_threshold),
                    len(feature_cols),
                )
                metrics_rows.append(
                    {
                        "k": int(k),
                        "bic": float(gmm.bic(X_scaled)),
                        "aic": float(gmm.aic(X_scaled)),
                    }
                )
                valid_ks.append(k)

            if not valid_ks:
                st.error(
                    "No se pudieron ajustar GMMs con los K seleccionados."
                )
                return
            if skipped_ks:
                st.warning(
                    "Ks omitidos por falta de muestras: "
                    + ", ".join(str(k) for k in skipped_ks)
                )

            metrics_df = pd.DataFrame(metrics_rows).sort_values("k")
            metrics_path = _save_gmm_comparison_metrics(
                metrics_df, valid_ks
            )
            quality_path = _save_gmm_comparison_quality(
                quality_map, valid_ks
            )
            metrics_params = {
                "ks_requested": list(ks),
                "ks_used": list(valid_ks),
                "ks_skipped": list(skipped_ks),
                "covariance_type": covariance_type,
                "n_init": int(n_init),
                "reg_covar": float(reg_covar),
                "random_state": int(random_state),
                "low_conf_threshold": float(low_conf_threshold),
                "use_sample": bool(use_sample),
                "rows_total": int(total_valid_rows),
                "rows_train_total": int(total_rows),
                "rows_used": int(len(X_scaled)),
                "source_file": source_path.name,
                "train_params": train_params,
                "train_distribution": train_distribution,
            }
            _log_gmm_comparison_run(
                feature_cols=feature_cols,
                rows=len(X_scaled),
                feature_path=source_path,
                metrics_path=metrics_path,
                metrics_params=metrics_params,
                quality_path=quality_path,
                train_params=train_params,
                train_distribution=train_distribution,
            )
            st.session_state["char_gmm_payload"] = {
                "ks": valid_ks,
                "metrics_df": metrics_df,
                "labels": labels_map,
                "max_prob": max_prob_map,
                "entropy": entropy_map,
                "quality": quality_map,
                "pca_coords": pca_coords,
                "rows": len(X_scaled),
            }
            st.session_state["char_gmm_params"] = current_params

    payload = st.session_state.get("char_gmm_payload")
    stored_params = st.session_state.get("char_gmm_params")
    if payload is None:
        return
    if stored_params != current_params:
        st.info(
            "Los parametros cambiaron. Ejecute nuevamente para actualizar "
            "los resultados."
        )

    ks = payload.get("ks", [])
    if not ks:
        st.warning("No hay resultados disponibles para los K seleccionados.")
        return

    metrics_df = payload.get("metrics_df")
    st.subheader("Resultados GMM")
    if payload.get("rows"):
        st.caption(f"Resultados sobre {payload['rows']:,} filas.")
    if metrics_df is not None and not metrics_df.empty:
        st.dataframe(metrics_df, width="stretch")

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.info("plotly no esta instalado. Los graficos no estan disponibles.")
        return

    tabs = st.tabs(["BIC/AIC", "PCA 2D", "Cruce", "Sankey", "Scorecard"])

    with tabs[0]:
        if metrics_df is None or metrics_df.empty:
            st.info("No hay metricas para graficar.")
        else:
            fig = px.line(
                metrics_df,
                x="k",
                y=["bic", "aic"],
                markers=True,
            )
            fig.update_layout(
                xaxis_title="K",
                yaxis_title="Score (menor es mejor)",
                legend_title_text="Metrica",
            )
            st.plotly_chart(fig, width="stretch")

    with tabs[1]:
        pca_coords = payload.get("pca_coords")
        if pca_coords is None:
            st.info("Se requieren al menos 2 variables para PCA.")
        else:
            k_tabs = st.tabs([f"K={k}" for k in ks])
            for idx, k in enumerate(ks):
                with k_tabs[idx]:
                    labels = payload["labels"][k]
                    max_prob = payload["max_prob"][k]
                    entropy = payload["entropy"][k]
                    plot_df = pd.DataFrame(
                        {
                            "pc1": pca_coords[:, 0],
                            "pc2": pca_coords[:, 1],
                            "cluster": labels.astype(str),
                            "max_prob": max_prob,
                            "entropy": entropy,
                        }
                    )
                    plot_df["conf_size"] = plot_df["max_prob"]
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig = px.scatter(
                            plot_df,
                            x="pc1",
                            y="pc2",
                            color="cluster",
                            size="conf_size",
                            size_max=16,
                            opacity=0.7,
                            hover_data=["max_prob", "entropy"],
                            render_mode="webgl",
                        )
                        fig.update_layout(
                            title=f"PCA 2D (K={k}) - color=cluster, size=confianza",
                            xaxis_title="PC1",
                            yaxis_title="PC2",
                            showlegend=True,
                        )
                        st.plotly_chart(fig, width="stretch")
                    with col_b:
                        fig = px.scatter(
                            plot_df,
                            x="pc1",
                            y="pc2",
                            color="entropy",
                            color_continuous_scale="Turbo",
                            opacity=0.7,
                            hover_data=["max_prob", "entropy"],
                            render_mode="webgl",
                        )
                        fig.update_layout(
                            title=f"PCA 2D (K={k}) - color=entropia",
                            xaxis_title="PC1",
                            yaxis_title="PC2",
                        )
                        st.plotly_chart(fig, width="stretch")

    with tabs[2]:
        if len(ks) < 2:
            st.info("Se requieren al menos dos K para comparar.")
        else:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                k1 = st.selectbox(
                    "K origen",
                    ks,
                    index=0,
                    key="char_gmm_pair_k1",
                )
            with comp_col2:
                default_index = 1 if len(ks) > 1 else 0
                k2 = st.selectbox(
                    "K destino",
                    ks,
                    index=default_index,
                    key="char_gmm_pair_k2",
                )
            if k1 == k2:
                st.info("Seleccione Ks distintos para comparar.")
            else:
                labels_a = payload["labels"][k1]
                labels_b = payload["labels"][k2]
                try:
                    C_aligned, col_order, assignment = (
                        _char_aligned_contingency(labels_a, labels_b)
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    return
                fig = px.imshow(
                    C_aligned,
                    x=[str(label) for label in col_order],
                    y=[str(label) for label in range(C_aligned.shape[0])],
                    aspect="auto",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    title=f"Contingencia alineada: K={k1} -> K={k2}",
                    xaxis_title="Clusters K destino (alineados)",
                    yaxis_title="Clusters K origen",
                    coloraxis_colorbar_title="count",
                )
                st.plotly_chart(fig, width="stretch")
                if assignment:
                    mapping_df = pd.DataFrame(
                        {
                            "cluster_origen": list(assignment.keys()),
                            "cluster_destino": [
                                assignment[k] for k in assignment.keys()
                            ],
                        }
                    ).sort_values("cluster_origen")
                    st.dataframe(mapping_df, width="stretch")
                elif C_aligned.size > 0:
                    st.caption(
                        "Alineacion sin Hungarian (scipy no disponible)."
                    )

    with tabs[3]:
        if len(ks) < 2:
            st.info("Se requieren al menos dos K para comparar.")
        else:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                k1 = st.selectbox(
                    "K origen",
                    ks,
                    index=0,
                    key="char_gmm_sankey_k1",
                )
            with comp_col2:
                default_index = 1 if len(ks) > 1 else 0
                k2 = st.selectbox(
                    "K destino",
                    ks,
                    index=default_index,
                    key="char_gmm_sankey_k2",
                )
            if k1 == k2:
                st.info("Seleccione Ks distintos para comparar.")
            else:
                labels_a = payload["labels"][k1]
                labels_b = payload["labels"][k2]
                try:
                    node_labels, sources, targets, values = _char_sankey_data(
                        labels_a,
                        labels_b,
                        prefix_a=f"K{k1}",
                        prefix_b=f"K{k2}",
                    )
                except ImportError as exc:
                    st.error(str(exc))
                    return
                fig = go.Figure(
                    data=[
                        go.Sankey(
                            node=dict(
                                label=node_labels, pad=12, thickness=16
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                            ),
                        )
                    ]
                )
                fig.update_layout(
                    title=f"Sankey: K={k1} -> K={k2}",
                    font_size=10,
                )
                st.plotly_chart(fig, width="stretch")

    with tabs[4]:
        selected_k = st.selectbox(
            "K para scorecard",
            ks,
            index=0,
            key="char_gmm_scorecard_k",
        )
        quality_df = payload["quality"].get(selected_k)
        if quality_df is None or quality_df.empty:
            st.info("No hay scorecard disponible.")
        else:
            threshold = low_conf_threshold
            if stored_params and "low_conf_threshold" in stored_params:
                threshold = stored_params["low_conf_threshold"]
            st.caption(f"Umbral baja confianza: {threshold:.2f}")
            st.dataframe(quality_df, width="stretch")


def _render_cluster_characterization_summary() -> None:
    summary_files = _char_list_summary_files()
    if not summary_files:
        st.warning("No se encontraron archivos cluster_summary*.csv en Resultados.")
        return

    file_names = [path.name for path in summary_files]
    st.subheader("Archivo de resumen")
    selected_name = st.selectbox(
        "Archivo",
        file_names,
        key="cluster_char_summary_file",
    )
    selected_path = RESULTS_DIR / selected_name

    try:
        summary_df = pd.read_csv(selected_path)
    except Exception as exc:
        st.error(f"No se pudo leer el resumen: {exc}")
        return

    method, k_value = _char_parse_summary_name(selected_path)
    run_info = _find_run_for_summary(
        _load_run_log_entries(RUN_LOG_PATH), selected_path.name
    )

    st.subheader("Detalles del proceso")
    if run_info:
        feature_file = run_info.get("feature_file")
        if feature_file:
            st.caption(f"Archivo de variables: {feature_file}")
        if run_info.get("feature_cols"):
            st.caption("Variables usadas: " + ", ".join(run_info.get("feature_cols")))
        if run_info.get("method"):
            st.caption(f"Metodo: {run_info.get('method')}")
        if run_info.get("timestamp"):
            st.caption(f"Fecha: {run_info.get('timestamp')}")
        if run_info.get("metrics_path"):
            metrics_path = _resolve_logged_path(run_info["metrics_path"])
            if metrics_path.exists():
                with st.expander("Metricas del proceso", expanded=False):
                    try:
                        metrics_df = pd.read_csv(metrics_path)
                    except Exception as exc:
                        st.error(f"No se pudo leer metricas: {exc}")
                    else:
                        st.dataframe(metrics_df, width="stretch")
    else:
        if method:
            st.caption(f"Metodo: {method}")
        if k_value is not None:
            st.caption(f"K: {k_value}")

    st.subheader("Resumen por cluster")
    st.dataframe(summary_df, width="stretch")

    plot_df = summary_df.copy()
    label_col = _char_pick_label_column(plot_df)
    if label_col is None:
        plot_df = plot_df.reset_index().rename(columns={"index": "cluster_label"})
        label_col = "cluster_label"
    size_col = _char_pick_size_column(plot_df)
    plot_df["cluster_label_str"] = plot_df[label_col].astype(str)

    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    excluded_cols = {label_col}
    if size_col:
        excluded_cols.add(size_col)
    metric_cols = [col for col in numeric_cols if col not in excluded_cols]

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.info("plotly no esta instalado. Los graficos no estan disponibles.")
    else:
        if metric_cols or size_col:
            cluster_values = (
                plot_df[label_col].dropna().unique().tolist()
            )
            try:
                cluster_values = sorted(cluster_values)
            except TypeError:
                cluster_values = list(cluster_values)

            stored_color_map = _viz_load_cluster_color_map(COLOR_MAP_PATH)
            palette = px.colors.qualitative.Plotly
            default_color_map = {
                str(cluster): palette[i % len(palette)]
                for i, cluster in enumerate(cluster_values)
            }
            cluster_color_map = {
                str(cluster): stored_color_map.get(
                    str(cluster), default_color_map[str(cluster)]
                )
                for cluster in cluster_values
            }

            st.subheader("Graficos de caracterizacion")
            tabs = st.tabs(
                ["Tamanos", "Perfil medio", "Radar", "Comparacion 2D"]
            )

            with tabs[0]:
                if not size_col:
                    st.info("No hay columna de tamanos en el resumen.")
                else:
                    size_fig = px.bar(
                        plot_df,
                        x="cluster_label_str",
                        y=size_col,
                        color="cluster_label_str",
                        color_discrete_map=cluster_color_map,
                        text=size_col,
                    )
                    size_fig.update_traces(textposition="outside")
                    size_fig.update_layout(
                        xaxis_title="Cluster",
                        yaxis_title="Tamanos",
                        showlegend=False,
                    )
                    st.plotly_chart(size_fig, width="stretch")

            with tabs[1]:
                if not metric_cols:
                    st.info("No hay variables numericas para perfil.")
                else:
                    default_profile = metric_cols[: min(8, len(metric_cols))]
                    profile_vars = st.multiselect(
                        "Variables",
                        metric_cols,
                        default=default_profile,
                        key="cluster_char_profile_vars",
                    )
                    norm_method = st.selectbox(
                        "Normalizacion",
                        ["Z-score", "Min-max", "Sin normalizar"],
                        index=0,
                        key="cluster_char_norm",
                    )
                    if not profile_vars:
                        st.info("Seleccione al menos una variable.")
                    else:
                        profile = (
                            plot_df.set_index(label_col)[profile_vars]
                            .sort_index()
                        )
                        profile_norm = _char_normalize_profile(
                            profile, norm_method
                        )
                        color_scale = (
                            "RdBu" if norm_method == "Z-score" else "Viridis"
                        )
                        heat_fig = px.imshow(
                            profile_norm,
                            aspect="auto",
                            color_continuous_scale=color_scale,
                        )
                        heat_fig.update_layout(
                            xaxis_title="Variables",
                            yaxis_title="Cluster",
                            coloraxis_colorbar_title=norm_method,
                        )
                        st.plotly_chart(heat_fig, width="stretch")

            with tabs[2]:
                if len(metric_cols) < 3:
                    st.info("Seleccione al menos tres variables para radar.")
                else:
                    default_radar = metric_cols[: min(6, len(metric_cols))]
                    radar_vars = st.multiselect(
                        "Variables para radar",
                        metric_cols,
                        default=default_radar,
                        key="cluster_char_radar_vars",
                    )
                    radar_clusters = st.multiselect(
                        "Clusters",
                        cluster_values,
                        default=cluster_values[: min(6, len(cluster_values))],
                        key="cluster_char_radar_clusters",
                    )
                    if len(radar_vars) < 3:
                        st.info("Seleccione al menos tres variables.")
                    elif not radar_clusters:
                        st.info("Seleccione al menos un cluster.")
                    else:
                        radar_profile = (
                            plot_df[plot_df[label_col].isin(radar_clusters)]
                            .set_index(label_col)[radar_vars]
                            .sort_index()
                        )
                        radar_profile = _char_normalize_profile(
                            radar_profile, "Min-max"
                        )
                        theta = radar_vars + [radar_vars[0]]
                        radar_fig = go.Figure()
                        for cluster_label, row in radar_profile.iterrows():
                            values = row.tolist()
                            values += values[:1]
                            radar_fig.add_trace(
                                go.Scatterpolar(
                                    r=values,
                                    theta=theta,
                                    name=f"Cluster {cluster_label}",
                                    fill="toself",
                                    opacity=0.6,
                                )
                            )
                        radar_fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 1]),
                            ),
                            showlegend=True,
                        )
                        st.plotly_chart(radar_fig, width="stretch")

            with tabs[3]:
                if len(metric_cols) < 2:
                    st.info("Seleccione al menos dos variables para comparar.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox(
                            "Eje X",
                            metric_cols,
                            index=0,
                            key="cluster_char_x",
                        )
                    with col2:
                        y_index = 1 if len(metric_cols) > 1 else 0
                        y_col = st.selectbox(
                            "Eje Y",
                            metric_cols,
                            index=y_index,
                            key="cluster_char_y",
                        )
                    bubble_fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        size=size_col if size_col else None,
                        color="cluster_label_str",
                        color_discrete_map=cluster_color_map,
                        hover_name="cluster_label_str",
                    )
                    bubble_fig.update_layout(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                    )
                    st.plotly_chart(bubble_fig, width="stretch")

    descriptive_name = _char_descriptive_filename(method, k_value)
    if descriptive_name:
        descriptive_path = RESULTS_DIR / descriptive_name
        if descriptive_path.exists():
            with st.expander("Estadistica descriptiva", expanded=False):
                try:
                    desc_df = pd.read_csv(descriptive_path)
                except Exception as exc:
                    st.error(f"No se pudo leer descriptivo: {exc}")
                else:
                    st.dataframe(desc_df, width="stretch")
        else:
            st.info("No se encontro archivo descriptivo para este resumen.")


def _render_cluster_characterization_tab() -> None:
    st.header("Cluster characterization")

    tab_gmm, tab_char = st.tabs(["Comparacion GMM", "Characterization"])
    with tab_gmm:
        _render_gmm_comparison_section()
    with tab_char:
        _render_cluster_characterization_summary()


def _render_history_tab() -> None:
    st.header("History")
    entries = _load_run_log_entries(RUN_LOG_PATH)
    if not entries:
        st.info("No hay historial disponible.")
        return

    events = sorted(
        {entry.get("event") for entry in entries if entry.get("event")}
    )
    methods = sorted(
        {entry.get("method") for entry in entries if entry.get("method")}
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if events:
            selected_events = st.multiselect(
                "Eventos",
                events,
                default=events,
                key="history_events",
            )
        else:
            selected_events = []
    with col2:
        if methods:
            selected_methods = st.multiselect(
                "Metodos",
                methods,
                default=methods,
                key="history_methods",
            )
        else:
            selected_methods = []
    with col3:
        max_entries = st.number_input(
            "Max entradas",
            min_value=1,
            max_value=max(1, len(entries)),
            value=min(50, len(entries)),
            step=1,
            key="history_max_entries",
        )

    filtered: list[dict] = []
    for entry in entries:
        event = entry.get("event")
        method = entry.get("method")
        if events and event not in selected_events:
            continue
        if methods and method not in selected_methods:
            continue
        filtered.append(entry)

    if not filtered:
        st.info("No hay entradas con los filtros seleccionados.")
        return

    entries_sorted = sorted(
        filtered,
        key=lambda item: str(item.get("timestamp", "")),
        reverse=True,
    )
    entries_sorted = entries_sorted[: int(max_entries)]

    for idx, entry in enumerate(entries_sorted, start=1):
        timestamp = entry.get("timestamp", "-")
        event = entry.get("event", "-")
        method = entry.get("method", "-")
        rows = entry.get("rows")
        title = f"{idx}. {timestamp} | {event} | {method}"
        if rows is not None:
            title = f"{title} | rows={rows}"
        with st.expander(title, expanded=False):
            st.caption(f"run_id: {entry.get('run_id', '-')}")
            run_id = entry.get("run_id")
            if st.button(
                "Eliminar registro",
                key=f"history_delete_cluster_{run_id or idx}",
            ):
                if _delete_run_log_entry(run_id):
                    st.success("Registro eliminado.")
                    st.rerun()
                else:
                    st.warning("No se pudo eliminar el registro.")

            st.markdown("**Features**")
            feature_source = entry.get("feature_source")
            feature_file = entry.get("feature_file")
            feature_path = entry.get("feature_path")
            if feature_source:
                st.caption(f"source: {feature_source}")
            if feature_file:
                st.caption(f"archivo: {feature_file}")
            if feature_path:
                st.caption(f"path: {feature_path}")

            feature_cols = entry.get("feature_cols")
            if isinstance(feature_cols, list) and feature_cols:
                st.caption(f"variables usadas: {len(feature_cols)}")
                st.dataframe(
                    pd.DataFrame({"variable": feature_cols}),
                    width="stretch",
                )

            metrics_params = entry.get("metrics_params")
            if metrics_params:
                st.markdown("**Parametros de metricas**")
                st.json(metrics_params)

            params = entry.get("params")
            if params:
                if entry.get("event") == "clustering":
                    title = "**Parametros de clustering**"
                elif entry.get("event") == "frequent_definition":
                    title = "**Parametros de frecuencia**"
                else:
                    title = "**Parametros**"
                st.markdown(title)
                st.json(params)

            train_params = entry.get("train_params")
            if train_params:
                st.markdown("**Parametros de train**")
                st.json(train_params)

            distribution = entry.get("distribution")
            if distribution:
                st.markdown("**Distribucion frecuentes/raros**")
                st.json(distribution)

            metrics_path = entry.get("metrics_path")
            if metrics_path:
                st.caption(f"metricas: {metrics_path}")
                metrics_resolved = _resolve_logged_path(metrics_path)
                if metrics_resolved.exists():
                    with st.expander("Ver metricas", expanded=False):
                        try:
                            metrics_df = pd.read_csv(metrics_resolved)
                        except Exception as exc:
                            st.error(f"No se pudo leer metricas: {exc}")
                        else:
                            st.dataframe(metrics_df, width="stretch")

            quality_path = entry.get("quality_path")
            if quality_path:
                st.caption(f"scorecard: {quality_path}")
                quality_resolved = _resolve_logged_path(quality_path)
                if quality_resolved.exists():
                    with st.expander("Ver scorecard", expanded=False):
                        try:
                            quality_df = pd.read_csv(quality_resolved)
                        except Exception as exc:
                            st.error(f"No se pudo leer scorecard: {exc}")
                        else:
                            st.dataframe(quality_df, width="stretch")

            if entry.get("event") == "clustering":
                labels_path = entry.get("labels_path")
                if labels_path:
                    st.caption(f"labels: {labels_path}")

                summary_path = entry.get("summary_path")
                if summary_path:
                    st.caption(f"summary: {summary_path}")
                    summary_resolved = _resolve_logged_path(summary_path)
                    if summary_resolved.exists():
                        with st.expander(
                            "Resumen por cluster", expanded=False
                        ):
                            try:
                                summary_df = pd.read_csv(summary_resolved)
                            except Exception as exc:
                                st.error(f"No se pudo leer resumen: {exc}")
                            else:
                                st.dataframe(summary_df, width="stretch")

                descriptive_path = entry.get("descriptive_path")
                if descriptive_path:
                    st.caption(f"descriptive: {descriptive_path}")
                    descriptive_resolved = _resolve_logged_path(
                        descriptive_path
                    )
                    if descriptive_resolved.exists():
                        with st.expander(
                            "Estadistica descriptiva", expanded=False
                        ):
                            try:
                                desc_df = pd.read_csv(descriptive_resolved)
                            except Exception as exc:
                                st.error(
                                    f"No se pudo leer descriptivo: {exc}"
                                )
                            else:
                                st.dataframe(desc_df, width="stretch")


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    _init_state()
    if set_page_config:
        st.set_page_config(page_title="Clustering", layout="wide")
    st.title("Clustering")

    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)

    tabs = st.tabs(
        [
            "Features",
            "Feature explorer",
            "Clustering",
            "K optimum",
            "History",
            "Cluster visualization",
            "Cluster characterization",
        ]
    )
    with tabs[0]:
        st.header("Features")
        _render_feature_loader()
    with tabs[1]:
        _render_feature_explorer_tab()
    with tabs[2]:
        st.header("Clustering")
        features_df = _get_features()
        if features_df is None or features_df.empty:
            st.info("Cargue o calcule variables en la pestana Features.")
        else:
            _render_clustering_section(features_df)
    with tabs[3]:
        features_df = _get_features()
        if features_df is None or features_df.empty:
            st.info("Cargue o calcule variables en la pestana Features.")
        else:
            _render_k_optimo_experiment(features_df)
    with tabs[4]:
        _render_history_tab()
    with tabs[5]:
        _render_cluster_visualization_tab()
    with tabs[6]:
        _render_cluster_characterization_tab()


if __name__ == "__main__":
    main()
