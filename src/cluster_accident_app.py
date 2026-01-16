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
from typing import Dict, List, Optional, Tuple

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
    build_model as _build_model,
    get_model_scores as _get_model_scores,
    select_threshold_for_far as _select_threshold_for_far,
    far_and_sensitivity as _far_and_sensitivity,
    temporal_train_test_split as _temporal_train_test_split,
    split_train_val_for_threshold as _split_train_val_for_threshold,
    train_model as _train_model,
    train_model_on_split as _train_model_on_split,
)
from src.experiments_logic import ExperimentsRunner

try:
    import duckdb
except ImportError:
    duckdb = None

RESULTS_DIR = ROOT_DIR / "Resultados"
DATA_DIR = ROOT_DIR / "Datos"
HISTORY_PATH = RESULTS_DIR / "experiment_history.jsonl"
MODELS_DIR = RESULTS_DIR / "model_history"
CLUSTER_LABEL_PATTERN = re.compile(
    r"^cluster_(?P<method>kmeans|gmm|hdbscan)(?:_k(?P<k>\d+))?\.csv$"
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
    st.session_state.setdefault("optuna_model_params_applied_signature", None)
    st.session_state.setdefault("optuna_n_trials", 30)
    st.session_state.setdefault("optuna_timeout", 3600)
    st.session_state.setdefault("optuna_random_state", 42)
    st.session_state.setdefault("optuna_pruner_enabled", True)
    st.session_state.setdefault("optuna_pruner_startup_trials", 5)
    st.session_state.setdefault("far_target", 0.2)
    st.session_state.setdefault("val_size", 0.2)
    st.session_state.setdefault("balance_last_stats", None)
    st.session_state.setdefault("balance_last_params", None)
    st.session_state.setdefault("history_entries", [])


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
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(RESULTS_DIR.glob(pattern))
    unique = {path.name: path for path in files}
    # Sort by name (which has timestamp) descending to show newest first
    return sorted(unique.values(), key=lambda p: p.name, reverse=True)



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
    return f"{source}:{len(features_df)}:{len(features_df.columns)}"


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


def _optuna_trials_path(optuna_id: str, model_choice: Optional[str] = None) -> Path:
    if model_choice:
        suffix = _slugify(model_choice)
        return RESULTS_DIR / f"optuna_{optuna_id}_{suffix}_trials.csv"
    return RESULTS_DIR / f"optuna_{optuna_id}_trials.csv"


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
    best_score: float,
    best_smote_params: Dict[str, object],
    best_model_params: Dict[str, object],
    trials_df: Optional[pd.DataFrame],
    optuna_settings: Optional[Dict[str, object]],
    search_space: Dict[str, object],
) -> None:
    store = st.session_state.setdefault("optuna_results_store", {})
    entry = store.get(optuna_key, {})
    if not isinstance(optuna_settings, dict):
        optuna_settings = {}
    results = entry.get("results")
    if not isinstance(results, dict):
        results = {}
        legacy_model = entry.get("model_choice")
        if legacy_model:
            results[str(legacy_model)] = {
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

    trials_csv = None
    if trials_df is not None and not trials_df.empty:
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            trials_path = _optuna_trials_path(optuna_id, model_choice)
            trials_df.to_csv(trials_path, index=False)
            trials_csv = str(trials_path)
        except Exception:
            trials_csv = None
    else:
        existing = results.get(model_choice, {})
        trials_csv = existing.get("trials_csv")

    result_entry = {
        "model_choice": model_choice,
        "best_score": float(best_score),
        "best_smote_params": dict(best_smote_params),
        "best_model_params": dict(best_model_params),
        "optuna_settings": dict(optuna_settings),
        "search_space": dict(search_space),
        "saved_at": datetime.now().isoformat(),
        "trials_df": trials_df,
        "trials_csv": trials_csv,
    }
    results[model_choice] = result_entry

    entry = {
        "optuna_id": optuna_id,
        "feature_key": feature_key,
        "feature_id": feature_id,
        "features_path": features_path,
        "features_source": features_source,
        "features_rows": int(len(features_df)),
        "features_cols": int(len(features_df.columns)),
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
        result_payload.pop("trials_df", None)
        payload_results[str(choice)] = result_payload
    payload["results"] = payload_results
    try:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
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
    summary["metrics"] = st.session_state.get("acc_flow_metrics")
    summary["categories"] = st.session_state.get("acc_flow_categories")
    summary["lanes"] = st.session_state.get("acc_flow_lanes")
    summary["include_cluster_vars"] = st.session_state.get(
        "acc_flow_include_cluster_vars"
    )
    summary["cluster_vars"] = st.session_state.get("acc_flow_cluster_vars")
    summary["cluster_choice"] = st.session_state.get("acc_flow_cluster_choice")
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
    for choice, data in results.items():
        if not isinstance(data, dict):
            continue
        models[str(choice)] = {
            "best_score": data.get("best_score"),
            "best_smote_params": data.get("best_smote_params", {}),
            "best_model_params": data.get("best_model_params", {}),
            "settings": data.get("optuna_settings", {}),
            "search_space": data.get("search_space", {}),
            "saved_at": data.get("saved_at"),
            "trials_csv": data.get("trials_csv"),
        }
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
    cluster_feature_cols: Optional[List[str]],
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


def _save_model_artifact(
    model: object, run_id: str, label: str, model_name: str
) -> Optional[str]:
    if model is None:
        return None
    try:
        import joblib  # type: ignore
    except Exception:
        return None
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _slugify(label)
    model_slug = _slugify(model_name)
    path = MODELS_DIR / f"{run_id}_{suffix}_{model_slug}.joblib"
    try:
        joblib.dump(model, path)
    except Exception:
        return None
    return str(path)


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
) -> Dict[str, object]:
    signature_payload = {
        "model_choice": model_choice,
        "model_params_base": model_params_base,
        "model_params_cluster": model_params_cluster,
        "random_state": random_state,
        "time": time.time(),
    }
    signature = hashlib.md5(
        json.dumps(signature_payload, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()[:8]
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signature}"

    models: Dict[str, object] = {}
    base_model_path = _save_model_artifact(
        base_result.get("model"),
        run_id,
        "base",
        model_choice,
    )
    models["Base"] = {
        "model_name": model_choice,
        "model_params": dict(model_params_base),
        "metrics": base_result.get("metrics", {}),
        "confusion_matrix": base_result.get("confusion_matrix"),
        "model_path": base_model_path,
        "feature_cols": list(base_feature_cols),
        "split_info": base_result.get("split_info", {}),
    }
    if cluster_result is not None and cluster_feature_cols is not None:
        cluster_model_path = _save_model_artifact(
            cluster_result.get("model"),
            run_id,
            "base_cluster",
            model_choice,
        )
        models["Base + Cluster"] = {
            "model_name": model_choice,
            "model_params": dict(model_params_cluster) if model_params_cluster else {},
            "metrics": cluster_result.get("metrics", {}),
            "confusion_matrix": cluster_result.get("confusion_matrix"),
            "model_path": cluster_model_path,
            "feature_cols": list(cluster_feature_cols),
            "split_info": cluster_result.get("split_info", {}),
        }

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

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": _summarize_dataset(base_df),
        "training": {
            "use_balanced": bool(use_balanced),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "far_target": float(far_target),
            "random_state": int(random_state),
        },
        "features": _summarize_flow_settings(features_df),
        "feature_selection": _summarize_feature_selection(features_df),
        "optuna": _summarize_optuna(
            feature_key=feature_key,
            feature_id=feature_id,
            base_feature_cols=base_feature_cols,
            cluster_feature_cols=cluster_feature_cols,
        ),
        "balance": _summarize_balance(
            base_df=balanced_base_df,
            cluster_df=balanced_cluster_df,
        ),
        "models": models,
    }
    _append_history_entry(entry)
    return entry

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
            tramo_tuple = _build_tramo_selector(
                st.session_state.get("accidents_df"),
                date_start=None,
                date_end=None,
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
        value=False,
        key="acc_flow_use_batches",
    )
    
    include_cluster_vars = st.checkbox(
        "Incluir variables de cluster",
        value=bool(st.session_state.get("acc_flow_include_cluster_vars", False)),
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
            default_vars = ["Proporciones por cluster"]
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
            X = base_df[feature_cols].fillna(0)
            progress.progress(10)
            y = base_df["target"].astype(int)
            progress.progress(20)
            if y.nunique() < 2:
                st.warning(
                    "No hay dos clases en el target para calcular importancia."
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
            progress.progress(100)
            st.success("Importancias calculadas.")
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
                    results: Dict[str, object] = {}
                    if payload and isinstance(payload.get("results"), dict):
                        for choice, data in payload["results"].items():
                            if not isinstance(data, dict):
                                continue
                            item = dict(data)
                            item.setdefault("model_choice", choice)
                            item.setdefault("optuna_settings", {})
                            item.setdefault("search_space", {})
                            results[str(choice)] = item
                    elif payload:
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
                        results[str(legacy_choice)] = legacy_result

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
                "selection_mode": entry.get(
                    "selection_mode",
                    "all" if selected_features is None else "selected",
                ),
                "selected_features": entry.get(
                    "selected_features",
                    list(selected_features) if selected_features else [],
                ),
                "feature_cols": entry.get("feature_cols", list(c_cols)),
                "results": results,
                "saved_at": entry.get("saved_at"),
            }
            store[c_key] = entry
            st.session_state["optuna_results_store"] = store

    st.caption(
        f"Filas: {len(base_df):,} | Variables numericas (Base): {len(feature_cols_base)}"
    )
    if len(configs) > 1:
        st.caption(
            f"Variables de cluster seleccionadas: {len(selected_cluster_cols)} (se optimizara con y sin ellas)"
        )
    objective_options = {
        "F1": {"key": "f1", "direction": "maximize"},
        "ROC-AUC": {"key": "roc_auc", "direction": "maximize"},
        "Accuracy": {"key": "accuracy", "direction": "maximize"},
        "Recall": {"key": "recall", "direction": "maximize"},
        "Precision": {"key": "precision", "direction": "maximize"},
        "FNR (menor es mejor)": {"key": "fnr", "direction": "minimize"},
    }
    objective_label = st.selectbox(
        "Metrica objetivo",
        list(objective_options.keys()),
        key="optuna_objective_metric",
    )
    objective_key = objective_options[objective_label]["key"]
    objective_direction = objective_options[objective_label]["direction"]
    objective_verb = "minimiza" if objective_direction == "minimize" else "optimiza"
    st.caption(
        f"Optuna {objective_verb} {objective_label} en el set de test con umbral calibrado por FAR."
    )

    model_choice = st.selectbox(
        "Modelo",
        ["XGBoost", "Random Forest", "SVM"],
        key="optuna_model_choice",
    )
    model_result: Optional[Dict[str, object]] = None
    if entry and isinstance(entry.get("results"), dict):
        model_result = entry["results"].get(model_choice)
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
            st.session_state["optuna_best_smote_params"] = model_result.get(
                "best_smote_params"
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
        max_value=300,
        value=int(st.session_state.get("optuna_n_trials", 30)),
        step=5,
        key="optuna_n_trials",
    )
    timeout = st.number_input(
        "timeout (segundos)",
        min_value=60,
        max_value=36000,
        value=int(st.session_state.get("optuna_timeout", 3600)),
        step=60,
        key="optuna_timeout",
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
    optuna_far_target = st.slider(
        "FAR (False alarm rate) target",
        min_value=0.0,
        max_value=0.5,
        value=float(st.session_state.get("far_target", 0.2)),
        step=0.01,
        key="optuna_far_target",
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
            min_value=0.05,
            max_value=1.0,
            value=0.5,
            step=0.05,
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
            value=0.05,
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
    else:
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
        except ImportError:
            st.error(
                "imbalanced-learn no esta instalado. "
                "Ejecute `pip install imbalanced-learn`."
            )
            return
        if model_choice == "XGBoost":
            try:
                import xgboost as xgb  # noqa: F401
            except ImportError:
                st.error(
                    "xgboost no esta instalado. Ejecute `pip install xgboost`."
                )
                return

        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
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
        if y_test.nunique() < 2:
            st.warning(
                "El split temporal dejo una sola clase en test. "
                "Ajuste el rango o el test_size."
            )
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
        if min_count < 2:
            st.warning("No hay suficientes ejemplos minoritarios para SMOTE.")
            return
        max_k = max(1, min_count - 1)
        k_low = max(1, int(smote_k_min))
        k_high = min(int(smote_k_max), max_k)
        if k_high < k_low:
            st.warning(
                "El rango de smote_k no es valido para este dataset."
            )
            return

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
        else:
            if not svm_kernels:
                st.warning("Seleccione al menos un kernel para SVM.")
                return
            if svm_c_min > svm_c_max:
                st.warning("svm_C_min > svm_C_max.")
                return
            if svm_c_step <= 0:
                st.warning("svm_C_step debe ser mayor a 0.")
                return

        def _run_optimization(cols: List[str], label: str):
            st.markdown(f"**Optimizando: {label}**")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
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
            X_test_run = X_test[cols].fillna(0).astype("float32")

            def objective(trial: "optuna.Trial") -> float:
                smote_k = trial.suggest_int(
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
                    X_res, y_res = smote.fit_resample(X_train_run, y_train)
                except ValueError as exc:
                    raise optuna.TrialPruned(str(exc)) from exc

                model_params: Dict[str, object]
                if model_choice == "Random Forest":
                    n_estimators = trial.suggest_int(
                        "rf_n_estimators",
                        int(rf_n_min),
                        int(rf_n_max),
                        step=int(rf_n_step),
                    )
                    max_depth = trial.suggest_int(
                        "rf_max_depth",
                        int(rf_depth_min),
                        int(rf_depth_max),
                        step=int(rf_depth_step),
                    )
                    model_params = {
                        "n_estimators": int(n_estimators),
                        "max_depth": None if max_depth == 0 else int(max_depth),
                    }
                elif model_choice == "XGBoost":
                    n_estimators = trial.suggest_int(
                        "xgb_n_estimators",
                        int(xgb_n_min),
                        int(xgb_n_max),
                        step=int(xgb_n_step),
                    )
                    max_depth = trial.suggest_int(
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
                    model_params = {
                        "n_estimators": int(n_estimators),
                        "max_depth": int(max_depth),
                        "learning_rate": float(learning_rate),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample),
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
                    scores_val = _get_model_scores(model, X_val_run)
                    thr_info = _select_threshold_for_far(
                        y_val.to_numpy(),
                        scores_val,
                        far_target=float(optuna_far_target),
                    )
                    threshold = float(thr_info["threshold"])
                    scores_test = _get_model_scores(model, X_test_run)
                    preds = (scores_test >= threshold).astype(int)
                except Exception as exc:
                    raise optuna.TrialPruned(str(exc)) from exc

                if objective_key == "f1":
                    score = float(f1_score(y_test, preds, zero_division=0))
                elif objective_key == "roc_auc":
                    try:
                        score = float(roc_auc_score(y_test, scores_test))
                    except ValueError:
                        score = 0.5
                elif objective_key == "accuracy":
                    score = float(accuracy_score(y_test, preds))
                elif objective_key == "recall":
                    score = float(recall_score(y_test, preds, zero_division=0))
                elif objective_key == "precision":
                    score = float(precision_score(y_test, preds, zero_division=0))
                elif objective_key == "fnr":
                    tn, fp, fn, tp = confusion_matrix(
                        y_test, preds, labels=[0, 1]
                    ).ravel()
                    score = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
                else:
                    score = float(f1_score(y_test, preds, zero_division=0))
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
                    f"Tiempo transcurrido: {elapsed:.1f}s",
                    f"Trials: {completed} completados | {pruned} podados | {total} total",
                    (
                        f"{best_prefix} "
                        f"{objective_label}: {best_score:.4f}"
                        if best_score is not None
                        else f"{best_prefix} {objective_label}: -"
                    ),
                    "Mejores parametros:",
                    _format_best_params(best_params),
                ]
                status_placeholder.code("\n".join(lines), language="text")

            _render_optuna_progress(study, None)
            with st.spinner(f"Optuna ({label}) en ejecucion..."):
                study.optimize(
                    objective,
                    n_trials=int(n_trials),
                    timeout=int(timeout),
                    callbacks=[_render_optuna_progress],
                )

            if not study.trials:
                st.warning(f"Optuna ({label}) no genero resultados.")
                return None

            completed_trials = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and t.value is not None
            ]
            if not completed_trials:
                st.warning(f"Optuna ({label}) no genero trials completos.")
                return None

            if objective_direction == "minimize":
                best_trial = min(completed_trials, key=lambda t: float(t.value))
            else:
                best_trial = max(completed_trials, key=lambda t: float(t.value))
            best_params = dict(best_trial.params)
            best_score = float(best_trial.value)
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
            
            return {
                "best_score": best_score,
                "smote_params": smote_params,
                "model_params": model_params,
                "trials_df": trials_df
            }

        # Run for each config
        for cfg in configs:
            res = _run_optimization(cfg["cols"], cfg["label"])
            if res:
                # Update session state if this is the primary config (Model tab compatibility)
                if cfg["key"] == primary_key:
                    st.session_state["optuna_best_smote_params"] = res["smote_params"]
                    st.session_state["optuna_best_model_params"] = res["model_params"]
                    st.session_state["optuna_best_score"] = res["best_score"]
                    st.session_state["optuna_best_model_choice"] = model_choice
                    st.session_state["optuna_trials_df"] = res["trials_df"]
                    st.session_state["smote_k_neighbors"] = res["smote_params"]["smote_k_neighbors"]
                    st.session_state["smote_sampling_strategy"] = res["smote_params"]["smote_sampling_strategy"]
                
                search_space: Dict[str, object] = {
                    "smote": {
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
                    "random_state": int(optuna_random_state),
                    "test_size": float(optuna_test_size),
                    "val_size": float(optuna_val_size),
                    "far_target": float(optuna_far_target),
                    "objective_metric": objective_key,
                    "objective_label": objective_label,
                    "objective_direction": objective_direction,
                    "pruner": {
                        "enabled": bool(pruner_enabled),
                        "type": "MedianPruner" if pruner_enabled else "NopPruner",
                        "startup_trials": int(pruner_startup_trials),
                    },
                }
                if cfg["key"] == primary_key:
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
                    best_score=res["best_score"],
                    best_smote_params=res["smote_params"],
                    best_model_params=res["model_params"],
                    trials_df=res["trials_df"],
                    optuna_settings=optuna_settings_payload,
                    search_space=search_space,
                )
                st.success(
                    f"Optuna ({cfg['label']}) finalizado. "
                    f"{'Menor' if objective_direction == 'minimize' else 'Mejor'} "
                    f"{objective_label}: {res['best_score']:.4f}"
                )
        
        st.rerun()

    st.subheader("Resultados guardados")
    res_tabs = st.tabs([c["label"] for c in configs])
    for idx, cfg in enumerate(configs):
        with res_tabs[idx]:
            entry_cfg = store.get(cfg["key"])
            res_cfg = None
            if entry_cfg and isinstance(entry_cfg.get("results"), dict):
                res_cfg = entry_cfg["results"].get(model_choice)
            
            if isinstance(res_cfg, dict):
                trials_df_cfg = res_cfg.get("trials_df")
                trials_csv_cfg = res_cfg.get("trials_csv")
                if trials_df_cfg is None and trials_csv_cfg and Path(str(trials_csv_cfg)).exists():
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
                    st.caption("Configuracion Optuna")
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
            else:
                st.info(f"No hay resultados guardados para {cfg['label']} con {model_choice}.")








def _render_balance_tab() -> None:
    st.subheader("Balance con SMOTE")
    _render_selected_features_info()

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

        optuna_smote = st.session_state.get("optuna_best_smote_params")
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
                st.warning(
                    "⚠️ Los resultados de Optuna no coinciden con el dataset "
                    "o las variables seleccionadas actualmente."
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
            progress = _StreamlitProgress(total=4)
            with st.spinner("Aplicando SMOTE..."):
                progress.set_description("Preparando dataset")
                dataset_df = base_df
                progress.update(1)

                try:
                    progress.set_description("Aplicando SMOTE (Base)")
                    selected_features = st.session_state.get("selected_features")
                    
                    # 1. Base (Flow only)
                    feature_cols_base, missing_base = _resolve_feature_cols(
                        dataset_df,
                        selected_features,
                        include_cluster_features=False,
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

                    # 2. Cluster (Flow + Cluster)
                    progress.set_description("Aplicando SMOTE (Base + Cluster)")
                    balanced_cluster_df = None
                    
                    has_cluster_cols = bool(_get_cluster_cols(dataset_df))
                    if has_cluster_cols:
                        feature_cols_cluster, missing_cluster = _resolve_feature_cols(
                            dataset_df,
                            selected_features,
                            include_cluster_features=True,
                        )
                        # Only run if cluster features are actually included and different
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
                    if balanced_cluster_df is not None:
                        msg.append(f"Cluster: {len(balanced_cluster_df):,} filas")
                    
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
        st.subheader("Dataset balanceado en memoria")
        balanced_base = st.session_state.get("balanced_base_df")
        balanced_cluster = st.session_state.get("balanced_cluster_df")

        if balanced_base is None and balanced_cluster is None:
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

                if balanced_cluster is not None:
                    missing_all = [
                        c for c in selected_features if c not in balanced_cluster.columns
                    ]
                    if missing_all:
                         st.warning(
                            "⚠️ El dataset Base + Cluster en memoria no contiene variables "
                            f"seleccionadas: {', '.join(missing_all)}"
                        )

            tabs = st.tabs(["Base (Flujo)", "Base + Cluster"])
            
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
                _show_balanced_info(balanced_cluster, "Base + Cluster")

        if balanced_base is not None or balanced_cluster is not None:
            st.subheader("Exportar dataset balanceado")
            export_name = st.text_input(
                "Nombre de archivo (sin .csv)",
                value="accident_balanced",
                key="export_balanced_name",
            )
            col_exp1, col_exp2 = st.columns(2)
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
                if balanced_cluster is not None and st.button("Exportar Cluster"):
                    out_path = RESULTS_DIR / f"{export_name.strip()}_cluster.csv"
                    try:
                        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                        balanced_cluster.to_csv(out_path, index=False)
                        st.success(f"Cluster exportado en {out_path}")
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
    if model_choice == "Random Forest":
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
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth) if max_depth else None,
        }
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
        params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample),
        }
    else:
        kernel = st.selectbox(
            "kernel",
            ["rbf", "linear", "poly", "sigmoid"],
            key=f"{prefix}model_svm_kernel",
        )
        c_value = st.number_input(
            "C", min_value=0.01, value=1.0, step=0.1, key=f"{prefix}model_svm_c"
        )
        params = {"kernel": kernel, "C": float(c_value)}
    return params


def _render_model_tab() -> None:
    st.subheader("Modelos de prediccion")
    _render_selected_features_info()

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
    balanced_cluster_df_chk = st.session_state.get("balanced_cluster_df")
    cluster_cols_in_balanced = balanced_df is not None and bool(
        _get_cluster_cols(balanced_cluster_df_chk if balanced_cluster_df_chk is not None else pd.DataFrame())
    )
    has_cluster_features = (
        isinstance(cluster_features_df, pd.DataFrame)
        and not cluster_features_df.empty
    ) or cluster_cols_in_features
    has_cluster_available = has_cluster_features or cluster_cols_in_balanced
    
    
    test_size = float(st.session_state.get("test_size", 0.2))
    st.caption(f"Test size: {test_size:.2f}")

    far_target = st.slider(
        "FAR target",
        min_value=0.0,
        max_value=0.5,
        value=float(st.session_state.get("far_target", 0.2)),
        step=0.01,
        key="far_target",
    )
    val_size = st.slider(
        "Validation size",
        min_value=0.05,
        max_value=0.4,
        value=float(st.session_state.get("val_size", 0.2)),
        step=0.05,
        key="val_size",
    )

    model_choice = st.selectbox(
        "Modelo",
        ["Random Forest", "XGBoost", "SVM"],
        key="model_choice",
    )

    random_state = st.number_input(
        "random_state", min_value=0, value=42, step=1, key="model_random_state"
    )

    optuna_model_params = st.session_state.get("optuna_best_model_params")
    optuna_model_choice = st.session_state.get("optuna_best_model_choice")
    optuna_active_key = st.session_state.get("optuna_active_key")
    features_path = st.session_state.get("flow_features_path")
    features_source = st.session_state.get("flow_features_source")
    feature_key = _feature_selection_key(
        features_path, features_source, features_df
    )
    
    selected_features = st.session_state.get("selected_features")
    numeric_cols = _get_feature_cols(base_df)
    cluster_cols = _get_cluster_cols(base_df)

    if selected_features is None:
        cols_all = numeric_cols
    else:
        cols_all = [col for col in selected_features if col in numeric_cols]
    
    cols_base = [c for c in cols_all if c not in cluster_cols]
    
    key_base = _optuna_result_key(feature_key, cols_base)
    key_cluster = _optuna_result_key(feature_key, cols_all)
    
    optuna_matches_base = optuna_active_key == key_base
    optuna_matches_cluster = optuna_active_key == key_cluster
    
    optuna_ready = (
        isinstance(optuna_model_params, dict)
        and optuna_model_choice == model_choice
        and (optuna_matches_base or optuna_matches_cluster)
    )
    if optuna_ready:
        try:
            params_signature = json.dumps(
                optuna_model_params, sort_keys=True
            )
        except TypeError:
            params_signature = str(optuna_model_params)
        optuna_signature = f"{optuna_active_key}|{optuna_model_choice}|{params_signature}"
        if model_choice == "Random Forest":
            required_keys = ["model_rf_n_estimators", "model_rf_max_depth"]
        elif model_choice == "XGBoost":
            required_keys = [
                "model_xgb_n_estimators",
                "model_xgb_max_depth",
                "model_xgb_learning_rate",
                "model_xgb_subsample",
                "model_xgb_colsample",
            ]
        else:
            required_keys = ["model_svm_kernel", "model_svm_c"]

        target_prefixes = []
        if optuna_matches_base:
            target_prefixes.append("base_")
        if optuna_matches_cluster:
            target_prefixes.append("cluster_")

        applied_signature = st.session_state.get(
            "optuna_model_params_applied_signature"
        )
        if applied_signature != optuna_signature:
            if model_choice == "Random Forest":
                n_estimators = int(optuna_model_params.get("n_estimators", 200))
                max_depth = optuna_model_params.get("max_depth")
                for prefix in target_prefixes:
                    st.session_state[f"{prefix}model_rf_n_estimators"] = max(50, n_estimators)
                    st.session_state[f"{prefix}model_rf_max_depth"] = int(max_depth or 0)
            elif model_choice == "XGBoost":
                n_estimators = int(optuna_model_params.get("n_estimators", 300))
                max_depth = int(optuna_model_params.get("max_depth", 6))
                learning_rate = float(
                    optuna_model_params.get("learning_rate", 0.1)
                )
                subsample = float(
                    optuna_model_params.get("subsample", 1.0)
                )
                colsample = float(
                    optuna_model_params.get("colsample_bytree", 1.0)
                )
                for prefix in target_prefixes:
                    st.session_state[f"{prefix}model_xgb_n_estimators"] = max(50, n_estimators)
                    st.session_state[f"{prefix}model_xgb_max_depth"] = max(2, max_depth)
                    st.session_state[f"{prefix}model_xgb_learning_rate"] = max(0.01, learning_rate)
                    st.session_state[f"{prefix}model_xgb_subsample"] = max(0.5, subsample)
                    st.session_state[f"{prefix}model_xgb_colsample"] = max(0.5, colsample)
            else:
                kernel_value = optuna_model_params.get("kernel", "rbf")
                if kernel_value not in {"rbf", "linear", "poly", "sigmoid"}:
                    kernel_value = "rbf"
                c_value = float(optuna_model_params.get("C", 1.0))
                for prefix in target_prefixes:
                    st.session_state[f"{prefix}model_svm_kernel"] = str(kernel_value)
                    st.session_state[f"{prefix}model_svm_c"] = max(0.01, c_value)
            st.session_state["optuna_model_params_applied_signature"] = (
                optuna_signature
            )
        st.caption("Parametros Optuna cargados en los selectores.")
    elif isinstance(optuna_model_params, dict) and optuna_model_choice:
        st.caption(f"Optuna disponible para {optuna_model_choice}.")

    param_tabs = st.tabs(["Parámetros Base", "Parámetros Base + Cluster"])
    with param_tabs[0]:
        model_params_base = _render_model_params_ui(model_choice, "base_")
    with param_tabs[1]:
        model_params_cluster = _render_model_params_ui(model_choice, "cluster_")

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

    if st.button("Entrenar modelos"):
        use_balanced = bool(st.session_state.get("use_balanced_base", False))
        balanced_df = st.session_state.get("balanced_base_df")
        if balanced_df is None:
            balanced_df = st.session_state.get("balanced_cluster_df")
        cluster_cols_in_balanced = balanced_df is not None and bool(
            _get_cluster_cols(balanced_df)
        )
        has_cluster = has_cluster_features or cluster_cols_in_balanced
        base_feature_cols_used: List[str] = []
        cluster_feature_cols_used: Optional[List[str]] = None
        total_steps = 2 + (1 if has_cluster else 0)
        progress = _StreamlitProgress(total=total_steps)
        with st.spinner("Entrenando modelos..."):
            progress.set_description("Preparando datasets")
            progress.update(1)

            base_result: Optional[Dict[str, object]] = None
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
                    selected_features = st.session_state.get("selected_features")

                    known_cluster_cols = set(_get_cluster_cols(base_df))
                    selected_features_base = None
                    if selected_features is not None:
                        selected_features_base = [f for f in selected_features if f not in known_cluster_cols]

                    base_feature_cols, missing = _resolve_feature_cols(
                        train_df,
                        selected_features_base,
                        include_cluster_features=False,
                    )
                    base_feature_cols_used = list(base_feature_cols)
                    if selected_features is not None:
                        if missing:
                            st.warning(
                                "Variables seleccionadas no estan en el dataset: "
                                + ", ".join(missing)
                            )
                        if not base_feature_cols:
                            progress.close()
                            st.warning(
                                "Seleccione al menos una variable en Feature selection."
                            )
                            return
                    if not base_feature_cols:
                        progress.close()
                        st.warning("No hay variables numericas para entrenar.")
                        return
                    try:
                        progress.set_description("Entrenando modelo Base")
                        base_result = _train_model_on_split(
                            train_df,
                            test_df,
                            base_feature_cols,
                            model_choice,
                            model_params_base,
                            val_size=float(val_size),
                            far_target=float(far_target),
                            random_state=int(random_state),
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
                selected_features = st.session_state.get("selected_features")
                base_feature_cols, missing = _resolve_feature_cols(
                    base_df,
                    selected_features,
                    include_cluster_features=False,
                )
                base_feature_cols_used = list(base_feature_cols)
                if selected_features is not None:
                    if missing:
                        st.warning(
                            "Variables seleccionadas no estan en el dataset: "
                            + ", ".join(missing)
                        )
                    if not base_feature_cols:
                        progress.close()
                        st.warning(
                            "Seleccione al menos una variable en Feature selection."
                        )
                        return
                if not base_feature_cols:
                    progress.close()
                    st.warning("No hay variables numericas para entrenar.")
                    return
                try:
                    progress.set_description("Entrenando modelo Base")
                    base_result = _train_model(
                        base_df,
                        base_feature_cols,
                        model_choice,
                        model_params_base,
                        test_size=float(test_size),
                        val_size=float(val_size),
                        far_target=float(far_target),
                        random_state=int(random_state),
                    )
                except Exception as exc:
                    progress.close()
                    st.error(f"No se pudo entrenar el modelo base: {exc}")
                    return

            progress.update(1)
            results = {"Base": base_result["metrics"]}

            cluster_result: Optional[Dict[str, object]] = None
            if has_cluster:
                progress.set_description("Entrenando modelo Base + Cluster")
                balanced_cluster_df = st.session_state.get("balanced_cluster_df")
                if use_balanced and balanced_cluster_df is None:
                    st.warning(
                        "No hay dataset balanceado para Base + Cluster (quizas no se genero). Usando original."
                    )

                if use_balanced and balanced_cluster_df is not None:
                    split = _split_balanced_dataset(balanced_cluster_df)
                    if split is None:
                        st.warning(
                            "El dataset balanceado no tiene split valido. "
                            "Usando dataset original."
                        )
                    else:
                        train_df, test_df = split
                        if not _get_cluster_cols(train_df):
                            st.warning(
                                "El dataset balanceado no incluye variables de cluster. "
                                "Usando dataset original."
                            )
                        else:
                            selected_features = st.session_state.get("selected_features")
                            cluster_feature_cols, missing = _resolve_feature_cols(
                                train_df,
                                selected_features,
                                include_cluster_features=True,
                            )
                            cluster_feature_cols_used = list(cluster_feature_cols)
                            if selected_features is not None:
                                if missing:
                                    st.warning(
                                        "Variables seleccionadas no estan en el dataset: "
                                        + ", ".join(missing)
                                    )
                                if not cluster_feature_cols:
                                    progress.close()
                                    st.warning(
                                        "Seleccione al menos una variable en Feature selection."
                                    )
                                    return
                            if not cluster_feature_cols:
                                progress.close()
                                st.warning("No hay variables numericas para entrenar.")
                                return
                            try:
                                cluster_result = _train_model_on_split(
                                    train_df,
                                    test_df,
                                    cluster_feature_cols,
                                    model_choice,
                                    model_params_cluster,
                                    val_size=float(val_size),
                                    far_target=float(far_target),
                                    random_state=int(random_state),
                                )
                            except Exception as exc:
                                progress.close()
                                st.error(
                                    f"No se pudo entrenar el modelo con clusters: {exc}"
                                )
                                return
                        if cluster_result is None:
                            split = None

                if cluster_result is None:
                    cluster_train_df = _build_cluster_dataset(
                        base_df,
                        cluster_features_df=cluster_features_df,
                    )
                    if cluster_train_df is not None:
                        selected_features = st.session_state.get("selected_features")
                        cluster_feature_cols, missing = _resolve_feature_cols(
                            cluster_train_df,
                            selected_features,
                            include_cluster_features=True,
                        )
                        cluster_feature_cols_used = list(cluster_feature_cols)
                        if selected_features is not None:
                            if missing:
                                st.warning(
                                    "Variables seleccionadas no estan en el dataset: "
                                    + ", ".join(missing)
                                )
                            if not cluster_feature_cols:
                                progress.close()
                                st.warning(
                                    "Seleccione al menos una variable en Feature selection."
                                )
                                return
                        if not cluster_feature_cols:
                            progress.close()
                            st.warning("No hay variables numericas para entrenar.")
                            return
                        try:
                            cluster_result = _train_model(
                                cluster_train_df,
                                cluster_feature_cols,
                                model_choice,
                                model_params_cluster,
                                test_size=float(test_size),
                                val_size=float(val_size),
                                far_target=float(far_target),
                                random_state=int(random_state),
                            )
                        except Exception as exc:
                            progress.close()
                            st.error(
                                f"No se pudo entrenar el modelo con clusters: {exc}"
                            )
                            return

                if cluster_result is not None:
                    results["Base + Cluster"] = cluster_result["metrics"]
                progress.update(1)

            metrics_df = pd.DataFrame(results).T
            st.subheader("Resultados")
            st.dataframe(metrics_df, width="stretch")
            if "Base + Cluster" in results:
                delta = metrics_df.loc["Base + Cluster"] - metrics_df.loc["Base"]
                delta_df = delta.to_frame(name="delta").T
                st.subheader("Delta vs Base")
                st.dataframe(delta_df, width="stretch")
            st.subheader("Matriz de confusion")
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Base")
                base_cm = base_result.get("confusion_matrix")
                if base_cm is not None:
                    base_cm_df = pd.DataFrame(
                        base_cm,
                        index=["Actual 0", "Actual 1"],
                        columns=["Pred 0", "Pred 1"],
                    )
                    st.dataframe(base_cm_df, width="stretch")
            with col_b:
                if "Base + Cluster" in results:
                    st.caption("Base + Cluster")
                    cluster_cm = cluster_result.get("confusion_matrix")
                    if cluster_cm is not None:
                        cluster_cm_df = pd.DataFrame(
                            cluster_cm,
                            index=["Actual 0", "Actual 1"],
                            columns=["Pred 0", "Pred 1"],
                        )
                        st.dataframe(cluster_cm_df, width="stretch")
            try:
                _record_experiment_history(
                    base_df=base_df,
                    features_df=features_df,
                    balanced_df=balanced_df,
                    base_feature_cols=base_feature_cols_used,
                    base_result=base_result,
                    cluster_feature_cols=cluster_feature_cols_used,
                    cluster_result=cluster_result,
                    model_choice=model_choice,
                    model_params_base=model_params_base,
                    model_params_cluster=model_params_cluster,
                    random_state=int(random_state),
                    test_size=float(test_size),
                    val_size=float(val_size),
                    far_target=float(far_target),
                    use_balanced=bool(use_balanced),
                )
                st.caption("Historial actualizado y modelos guardados.")
            except Exception as exc:
                st.warning(f"No se pudo guardar en History: {exc}")
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
        "usa ese periodo como train y el resto como val/test."
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

    st.markdown("**Busqueda de ventana temporal**")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        min_window_days = st.number_input(
            "Ventana min (dias)",
            min_value=1,
            value=30,
            step=1,
            key="exp_samples_window_min",
        )
    with col_w2:
        max_window_days = st.number_input(
            "Ventana max (dias)",
            min_value=1,
            value=180,
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

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        max_segments = st.number_input(
            "Max segmentos a evaluar (0 = todos)",
            min_value=0,
            value=25,
            step=1,
            key="exp_samples_max_segments",
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
        objective_options = {
            "F1": {"key": "best_f1", "direction": "maximize"},
            "ROC-AUC": {"key": "roc_auc", "direction": "maximize"},
            "Accuracy": {"key": "accuracy", "direction": "maximize"},
            "Recall": {"key": "recall", "direction": "maximize"},
            "Precision": {"key": "precision", "direction": "maximize"},
            "FNR (menor es mejor)": {"key": "fnr", "direction": "minimize"},
            "FAR - Sensibilidad (menor es mejor)": {
                "key": "far_sens",
                "direction": "minimize",
            },
        }
        objective_label = st.selectbox(
            "Metrica objetivo (mejor mix)",
            list(objective_options.keys()),
            key="exp_samples_objective_metric",
        )
        objective_cfg = objective_options.get(
            objective_label, {"key": "best_f1", "direction": "maximize"}
        )
        objective_key = objective_cfg["key"]
        objective_direction = objective_cfg["direction"]
    use_cluster_features = st.checkbox(
        "Incluir variables de cluster (si existen)",
        value=True,
        key="exp_samples_use_cluster",
    )
    st.markdown("**Feature selection**")
    use_feature_selection = st.checkbox(
        "Usar seleccion de features (top %)",
        value=False,
        key="exp_samples_use_feature_selection",
    )
    feature_percent = 100
    if use_feature_selection:
        feature_percent = st.slider(
            "Porcentaje de variables mas importantes",
            5,
            100,
            30,
            5,
            key="exp_samples_feature_percent",
        )

    st.markdown("**Configuracion del modelo**")
    model_choice = st.selectbox(
        "Modelo para Experimento",
        ["Random Forest", "XGBoost", "SVM"],
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

    far_target = 0.2
    threshold_strategy = "optuna"
    threshold_strategy_label = "Optimizar threshold"
    with st.expander("Configuracion avanzada (parametros y rangos)"):
        st.markdown("**Split de datos (del resto)**")
        c_split1, c_split2 = st.columns(2)
        with c_split1:
            val_size = st.slider(
                "Validation Size (relativo)",
                0.1,
                0.9,
                0.4,
                0.05,
                key="exp_samples_val_size",
            )
        with c_split2:
            test_size = st.slider(
                "Test Size (relativo)",
                0.1,
                0.9,
                0.6,
                0.05,
                key="exp_samples_test_size",
            )
        st.markdown("**Calibracion de umbral**")
        far_target = st.slider(
            "FAR target",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.01,
            key="exp_samples_far_target",
        )
        threshold_options = {
            "Optimizar threshold": "optuna",
            "Calibrar por FAR": "far",
        }
        threshold_strategy_label = st.selectbox(
            "Estrategia de umbral",
            list(threshold_options.keys()),
            key="exp_samples_threshold_strategy",
        )
        threshold_strategy = threshold_options[threshold_strategy_label]

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
                "val_size": float(val_size),
                "test_size": float(test_size),
                "window_min_days": int(min_window_days),
                "window_max_days": int(max_window_days),
                "window_step_days": int(step_window_days),
                "max_segments": int(max_segments),
                "eval_top_n": int(eval_top_n),
                "use_cluster_features": bool(use_cluster_features),
                "feature_selection_enabled": bool(use_feature_selection),
                "feature_selection_percent": int(feature_percent),
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

        if max_segments and max_segments > 0:
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

        val_weight = float(val_size)
        test_weight = float(test_size)
        if val_weight <= 0 or test_weight <= 0:
            st.error("Validation/Test deben ser mayores que 0.")
            return
        test_ratio = test_weight / (val_weight + test_weight)
        val_ratio = 1 - test_ratio

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
                "run_id": run_id,
                "feature_selection_enabled": bool(use_feature_selection),
                "feature_selection_percent": int(feature_percent),
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
                "n_trials": int(n_trials),
                "timeout": int(timeout),
                "far_target": float(far_target),
                "val_weight": float(val_weight),
                "test_weight": float(test_weight),
                "rest_val_ratio": float(val_ratio),
                "rest_test_ratio": float(test_ratio),
                "use_cluster_features": bool(use_cluster_features),
                "search_space_config": json.dumps(search_space),
            }

        def _evaluate_candidate(
            row: pd.Series, *, rank: int
        ) -> Tuple[Dict[str, object], Optional[object]]:
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
                return payload, None

            segment_accidents = acc_seg.loc[
                (acc_seg["portico_last"] == row["portico_last"])
                & (acc_seg["portico_next"] == row["portico_next"])
            ].copy()

            segment_base_df = add_accident_target(
                segment_features, segment_accidents
            )
            if segment_base_df.empty:
                payload["error"] = "Dataset vacio tras merge."
                return payload, None

            train_mask = (
                (segment_base_df["interval_start"] >= row["window_start"])
                & (segment_base_df["interval_start"] <= row["window_end"])
            )
            train_df = segment_base_df.loc[train_mask].copy()
            rest_df = segment_base_df.loc[~train_mask].copy()
            if train_df.empty or rest_df.empty:
                payload["error"] = "No hay datos suficientes en train/rest."
                return payload, None

            try:
                val_df, test_df = _temporal_train_test_split(
                    rest_df, test_size=float(test_ratio)
                )
            except Exception as exc:
                payload["error"] = f"Split val/test fallo: {exc}"
                return payload, None

            if train_df["target"].nunique() < 2:
                payload["error"] = "Train solo tiene una clase."
                return payload, None
            if val_df["target"].nunique() < 2:
                payload["error"] = "Val solo tiene una clase."
                return payload, None
            if test_df["target"].nunique() < 2:
                payload["error"] = "Test solo tiene una clase."
                return payload, None

            all_feature_cols = _get_feature_cols(segment_base_df)
            if not use_cluster_features:
                cluster_cols = set(_get_cluster_cols(segment_base_df))
                all_feature_cols = [
                    c for c in all_feature_cols if c not in cluster_cols
                ]
            if not all_feature_cols:
                payload["error"] = "No hay variables numericas para entrenar."
                return payload, None

            runner = ExperimentsRunner()
            selected_feature_cols = all_feature_cols
            if use_feature_selection:
                try:
                    importance_df = runner.calculate_feature_importance(
                        segment_base_df, all_feature_cols
                    )
                    ordered = importance_df["variable"].tolist()
                    top_n = max(
                        1,
                        int(
                            round(
                                len(ordered)
                                * (float(feature_percent) / 100.0)
                            )
                        ),
                    )
                    selected_feature_cols = ordered[:top_n]
                    payload["feature_selection_total"] = len(all_feature_cols)
                    payload["feature_selection_selected"] = len(selected_feature_cols)
                except Exception as exc:
                    payload["error"] = f"Feature selection fallo: {exc}"
                    return payload, None
            try:
                result = runner.run_optimization_loop(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    feature_cols=selected_feature_cols,
                    model_choice=model_choice,
                    n_trials=int(n_trials),
                    timeout=int(timeout),
                    far_target=float(far_target),
                    search_space_config=search_space,
                    objective_key=objective_key,
                    objective_direction=objective_direction,
                    threshold_strategy=threshold_strategy,
                    return_model=True,
                )
            except Exception as exc:
                payload["error"] = f"Error en entrenamiento: {exc}"
                return payload, None

            model_obj = result.pop("model", None)
            payload.update(result)
            return payload, model_obj

        eval_limit = max(1, int(eval_top_n))
        eval_limit = min(eval_limit, len(candidates_df))
        eval_candidates = candidates_df.head(eval_limit)
        progress_eval = st.progress(0, text="Evaluando candidatos...")
        eval_results: List[Dict[str, object]] = []
        eval_models: List[Optional[object]] = []
        for idx, (_, row) in enumerate(eval_candidates.iterrows(), start=1):
            payload, model_obj = _evaluate_candidate(row, rank=idx)
            eval_results.append(payload)
            eval_models.append(model_obj)
            _append_experiment_result(exp_db_path, payload)
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

        res_df["is_best"] = False
        if best_rank:
            res_df.loc[
                res_df["candidate_rank"] == best_rank, "is_best"
            ] = True

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = None
        if best_rank and best_rank <= len(eval_models):
            best_model = eval_models[best_rank - 1]
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
            res_df.loc[
                res_df["candidate_rank"] == best_rank, "model_path"
            ] = model_path
            if best_row is not None:
                best_row = best_row.copy()
                best_row["model_path"] = model_path

        st.subheader("Resultados")
        st.dataframe(res_df, width="stretch")

        if best_row is not None:
            st.success(
                "Mejor mix segun "
                f"{objective_label}: "
                f"{best_row['segment_portico_last']} -> {best_row['segment_portico_next']} | "
                f"{best_row['window_start']} a {best_row['window_end']}"
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

    objective_options = {
        "F1": {"key": "best_f1", "direction": "maximize"},
        "ROC-AUC": {"key": "roc_auc", "direction": "maximize"},
        "Accuracy": {"key": "accuracy", "direction": "maximize"},
        "Recall": {"key": "recall", "direction": "maximize"},
        "Precision": {"key": "precision", "direction": "maximize"},
        "FNR (menor es mejor)": {"key": "fnr", "direction": "minimize"},
        "FAR - Sensibilidad (menor es mejor)": {
            "key": "far_sens",
            "direction": "minimize",
        },
    }
    objective_label = st.selectbox(
        "Metrica objetivo (mejor mix)",
        list(objective_options.keys()),
        key="exp_best_section_objective_metric",
    )
    objective_cfg = objective_options.get(
        objective_label, {"key": "best_f1", "direction": "maximize"}
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
        far_target = st.slider(
            "FAR target",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.01,
            key="exp_best_section_far_target",
        )
        threshold_options = {
            "Optimizar threshold": "optuna",
            "Calibrar por FAR": "far",
        }
        threshold_strategy_label = st.selectbox(
            "Estrategia de umbral",
            list(threshold_options.keys()),
            key="exp_best_section_threshold_strategy",
        )
        threshold_strategy = threshold_options[threshold_strategy_label]

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
                    "far_target": float(far_target),
                    "threshold_strategy": threshold_strategy,
                    "threshold_strategy_label": threshold_strategy_label,
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
                ) -> Dict[str, object]:
                    payload = dict(payload_common)
                    payload["type"] = dataset_type
                    payload["feature_selection_total"] = int(len(candidate_cols))
                    if not candidate_cols:
                        payload["error"] = "No hay variables numericas para entrenar."
                        return payload
                    if not selected_cols_override:
                        payload["error"] = (
                            "No hay ranking de importancia para seleccionar "
                            "variables."
                        )
                        return payload
                    selected_cols = [
                        col
                        for col in selected_cols_override
                        if col in candidate_cols
                    ]
                    if not selected_cols:
                        payload["error"] = (
                            "No hay variables numericas para entrenar."
                        )
                        return payload

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
                            far_target=float(far_target),
                            search_space_config=search_space,
                            objective_key=objective_key,
                            objective_direction=objective_direction,
                            threshold_strategy=threshold_strategy,
                        )
                        payload.update(result)
                    except Exception as exc:
                        payload["error"] = f"Error en Optuna: {exc}"
                    return payload

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
                    payload_base = _run_dataset(
                        dataset_type="Base",
                        candidate_cols=base_cols,
                        selected_cols_override=base_selected_cols,
                    )
                results.append(payload_base)
                _append_experiment_result(exp_db_path, payload_base)

                if cluster_cols:
                    payload_cluster = _run_dataset(
                        dataset_type="Base + Cluster",
                        candidate_cols=all_feature_cols,
                        selected_cols_override=combined_selected_cols,
                    )
                    results.append(payload_cluster)
                    _append_experiment_result(exp_db_path, payload_cluster)
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
            past_names = [p.name for p in past_files]
            sel_past = st.selectbox("Seleccionar archivo de resultados previos", past_names, key="history_exp_select")
            
            if sel_past:
                path = next(p for p in past_files if p.name == sel_past)
                
                # --- Export Logic ---
                # Attempt to extract timestamp from filename: experiments_results_YYYYMMDD_HHMMSS.csv
                # Pattern: experiments_results_{timestamp}.csv
                match = re.search(
                    r"(?:experiments_results|find_samples_sizes_results|best_highway_section_results)_(\d{8}_\d{6})\.csv",
                    sel_past,
                )
                timestamp = match.group(1) if match else None
                
                col_view, col_export = st.columns([0.8, 0.2])
                with col_export:
                    if timestamp:
                        # Find all files with this timestamp
                        related_files = sorted(RESULTS_DIR.glob(f"*{timestamp}*"))
                        if related_files:
                            try:
                                import zipfile
                                import io
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                    for f in related_files:
                                        zf.write(f, arcname=f.name)
                                zip_buffer.seek(0)
                                st.download_button(
                                    label="Exportar Experimento (ZIP)",
                                    data=zip_buffer,
                                    file_name=f"experiment_{timestamp}.zip",
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
                    is_find_samples = False
                    is_best_section = False
                    if "experiment" in past_df.columns:
                        is_find_samples = past_df["experiment"].astype(str).str.contains(
                            "find samples", case=False, na=False
                        ).any()
                        is_best_section = past_df["experiment"].astype(str).str.contains(
                            "best highway section", case=False, na=False
                        ).any()
                    if not is_find_samples and "type" in past_df.columns:
                        is_find_samples = past_df["type"].astype(str).str.contains(
                            "find samples", case=False, na=False
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

                    if is_find_samples:
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
                                if metric_key in {"fnr", "far_sens"}:
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
                                if metric_key in {"fnr", "far_sens"}:
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
                                    "fnr": "FNR"
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
            ["Features sampler", "Find samples sizes", "Best highway section"],
            key="exp_kind_choice",
        )
        if exp_kind == "Find samples sizes":
            _render_find_samples_sizes_experiment()
            return
        if exp_kind == "Best highway section":
            _render_best_highway_section_experiment()
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

        objective_options = {
            "F1": {"key": "best_f1", "direction": "maximize"},
            "ROC-AUC": {"key": "roc_auc", "direction": "maximize"},
            "Accuracy": {"key": "accuracy", "direction": "maximize"},
            "Recall": {"key": "recall", "direction": "maximize"},
            "Precision": {"key": "precision", "direction": "maximize"},
            "FNR (menor es mejor)": {"key": "fnr", "direction": "minimize"},
            "FAR - Sensibilidad (menor es mejor)": {
                "key": "far_sens",
                "direction": "minimize",
            },
        }
        objective_label = st.selectbox(
            "Metrica objetivo (Optuna/SMOTE)",
            list(objective_options.keys()),
            key="exp_features_objective_metric",
        )
        objective_cfg = objective_options.get(
            objective_label, {"key": "best_f1", "direction": "maximize"}
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
            far_target = st.slider(
                "FAR target",
                min_value=0.0,
                max_value=0.5,
                value=0.2,
                step=0.01,
                key="exp_far_target",
            )
            threshold_options = {
                "Optimizar threshold": "optuna",
                "Calibrar por FAR": "far",
            }
            threshold_strategy_label = st.selectbox(
                "Estrategia de umbral",
                list(threshold_options.keys()),
                key="exp_features_threshold_strategy",
            )
            threshold_strategy = threshold_options[threshold_strategy_label]

                
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
                    "test_size": float(test_size),
                    "val_size": float(val_size),
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

                results = runner.run_iterative_experiment(
                    base_df=base_df,
                    base_features_ordered=base_ordered,
                    cluster_features=combined_ordered_for_run,
                    model_choice=model_choice,
                    n_trials=int(n_trials),
                    timeout=int(timeout),
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
                    progress_bar=progress_bar,
                    dataset_name=selected_event,
                    features_name=selected_features,
                    max_k_limit=int(max_k_limit),
                    result_callback=_db_callback,
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
    st.title("Drivers Behavior Modeling and Simulation")



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
