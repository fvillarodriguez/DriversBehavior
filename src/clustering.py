#!/usr/bin/env python3
"""
clustering.py
=============
Funciones para calcular variables de clusterizacion y ejecutar clustering.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    FLOW_TABLE_NAME,
    FlowColumns,
    ensure_flow_db_summary,
    load_flujos,
    load_flujos_range,
    load_portico_maxspeed_map,
    normalize_plate_series,
    prompt_flow_sample_selection,
)

PLATE_CLEAN_COL = "plate_clean"
LANE_CLEAN_COL = "lane_numeric"

RESULTS_DIR = ROOT_DIR / "Resultados"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
CLUSTER_DB_PATH = CLUSTERING_RESULTS_DIR / "cluster_features.duckdb"
CLUSTER_TABLE_NAME = "cluster_features"
CLUSTER_BATCH_TABLE_NAME = "cluster_features_batches"
CLUSTER_META_TABLE_NAME = "cluster_features_meta"
DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME = "dynamic_assignments"
DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME = "dynamic_window_summary"
DYNAMIC_GMM_METADATA_TABLE_NAME = "dynamic_metadata"
DYNAMIC_GMM_LIVE_EVENTS_TABLE_NAME = "dynamic_live_events"
DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME = "dynamic_window_checkpoint"
DYNAMIC_GMM_RUN_STATUS_TABLE_NAME = "dynamic_run_status"
DEFAULT_CLUSTER_FEATURES = [
    "avg_speed_kmh",
    "exceso_velocidad",
    "avg_relative_speed",
    "avg_headway_s",
    "conflict_rate",
    "lane_prop_1",
    "lane_prop_2",
    "lane_change_rate",
]
CLUSTER_SUMMARY_PATTERN = re.compile(
    r"^cluster_summary(?:_(?P<method>kmeans|gmm|hdbscan))?(?:_k(?P<k>\d+))?\.csv$"
)
CLUSTER_LABEL_PATTERN = re.compile(
    r"^cluster_(?P<method>kmeans|gmm|hdbscan)(?:_k(?P<k>\d+))?\.csv$"
)
TTC_MAX_BY_PORTICO = {
    1: 5.5,
    2: 15,
    3: 13,
    4: 14,
    5: 10.5,
    6: 4.5,
    7: 13.5,
    8: 13.5,
    9: 15,
    10: 14.5,
    11: 11.5,
    12: 15,
    13: 11.5,
    14: 4.5,
    15: 14,
    16: 13,
    17: 9,
    18: 8.5,
    19: 12,
    20: 14,
    21: 10.5,
    22: 9.5,
    23: 13,
    24: 9.5,
    25: 15,
    26: 12,
    28: 7.5,
    29: 14,
    30: 15,
    31: 10,
    32: 8,
}
DEFAULT_FIXED_TTC_SECONDS = 1.5


def build_ttc_feature_metadata(
    ttc_mode: str = "dynamic",
    fixed_ttc_s: Optional[float] = None,
    ttc_max_map: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    normalized_mode = str(ttc_mode or "dynamic").strip().lower()
    if normalized_mode not in {"dynamic", "fixed"}:
        normalized_mode = "dynamic"

    metadata: Dict[str, object] = {
        "ttc_mode": normalized_mode,
        "ttc_mode_label": "Dinamico" if normalized_mode == "dynamic" else "Fijo",
    }
    if normalized_mode == "fixed":
        threshold = (
            float(fixed_ttc_s)
            if fixed_ttc_s is not None
            else float(DEFAULT_FIXED_TTC_SECONDS)
        )
        metadata["ttc_fixed_seconds"] = threshold
        metadata["ttc_threshold_map"] = None
        return metadata

    threshold_map = ttc_max_map or TTC_MAX_BY_PORTICO
    metadata["ttc_fixed_seconds"] = None
    metadata["ttc_threshold_map"] = {
        str(int(portico)): float(value)
        for portico, value in sorted(threshold_map.items())
    }
    return metadata


def list_cluster_feature_db_paths() -> List[Path]:
    return _glob_clustering_results("cluster_features*.duckdb")


def _clustering_result_dirs() -> List[Path]:
    dirs = [CLUSTERING_RESULTS_DIR, RESULTS_DIR]
    unique: List[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _glob_clustering_results(pattern: str) -> List[Path]:
    files: List[Path] = []
    for directory in _clustering_result_dirs():
        if directory.exists():
            files.extend(directory.glob(pattern))
    unique = {str(path.resolve()): path for path in files}
    return sorted(unique.values(), key=lambda path: (path.name, str(path)))


def _ensure_clustering_results_dir() -> Path:
    CLUSTERING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return CLUSTERING_RESULTS_DIR


def _normalize_feature_db_suffix(value: str) -> str:
    sanitized = value.strip()
    if not sanitized:
        return ""
    for ch in ("/", "\\", ":"):
        sanitized = sanitized.replace(ch, "_")
    return sanitized


def _build_cluster_feature_db_path(suffix: str) -> Path:
    normalized = _normalize_feature_db_suffix(suffix)
    if not normalized:
        return CLUSTER_DB_PATH
    filename = f"cluster_features({normalized}).duckdb"
    return CLUSTER_DB_PATH.with_name(filename)


def _prompt_cluster_feature_db_suffix() -> str:
    raw = input(
        "Ingrese texto para el archivo (se guardara como "
        "cluster_features(<texto>).duckdb). Enter=sin sufijo: "
    ).strip()
    if raw.lower() in {"q", "quit", "salir"}:
        return ""
    return raw


def _prompt_select_feature_db(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    print("\nArchivos de variables disponibles:")
    for idx, path in enumerate(paths, start=1):
        print(f"  [{idx}] {path.name}")
    choice = _prompt_int_value(
        "Seleccione un archivo (q para cancelar): ",
        default=None,
        min_value=1,
        max_value=len(paths),
    )
    if choice is None:
        return None
    return paths[choice - 1]


def ensure_plate_clean_column(df: pd.DataFrame, flow_cols: FlowColumns) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if flow_cols.plate_id not in df.columns:
        return df
    if PLATE_CLEAN_COL not in df.columns:
        df[PLATE_CLEAN_COL] = normalize_plate_series(df[flow_cols.plate_id])
        return df
    missing = df[PLATE_CLEAN_COL].isna()
    if missing.any():
        df.loc[missing, PLATE_CLEAN_COL] = normalize_plate_series(
            df.loc[missing, flow_cols.plate_id]
        )
    return df


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


def clean_flujos_for_clustering(
    flujos_df: pd.DataFrame,
    flow_cols: FlowColumns,
    outlier_action: str = "winsorize",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if flujos_df is None or flujos_df.empty:
        return pd.DataFrame()

    ensure_plate_clean_column(flujos_df, flow_cols)
    required = [
        flow_cols.timestamp,
        flow_cols.speed_kmh,
        flow_cols.portico,
        flow_cols.lane,
        PLATE_CLEAN_COL,
    ]
    if extra_cols:
        for col in extra_cols:
            if col in flujos_df.columns and col not in required:
                required.append(col)
    df = flujos_df[required].copy()
    df = df[df[PLATE_CLEAN_COL].notna()]
    plate_len = df[PLATE_CLEAN_COL].str.len().between(5, 6)
    df = df[plate_len.fillna(False)]
    if df.empty:
        return df

    df[flow_cols.portico] = _normalize_portico_series(df[flow_cols.portico])
    df[flow_cols.timestamp] = pd.to_datetime(df[flow_cols.timestamp], errors="coerce")
    df[flow_cols.speed_kmh] = pd.to_numeric(df[flow_cols.speed_kmh], errors="coerce")
    df[LANE_CLEAN_COL] = pd.to_numeric(df[flow_cols.lane], errors="coerce")

    df = df.dropna(subset=[flow_cols.timestamp, flow_cols.speed_kmh, LANE_CLEAN_COL])
    df = df[df[LANE_CLEAN_COL].isin([1, 2, 3])]
    if df.empty:
        return df

    dedup_cols = [PLATE_CLEAN_COL, flow_cols.portico, LANE_CLEAN_COL, flow_cols.timestamp]
    dup_mask = df.duplicated(subset=dedup_cols, keep=False)
    if dup_mask.any():
        # Ensure deterministic deduping when duplicates have different speeds.
        dup_df = df.loc[dup_mask].sort_values(
            dedup_cols + [flow_cols.speed_kmh],
            kind="mergesort",
        )
        df = pd.concat(
            [df.loc[~dup_mask], dup_df.drop_duplicates(subset=dedup_cols, keep="first")],
            ignore_index=True,
        )
    else:
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
    if df.empty:
        return df

    if outlier_action not in {"winsorize", "filter", "none"}:
        raise ValueError("outlier_action must be 'winsorize', 'filter', or 'none'")

    group_cols = [flow_cols.portico, LANE_CLEAN_COL]
    lower = df.groupby(group_cols)[flow_cols.speed_kmh].transform(
        lambda s: s.quantile(lower_q)
    )
    upper = df.groupby(group_cols)[flow_cols.speed_kmh].transform(
        lambda s: s.quantile(upper_q)
    )
    if outlier_action == "none":
        pass
    elif outlier_action == "winsorize":
        df[flow_cols.speed_kmh] = df[flow_cols.speed_kmh].clip(lower, upper)
    else:
        df = df[
            (df[flow_cols.speed_kmh] >= lower)
            & (df[flow_cols.speed_kmh] <= upper)
        ]

    return df


def Clusterization(
    flujos_df: pd.DataFrame,
    flow_cols: FlowColumns,
    ttc_max_map: Optional[Dict[int, float]] = None,
    monthly_weighting: bool = False,
    overlap_col: Optional[str] = None,
    include_counts: bool = False,
    max_headway_s: Optional[float] = 60.0,
    ttc_mode: str = "dynamic",
    fixed_ttc_s: Optional[float] = None,
    speed_limit_map: Optional[Dict[str, float]] = None,

    progress: Optional[object] = None,
    group_progress: Optional[object] = None,
    **clean_kwargs,
) -> pd.DataFrame:
    """
    Calcula indicadores por matricula para preparar la clusterizacion (K-means).
    Nota: lane_prop_3 es redundante para K-means y se debe omitir al entrenar.
    ttc_max_map: umbral TTC por portico usado para conflicto.
    monthly_weighting: si True, calcula variables por mes y pondera por total_passes.
    overlap_col: columna booleana para marcar filas de solape a excluir de agregados.
    include_counts: si True, agrega columnas de conteo para ponderacion posterior.
    max_headway_s: headways mayores a este umbral se tratan como NaN.
    ttc_mode: "dynamic" usa umbral por portico; "fixed" usa un umbral unico.
    fixed_ttc_s: umbral TTC fijo en segundos cuando ttc_mode == "fixed".
    speed_limit_map: mapa portico -> maxspeed para calcular exceso_velocidad.
    progress: barra de progreso para pasos principales.
    group_progress: barra de progreso para el loop de headway/TTC por grupo.
    """
    def _tick(label: str) -> None:
        if progress is None:
            return
        progress.set_description(label)
        progress.update(1)

    if flujos_df is None or flujos_df.empty:
        return pd.DataFrame()

    ensure_plate_clean_column(flujos_df, flow_cols)
    required = [
        flow_cols.timestamp,
        flow_cols.speed_kmh,
        flow_cols.portico,
        flow_cols.lane,
        PLATE_CLEAN_COL,
    ]
    missing = [col for col in required if col not in flujos_df.columns]
    if missing:
        raise ValueError(
            f"Missing required flow columns: {', '.join(missing)}"
        )

    extra_cols = [overlap_col] if overlap_col else None
    df = clean_flujos_for_clustering(
        flujos_df, flow_cols, extra_cols=extra_cols, **clean_kwargs
    )
    if df.empty:
        return pd.DataFrame()

    if ttc_max_map is None:
        ttc_max_map = TTC_MAX_BY_PORTICO
    ttc_mode = str(ttc_mode or "dynamic").strip().lower()
    if ttc_mode not in {"dynamic", "fixed"}:
        raise ValueError("ttc_mode must be 'dynamic' or 'fixed'.")
    if ttc_mode == "fixed":
        if fixed_ttc_s is None:
            fixed_ttc_s = DEFAULT_FIXED_TTC_SECONDS
        fixed_ttc_s = float(fixed_ttc_s)
        if fixed_ttc_s <= 0:
            raise ValueError("fixed_ttc_s must be > 0.")

    month_col = "month"
    if monthly_weighting:
        df[month_col] = df[flow_cols.timestamp].dt.to_period("M").astype(str)
        group_cols = [PLATE_CLEAN_COL, month_col]
    else:
        group_cols = [PLATE_CLEAN_COL]

    valid_mask = pd.Series(True, index=df.index)
    if overlap_col and overlap_col in df.columns:
        valid_mask = ~df[overlap_col].fillna(False)
    df_valid = df.loc[valid_mask]
    if df_valid.empty:
        return pd.DataFrame()

    timestamps = df_valid[flow_cols.timestamp]
    plates_clean = df_valid[PLATE_CLEAN_COL]
    n_days_active = timestamps.dt.normalize().groupby(plates_clean, sort=False).nunique()
    iso = timestamps.dt.isocalendar()
    week_id = (iso["year"] * 100 + iso["week"]).astype(int)
    n_weeks_active = week_id.groupby(plates_clean, sort=False).nunique()
    n_months_active = (
        timestamps.dt.to_period("M").groupby(plates_clean, sort=False).nunique()
    )
    n_years_active = timestamps.dt.year.groupby(plates_clean, sort=False).nunique()

    _tick("Preparando datos")

    plate_groups = df_valid.groupby(group_cols, sort=False)
    total_passes = plate_groups.size()
    sum_speed = plate_groups[flow_cols.speed_kmh].sum()
    summary = pd.DataFrame(index=total_passes.index)
    summary["total_passes"] = total_passes
    summary["avg_speed_kmh"] = sum_speed / total_passes

    if speed_limit_map is None:
        speed_limit_map = load_portico_maxspeed_map()
    speed_limit_lookup: Dict[str, float] = {}
    for portico, maxspeed in (speed_limit_map or {}).items():
        try:
            value = float(maxspeed)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            key = _normalize_portico_series(pd.Series([portico])).iloc[0]
            if pd.notna(key):
                speed_limit_lookup[str(key).strip().upper()] = value

    speed_limit = (
        df_valid[flow_cols.portico]
        .astype("string")
        .str.strip()
        .str.upper()
        .map(speed_limit_lookup)
    )
    speed_limit_mask = speed_limit.notna()
    if speed_limit_mask.any():
        exceso = (
            df_valid.loc[speed_limit_mask, flow_cols.speed_kmh].to_numpy(dtype=float)
            > speed_limit.loc[speed_limit_mask].to_numpy(dtype=float)
        ).astype(float)
        speed_limit_stats = (
            df_valid.loc[speed_limit_mask, group_cols]
            .assign(exceso_velocidad=exceso)
            .groupby(group_cols, sort=False)["exceso_velocidad"]
            .agg(["sum", "count"])
        )
        speed_limit_sum_s = speed_limit_stats["sum"]
        speed_limit_count_s = speed_limit_stats["count"]
    else:
        speed_limit_sum_s = pd.Series(dtype=float)
        speed_limit_count_s = pd.Series(dtype=float)
    speed_limit_den = speed_limit_count_s.reindex(summary.index).replace(0, np.nan)
    summary["exceso_velocidad"] = (
        speed_limit_sum_s.reindex(summary.index) / speed_limit_den
    ).fillna(0.0)
    summary["speed_limit_count"] = (
        speed_limit_count_s.reindex(summary.index).fillna(0).astype(int)
    )

    _tick("Agregando totales por matricula")

    lane_counts = (
        df_valid.groupby(group_cols + [LANE_CLEAN_COL])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3], fill_value=0)
    )
    lane_props = lane_counts.div(summary["total_passes"], axis=0).rename(
        columns={1: "lane_prop_1", 2: "lane_prop_2", 3: "lane_prop_3"}
    )
    summary = summary.join(lane_props, how="left")

    _tick("Calculando proporciones de carril")

    df_valid = df_valid.copy()
    df_valid["interval_5min"] = df_valid[flow_cols.timestamp].dt.floor("5min")
    interval_speed = (
        df_valid.groupby(["interval_5min", flow_cols.portico], sort=False)[flow_cols.speed_kmh]
        .mean()
        .rename("interval_speed_mean")
        .reset_index()
    )
    df_valid = df_valid.merge(interval_speed, on=["interval_5min", flow_cols.portico], how="left")
    df_valid["relative_speed"] = df_valid[flow_cols.speed_kmh] / df_valid["interval_speed_mean"]
    rel_stats = (
        df_valid.groupby(group_cols, sort=False)["relative_speed"]
        .agg(["sum", "count"])
    )

    key_cols = [PLATE_CLEAN_COL]
    if monthly_weighting:
        key_cols.append(month_col)

    if monthly_weighting:
        headway_group_cols = [month_col, flow_cols.portico, LANE_CLEAN_COL]
    else:
        headway_group_cols = [flow_cols.portico, LANE_CLEAN_COL]

    local_progress = None
    hw_progress = group_progress
    if hw_progress is None:
        local_progress = tqdm(
            total=3,
            desc="Headway/TTC por portico-carril",
            unit="paso",
            leave=False,
        )
        hw_progress = local_progress
    else:
        if hasattr(hw_progress, "set_description"):
            hw_progress.set_description("Headway/TTC por portico-carril")
        if hasattr(hw_progress, "reset"):
            try:
                hw_progress.reset(total=3)
            except TypeError:
                hw_progress.reset(3)

    try:
        ordered_hw = df.sort_values(
            headway_group_cols + [flow_cols.timestamp, PLATE_CLEAN_COL],
            kind="mergesort",
        )
        if hw_progress is not None and hasattr(hw_progress, "update"):
            hw_progress.update(1)

        prev_time = ordered_hw.groupby(headway_group_cols, sort=False)[
            flow_cols.timestamp
        ].shift(1)
        prev_speed = ordered_hw.groupby(headway_group_cols, sort=False)[
            flow_cols.speed_kmh
        ].shift(1)
        headway = (ordered_hw[flow_cols.timestamp] - prev_time).dt.total_seconds()
        if max_headway_s is not None:
            headway = headway.where(headway <= max_headway_s)
        speed = ordered_hw[flow_cols.speed_kmh]
        speed_diff = speed - prev_speed

        group_valid = valid_mask.reindex(ordered_hw.index).fillna(False)
        hw_mask = headway > 0
        valid_hw_mask = hw_mask & group_valid
        headway_stats = (
            ordered_hw.loc[valid_hw_mask, key_cols]
            .assign(headway=headway.loc[valid_hw_mask].to_numpy())
            .groupby(key_cols, sort=False)["headway"]
            .agg(["sum", "count"])
        )
        headway_sum_s = headway_stats["sum"]
        headway_count_s = headway_stats["count"]
        if hw_progress is not None and hasattr(hw_progress, "update"):
            hw_progress.update(1)

        conf_mask = hw_mask & prev_speed.notna()
        valid_conf_mask = conf_mask & group_valid
        if ttc_mode == "fixed":
            ttc_max = pd.Series(float(fixed_ttc_s), index=ordered_hw.index)
        elif ttc_max_map:
            portico_key = pd.to_numeric(
                ordered_hw[flow_cols.portico], errors="coerce"
            )
            portico_key = portico_key.where(
                portico_key.notna() & (portico_key % 1 == 0)
            ).astype("Int64")
            ttc_max = portico_key.map(ttc_max_map)
        else:
            ttc_max = pd.Series(math.nan, index=ordered_hw.index)

        ttc = pd.Series(math.nan, index=ordered_hw.index)
        valid_ttc = hw_mask & prev_speed.notna() & (speed_diff > 0)
        if valid_ttc.any():
            ttc.loc[valid_ttc] = (
                headway.loc[valid_ttc] * speed.loc[valid_ttc]
            ) / speed_diff.loc[valid_ttc]
        if ttc_max.notna().any():
            ttc = ttc.where(ttc_max.isna() | (ttc <= ttc_max), ttc_max)
        conflict = (ttc < ttc_max).astype(int)
        conflict_stats = (
            ordered_hw.loc[valid_conf_mask, key_cols]
            .assign(conflict=conflict.loc[valid_conf_mask].to_numpy())
            .groupby(key_cols, sort=False)["conflict"]
            .agg(["sum", "count"])
        )
        conf_sum_s = conflict_stats["sum"]
        conf_count_s = conflict_stats["count"]
        if hw_progress is not None and hasattr(hw_progress, "update"):
            hw_progress.update(1)
    finally:
        if local_progress is not None:
            local_progress.close()

    rel_sum_s = rel_stats["sum"]
    rel_count_s = rel_stats["count"]

    summary["avg_headway_s"] = headway_sum_s.reindex(summary.index) / headway_count_s.reindex(summary.index)
    summary["avg_relative_speed"] = rel_sum_s.reindex(summary.index) / rel_count_s.reindex(summary.index)
    summary["conflict_rate"] = conf_sum_s.reindex(summary.index) / conf_count_s.reindex(summary.index)
    if include_counts:
        summary["rel_speed_count"] = (
            rel_count_s.reindex(summary.index).fillna(0).astype(int)
        )
        summary["headway_count"] = (
            headway_count_s.reindex(summary.index).fillna(0).astype(int)
        )
        summary["conflict_count"] = (
            conf_count_s.reindex(summary.index).fillna(0).astype(int)
        )

    _tick("Calculando headway, velocidad relativa y conflicto")

    ordered = df_valid.sort_values(key_cols + [flow_cols.timestamp, flow_cols.portico])
    lane_prev = ordered.groupby(key_cols, sort=False)[LANE_CLEAN_COL].shift()
    lane_changed = ordered[LANE_CLEAN_COL].ne(lane_prev)
    group_keys = [ordered[col] for col in key_cols]
    lane_changes_s = lane_changed.groupby(group_keys, sort=False).sum()
    lane_changes_s = lane_changes_s.sub(1).clip(lower=0).astype(int)
    summary["lane_changes"] = lane_changes_s.reindex(summary.index).fillna(0).astype(int)
    summary["lane_change_rate"] = 0.0
    valid_rate = summary["total_passes"] > 1
    summary.loc[valid_rate, "lane_change_rate"] = (
        summary.loc[valid_rate, "lane_changes"] / (summary.loc[valid_rate, "total_passes"] - 1)
    )

    _tick("Calculando cambios de pista")

    if monthly_weighting:
        summary = summary.reset_index()
        transitions_sum = (
            (summary["total_passes"] - 1)
            .clip(lower=0)
            .groupby(summary[PLATE_CLEAN_COL], sort=False)
            .sum()
        )
        weighted_cols = [
            "avg_speed_kmh",
            "avg_relative_speed",
            "avg_headway_s",
            "conflict_rate",
            "lane_prop_1",
            "lane_prop_2",
            "lane_prop_3",
        ]
        weighted = summary.copy()
        weighted[weighted_cols] = weighted[weighted_cols].multiply(
            weighted["total_passes"], axis=0
        )
        weighted_grouped = weighted.groupby(PLATE_CLEAN_COL, sort=False)
        weighted_sum = weighted_grouped[weighted_cols].sum()
        total_passes_sum = weighted_grouped["total_passes"].sum()
        lane_changes_sum = weighted_grouped["lane_changes"].sum()
        speed_limit_count_sum = weighted_grouped["speed_limit_count"].sum()
        speed_limit_excess_sum = (
            weighted["exceso_velocidad"] * weighted["speed_limit_count"]
        ).groupby(weighted[PLATE_CLEAN_COL], sort=False).sum()
        summary = weighted_sum.div(total_passes_sum, axis=0)
        speed_limit_den = speed_limit_count_sum.replace(0, np.nan)
        summary["exceso_velocidad"] = (
            speed_limit_excess_sum / speed_limit_den
        ).fillna(0.0)
        summary["total_passes"] = total_passes_sum
        summary["lane_changes"] = lane_changes_sum
        summary["lane_change_rate"] = lane_changes_sum.div(transitions_sum).fillna(0.0)
        if include_counts:
            summary["speed_limit_count"] = speed_limit_count_sum
        summary = summary.reset_index()
    else:
        summary = summary.reset_index()

    summary = summary.rename(columns={PLATE_CLEAN_COL: "plate"})
    if not include_counts and "speed_limit_count" in summary.columns:
        summary = summary.drop(columns=["speed_limit_count"])
    summary["n_days_active"] = summary["plate"].map(n_days_active).fillna(0).astype(int)
    summary["n_weeks_active"] = summary["plate"].map(n_weeks_active).fillna(0).astype(int)
    summary["n_months_active"] = (
        summary["plate"].map(n_months_active).fillna(0).astype(int)
    )
    summary["n_years_active"] = (
        summary["plate"].map(n_years_active).fillna(0).astype(int)
    )
    summary = summary.sort_values(
        by=["total_passes", "plate"], ascending=[False, True]
    ).reset_index(drop=True)
    return summary


def _prompt_float_value(
    prompt: str,
    default: Optional[float] = None,
    min_value: Optional[float] = None,
) -> Optional[float]:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"q", "quit", "salir"}:
            return None
        if raw == "" and default is not None:
            return default
        raw = raw.replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            print("Entrada invalida. Ingrese un valor numerico.")
            continue
        if min_value is not None and value < min_value:
            print(f"El valor debe ser >= {min_value}.")
            continue
        return value


def _prompt_int_value(
    prompt: str,
    default: Optional[int] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[int]:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"q", "quit", "salir"}:
            return None
        if raw == "":
            if default is not None:
                return default
            print("Ingrese un valor.")
            continue
        try:
            value = int(raw)
        except ValueError:
            print("Entrada invalida. Ingrese un entero.")
            continue
        if min_value is not None and value < min_value:
            print(f"El valor debe ser >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"El valor debe ser <= {max_value}.")
            continue
        return value


def _order_feature_columns(available: List[str]) -> List[str]:
    preferred = [col for col in DEFAULT_CLUSTER_FEATURES if col in available]
    remaining = [col for col in available if col not in preferred]
    return preferred + remaining


def _parse_selection_indices(raw: str, max_index: int) -> Optional[List[int]]:
    tokens = re.split(r"[,\s]+", raw.strip())
    indices: List[int] = []
    for token in tokens:
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                return None
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                start, end = end, start
            if start < 1 or end > max_index:
                return None
            indices.extend(range(start, end + 1))
        else:
            if not token.isdigit():
                return None
            value = int(token)
            if value < 1 or value > max_index:
                return None
            indices.append(value)
    if not indices:
        return None
    seen = set()
    unique: List[int] = []
    for idx in indices:
        if idx in seen:
            continue
        unique.append(idx)
        seen.add(idx)
    return unique


def _prompt_feature_selection(features_df: pd.DataFrame) -> Optional[List[str]]:
    available = features_df.select_dtypes(include=["number"]).columns.tolist()
    available = _order_feature_columns(available)
    if not available:
        print("⚠️ No se encontraron variables numericas para clustering.")
        return None
    default_cols = _choose_feature_columns(features_df)
    if not default_cols:
        default_cols = available

    print("\nVariables disponibles para clustering:")
    for idx, col in enumerate(available, start=1):
        print(f"  [{idx}] {col}")
    print(f"Recomendadas: {', '.join(default_cols)}")

    while True:
        raw = input(
            "Seleccione variables (ej: 1,2,5; Enter=recomendadas; "
            "todo=todas; q=salir): "
        ).strip().lower()
        if raw in {"q", "quit", "salir"}:
            return None
        if raw == "":
            return default_cols
        if raw in {"todo", "todas", "all", "*"}:
            return available
        indices = _parse_selection_indices(raw, len(available))
        if indices is None:
            print("Entrada invalida. Use numeros separados por coma o rangos (ej: 1-3).")
            continue
        return [available[i - 1] for i in indices]


def _prompt_cluster_method() -> Optional[str]:
    options = {
        "1": "kmeans",
        "kmeans": "kmeans",
        "k-means": "kmeans",
        "kmean": "kmeans",
        "2": "gmm",
        "gmm": "gmm",
        "gaussian": "gmm",
        "mixture": "gmm",
        "3": "hdbscan",
        "hdbscan": "hdbscan",
    }
    while True:
        raw = input(
            "\nSeleccione metodo de clustering: "
            "[1] K-means [2] GMM [3] HDBSCAN (q para salir): "
        ).strip().lower()
        if raw in {"q", "quit", "salir"}:
            return None
        method = options.get(raw)
        if method:
            return method
        print("Entrada invalida. Intente nuevamente.")


def _maybe_export_cluster_inputs(
    features_df: pd.DataFrame, metrics_df: Optional[pd.DataFrame]
) -> None:
    if metrics_df is None:
        prompt = "\n¿Exportar variables a CSV? (s/n): "
    else:
        prompt = "\n¿Exportar variables y metricas a CSV? (s/n): "
    export = input(prompt).strip().lower()
    if export not in {"s", "si", "y", "yes"}:
        return
    features_path = save_cluster_features(features_df)
    print(f"📁 Variables guardadas en: {features_path}")
    if metrics_df is not None:
        metrics_path = save_cluster_metrics(metrics_df)
        print(f"📁 Metricas guardadas en: {metrics_path}")


def _prepare_cluster_features(
    features_df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    df = features_df.copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([math.inf, -math.inf], math.nan)
    return df.dropna(subset=feature_cols)


def split_frequent_drivers(
    features_df: pd.DataFrame,
    min_total_passes: int = 20,
    min_days_active: int = 5,
    min_weeks_active: Optional[int] = None,
    min_months_active: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en dos: conductores frecuentes (train set) e infrecuentes (rare set).
    Usa las columnas disponibles (si existen): total_passes, n_days_active, n_weeks_active,
    n_months_active.
    Retorna (df_frequent, df_rare).
    """
    if features_df is None or features_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    mask = pd.Series(True, index=features_df.index)

    if "total_passes" in features_df.columns:
        mask &= features_df["total_passes"] >= min_total_passes

    if "n_days_active" in features_df.columns:
        mask &= features_df["n_days_active"] >= min_days_active

    if min_weeks_active is not None and "n_weeks_active" in features_df.columns:
        mask &= features_df["n_weeks_active"] >= min_weeks_active

    if min_months_active is not None and "n_months_active" in features_df.columns:
        mask &= features_df["n_months_active"] >= min_months_active

    return features_df[mask], features_df[~mask]


def _normalize_plate_values(values: object) -> List[str]:
    if values is None:
        return []
    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, (list, tuple, set, np.ndarray, pd.Index)):
        series = pd.Series(list(values), dtype="object")
    else:
        series = pd.Series([values], dtype="object")
    normalized = normalize_plate_series(series).dropna().astype(str)
    return list(dict.fromkeys(normalized.tolist()))


def _dynamic_gmm_plate_filter_hash(plates: List[str]) -> str:
    payload = json.dumps(list(plates), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prevalent_plate_selection(
    features_df: pd.DataFrame,
    fraction_pct: float = 10.0,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Select the exact top fraction of historical plates ranked by prevalence.
    """
    fraction = float(fraction_pct)
    if not math.isfinite(fraction):
        raise ValueError("fraction_pct must be finite.")
    fraction = min(max(fraction, 0.0), 100.0)
    empty_ranked = pd.DataFrame(
        columns=[
            "plate",
            "n_months_active",
            "n_years_active",
            "total_passes",
        ]
    )
    if features_df is None or features_df.empty or "plate" not in features_df.columns:
        return {
            "plates": [],
            "ranked": empty_ranked,
            "selected_count": 0,
            "total_valid_plates": 0,
            "fraction_pct": fraction,
            "plate_hash": _dynamic_gmm_plate_filter_hash([]),
        }

    work = features_df.copy()
    if feature_cols:
        valid_feature_cols = [col for col in feature_cols if col in work.columns]
        if valid_feature_cols:
            work = _prepare_cluster_features(work, valid_feature_cols)
    if work.empty:
        return {
            "plates": [],
            "ranked": empty_ranked,
            "selected_count": 0,
            "total_valid_plates": 0,
            "fraction_pct": fraction,
            "plate_hash": _dynamic_gmm_plate_filter_hash([]),
        }

    work["plate"] = normalize_plate_series(work["plate"])
    work = work[work["plate"].notna()].copy()
    if work.empty:
        return {
            "plates": [],
            "ranked": empty_ranked,
            "selected_count": 0,
            "total_valid_plates": 0,
            "fraction_pct": fraction,
            "plate_hash": _dynamic_gmm_plate_filter_hash([]),
        }

    for col in ["n_months_active", "n_years_active", "total_passes"]:
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

    ranked = (
        work.groupby("plate", sort=False)
        .agg(
            n_months_active=("n_months_active", "max"),
            n_years_active=("n_years_active", "max"),
            total_passes=("total_passes", "max"),
        )
        .reset_index()
    )
    ranked = ranked.sort_values(
        ["n_months_active", "n_years_active", "total_passes", "plate"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    total_valid = int(len(ranked))
    selected_count = int(math.ceil(total_valid * fraction / 100.0)) if total_valid else 0
    if total_valid and fraction > 0:
        selected_count = max(1, selected_count)
    selected_count = min(selected_count, total_valid)
    plates = ranked["plate"].head(selected_count).astype(str).tolist()
    return {
        "plates": plates,
        "ranked": ranked,
        "selected_count": int(selected_count),
        "total_valid_plates": int(total_valid),
        "fraction_pct": fraction,
        "plate_hash": _dynamic_gmm_plate_filter_hash(plates),
    }


def _scale_cluster_features(
    cluster_df: pd.DataFrame,
    feature_cols: List[str],
    train_df: Optional[pd.DataFrame] = None
):
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for clustering. "
            "Install it with: pip install scikit-learn"
        ) from exc
        
    X_all = cluster_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    
    if train_df is not None and not train_df.empty:
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        scaler.fit(X_train)
    else:
        scaler.fit(X_all)
        
    return scaler.transform(X_all), scaler


def _distance_to_confidence_score(distances: np.ndarray) -> np.ndarray:
    """Map centroid distances to a bounded confidence score where higher is better."""
    clipped = np.clip(np.asarray(distances, dtype=float), a_min=0.0, a_max=None)
    return 1.0 / (1.0 + clipped)


def assign_clusters_kmeans(
    frequent_df: pd.DataFrame,
    rare_df: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    confidence_threshold_percentile: float = 95.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, object, float]:
    """
    Entrena KMeans en frequent_df. Asigna clusters a rare_df con umbral de distancia.
    `confidence_score` queda normalizado a [0, 1] donde mayor es mejor.
    `distance_to_centroid` preserva la distancia cruda usada para el umbral.
    Retorna (df_consolidado, model, threshold_used).
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError("scikit-learn required") from exc

    # 1. Scale based on frequent
    X_freq_scaled, scaler = _scale_cluster_features(frequent_df, feature_cols)
    
    # 2. Train KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(X_freq_scaled)
    
    # 3. Calculate distances for frequent to establish threshold
    # transform() returns distance to ALL centroids. We need min distance (to assigned centroid).
    freq_dists_all = kmeans.transform(X_freq_scaled)
    freq_min_dists = freq_dists_all.min(axis=1)
    
    # Calculate threshold (e.g. 95th percentile of training distances)
    threshold = float(np.percentile(freq_min_dists, confidence_threshold_percentile))
    
    # 4. Process Rare drivers
    # Scale rare using the SAME scaler
    if not rare_df.empty:
        X_rare = rare_df[feature_cols].to_numpy(dtype=float)
        X_rare_scaled = scaler.transform(X_rare)
        
        rare_dists_all = kmeans.transform(X_rare_scaled)
        rare_min_dists = rare_dists_all.min(axis=1)
        rare_conf_scores = _distance_to_confidence_score(rare_min_dists)
        rare_labels = kmeans.predict(X_rare_scaled)
        
        # Apply threshold
        # If dist > threshold -> -1
        mask_unknown = rare_min_dists > threshold
        rare_labels[mask_unknown] = -1
        
        # Build Rare result
        rare_result = rare_df.copy()
        rare_result["cluster_label"] = rare_labels
        rare_result["distance_to_centroid"] = rare_min_dists
        rare_result["confidence_score"] = rare_conf_scores
        rare_result["is_rare"] = True
    else:
        rare_result = pd.DataFrame()

    # 5. Build Frequent result
    freq_labels = kmeans.labels_
    freq_result = frequent_df.copy()
    freq_result["cluster_label"] = freq_labels
    freq_result["distance_to_centroid"] = freq_min_dists
    freq_result["confidence_score"] = _distance_to_confidence_score(freq_min_dists)
    freq_result["is_rare"] = False
    
    # Consolidate
    full_df = pd.concat([freq_result, rare_result], axis=0)
    return full_df, kmeans, threshold


def fit_gmm_cluster_model(
    frequent_df: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    random_state: int = 42,
    covariance_type: str = "full",
    max_iter: int = 100,
    n_init: int = 3,
) -> Tuple[object, object]:
    """
    Fit a GaussianMixture and its StandardScaler on the frequent-driver subset.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("scikit-learn required") from exc

    if frequent_df is None or frequent_df.empty:
        raise ValueError("frequent_df must contain rows to fit GMM.")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")
    missing = [col for col in feature_cols if col not in frequent_df.columns]
    if missing:
        raise ValueError(f"Missing GMM feature columns: {', '.join(missing)}")

    X_freq_scaled, scaler = _scale_cluster_features(frequent_df, feature_cols)
    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )
    gmm.fit(X_freq_scaled)
    return gmm, scaler


def _cluster_probability_columns(k: int) -> List[str]:
    return [f"cluster_prob_{cluster_idx}" for cluster_idx in range(int(k))]


def _add_soft_entropy(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    prob_cols = [
        col for col in result.columns if re.match(r"^cluster_prob_\d+$", str(col))
    ]
    prob_cols = sorted(prob_cols, key=lambda col: int(str(col).rsplit("_", 1)[1]))
    if not prob_cols:
        result["soft_entropy"] = math.nan
        return result
    probs = result[prob_cols].to_numpy(dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log(probs), 0.0)
    entropy = -(probs * log_probs).sum(axis=1)
    if len(prob_cols) > 1:
        entropy = entropy / math.log(len(prob_cols))
    result["soft_entropy"] = entropy
    return result


def predict_gmm_cluster_membership(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    model: object,
    scaler: object,
    confidence_threshold_proba: float = 0.70,
    min_window_passes: Optional[int] = None,
    include_membership_probabilities: bool = True,
    apply_confidence_threshold: bool = True,
) -> pd.DataFrame:
    """
    Apply a fitted GMM/scaler pair to a feature table and return hard labels,
    soft memberships and assignment diagnostics.
    """
    if features_df is None:
        features_df = pd.DataFrame()
    result = features_df.copy()
    k = int(getattr(model, "n_components", 0) or 0)
    if k <= 0:
        k = 0

    if result.empty:
        for col in [
            "raw_cluster_label",
            "cluster_label",
            "confidence_score",
            "assignment_status",
            "is_low_support",
        ]:
            result[col] = pd.Series(dtype="float64" if col != "assignment_status" else "object")
        if include_membership_probabilities:
            for col in _cluster_probability_columns(k):
                result[col] = pd.Series(dtype="float64")
        result["soft_entropy"] = pd.Series(dtype="float64")
        return result

    missing = [col for col in feature_cols if col not in result.columns]
    if missing:
        raise ValueError(f"Missing GMM prediction columns: {', '.join(missing)}")

    working = result.copy()
    for col in feature_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.replace([math.inf, -math.inf], math.nan)
    working = working.dropna(subset=feature_cols)
    if working.empty:
        result = result.iloc[0:0].copy()
        for col in [
            "raw_cluster_label",
            "cluster_label",
            "confidence_score",
            "assignment_status",
            "is_low_support",
        ]:
            result[col] = pd.Series(dtype="float64" if col != "assignment_status" else "object")
        if include_membership_probabilities:
            for col in _cluster_probability_columns(k):
                result[col] = pd.Series(dtype="float64")
        result["soft_entropy"] = pd.Series(dtype="float64")
        return result

    X = working[feature_cols].to_numpy(dtype=float)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)
    k = probs.shape[1]
    raw_labels = probs.argmax(axis=1).astype(int)
    max_probs = probs.max(axis=1)

    labels = raw_labels.copy()
    low_confidence = (
        max_probs < float(confidence_threshold_proba)
        if apply_confidence_threshold
        else np.zeros(len(working), dtype=bool)
    )
    if min_window_passes is not None and "total_passes" in working.columns:
        total_passes = pd.to_numeric(working["total_passes"], errors="coerce").fillna(0)
        low_support = total_passes.to_numpy(dtype=float) < float(min_window_passes)
    else:
        low_support = np.zeros(len(working), dtype=bool)
    labels[low_confidence | low_support] = -1

    status = np.full(len(working), "assigned", dtype=object)
    status[low_confidence] = "low_confidence"
    status[low_support] = "low_support"
    status[low_confidence & low_support] = "low_support_low_confidence"

    predicted = working.copy()
    predicted["raw_cluster_label"] = raw_labels
    predicted["cluster_label"] = labels
    predicted["confidence_score"] = max_probs
    predicted["assignment_status"] = status
    predicted["is_low_support"] = low_support
    if include_membership_probabilities:
        for cluster_idx in range(k):
            predicted[f"cluster_prob_{cluster_idx}"] = probs[:, cluster_idx]
    predicted = _add_soft_entropy(predicted)
    return predicted


def assign_clusters_gmm(
    frequent_df: pd.DataFrame,
    rare_df: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    confidence_threshold_proba: float = 0.70,
    random_state: int = 42,
    covariance_type: str = "full",
    include_membership_probabilities: bool = False,
) -> Tuple[pd.DataFrame, object, float]:
    """
    Entrena GMM en frequent_df. Asigna clusters a rare_df con umbral de probabilidad.
    Retorna (df_consolidado, model, threshold_used).
    """
    gmm, scaler = fit_gmm_cluster_model(
        frequent_df,
        feature_cols,
        k=int(k),
        random_state=random_state,
        covariance_type=covariance_type,
        n_init=3,
    )

    freq_result = predict_gmm_cluster_membership(
        frequent_df,
        feature_cols,
        gmm,
        scaler,
        confidence_threshold_proba=confidence_threshold_proba,
        include_membership_probabilities=include_membership_probabilities,
        apply_confidence_threshold=False,
    )
    freq_result["is_rare"] = False

    if not rare_df.empty:
        rare_result = predict_gmm_cluster_membership(
            rare_df,
            feature_cols,
            gmm,
            scaler,
            confidence_threshold_proba=confidence_threshold_proba,
            include_membership_probabilities=include_membership_probabilities,
            apply_confidence_threshold=True,
        )
        rare_result["is_rare"] = True
    else:
        rare_result = pd.DataFrame()

    full_df = pd.concat([freq_result, rare_result], axis=0)
    return full_df, gmm, confidence_threshold_proba



def _ensure_duckdb_available() -> None:
    if duckdb is None:
        raise ImportError(
            "duckdb no esta instalado. Ejecute `pip install duckdb` para habilitar esta funcion."
        )


def _connect_cluster_duckdb(
    read_only: bool = False, db_path: Optional[Path] = None
):
    _ensure_duckdb_available()
    target_path = db_path or CLUSTER_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ro_flag = read_only and target_path.exists()
    return duckdb.connect(str(target_path), read_only=ro_flag)


def _feature_metadata_to_rows(metadata: Optional[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for key, value in (metadata or {}).items():
        rows.append({"key": str(key), "value_json": json.dumps(value, ensure_ascii=True)})
    return pd.DataFrame(rows, columns=["key", "value_json"])


def _load_cluster_feature_metadata_from_conn(conn) -> Dict[str, object]:
    try:
        info = conn.execute(
            f"PRAGMA table_info('{CLUSTER_META_TABLE_NAME}')"
        ).fetchall()
    except Exception:
        return {}
    if not info:
        return {}
    try:
        meta_df = conn.execute(
            f"SELECT key, value_json FROM {CLUSTER_META_TABLE_NAME}"
        ).df()
    except Exception:
        return {}
    metadata: Dict[str, object] = {}
    for row in meta_df.itertuples(index=False):
        key = str(getattr(row, "key", "") or "").strip()
        if not key:
            continue
        raw_value = getattr(row, "value_json", None)
        if raw_value is None or pd.isna(raw_value):
            metadata[key] = None
            continue
        try:
            metadata[key] = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            metadata[key] = raw_value
    return metadata


def save_cluster_features_duckdb(
    features_df: pd.DataFrame,
    db_path: Optional[Path] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    target_path = db_path or CLUSTER_DB_PATH
    conn = _connect_cluster_duckdb(read_only=False, db_path=target_path)
    try:
        conn.register("cluster_features_df", features_df)
        conn.execute(
            f"CREATE OR REPLACE TABLE {CLUSTER_TABLE_NAME} AS "
            "SELECT * FROM cluster_features_df"
        )
        conn.unregister("cluster_features_df")
        conn.execute(f"DROP TABLE IF EXISTS {CLUSTER_META_TABLE_NAME}")
        meta_rows = _feature_metadata_to_rows(metadata)
        if not meta_rows.empty:
            conn.register("cluster_features_meta_df", meta_rows)
            conn.execute(
                f"CREATE TABLE {CLUSTER_META_TABLE_NAME} AS "
                "SELECT * FROM cluster_features_meta_df"
            )
            conn.unregister("cluster_features_meta_df")
    finally:
        conn.close()
    return target_path


def load_cluster_feature_bundle_duckdb(
    db_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    target_path = db_path or CLUSTER_DB_PATH
    if not target_path.exists():
        return pd.DataFrame(), {}
    conn = _connect_cluster_duckdb(read_only=True, db_path=target_path)
    try:
        info = conn.execute(
            f"PRAGMA table_info('{CLUSTER_TABLE_NAME}')"
        ).fetchall()
        if not info:
            return pd.DataFrame(), {}
        features_df = conn.execute(f"SELECT * FROM {CLUSTER_TABLE_NAME}").df()
        metadata = _load_cluster_feature_metadata_from_conn(conn)
        return features_df, metadata
    finally:
        conn.close()


def load_cluster_feature_metadata_duckdb(
    db_path: Optional[Path] = None
) -> Dict[str, object]:
    _features_df, metadata = load_cluster_feature_bundle_duckdb(db_path)
    return metadata


def load_cluster_features_duckdb(
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    features_df, _metadata = load_cluster_feature_bundle_duckdb(db_path)
    return features_df


def _require_joblib():
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "joblib is required to persist dynamic GMM artifacts."
        ) from exc
    return joblib


def list_dynamic_gmm_db_paths(output_dir: Optional[Path] = None) -> List[Path]:
    if output_dir is not None:
        target_dir = Path(output_dir)
        if not target_dir.exists():
            return []
        return sorted(target_dir.glob("dynamic_gmm_*.duckdb"))
    return _glob_clustering_results("dynamic_gmm_*.duckdb")


def list_dynamic_gmm_checkpoint_db_paths(output_dir: Optional[Path] = None) -> List[Path]:
    paths = list_dynamic_gmm_db_paths(output_dir=output_dir)
    if not paths or duckdb is None:
        return []
    checkpoint_paths: List[Path] = []
    for path in paths:
        try:
            conn = _connect_dynamic_gmm_duckdb(path, read_only=False)
            try:
                if _dynamic_gmm_table_exists(
                    conn,
                    DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME,
                ):
                    checkpoint_paths.append(path)
            finally:
                conn.close()
        except Exception:
            continue
    return checkpoint_paths


def _dynamic_gmm_now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def _dynamic_gmm_stamp() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")


def _dynamic_gmm_table_exists(conn, table_name: str) -> bool:
    try:
        info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    except Exception:
        return False
    return bool(info)


def _dynamic_gmm_quote_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _dynamic_gmm_table_column_types(conn, table_name: str) -> Dict[str, str]:
    if not _dynamic_gmm_table_exists(conn, table_name):
        return {}
    try:
        rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    except Exception:
        return {}
    return {str(row[1]): str(row[2] or "VARCHAR") for row in rows}


def _dynamic_gmm_duckdb_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "VARCHAR"


def _connect_dynamic_gmm_duckdb(path: Path, *, read_only: bool = False):
    _ensure_duckdb_available()
    try:
        return duckdb.connect(str(path), read_only=bool(read_only))
    except Exception as exc:
        message = str(exc).lower()
        if read_only and "different configuration" in message:
            return duckdb.connect(str(path))
        raise


def _load_dynamic_gmm_metadata_from_conn(conn) -> Dict[str, object]:
    if not _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_METADATA_TABLE_NAME):
        return {}
    try:
        meta_df = conn.execute(
            f"SELECT key, value_json FROM {DYNAMIC_GMM_METADATA_TABLE_NAME}"
        ).df()
    except Exception:
        return {}
    metadata: Dict[str, object] = {}
    for row in meta_df.itertuples(index=False):
        key = str(getattr(row, "key", "") or "").strip()
        if not key:
            continue
        raw_value = getattr(row, "value_json", None)
        try:
            metadata[key] = json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            metadata[key] = raw_value
    return metadata


def _dynamic_gmm_config_fingerprint(config: Dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dynamic_gmm_normalize_config_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _dynamic_gmm_normalize_config_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_dynamic_gmm_normalize_config_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _dynamic_gmm_config_differences(
    checkpoint_value: object,
    current_value: object,
    *,
    path: str = "",
) -> List[Tuple[str, object, object]]:
    checkpoint_normalized = _dynamic_gmm_normalize_config_value(checkpoint_value)
    current_normalized = _dynamic_gmm_normalize_config_value(current_value)
    if isinstance(checkpoint_normalized, dict) and isinstance(current_normalized, dict):
        diffs: List[Tuple[str, object, object]] = []
        for key in sorted(set(checkpoint_normalized) | set(current_normalized)):
            child_path = f"{path}.{key}" if path else str(key)
            diffs.extend(
                _dynamic_gmm_config_differences(
                    checkpoint_normalized.get(key),
                    current_normalized.get(key),
                    path=child_path,
                )
            )
        return diffs
    if checkpoint_normalized != current_normalized:
        return [(path, checkpoint_normalized, current_normalized)]
    return []


def _dynamic_gmm_format_config_value(value: object, *, max_len: int = 160) -> str:
    try:
        text = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    except TypeError:
        text = str(value)
    if len(text) > int(max_len):
        return text[: int(max_len) - 3] + "..."
    return text


def _dynamic_gmm_config_mismatch_details(
    checkpoint_metadata: Dict[str, object],
    current_config: Dict[str, object],
    *,
    max_differences: int = 12,
) -> str:
    checkpoint_config = {
        key: checkpoint_metadata.get(key)
        for key in current_config.keys()
    }
    diffs = _dynamic_gmm_config_differences(checkpoint_config, current_config)
    if not diffs:
        return (
            "No se pudo aislar diferencias campo a campo; el fingerprint guardado "
            "difiere del fingerprint actual."
        )
    lines = ["Variables diferentes:"]
    for key, checkpoint_value, current_value in diffs[: int(max_differences)]:
        lines.append(
            "- "
            + str(key)
            + ": checkpoint="
            + _dynamic_gmm_format_config_value(checkpoint_value)
            + " | actual="
            + _dynamic_gmm_format_config_value(current_value)
        )
    remaining = len(diffs) - int(max_differences)
    if remaining > 0:
        lines.append(f"- ... y {remaining} diferencia(s) adicional(es).")
    return "\n".join(lines)


def build_dynamic_gmm_config_payload(
    *,
    base_features_df: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    confidence_threshold_proba: float,
    window_days: int,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    min_window_passes: int = 5,
    train_params: Optional[Dict[str, int]] = None,
    random_state: int = 42,
    covariance_type: str = "full",
    include_membership_probabilities: bool = True,
    ttc_mode: str = "dynamic",
    fixed_ttc_s: Optional[float] = None,
    assignment_scope: str = "all",
    prevalent_fraction_pct: Optional[float] = None,
    prevalent_plate_count: Optional[int] = None,
    prevalent_valid_plate_count: Optional[int] = None,
    prevalent_plate_hash: Optional[str] = None,
    prevalent_source: Optional[str] = None,
) -> Dict[str, object]:
    train_params = dict(train_params or {})
    min_total_passes = int(train_params.get("min_total_passes", 20))
    min_days_active = int(train_params.get("min_days_active", 1))
    min_months_active = int(train_params.get("min_months_active", 1))
    effective_feature_cols = [
        col for col in feature_cols if col in base_features_df.columns
    ]
    windows = build_dynamic_gmm_windows(date_start, date_end, int(window_days))
    payload: Dict[str, object] = {
        "method": "gmm_dynamic",
        "feature_cols": list(effective_feature_cols),
        "requested_feature_cols": list(feature_cols),
        "k": int(k),
        "confidence_threshold_proba": float(confidence_threshold_proba),
        "window_days": int(window_days),
        "window_step_days": 1,
        "min_window_passes": int(min_window_passes),
        "date_start": str(pd.Timestamp(date_start).date()),
        "date_end": str(pd.Timestamp(date_end).date()),
        "n_windows": int(len(windows)),
        "covariance_type": covariance_type,
        "random_state": int(random_state),
        "include_membership_probabilities": bool(include_membership_probabilities),
        "train_params": {
            "min_total_passes": min_total_passes,
            "min_days_active": min_days_active,
            "min_months_active": min_months_active,
        },
        "ttc_mode": ttc_mode,
        "ttc_fixed_seconds": fixed_ttc_s,
    }
    normalized_scope = str(assignment_scope or "all").strip().lower()
    if normalized_scope != "all":
        payload.update(
            {
                "assignment_scope": normalized_scope,
                "prevalent_fraction_pct": (
                    None
                    if prevalent_fraction_pct is None
                    else float(prevalent_fraction_pct)
                ),
                "prevalent_plate_count": (
                    None if prevalent_plate_count is None else int(prevalent_plate_count)
                ),
                "prevalent_valid_plate_count": (
                    None
                    if prevalent_valid_plate_count is None
                    else int(prevalent_valid_plate_count)
                ),
                "prevalent_plate_hash": prevalent_plate_hash,
                "prevalent_source": prevalent_source,
            }
        )
    return payload


def check_dynamic_gmm_checkpoint_compatibility(
    db_path: Path,
    current_config: Dict[str, object],
) -> Dict[str, object]:
    _ensure_duckdb_available()
    path = Path(db_path)
    if not path.exists():
        return {
            "compatible": False,
            "details": f"No existe el checkpoint: {path}",
            "metadata": {},
        }
    conn = _connect_dynamic_gmm_duckdb(path, read_only=False)
    try:
        metadata = _load_dynamic_gmm_metadata_from_conn(conn)
    finally:
        conn.close()
    if not metadata:
        return {
            "compatible": False,
            "details": "El DuckDB seleccionado no contiene metadata de GMM dinamico.",
            "metadata": {},
        }
    current_fingerprint = _dynamic_gmm_config_fingerprint(current_config)
    checkpoint_fingerprint = str(metadata.get("config_fingerprint") or "")
    if checkpoint_fingerprint and checkpoint_fingerprint != current_fingerprint:
        return {
            "compatible": False,
            "details": _dynamic_gmm_config_mismatch_details(
                metadata,
                current_config,
            ),
            "metadata": metadata,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "current_fingerprint": current_fingerprint,
        }
    if not checkpoint_fingerprint:
        return {
            "compatible": False,
            "details": "El checkpoint no tiene fingerprint de configuracion para validar resume.",
            "metadata": metadata,
            "current_fingerprint": current_fingerprint,
        }
    return {
        "compatible": True,
        "details": "",
        "metadata": metadata,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "current_fingerprint": current_fingerprint,
    }


def _dynamic_gmm_empty_assignments_df(
    effective_feature_cols: List[str],
    k: int,
    include_membership_probabilities: bool,
) -> pd.DataFrame:
    probability_cols = (
        _cluster_probability_columns(int(k))
        if include_membership_probabilities
        else []
    )
    return pd.DataFrame(
        columns=[
            "run_id",
            "window_index",
            "window_label",
            "window_start",
            "window_end",
            "plate",
            *effective_feature_cols,
            "raw_cluster_label",
            "cluster_label",
            "confidence_score",
            "assignment_status",
            "is_low_support",
            *probability_cols,
            "soft_entropy",
        ]
    )


def _dynamic_gmm_empty_window_summary(
    *,
    run_id: str,
    window_index: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    window_label: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": str(run_id),
                "window_index": int(window_index),
                "window_label": str(window_label),
                "window_start": pd.Timestamp(window_start),
                "window_end": pd.Timestamp(window_end),
                "rows": 0,
                "assigned_rows": 0,
                "unknown_rows": 0,
                "low_support_rows": 0,
                "low_confidence_rows": 0,
                "mean_confidence": math.nan,
                "mean_soft_entropy": math.nan,
            }
        ]
    )


def _dynamic_gmm_append_df(conn, table_name: str, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]) or col in {
            "window_start",
            "window_end",
        }:
            converted = pd.to_datetime(work[col], errors="coerce")
            if converted.notna().any():
                work[col] = converted.astype("datetime64[us]")
    view_name = f"_{table_name}_view"
    conn.register(view_name, work)
    try:
        if not _dynamic_gmm_table_exists(conn, table_name):
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {view_name}")
        else:
            table_types = _dynamic_gmm_table_column_types(conn, table_name)
            for col in work.columns:
                if col in table_types:
                    continue
                col_type = _dynamic_gmm_duckdb_type_for_series(work[col])
                conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN "
                    f"{_dynamic_gmm_quote_identifier(col)} {col_type}"
                )
                table_types[col] = col_type
                if col.startswith("cluster_count_"):
                    conn.execute(
                        f"UPDATE {table_name} SET "
                        f"{_dynamic_gmm_quote_identifier(col)} = 0 "
                        f"WHERE {_dynamic_gmm_quote_identifier(col)} IS NULL"
                    )
                elif col.startswith("cluster_share_"):
                    conn.execute(
                        f"UPDATE {table_name} SET "
                        f"{_dynamic_gmm_quote_identifier(col)} = 0.0 "
                        f"WHERE {_dynamic_gmm_quote_identifier(col)} IS NULL"
                    )
            insert_cols = list(table_types.keys())
            select_exprs = []
            for col in insert_cols:
                quoted_col = _dynamic_gmm_quote_identifier(col)
                if col in work.columns:
                    select_exprs.append(quoted_col)
                elif col.startswith("cluster_count_"):
                    select_exprs.append(f"CAST(0 AS {table_types[col]}) AS {quoted_col}")
                elif col.startswith("cluster_share_"):
                    select_exprs.append(f"CAST(0.0 AS {table_types[col]}) AS {quoted_col}")
                else:
                    select_exprs.append(f"CAST(NULL AS {table_types[col]}) AS {quoted_col}")
            conn.execute(
                f"INSERT INTO {table_name} ("
                + ", ".join(_dynamic_gmm_quote_identifier(col) for col in insert_cols)
                + ") SELECT "
                + ", ".join(select_exprs)
                + f" FROM {view_name}"
            )
    finally:
        conn.unregister(view_name)


def _dynamic_gmm_replace_metadata(conn, metadata: Dict[str, object]) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {DYNAMIC_GMM_METADATA_TABLE_NAME}")
    meta_rows = _feature_metadata_to_rows(metadata)
    if meta_rows.empty:
        meta_rows = pd.DataFrame(columns=["key", "value_json"])
    conn.register("dynamic_metadata_df", meta_rows)
    try:
        conn.execute(
            f"CREATE TABLE {DYNAMIC_GMM_METADATA_TABLE_NAME} AS "
            "SELECT * FROM dynamic_metadata_df"
        )
    finally:
        conn.unregister("dynamic_metadata_df")


def _dynamic_gmm_replace_run_status(conn, row: Dict[str, object]) -> None:
    df = pd.DataFrame([row])
    conn.register("dynamic_run_status_df", df)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {DYNAMIC_GMM_RUN_STATUS_TABLE_NAME}")
        conn.execute(
            f"CREATE TABLE {DYNAMIC_GMM_RUN_STATUS_TABLE_NAME} AS "
            "SELECT * FROM dynamic_run_status_df"
        )
    finally:
        conn.unregister("dynamic_run_status_df")


def _dynamic_gmm_checkpoint_rows(
    run_id: str,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
) -> pd.DataFrame:
    now = _dynamic_gmm_now()
    rows = pd.DataFrame(
        [
            {
                "run_id": str(run_id),
                "window_index": int(idx),
                "window_label": str(window_label),
                "window_start": pd.Timestamp(window_start),
                "window_end": pd.Timestamp(window_end),
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "updated_at": now,
                "worker_id": None,
                "rows": 0,
                "error": None,
                "attempts": 0,
            }
            for idx, (window_start, window_end, window_label) in enumerate(
                windows,
                start=1,
            )
        ]
    )
    for col in [
        "run_id",
        "window_label",
        "status",
        "started_at",
        "completed_at",
        "updated_at",
        "worker_id",
        "error",
    ]:
        if col in rows.columns:
            rows[col] = rows[col].astype("string")
    return rows


def _dynamic_gmm_insert_initial_checkpoint(
    conn,
    run_id: str,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
) -> None:
    rows = _dynamic_gmm_checkpoint_rows(run_id, windows)
    conn.register("dynamic_checkpoint_df", rows)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME}")
        conn.execute(
            f"CREATE TABLE {DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME} AS "
            "SELECT * FROM dynamic_checkpoint_df"
        )
    finally:
        conn.unregister("dynamic_checkpoint_df")


def _dynamic_gmm_mark_running_stale(conn, run_id: str) -> None:
    if not _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME):
        return
    now = _dynamic_gmm_now()
    conn.execute(
        f"""
        UPDATE {DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME}
        SET status = 'failed_stale',
            error = COALESCE(error, 'Run interrupted while window was running.'),
            updated_at = ?
        WHERE run_id = ? AND status = 'running'
        """,
        [now, str(run_id)],
    )


def _dynamic_gmm_checkpoint_df(conn, run_id: str) -> pd.DataFrame:
    if not _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME):
        return pd.DataFrame()
    return conn.execute(
        f"""
        SELECT *
        FROM {DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME}
        WHERE run_id = ?
        ORDER BY window_index
        """,
        [str(run_id)],
    ).df()


def _dynamic_gmm_todo_windows(
    conn,
    run_id: str,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
) -> List[Tuple[int, pd.Timestamp, pd.Timestamp, str]]:
    checkpoint = _dynamic_gmm_checkpoint_df(conn, run_id)
    if checkpoint.empty:
        return [
            (idx, window_start, window_end, window_label)
            for idx, (window_start, window_end, window_label) in enumerate(
                windows,
                start=1,
            )
        ]
    completed = set(
        pd.to_numeric(
            checkpoint.loc[checkpoint["status"].astype(str) == "completed", "window_index"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )
    return [
        (idx, window_start, window_end, window_label)
        for idx, (window_start, window_end, window_label) in enumerate(
            windows,
            start=1,
        )
        if idx not in completed
    ]


def _dynamic_gmm_update_checkpoint(
    conn,
    *,
    run_id: str,
    window_index: int,
    status: str,
    worker_id: Optional[str] = None,
    rows: Optional[int] = None,
    error: Optional[str] = None,
    increment_attempts: bool = False,
) -> None:
    if not _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME):
        return
    now = _dynamic_gmm_now()
    if increment_attempts:
        conn.execute(
            f"""
            UPDATE {DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME}
            SET status = ?,
                started_at = COALESCE(started_at, ?),
                updated_at = ?,
                worker_id = ?,
                attempts = COALESCE(attempts, 0) + 1,
                error = NULL
            WHERE run_id = ? AND window_index = ?
            """,
            [
                str(status),
                now,
                now,
                None if worker_id is None else str(worker_id),
                str(run_id),
                int(window_index),
            ],
        )
        return
    completed_at = now if str(status) == "completed" else None
    conn.execute(
        f"""
        UPDATE {DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME}
        SET status = ?,
            completed_at = COALESCE(?, completed_at),
            updated_at = ?,
            worker_id = COALESCE(?, worker_id),
            rows = COALESCE(?, rows),
            error = ?
        WHERE run_id = ? AND window_index = ?
        """,
        [
            str(status),
            completed_at,
            now,
            None if worker_id is None else str(worker_id),
            None if rows is None else int(rows),
            error,
            str(run_id),
            int(window_index),
        ],
    )


def _dynamic_gmm_counts(conn, run_id: str, total_windows: int) -> Dict[str, int]:
    completed = 0
    failed = 0
    running = 0
    pending = int(total_windows)
    if _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME):
        rows = conn.execute(
            f"""
            SELECT status, COUNT(*) AS n
            FROM {DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME}
            WHERE run_id = ?
            GROUP BY status
            """,
            [str(run_id)],
        ).fetchall()
        counts = {str(status): int(n or 0) for status, n in rows}
        completed = int(counts.get("completed", 0))
        running = int(counts.get("running", 0))
        failed = int(
            counts.get("failed", 0)
            + counts.get("failed_stale", 0)
            + counts.get("skipped", 0)
        )
        pending = int(counts.get("pending", 0))
    assignment_rows = 0
    if _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME):
        try:
            assignment_rows = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME}
                    WHERE run_id = ?
                    """,
                    [str(run_id)],
                ).fetchone()[0]
                or 0
            )
        except Exception:
            assignment_rows = 0
    return {
        "completed_windows": completed,
        "failed_windows": failed,
        "running_windows": running,
        "pending_windows": pending,
        "assignment_rows": assignment_rows,
    }


def _dynamic_gmm_append_event(
    conn,
    *,
    run_id: str,
    event_type: str,
    status: str,
    total_windows: int,
    message: str,
    window_index: Optional[int] = None,
    window_label: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    counts = _dynamic_gmm_counts(conn, run_id, total_windows)
    completed = int(counts["completed_windows"])
    progress_ratio = completed / float(max(1, int(total_windows)))
    event_index = 1
    if _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_LIVE_EVENTS_TABLE_NAME):
        try:
            event_index = int(
                conn.execute(
                    f"SELECT COALESCE(MAX(event_index), 0) + 1 FROM {DYNAMIC_GMM_LIVE_EVENTS_TABLE_NAME}"
                ).fetchone()[0]
                or 1
            )
        except Exception:
            event_index = 1
    row = pd.DataFrame(
        [
            {
                "event_index": int(event_index),
                "timestamp": _dynamic_gmm_now(),
                "run_id": str(run_id),
                "event_type": str(event_type),
                "status": str(status),
                "window_index": None if window_index is None else int(window_index),
                "window_label": window_label,
                "message": str(message),
                "completed_windows": completed,
                "failed_windows": int(counts["failed_windows"]),
                "running_windows": int(counts["running_windows"]),
                "pending_windows": int(counts["pending_windows"]),
                "total_windows": int(total_windows),
                "assignment_rows": int(counts["assignment_rows"]),
                "progress_ratio": float(progress_ratio),
                "error": error,
            }
        ]
    )
    for col in [
        "timestamp",
        "run_id",
        "event_type",
        "status",
        "window_label",
        "message",
        "error",
    ]:
        if col in row.columns:
            row[col] = row[col].astype("string")
    if "window_index" in row.columns:
        row["window_index"] = pd.to_numeric(row["window_index"], errors="coerce").astype(
            "Int64"
        )
    _dynamic_gmm_append_df(conn, DYNAMIC_GMM_LIVE_EVENTS_TABLE_NAME, row)


def _dynamic_gmm_write_run_status(
    conn,
    *,
    run_id: str,
    status: str,
    result_status: str,
    total_windows: int,
    duckdb_path: Path,
    model_path: Optional[Path],
    config_fingerprint: str,
    parallel_jobs: int,
    started_at: str,
    message: str,
    error: Optional[str] = None,
) -> None:
    counts = _dynamic_gmm_counts(conn, run_id, total_windows)
    completed = int(counts["completed_windows"])
    row = {
        "run_id": str(run_id),
        "status": str(status),
        "result_status": str(result_status),
        "started_at": str(started_at),
        "updated_at": _dynamic_gmm_now(),
        "completed_windows": completed,
        "failed_windows": int(counts["failed_windows"]),
        "running_windows": int(counts["running_windows"]),
        "pending_windows": int(counts["pending_windows"]),
        "total_windows": int(total_windows),
        "assignment_rows": int(counts["assignment_rows"]),
        "progress_ratio": float(completed) / float(max(1, int(total_windows))),
        "duckdb_path": str(duckdb_path),
        "model_path": str(model_path) if model_path is not None else None,
        "config_fingerprint": str(config_fingerprint),
        "parallel_jobs": int(parallel_jobs),
        "last_message": str(message),
        "last_error": error,
    }
    _dynamic_gmm_replace_run_status(conn, row)


def _dynamic_gmm_delete_window_rows(
    conn,
    *,
    run_id: str,
    window_index: int,
) -> None:
    for table_name in [
        DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME,
        DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME,
    ]:
        if not _dynamic_gmm_table_exists(conn, table_name):
            continue
        try:
            conn.execute(
                f"DELETE FROM {table_name} WHERE run_id = ? AND window_index = ?",
                [str(run_id), int(window_index)],
            )
        except Exception:
            continue


def _dynamic_gmm_save_model_artifact(
    *,
    model: object,
    scaler: object,
    metadata: Dict[str, object],
    model_path: Path,
) -> None:
    joblib = _require_joblib()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "metadata": dict(metadata),
        },
        model_path,
    )


def _compute_dynamic_gmm_window_job(payload: Dict[str, object]) -> Dict[str, object]:
    run_id = str(payload["run_id"])
    window_index = int(payload["window_index"])
    window_start = pd.Timestamp(payload["window_start"])
    window_end = pd.Timestamp(payload["window_end"])
    window_label = str(payload["window_label"])
    effective_feature_cols = list(payload["effective_feature_cols"])
    k = int(payload["k"])
    include_membership_probabilities = bool(payload["include_membership_probabilities"])

    def _empty_result(event_status: str) -> Dict[str, object]:
        return {
            "run_id": run_id,
            "window_index": window_index,
            "window_label": window_label,
            "status": "completed",
            "event_status": event_status,
            "rows": 0,
            "assignments": _dynamic_gmm_empty_assignments_df(
                effective_feature_cols,
                k,
                include_membership_probabilities,
            ),
            "window_summary": _dynamic_gmm_empty_window_summary(
                run_id=run_id,
                window_index=window_index,
                window_start=window_start,
                window_end=window_end,
                window_label=window_label,
            ),
            "error": None,
        }

    try:
        flows_df = load_flujos_range(window_start, window_end)
        if flows_df is None or flows_df.empty:
            return _empty_result("empty_flows")
        assignment_plate_filter = _normalize_plate_values(
            payload.get("assignment_plate_filter")
        )
        if assignment_plate_filter:
            flows_df = flows_df.copy()
            flow_cols = payload["flow_cols"]
            ensure_plate_clean_column(flows_df, flow_cols)
            if PLATE_CLEAN_COL not in flows_df.columns:
                return _empty_result("empty_prevalent_flows")
            selected = set(assignment_plate_filter)
            flows_df = flows_df[flows_df[PLATE_CLEAN_COL].astype(str).isin(selected)].copy()
            if flows_df.empty:
                return _empty_result("empty_prevalent_flows")
        window_features = Clusterization(
            flows_df,
            payload["flow_cols"],
            ttc_max_map=payload.get("ttc_max_map"),
            monthly_weighting=False,
            include_counts=False,
            ttc_mode=str(payload.get("ttc_mode") or "dynamic"),
            fixed_ttc_s=payload.get("fixed_ttc_s"),
            speed_limit_map=payload.get("speed_limit_map"),
            progress=None,
            group_progress=None,
            **dict(payload.get("clean_kwargs") or {}),
        )
        if window_features is None or window_features.empty:
            return _empty_result("empty_features")
        missing_window_cols = [
            col for col in effective_feature_cols if col not in window_features.columns
        ]
        if missing_window_cols:
            raise ValueError(
                "La ventana no contiene columnas usadas por el GMM dinamico: "
                + ", ".join(missing_window_cols)
            )
        window_cluster_df = _prepare_cluster_features(
            window_features,
            effective_feature_cols,
        )
        if window_cluster_df.empty:
            return _empty_result("empty_cluster_features")
        predicted = predict_gmm_cluster_membership(
            window_cluster_df,
            effective_feature_cols,
            payload["model"],
            payload["scaler"],
            confidence_threshold_proba=float(payload["confidence_threshold_proba"]),
            min_window_passes=int(payload["min_window_passes"]),
            include_membership_probabilities=include_membership_probabilities,
            apply_confidence_threshold=True,
        )
        if predicted.empty:
            assignments = _dynamic_gmm_empty_assignments_df(
                effective_feature_cols,
                k,
                include_membership_probabilities,
            )
            summary = _dynamic_gmm_empty_window_summary(
                run_id=run_id,
                window_index=window_index,
                window_start=window_start,
                window_end=window_end,
                window_label=window_label,
            )
            event_status = "empty_predictions"
        else:
            predicted.insert(0, "run_id", run_id)
            predicted.insert(1, "window_index", int(window_index))
            predicted.insert(2, "window_label", window_label)
            predicted.insert(3, "window_start", pd.Timestamp(window_start))
            predicted.insert(4, "window_end", pd.Timestamp(window_end))
            assignments = predicted
            summary = build_dynamic_gmm_window_summary(assignments)
            if "run_id" not in summary.columns:
                summary.insert(0, "run_id", run_id)
            event_status = "completed"
        return {
            "run_id": run_id,
            "window_index": window_index,
            "window_label": window_label,
            "status": "completed",
            "event_status": event_status,
            "rows": int(len(assignments)),
            "assignments": assignments,
            "window_summary": summary,
            "error": None,
        }
    except Exception as exc:
        return {
            "run_id": run_id,
            "window_index": window_index,
            "window_label": window_label,
            "status": "failed",
            "event_status": "failed",
            "rows": 0,
            "assignments": _dynamic_gmm_empty_assignments_df(
                effective_feature_cols,
                k,
                include_membership_probabilities,
            ),
            "window_summary": _dynamic_gmm_empty_window_summary(
                run_id=run_id,
                window_index=window_index,
                window_start=window_start,
                window_end=window_end,
                window_label=window_label,
            ),
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def _dynamic_gmm_available_memory_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.5)
        except Exception:
            return None
    return None


def estimate_dynamic_gmm_parallelism(
    *,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    window_days: int,
    memory_fraction: float = 0.60,
    bytes_per_flow_row: int = 900,
    worker_overhead_bytes: int = 512 * 1024 * 1024,
    max_cpu_count: Optional[int] = None,
) -> Dict[str, object]:
    _ensure_duckdb_available()
    windows = build_dynamic_gmm_windows(date_start, date_end, int(window_days))
    if not windows:
        return {
            "n_windows": 0,
            "recommended_parallel_jobs": 1,
            "max_parallel_jobs_by_memory": 1,
            "max_parallel_jobs_by_cpu": 1,
            "available_memory_bytes": _dynamic_gmm_available_memory_bytes(),
            "max_window_rows": 0,
            "mean_window_rows": 0.0,
            "estimated_worker_bytes": int(worker_overhead_bytes),
        }
    summary = ensure_flow_db_summary()
    if summary is None:
        raise RuntimeError("No se pudo obtener resumen de la base de flujos.")
    conn = duckdb.connect(str(summary.db_path), read_only=True)
    counts: List[int] = []
    try:
        for window_start, window_end, _label in windows:
            count = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {FLOW_TABLE_NAME}
                    WHERE FECHA >= ? AND FECHA < ?
                    """,
                    [pd.Timestamp(window_start), pd.Timestamp(window_end)],
                ).fetchone()[0]
                or 0
            )
            counts.append(count)
    finally:
        conn.close()
    max_window_rows = max(counts) if counts else 0
    mean_window_rows = float(np.mean(counts)) if counts else 0.0
    estimated_worker_bytes = int(
        worker_overhead_bytes + max_window_rows * int(bytes_per_flow_row)
    )
    available_memory = _dynamic_gmm_available_memory_bytes()
    memory_budget = int((available_memory or estimated_worker_bytes) * float(memory_fraction))
    max_by_memory = max(1, int(memory_budget // max(1, estimated_worker_bytes)))
    cpu_count = max_cpu_count or os.cpu_count() or 1
    max_by_cpu = max(1, int(cpu_count) - 1)
    recommended = max(1, min(max_by_memory, max_by_cpu))
    return {
        "n_windows": int(len(windows)),
        "recommended_parallel_jobs": int(recommended),
        "max_parallel_jobs_by_memory": int(max_by_memory),
        "max_parallel_jobs_by_cpu": int(max_by_cpu),
        "available_memory_bytes": available_memory,
        "memory_budget_bytes": int(memory_budget),
        "max_window_rows": int(max_window_rows),
        "mean_window_rows": float(mean_window_rows),
        "estimated_worker_bytes": int(estimated_worker_bytes),
        "bytes_per_flow_row": int(bytes_per_flow_row),
        "worker_overhead_bytes": int(worker_overhead_bytes),
    }


def build_dynamic_gmm_windows(
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    window_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Build full daily sliding windows [start, end) over inclusive date bounds.
    """
    window_days = int(window_days)
    if window_days < 1:
        raise ValueError("window_days must be >= 1.")
    start = pd.Timestamp(date_start).normalize()
    end_inclusive = pd.Timestamp(date_end).normalize()
    if end_inclusive < start:
        return []

    last_start = end_inclusive - pd.Timedelta(days=window_days - 1)
    if last_start < start:
        return []

    windows: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    current = start
    while current <= last_start:
        window_end = current + pd.Timedelta(days=window_days)
        label = (
            f"{current:%Y-%m-%d}_to_"
            f"{(window_end - pd.Timedelta(days=1)):%Y-%m-%d}"
        )
        windows.append((current, window_end, label))
        current += pd.Timedelta(days=1)
    return windows


def build_dynamic_gmm_driver_summary(assignments_df: pd.DataFrame) -> pd.DataFrame:
    if assignments_df is None or assignments_df.empty or "plate" not in assignments_df.columns:
        return pd.DataFrame(
            columns=[
                "plate",
                "n_windows",
                "n_assigned_windows",
                "transitions",
                "changes",
                "stability_score",
                "change_rate",
                "dominant_cluster",
                "dominant_cluster_share",
                "mean_confidence",
                "mean_soft_entropy",
                "low_support_windows",
                "unknown_windows",
            ]
        )

    df = assignments_df.copy()
    if "window_start" in df.columns:
        df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
        df = df.sort_values(["plate", "window_start"], kind="mergesort")
    rows = []
    for plate, group in df.groupby("plate", sort=False):
        labels = pd.to_numeric(group["cluster_label"], errors="coerce").astype("Int64")
        assigned = labels[labels != -1].dropna().astype(int)
        transitions = max(len(assigned) - 1, 0)
        changes = int((assigned.diff().dropna() != 0).sum()) if transitions else 0
        stability = (
            1.0 - (changes / transitions)
            if transitions > 0
            else (1.0 if len(assigned) > 0 else math.nan)
        )
        if len(assigned) > 0:
            counts = assigned.value_counts()
            dominant_cluster = int(counts.index[0])
            dominant_share = float(counts.iloc[0] / len(assigned))
        else:
            dominant_cluster = -1
            dominant_share = math.nan
        low_support_windows = (
            int(group["is_low_support"].fillna(False).astype(bool).sum())
            if "is_low_support" in group.columns
            else 0
        )
        rows.append(
            {
                "plate": plate,
                "n_windows": int(len(group)),
                "n_assigned_windows": int(len(assigned)),
                "transitions": int(transitions),
                "changes": int(changes),
                "stability_score": float(stability) if not pd.isna(stability) else math.nan,
                "change_rate": (
                    float(1.0 - stability) if not pd.isna(stability) else math.nan
                ),
                "dominant_cluster": int(dominant_cluster),
                "dominant_cluster_share": float(dominant_share)
                if not pd.isna(dominant_share)
                else math.nan,
                "mean_confidence": float(
                    pd.to_numeric(group.get("confidence_score"), errors="coerce").mean()
                ),
                "mean_soft_entropy": float(
                    pd.to_numeric(group.get("soft_entropy"), errors="coerce").mean()
                )
                if "soft_entropy" in group.columns
                else math.nan,
                "low_support_windows": low_support_windows,
                "unknown_windows": int((labels == -1).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["change_rate", "n_assigned_windows", "plate"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_dynamic_gmm_window_summary(
    assignments_df: pd.DataFrame,
    windows: Optional[List[Tuple[pd.Timestamp, pd.Timestamp, str]]] = None,
) -> pd.DataFrame:
    base_rows = []
    if windows:
        for window_index, (window_start, window_end, window_label) in enumerate(
            windows,
            start=1,
        ):
            base_rows.append(
                {
                    "window_index": int(window_index),
                    "window_label": window_label,
                    "window_start": pd.Timestamp(window_start),
                    "window_end": pd.Timestamp(window_end),
                }
            )
    base = pd.DataFrame(base_rows)
    if assignments_df is None or assignments_df.empty:
        if base.empty:
            return pd.DataFrame()
        for col in [
            "rows",
            "assigned_rows",
            "unknown_rows",
            "low_support_rows",
            "low_confidence_rows",
            "mean_confidence",
            "mean_soft_entropy",
        ]:
            base[col] = 0.0 if col.startswith("mean") else 0
        return base

    df = assignments_df.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce")
    df["cluster_label"] = pd.to_numeric(df["cluster_label"], errors="coerce")
    grouped = df.groupby(["window_label", "window_start", "window_end"], sort=False)
    rows = []
    for keys, group in grouped:
        window_label, window_start, window_end = keys
        status = group.get("assignment_status", pd.Series(index=group.index, dtype=object))
        labels = pd.to_numeric(group["cluster_label"], errors="coerce")
        row = {
            "window_label": window_label,
            "window_start": window_start,
            "window_end": window_end,
            "rows": int(len(group)),
            "assigned_rows": int((labels != -1).sum()),
            "unknown_rows": int((labels == -1).sum()),
            "low_support_rows": int(
                group.get("is_low_support", pd.Series(False, index=group.index))
                .fillna(False)
                .astype(bool)
                .sum()
            ),
            "low_confidence_rows": int(status.astype(str).str.contains("low_confidence").sum()),
            "mean_confidence": float(
                pd.to_numeric(group.get("confidence_score"), errors="coerce").mean()
            ),
            "mean_soft_entropy": float(
                pd.to_numeric(group.get("soft_entropy"), errors="coerce").mean()
            )
            if "soft_entropy" in group.columns
            else math.nan,
        }
        for label, count in labels.value_counts(dropna=True).items():
            label_int = int(label)
            row[f"cluster_count_{label_int}"] = int(count)
            row[f"cluster_share_{label_int}"] = float(count / len(group)) if len(group) else 0.0
        if base.empty and "window_index" in group.columns:
            group_window_indexes = pd.to_numeric(
                group["window_index"],
                errors="coerce",
            ).dropna()
            if not group_window_indexes.empty:
                row["window_index"] = int(group_window_indexes.iloc[0])
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not base.empty:
        summary = base.merge(
            summary,
            on=["window_label", "window_start", "window_end"],
            how="left",
        )
    count_cols = [col for col in summary.columns if col.startswith("cluster_count_")]
    share_cols = [col for col in summary.columns if col.startswith("cluster_share_")]
    for col in [
        "rows",
        "assigned_rows",
        "unknown_rows",
        "low_support_rows",
        "low_confidence_rows",
        *count_cols,
    ]:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0).astype(int)
    for col in ["mean_confidence", "mean_soft_entropy", *share_cols]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    if "window_index" in summary.columns:
        summary["window_index"] = pd.to_numeric(
            summary["window_index"],
            errors="coerce",
        ).astype("Int64")
        return summary.sort_values(["window_index", "window_start"]).reset_index(drop=True)
    return summary.sort_values("window_start").reset_index(drop=True)


def save_dynamic_gmm_results(
    assignments_df: pd.DataFrame,
    window_summary_df: pd.DataFrame,
    model: object,
    scaler: object,
    metadata: Dict[str, object],
    output_dir: Optional[Path] = None,
    stem: Optional[str] = None,
) -> Tuple[Path, Path]:
    target_dir = Path(output_dir) if output_dir is not None else _ensure_clustering_results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    run_stem = stem or f"dynamic_gmm_{pd.Timestamp.now():%Y%m%d_%H%M%S_%f}"
    model_path = target_dir / f"{run_stem}.joblib"
    db_path = target_dir / f"{run_stem}.duckdb"

    joblib = _require_joblib()
    model_metadata = dict(metadata or {})
    model_metadata["model_path"] = str(model_path)
    model_metadata["duckdb_path"] = str(db_path)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "metadata": model_metadata,
        },
        model_path,
    )

    _ensure_duckdb_available()
    if db_path.exists():
        db_path.unlink()
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("dynamic_assignments_df", assignments_df)
        conn.execute(
            f"CREATE TABLE {DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME} AS "
            "SELECT * FROM dynamic_assignments_df"
        )
        conn.unregister("dynamic_assignments_df")

        conn.register("dynamic_window_summary_df", window_summary_df)
        conn.execute(
            f"CREATE TABLE {DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME} AS "
            "SELECT * FROM dynamic_window_summary_df"
        )
        conn.unregister("dynamic_window_summary_df")

        meta_rows = _feature_metadata_to_rows(model_metadata)
        conn.register("dynamic_metadata_df", meta_rows)
        conn.execute(
            f"CREATE TABLE {DYNAMIC_GMM_METADATA_TABLE_NAME} AS "
            "SELECT * FROM dynamic_metadata_df"
        )
        conn.unregister("dynamic_metadata_df")
    finally:
        conn.close()
    return model_path, db_path


def load_dynamic_gmm_results_duckdb(
    db_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    _ensure_duckdb_available()
    if not db_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    conn = _connect_dynamic_gmm_duckdb(db_path, read_only=False)
    try:
        if _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME):
            assignments = conn.execute(
                f"SELECT * FROM {DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME}"
            ).df()
        else:
            assignments = pd.DataFrame()
        if _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME):
            window_summary = conn.execute(
                f"SELECT * FROM {DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME}"
            ).df()
        else:
            window_summary = pd.DataFrame()
        metadata = _load_dynamic_gmm_metadata_from_conn(conn)
    finally:
        conn.close()
    return assignments, window_summary, metadata


def run_dynamic_gmm_clustering(
    base_features_df: pd.DataFrame,
    feature_cols: List[str],
    flow_cols: FlowColumns,
    ttc_max_map: Optional[Dict[int, float]],
    k: int,
    confidence_threshold_proba: float,
    window_days: int,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    min_window_passes: int = 5,
    train_params: Optional[Dict[str, int]] = None,
    random_state: int = 42,
    covariance_type: str = "full",
    ttc_mode: str = "dynamic",
    fixed_ttc_s: Optional[float] = None,
    speed_limit_map: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, object]] = None,
    output_dir: Optional[Path] = None,
    persist: bool = True,
    progress: Optional[object] = None,
    include_membership_probabilities: bool = True,
    window_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    parallel_jobs: int = 1,
    checkpoint_enabled: bool = True,
    incremental_db_path: Optional[Path] = None,
    resume_existing: bool = False,
    load_final_result: bool = True,
    run_id: Optional[str] = None,
    assignment_scope: str = "all",
    prevalent_fraction_pct: Optional[float] = None,
    assignment_plate_filter: Optional[List[str]] = None,
    **clean_kwargs,
) -> Dict[str, object]:
    train_params = dict(train_params or {})
    min_total_passes = int(train_params.get("min_total_passes", 20))
    min_days_active = int(train_params.get("min_days_active", 1))
    min_months_active = int(train_params.get("min_months_active", 1))

    effective_feature_cols = [
        col for col in feature_cols if col in base_features_df.columns
    ]
    if not effective_feature_cols:
        raise ValueError("No hay columnas de features disponibles para GMM dinamico.")

    normalized_assignment_scope = str(assignment_scope or "all").strip().lower()
    if normalized_assignment_scope not in {"all", "prevalent"}:
        raise ValueError("assignment_scope must be 'all' or 'prevalent'.")

    base_cluster_df = _prepare_cluster_features(base_features_df, effective_feature_cols)
    frequent_df, _rare_df = split_frequent_drivers(
        base_features_df,
        min_total_passes=min_total_passes,
        min_days_active=min_days_active,
        min_months_active=min_months_active,
    )
    cluster_freq = base_cluster_df.loc[
        base_cluster_df.index.intersection(frequent_df.index)
    ]
    if len(cluster_freq) <= int(k):
        raise ValueError(
            "No hay suficientes conductores frecuentes para entrenar GMM "
            f"con K={int(k)}."
        )

    model, scaler = fit_gmm_cluster_model(
        cluster_freq,
        effective_feature_cols,
        k=int(k),
        random_state=random_state,
        covariance_type=covariance_type,
        n_init=3,
    )
    windows = build_dynamic_gmm_windows(date_start, date_end, int(window_days))
    if not windows:
        raise ValueError("El rango seleccionado no contiene ventanas completas.")

    prevalent_selection: Dict[str, object] = {
        "plates": [],
        "ranked": pd.DataFrame(),
        "selected_count": 0,
        "total_valid_plates": int(base_features_df["plate"].nunique())
        if "plate" in base_features_df.columns
        else 0,
        "fraction_pct": None,
        "plate_hash": _dynamic_gmm_plate_filter_hash([]),
    }
    selected_assignment_plates: List[str] = []
    if normalized_assignment_scope == "prevalent":
        if assignment_plate_filter is None:
            prevalent_selection = build_prevalent_plate_selection(
                base_features_df,
                fraction_pct=(
                    10.0
                    if prevalent_fraction_pct is None
                    else float(prevalent_fraction_pct)
                ),
                feature_cols=effective_feature_cols,
            )
            selected_assignment_plates = list(prevalent_selection["plates"])
        else:
            selected_assignment_plates = _normalize_plate_values(assignment_plate_filter)
            fraction = (
                100.0
                if prevalent_fraction_pct is None
                else float(prevalent_fraction_pct)
            )
            prevalent_selection = {
                "plates": selected_assignment_plates,
                "ranked": pd.DataFrame(),
                "selected_count": int(len(selected_assignment_plates)),
                "total_valid_plates": int(len(selected_assignment_plates)),
                "fraction_pct": fraction,
                "plate_hash": _dynamic_gmm_plate_filter_hash(selected_assignment_plates),
            }
        if not selected_assignment_plates:
            raise ValueError(
                "No hay patentes prevalentes validas para asignacion dinamica."
            )

    config_payload = build_dynamic_gmm_config_payload(
        base_features_df=base_features_df,
        feature_cols=feature_cols,
        k=int(k),
        confidence_threshold_proba=float(confidence_threshold_proba),
        window_days=int(window_days),
        date_start=pd.Timestamp(date_start),
        date_end=pd.Timestamp(date_end),
        min_window_passes=int(min_window_passes),
        train_params={
            "min_total_passes": min_total_passes,
            "min_days_active": min_days_active,
            "min_months_active": min_months_active,
        },
        random_state=int(random_state),
        covariance_type=covariance_type,
        include_membership_probabilities=bool(include_membership_probabilities),
        ttc_mode=ttc_mode,
        fixed_ttc_s=fixed_ttc_s,
        assignment_scope=normalized_assignment_scope,
        prevalent_fraction_pct=prevalent_selection.get("fraction_pct"),
        prevalent_plate_count=prevalent_selection.get("selected_count"),
        prevalent_valid_plate_count=prevalent_selection.get("total_valid_plates"),
        prevalent_plate_hash=prevalent_selection.get("plate_hash"),
        prevalent_source=(
            "historical_features"
            if normalized_assignment_scope == "prevalent"
            else None
        ),
    )
    config_fingerprint = _dynamic_gmm_config_fingerprint(config_payload)
    run_stem = run_id or f"dynamic_gmm_{_dynamic_gmm_stamp()}"
    result_metadata: Dict[str, object] = {
        **config_payload,
        "created_at": pd.Timestamp.now().isoformat(),
        "run_id": str(run_stem),
        "config_fingerprint": config_fingerprint,
        "checkpoint_enabled": bool(checkpoint_enabled and persist),
        "parallel_jobs": max(1, int(parallel_jobs)),
        "assignment_scope": normalized_assignment_scope,
    }
    if normalized_assignment_scope == "prevalent":
        result_metadata.update(
            {
                "prevalent_fraction_pct": float(
                    prevalent_selection.get("fraction_pct") or 0.0
                ),
                "prevalent_plate_count": int(
                    prevalent_selection.get("selected_count") or 0
                ),
                "prevalent_valid_plate_count": int(
                    prevalent_selection.get("total_valid_plates") or 0
                ),
                "prevalent_source": "historical_features",
                "prevalent_plate_hash": str(prevalent_selection.get("plate_hash") or ""),
                "prevalent_plates": list(selected_assignment_plates),
            }
        )
    if metadata:
        result_metadata.update(metadata)

    def _build_payload(
        window_index: int,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        window_label: str,
        payload_run_id: str,
    ) -> Dict[str, object]:
        return {
            "run_id": str(payload_run_id),
            "window_index": int(window_index),
            "window_start": pd.Timestamp(window_start),
            "window_end": pd.Timestamp(window_end),
            "window_label": str(window_label),
            "effective_feature_cols": list(effective_feature_cols),
            "flow_cols": flow_cols,
            "ttc_max_map": ttc_max_map,
            "k": int(k),
            "model": model,
            "scaler": scaler,
            "confidence_threshold_proba": float(confidence_threshold_proba),
            "min_window_passes": int(min_window_passes),
            "include_membership_probabilities": bool(include_membership_probabilities),
            "ttc_mode": ttc_mode,
            "fixed_ttc_s": fixed_ttc_s,
            "speed_limit_map": speed_limit_map,
            "assignment_plate_filter": list(selected_assignment_plates),
            "clean_kwargs": dict(clean_kwargs),
        }

    if not persist or not checkpoint_enabled:
        memory_run_id = str(run_stem)
        assignment_parts: List[pd.DataFrame] = []
        task_specs = [
            (idx, window_start, window_end, window_label)
            for idx, (window_start, window_end, window_label) in enumerate(
                windows,
                start=1,
            )
        ]

        def _handle_memory_result(result: Dict[str, object]) -> None:
            status = str(result.get("event_status") or result.get("status") or "")
            assignments = result.get("assignments")
            if str(result.get("status")) == "failed":
                raise RuntimeError(str(result.get("error") or "Error en GMM dinamico."))
            if isinstance(assignments, pd.DataFrame) and not assignments.empty:
                assignment_parts.append(assignments.copy())
            if window_callback is not None:
                window_callback(
                    {
                        "window_index": int(result.get("window_index", 0) or 0),
                        "total_windows": int(len(windows)),
                        "window_label": str(result.get("window_label") or ""),
                        "status": status,
                        "assignments": assignments if isinstance(assignments, pd.DataFrame) else pd.DataFrame(),
                    }
                )
            if progress is not None and hasattr(progress, "update"):
                progress.update(1)

        effective_parallel = max(1, int(parallel_jobs))
        if effective_parallel > 1 and len(task_specs) > 1:
            with ProcessPoolExecutor(max_workers=effective_parallel) as executor:
                future_map = {}
                for idx, window_start, window_end, window_label in task_specs:
                    if progress is not None and hasattr(progress, "set_description"):
                        progress.set_description(
                            f"Encolando ventana {idx}/{len(windows)}: {window_label}"
                        )
                    payload = _build_payload(
                        idx,
                        window_start,
                        window_end,
                        window_label,
                        memory_run_id,
                    )
                    future_map[executor.submit(_compute_dynamic_gmm_window_job, payload)] = (
                        idx,
                        window_label,
                    )
                for future in as_completed(future_map):
                    idx, window_label = future_map[future]
                    if progress is not None and hasattr(progress, "set_description"):
                        progress.set_description(
                            f"Ventana {idx}/{len(windows)} lista: {window_label}"
                        )
                    _handle_memory_result(future.result())
        else:
            for idx, window_start, window_end, window_label in task_specs:
                if progress is not None and hasattr(progress, "set_description"):
                    progress.set_description(
                        f"Ventana {idx}/{len(windows)}: {window_label}"
                    )
                _handle_memory_result(
                    _compute_dynamic_gmm_window_job(
                        _build_payload(
                            idx,
                            window_start,
                            window_end,
                            window_label,
                            memory_run_id,
                        )
                    )
                )
        if assignment_parts:
            assignments_df = pd.concat(assignment_parts, ignore_index=True, sort=False)
        else:
            assignments_df = _dynamic_gmm_empty_assignments_df(
                effective_feature_cols,
                int(k),
                bool(include_membership_probabilities),
            )
        window_summary_df = build_dynamic_gmm_window_summary(assignments_df, windows)
        driver_summary_df = build_dynamic_gmm_driver_summary(assignments_df)
        model_path = None
        duckdb_path = None
        if persist:
            model_path, duckdb_path = save_dynamic_gmm_results(
                assignments_df,
                window_summary_df,
                model,
                scaler,
                result_metadata,
                output_dir=output_dir,
            )
            result_metadata["model_path"] = str(model_path)
            result_metadata["duckdb_path"] = str(duckdb_path)
        return {
            "assignments": assignments_df,
            "window_summary": window_summary_df,
            "driver_summary": driver_summary_df,
            "metadata": result_metadata,
            "model": model,
            "scaler": scaler,
            "model_path": model_path,
            "duckdb_path": duckdb_path,
            "feature_cols": effective_feature_cols,
            "windows": windows,
        }

    _ensure_duckdb_available()
    target_dir = Path(output_dir) if output_dir is not None else _ensure_clustering_results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = Path(incremental_db_path) if incremental_db_path is not None else target_dir / f"{run_stem}.duckdb"
    model_path = duckdb_path.with_suffix(".joblib")
    started_at = _dynamic_gmm_now()
    if duckdb_path.exists() and not resume_existing:
        duckdb_path.unlink()
    conn = duckdb.connect(str(duckdb_path))
    try:
        existing_metadata = _load_dynamic_gmm_metadata_from_conn(conn)
        if resume_existing and not existing_metadata:
            raise ValueError(
                "El DuckDB seleccionado no contiene metadata de GMM dinamico para retomar."
            )
        if resume_existing and existing_metadata:
            existing_fingerprint = str(existing_metadata.get("config_fingerprint") or "")
            if existing_fingerprint and existing_fingerprint != config_fingerprint:
                mismatch_details = _dynamic_gmm_config_mismatch_details(
                    existing_metadata,
                    config_payload,
                )
                raise ValueError(
                    "El checkpoint existe, pero sus parametros no coinciden con "
                    "la configuracion actual. Inicie un nuevo run o use los mismos parametros.\n"
                    f"{mismatch_details}"
                )
            run_id_value = str(existing_metadata.get("run_id") or run_stem)
            result_metadata["run_id"] = run_id_value
            started_at = str(existing_metadata.get("created_at") or started_at)
            _dynamic_gmm_mark_running_stale(conn, run_id_value)
            if not _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME):
                raise ValueError(
                    "El archivo seleccionado no contiene checkpoint incremental para retomar."
                )
        else:
            run_id_value = str(run_stem)
            result_metadata["run_id"] = run_id_value
            for table_name in [
                DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME,
                DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME,
                DYNAMIC_GMM_METADATA_TABLE_NAME,
                DYNAMIC_GMM_LIVE_EVENTS_TABLE_NAME,
                DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME,
                DYNAMIC_GMM_RUN_STATUS_TABLE_NAME,
            ]:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            _dynamic_gmm_insert_initial_checkpoint(conn, run_id_value, windows)

        result_metadata["duckdb_path"] = str(duckdb_path)
        result_metadata["model_path"] = str(model_path)
        _dynamic_gmm_save_model_artifact(
            model=model,
            scaler=scaler,
            metadata=result_metadata,
            model_path=model_path,
        )
        _dynamic_gmm_replace_metadata(conn, result_metadata)
        effective_parallel = max(1, int(parallel_jobs))
        todo_windows = _dynamic_gmm_todo_windows(conn, str(result_metadata["run_id"]), windows)
        completed_before = len(windows) - len(todo_windows)
        if progress is not None and completed_before > 0 and hasattr(progress, "update"):
            progress.update(completed_before)
        _dynamic_gmm_write_run_status(
            conn,
            run_id=str(result_metadata["run_id"]),
            status="running",
            result_status="partial",
            total_windows=len(windows),
            duckdb_path=duckdb_path,
            model_path=model_path,
            config_fingerprint=config_fingerprint,
            parallel_jobs=effective_parallel,
            started_at=started_at,
            message=f"Procesando {len(todo_windows)} ventanas pendientes.",
        )
        _dynamic_gmm_append_event(
            conn,
            run_id=str(result_metadata["run_id"]),
            event_type="run_start" if not resume_existing else "run_resume",
            status="running",
            total_windows=len(windows),
            message=f"Ventanas pendientes: {len(todo_windows):,}.",
        )

        def _handle_incremental_result(result: Dict[str, object]) -> None:
            window_index = int(result.get("window_index") or 0)
            window_label = str(result.get("window_label") or "")
            event_status = str(result.get("event_status") or result.get("status") or "")
            assignments = result.get("assignments")
            window_summary = result.get("window_summary")
            if str(result.get("status")) == "failed":
                error = str(result.get("error") or "Error en ventana GMM dinamico.")
                _dynamic_gmm_update_checkpoint(
                    conn,
                    run_id=str(result_metadata["run_id"]),
                    window_index=window_index,
                    status="failed",
                    rows=0,
                    error=error,
                )
                _dynamic_gmm_append_event(
                    conn,
                    run_id=str(result_metadata["run_id"]),
                    event_type="window_failed",
                    status="failed",
                    total_windows=len(windows),
                    message=f"Ventana {window_index}/{len(windows)} fallo.",
                    window_index=window_index,
                    window_label=window_label,
                    error=error,
                )
                if window_callback is not None:
                    window_callback(
                        {
                            "window_index": window_index,
                            "total_windows": int(len(windows)),
                            "window_label": window_label,
                            "status": "failed",
                            "assignments": pd.DataFrame(),
                            "error": error,
                        }
                    )
                if progress is not None and hasattr(progress, "update"):
                    progress.update(1)
                _dynamic_gmm_write_run_status(
                    conn,
                    run_id=str(result_metadata["run_id"]),
                    status="running",
                    result_status="partial",
                    total_windows=len(windows),
                    duckdb_path=duckdb_path,
                    model_path=model_path,
                    config_fingerprint=config_fingerprint,
                    parallel_jobs=effective_parallel,
                    started_at=started_at,
                    message=f"Ventana fallida: {window_label}.",
                    error=error,
                )
                return
            _dynamic_gmm_delete_window_rows(
                conn,
                run_id=str(result_metadata["run_id"]),
                window_index=window_index,
            )
            if isinstance(assignments, pd.DataFrame) and not assignments.empty:
                _dynamic_gmm_append_df(conn, DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME, assignments)
            if isinstance(window_summary, pd.DataFrame) and not window_summary.empty:
                _dynamic_gmm_append_df(conn, DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME, window_summary)
            rows = int(result.get("rows") or 0)
            _dynamic_gmm_update_checkpoint(
                conn,
                run_id=str(result_metadata["run_id"]),
                window_index=window_index,
                status="completed",
                rows=rows,
                error=None,
            )
            _dynamic_gmm_append_event(
                conn,
                run_id=str(result_metadata["run_id"]),
                event_type="window_done",
                status=event_status,
                total_windows=len(windows),
                message=f"Ventana {window_index}/{len(windows)} completada ({rows:,} asignaciones).",
                window_index=window_index,
                window_label=window_label,
            )
            if window_callback is not None:
                window_callback(
                    {
                        "window_index": window_index,
                        "total_windows": int(len(windows)),
                        "window_label": window_label,
                        "status": event_status,
                        "assignments": assignments if isinstance(assignments, pd.DataFrame) else pd.DataFrame(),
                    }
                )
            if progress is not None and hasattr(progress, "update"):
                progress.update(1)
            if progress is not None and hasattr(progress, "set_description"):
                progress.set_description(
                    f"Ventana {window_index}/{len(windows)}: {window_label}"
                )
            _dynamic_gmm_write_run_status(
                conn,
                run_id=str(result_metadata["run_id"]),
                status="running",
                result_status="partial",
                total_windows=len(windows),
                duckdb_path=duckdb_path,
                model_path=model_path,
                config_fingerprint=config_fingerprint,
                parallel_jobs=effective_parallel,
                started_at=started_at,
                message=f"Ultima ventana completada: {window_label}.",
            )

        if effective_parallel > 1 and len(todo_windows) > 1:
            with ProcessPoolExecutor(max_workers=effective_parallel) as executor:
                future_map = {}
                for idx, window_start, window_end, window_label in todo_windows:
                    _dynamic_gmm_update_checkpoint(
                        conn,
                        run_id=str(result_metadata["run_id"]),
                        window_index=idx,
                        status="running",
                        worker_id=f"process_pool_{idx}",
                        increment_attempts=True,
                    )
                    _dynamic_gmm_append_event(
                        conn,
                        run_id=str(result_metadata["run_id"]),
                        event_type="window_start",
                        status="running",
                        total_windows=len(windows),
                        message=f"Ventana {idx}/{len(windows)} iniciada.",
                        window_index=idx,
                        window_label=window_label,
                    )
                    payload = _build_payload(
                        idx,
                        window_start,
                        window_end,
                        window_label,
                        str(result_metadata["run_id"]),
                    )
                    future_map[executor.submit(_compute_dynamic_gmm_window_job, payload)] = (
                        idx,
                        window_label,
                    )
                for future in as_completed(future_map):
                    idx, window_label = future_map[future]
                    try:
                        _handle_incremental_result(future.result())
                    except Exception as exc:
                        error = f"{exc}\n{traceback.format_exc()}"
                        _dynamic_gmm_update_checkpoint(
                            conn,
                            run_id=str(result_metadata["run_id"]),
                            window_index=idx,
                            status="failed",
                            rows=0,
                            error=error,
                        )
                        _dynamic_gmm_append_event(
                            conn,
                            run_id=str(result_metadata["run_id"]),
                            event_type="window_failed",
                            status="failed",
                            total_windows=len(windows),
                            message=f"Ventana {idx}/{len(windows)} fallo.",
                            window_index=idx,
                            window_label=window_label,
                            error=error,
                        )
                        if progress is not None and hasattr(progress, "update"):
                            progress.update(1)
                        if progress is not None and hasattr(progress, "set_description"):
                            progress.set_description(
                                f"Ventana {idx}/{len(windows)} fallo: {window_label}"
                            )
                        _dynamic_gmm_write_run_status(
                            conn,
                            run_id=str(result_metadata["run_id"]),
                            status="running",
                            result_status="partial",
                            total_windows=len(windows),
                            duckdb_path=duckdb_path,
                            model_path=model_path,
                            config_fingerprint=config_fingerprint,
                            parallel_jobs=effective_parallel,
                            started_at=started_at,
                            message=f"Ventana fallida: {window_label}.",
                            error=error,
                        )
        else:
            for idx, window_start, window_end, window_label in todo_windows:
                _dynamic_gmm_update_checkpoint(
                    conn,
                    run_id=str(result_metadata["run_id"]),
                    window_index=idx,
                    status="running",
                    worker_id="main",
                    increment_attempts=True,
                )
                _dynamic_gmm_append_event(
                    conn,
                    run_id=str(result_metadata["run_id"]),
                    event_type="window_start",
                    status="running",
                    total_windows=len(windows),
                    message=f"Ventana {idx}/{len(windows)} iniciada.",
                    window_index=idx,
                    window_label=window_label,
                )
                _handle_incremental_result(
                    _compute_dynamic_gmm_window_job(
                        _build_payload(
                            idx,
                            window_start,
                            window_end,
                            window_label,
                            str(result_metadata["run_id"]),
                        )
                    )
                )

        counts = _dynamic_gmm_counts(conn, str(result_metadata["run_id"]), len(windows))
        failed_windows = int(counts["failed_windows"])
        pending_windows = int(counts["pending_windows"]) + int(counts["running_windows"])
        final_status = "completed" if failed_windows == 0 and pending_windows == 0 else "failed_partial"
        result_status = "completed" if final_status == "completed" else "partial"
        result_metadata["n_assignments"] = int(counts["assignment_rows"])
        result_metadata["status"] = final_status
        _dynamic_gmm_replace_metadata(conn, result_metadata)
        _dynamic_gmm_write_run_status(
            conn,
            run_id=str(result_metadata["run_id"]),
            status=final_status,
            result_status=result_status,
            total_windows=len(windows),
            duckdb_path=duckdb_path,
            model_path=model_path,
            config_fingerprint=config_fingerprint,
            parallel_jobs=effective_parallel,
            started_at=started_at,
            message="GMM dinamico completado." if final_status == "completed" else "GMM dinamico finalizo con ventanas fallidas.",
            error=None if final_status == "completed" else "Hay ventanas fallidas; revise dynamic_window_checkpoint.",
        )
        _dynamic_gmm_append_event(
            conn,
            run_id=str(result_metadata["run_id"]),
            event_type="run_completed",
            status=final_status,
            total_windows=len(windows),
            message="Run completado." if final_status == "completed" else "Run con resultados parciales.",
        )
    except Exception:
        try:
            _dynamic_gmm_write_run_status(
                conn,
                run_id=str(result_metadata.get("run_id") or run_stem),
                status="failed",
                result_status="failed",
                total_windows=len(windows),
                duckdb_path=duckdb_path,
                model_path=model_path,
                config_fingerprint=config_fingerprint,
                parallel_jobs=max(1, int(parallel_jobs)),
                started_at=started_at,
                message="GMM dinamico fallo.",
                error=traceback.format_exc(),
            )
        except Exception:
            pass
        raise
    finally:
        conn.close()

    if load_final_result:
        assignments_df, window_summary_df, loaded_metadata = load_dynamic_gmm_results_duckdb(duckdb_path)
        result_metadata.update(loaded_metadata)
    else:
        assignments_df = _dynamic_gmm_empty_assignments_df(
            effective_feature_cols,
            int(k),
            bool(include_membership_probabilities),
        )
        conn = _connect_dynamic_gmm_duckdb(duckdb_path, read_only=False)
        try:
            if _dynamic_gmm_table_exists(conn, DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME):
                window_summary_df = conn.execute(
                    f"SELECT * FROM {DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME} ORDER BY window_index"
                ).df()
            else:
                window_summary_df = pd.DataFrame()
        finally:
            conn.close()
    driver_summary_df = (
        build_dynamic_gmm_driver_summary(assignments_df)
        if not assignments_df.empty
        else pd.DataFrame()
    )

    return {
        "assignments": assignments_df,
        "window_summary": window_summary_df,
        "driver_summary": driver_summary_df,
        "metadata": result_metadata,
        "model": model,
        "scaler": scaler,
        "model_path": model_path,
        "duckdb_path": duckdb_path,
        "feature_cols": effective_feature_cols,
        "windows": windows,
    }


def compute_kmeans_metrics(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    k_min: int,
    k_max: int,
    random_state: int = 42,
    use_minibatch: bool = True,
    batch_size: int = 4096,
    max_iter: int = 100,
    n_init: int = 3,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, object, object]:
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for clustering metrics. "
            "Install it with: pip install scikit-learn"
        ) from exc

    X = features_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_rows: List[Dict[str, float]] = []
    k_values = range(k_min, k_max + 1)
    if show_progress:
        progress = tqdm(k_values, desc="Evaluando K", unit="k")
        for k in progress:
            progress.set_description(f"Evaluando K={k}")
            if use_minibatch:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=n_init,
                    batch_size=batch_size,
                    max_iter=max_iter,
                    verbose=1,
                )
            else:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=max(n_init, 1),
                    max_iter=max_iter,
                    verbose=0,
                )
            labels = kmeans.fit_predict(X_scaled)
            metrics_rows.append(
                {
                    "k": int(k),
                    "silhouette": float(silhouette_score(X_scaled, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
                    "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                }
            )
    else:
        for k in k_values:
            if use_minibatch:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=n_init,
                    batch_size=batch_size,
                    max_iter=max_iter,
                    verbose=0,
                )
            else:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=max(n_init, 1),
                    max_iter=max_iter,
                    verbose=0,
                )
            labels = kmeans.fit_predict(X_scaled)
            metrics_rows.append(
                {
                    "k": int(k),
                    "silhouette": float(silhouette_score(X_scaled, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
                    "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, scaler, X_scaled


def compute_gmm_metrics(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    k_min: int,
    k_max: int,
    random_state: int = 42,
    covariance_type: str = "full",
    max_iter: int = 200,
    n_init: int = 3,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, object, object]:
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for GMM metrics. "
            "Install it with: pip install scikit-learn"
        ) from exc

    X = features_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_rows: List[Dict[str, float]] = []
    k_values = range(k_min, k_max + 1)
    if show_progress:
        progress = tqdm(k_values, desc="Evaluando K (GMM)", unit="k")
        for k in progress:
            progress.set_description(f"Evaluando K={k} (GMM)")
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=random_state,
                max_iter=max_iter,
                n_init=n_init,
            )
            gmm.fit(X_scaled)
            metrics_rows.append(
                {
                    "k": int(k),
                    "bic": float(gmm.bic(X_scaled)),
                    "aic": float(gmm.aic(X_scaled)),
                }
            )
    else:
        for k in k_values:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=random_state,
                max_iter=max_iter,
                n_init=n_init,
            )
            gmm.fit(X_scaled)
            metrics_rows.append(
                {
                    "k": int(k),
                    "bic": float(gmm.bic(X_scaled)),
                    "aic": float(gmm.aic(X_scaled)),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, scaler, X_scaled


def save_cluster_features(features_df: pd.DataFrame) -> Path:
    output_dir = _ensure_clustering_results_dir()
    path = output_dir / "cluster_features.csv"
    features_df.to_csv(path, index=False)
    return path


def save_cluster_metrics(metrics_df: pd.DataFrame) -> Path:
    output_dir = _ensure_clustering_results_dir()
    path = output_dir / "cluster_metrics.csv"
    metrics_df.to_csv(path, index=False)
    return path


def _cluster_label_filename(method: str, k: Optional[int]) -> str:
    method = method.lower().strip()
    if method in {"kmeans", "gmm"}:
        if k is None:
            raise ValueError("k is required for kmeans/gmm outputs.")
        return f"cluster_{method}_k{k}.csv"
    if method == "hdbscan":
        return "cluster_hdbscan.csv"
    raise ValueError(f"Unsupported cluster method: {method}")


def _cluster_summary_filename(method: str, k: Optional[int]) -> str:
    method = method.lower().strip()
    if method == "kmeans":
        if k is None:
            raise ValueError("k is required for kmeans/gmm outputs.")
        return f"cluster_summary_k{k}.csv"
    if method == "gmm":
        if k is None:
            raise ValueError("k is required for kmeans/gmm outputs.")
        return f"cluster_summary_gmm_k{k}.csv"
    if method == "hdbscan":
        return "cluster_summary_hdbscan.csv"
    raise ValueError(f"Unsupported cluster method: {method}")


def _cluster_descriptive_filename(method: str, k: Optional[int]) -> str:
    method = method.lower().strip()
    if method == "kmeans":
        if k is None:
            raise ValueError("k is required for kmeans/gmm outputs.")
        return f"cluster_descriptive_k{k}.csv"
    if method == "gmm":
        if k is None:
            raise ValueError("k is required for kmeans/gmm outputs.")
        return f"cluster_descriptive_gmm_k{k}.csv"
    if method == "hdbscan":
        return "cluster_descriptive_hdbscan.csv"
    raise ValueError(f"Unsupported cluster method: {method}")


def save_cluster_labels(
    cluster_df: pd.DataFrame, method: str, k: Optional[int] = None
) -> Path:
    output_dir = _ensure_clustering_results_dir()
    path = output_dir / _cluster_label_filename(method, k)
    cluster_df.to_csv(path, index=False)
    return path


def _flatten_columns(columns: pd.Index) -> List[str]:
    if not isinstance(columns, pd.MultiIndex):
        return [str(col) for col in columns]
    return ["_".join(str(part) for part in col if part) for col in columns]


def build_cluster_summary(clustered_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    grouped = clustered_df.groupby("cluster_label", sort=True)
    summary = grouped[feature_cols].mean()
    summary.insert(0, "cluster_size", grouped.size())
    summary = summary.reset_index()
    return summary


def build_cluster_descriptive(clustered_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    stats = clustered_df.groupby("cluster_label", sort=True)[feature_cols].agg(
        ["count", "mean", "std", "min", "max"]
    )
    stats.columns = _flatten_columns(stats.columns)
    return stats.reset_index()


def save_cluster_summary(
    summary_df: pd.DataFrame, method: str, k: Optional[int] = None
) -> Path:
    output_dir = _ensure_clustering_results_dir()
    path = output_dir / _cluster_summary_filename(method, k)
    summary_df.to_csv(path, index=False)
    return path


def save_cluster_descriptive(
    stats_df: pd.DataFrame, method: str, k: Optional[int] = None
) -> Path:
    output_dir = _ensure_clustering_results_dir()
    path = output_dir / _cluster_descriptive_filename(method, k)
    stats_df.to_csv(path, index=False)
    return path


def list_cluster_summary_files() -> List[Path]:
    candidates = _glob_clustering_results("cluster_summary*.csv")
    return [path for path in candidates if _parse_cluster_summary_file(path) is not None]


def list_cluster_label_files() -> List[Path]:
    candidates = _glob_clustering_results("cluster_*.csv")
    return [path for path in candidates if _parse_cluster_label_file(path) is not None]


def has_cluster_features_db() -> bool:
    return bool(list_cluster_feature_db_paths())

def _parse_cluster_summary_file(path: Path) -> Optional[Tuple[str, Optional[int]]]:
    match = CLUSTER_SUMMARY_PATTERN.match(path.name)
    if not match:
        return None
    method = (match.group("method") or "kmeans").lower()
    k_raw = match.group("k")
    if method in {"kmeans", "gmm"}:
        if not k_raw:
            return None
        try:
            return method, int(k_raw)
        except ValueError:
            return None
    if method == "hdbscan":
        return method, None
    return None


def _parse_cluster_label_file(path: Path) -> Optional[Tuple[str, Optional[int]]]:
    match = CLUSTER_LABEL_PATTERN.match(path.name)
    if not match:
        return None
    method = match.group("method").lower()
    k_raw = match.group("k")
    if method in {"kmeans", "gmm"}:
        if not k_raw:
            return None
        try:
            return method, int(k_raw)
        except ValueError:
            return None
    if method == "hdbscan":
        return method, None
    return None


def handle_cluster_statistics() -> None:
    summary_files = list_cluster_summary_files()
    if not summary_files:
        print("⚠️ No se encontraron archivos cluster_summary*.csv.")
        return

    selected = summary_files[0]
    if len(summary_files) > 1:
        print("\nArchivos de resumen disponibles:")
        for idx, path in enumerate(summary_files, start=1):
            print(f"  [{idx}] {path.name}")
        choice = _prompt_int_value(
            "Seleccione un archivo (q para cancelar): ",
            default=None,
            min_value=1,
            max_value=len(summary_files),
        )
        if choice is None:
            return
        selected = summary_files[choice - 1]

    summary_info = _parse_cluster_summary_file(selected)
    if summary_info is None:
        print("⚠️ No se pudo determinar el metodo/K desde el nombre del archivo.")
        return
    method, k_value = summary_info

    labels_path = CLUSTERING_RESULTS_DIR / _cluster_label_filename(method, k_value)
    if not labels_path.exists():
        legacy_path = RESULTS_DIR / _cluster_label_filename(method, k_value)
        if legacy_path.exists():
            labels_path = legacy_path
    if labels_path.exists():
        clustered = pd.read_csv(labels_path)
        feature_cols = _choose_feature_columns(clustered)
        required_cols = set(feature_cols) | {"cluster_label"}
        missing_cols = required_cols - set(clustered.columns)
        if missing_cols:
            print(
                "⚠️ El archivo de clusters no contiene las columnas requeridas: "
                f"{', '.join(sorted(missing_cols))}."
            )
            return

        summary_df = build_cluster_summary(clustered, feature_cols)
        descriptive_df = build_cluster_descriptive(clustered, feature_cols)
        summary_path = save_cluster_summary(summary_df, method, k_value)
        descriptive_path = save_cluster_descriptive(descriptive_df, method, k_value)
        print(f"📁 Resumen por cluster guardado en: {summary_path}")
        print(f"📁 Estadistica descriptiva guardada en: {descriptive_path}")

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("\nResumen por cluster (medias y tamanos):")
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return

    print(
        "⚠️ No se encontro el archivo de etiquetas para recalcular estadisticas: "
        f"{labels_path}"
    )
    try:
        summary_df = pd.read_csv(selected)
    except Exception as exc:
        print(f"❌ No se pudo leer el archivo: {exc}")
        return
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("\nResumen por cluster (archivo existente):")
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _choose_feature_columns(clustered: pd.DataFrame) -> List[str]:
    preferred = DEFAULT_CLUSTER_FEATURES
    cols = [col for col in preferred if col in clustered.columns]
    if len(cols) >= 3:
        return cols
    numeric_cols = [
        col
        for col in clustered.select_dtypes(include=["number"]).columns
        if col != "cluster_label"
    ]
    return numeric_cols


def _sample_cluster_data(
    clustered: pd.DataFrame,
    sample_size: Optional[int],
    random_state: int = 42,
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(clustered):
        return clustered
    return clustered.sample(sample_size, random_state=random_state)


def build_cluster_visualization_html(
    clustered: pd.DataFrame,
    feature_cols: List[str],
    title: str,
) -> str:
    data_cols = ["cluster_label"] + feature_cols
    data_payload = {col: clustered[col].tolist() for col in data_cols}
    payload = json.dumps(data_payload)
    features = json.dumps(feature_cols)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2933;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }}
    .panel {{
      margin-bottom: 32px;
    }}
    label {{
      font-weight: 600;
      margin-right: 8px;
    }}
    select {{
      padding: 4px 8px;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>

  <div class="panel">
    <h2>3D Scatter</h2>
    <div class="controls">
      <div>
        <label for="xSelect">X</label>
        <select id="xSelect"></select>
      </div>
      <div>
        <label for="ySelect">Y</label>
        <select id="ySelect"></select>
      </div>
      <div>
        <label for="zSelect">Z</label>
        <select id="zSelect"></select>
      </div>
    </div>
    <div id="scatter3d" style="width:100%;height:600px;"></div>
  </div>

  <div class="panel">
    <h2>Distribution</h2>
    <div class="controls">
      <div>
        <label for="histSelect">Variable</label>
        <select id="histSelect"></select>
      </div>
    </div>
    <div id="histogram" style="width:100%;height:420px;"></div>
  </div>

  <script>
    const data = {payload};
    const featureCols = {features};

    function fillSelect(selectId, defaultValue) {{
      const select = document.getElementById(selectId);
      featureCols.forEach((col) => {{
        const opt = document.createElement('option');
        opt.value = col;
        opt.textContent = col;
        select.appendChild(opt);
      }});
      if (defaultValue && featureCols.includes(defaultValue)) {{
        select.value = defaultValue;
      }} else {{
        select.selectedIndex = 0;
      }}
    }}

    function plot3d(xCol, yCol, zCol) {{
      const trace = {{
        x: data[xCol],
        y: data[yCol],
        z: data[zCol],
        mode: 'markers',
        type: 'scatter3d',
        marker: {{
          size: 2,
          opacity: 0.7,
          color: data.cluster_label,
          colorscale: 'Turbo',
          colorbar: {{ title: 'cluster' }}
        }}
      }};
      const layout = {{
        margin: {{ l: 0, r: 0, b: 0, t: 30 }},
        scene: {{
          xaxis: {{ title: xCol }},
          yaxis: {{ title: yCol }},
          zaxis: {{ title: zCol }},
        }}
      }};
      Plotly.react('scatter3d', [trace], layout, {{responsive: true}});
    }}

    function plotHistogram(col) {{
      const trace = {{
        x: data[col],
        type: 'histogram',
        nbinsx: 50,
        marker: {{ color: '#2b6cb0' }}
      }};
      const layout = {{
        margin: {{ l: 40, r: 20, b: 40, t: 30 }},
        xaxis: {{ title: col }},
        yaxis: {{ title: 'count' }}
      }};
      Plotly.react('histogram', [trace], layout, {{responsive: true}});
    }}

    fillSelect('xSelect', featureCols[0]);
    fillSelect('ySelect', featureCols[1] || featureCols[0]);
    fillSelect('zSelect', featureCols[2] || featureCols[0]);
    fillSelect('histSelect', featureCols[0]);

    plot3d(
      document.getElementById('xSelect').value,
      document.getElementById('ySelect').value,
      document.getElementById('zSelect').value
    );
    plotHistogram(document.getElementById('histSelect').value);

    document.getElementById('xSelect').addEventListener('change', () => {{
      plot3d(
        document.getElementById('xSelect').value,
        document.getElementById('ySelect').value,
        document.getElementById('zSelect').value
      );
    }});
    document.getElementById('ySelect').addEventListener('change', () => {{
      plot3d(
        document.getElementById('xSelect').value,
        document.getElementById('ySelect').value,
        document.getElementById('zSelect').value
      );
    }});
    document.getElementById('zSelect').addEventListener('change', () => {{
      plot3d(
        document.getElementById('xSelect').value,
        document.getElementById('ySelect').value,
        document.getElementById('zSelect').value
      );
    }});
    document.getElementById('histSelect').addEventListener('change', () => {{
      plotHistogram(document.getElementById('histSelect').value);
    }});
  </script>
</body>
</html>
"""


def save_cluster_visualization_html(html: str, k: int) -> Path:
    output_dir = _ensure_clustering_results_dir()
    path = output_dir / f"cluster_visualization_k{k}.html"
    path.write_text(html, encoding="utf-8")
    return path


def handle_cluster_visualization() -> None:
    label_files = list_cluster_label_files()
    if not label_files:
        print("⚠️ No se encontraron archivos cluster_*.csv de clustering.")
        return

    app_path = ROOT_DIR / "src" / "cluster_visualization_app.py"
    if not app_path.exists():
        print("⚠️ No se encontro el archivo de visualizacion Streamlit.")
        return

    try:
        import streamlit  # type: ignore  # noqa: F401
    except ImportError:
        print("❌ streamlit no esta instalado. Ejecute `pip install streamlit`.")
        return

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
    ]
    print("\n🚀 Lanzando Streamlit para visualizacion de clusters...")
    print("   Abra el enlace local que mostrara Streamlit en la terminal.")
    subprocess.run(cmd, check=False)


def handle_cluster_features_visualization() -> None:
    if not has_cluster_features_db():
        print("⚠️ No se encontraron archivos cluster_features*.duckdb en Resultados.")
        return

    app_path = ROOT_DIR / "src" / "cluster_features_app.py"
    if not app_path.exists():
        print("⚠️ No se encontro el archivo de visualizacion de features.")
        return

    try:
        import streamlit  # type: ignore  # noqa: F401
    except ImportError:
        print("❌ streamlit no esta instalado. Ejecute `pip install streamlit`.")
        return

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
    ]
    print("\n🚀 Lanzando Streamlit para visualizacion de features...")
    print("   Abra el enlace local que mostrara Streamlit en la terminal.")
    subprocess.run(cmd, check=False)


def _build_batch_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    mode: str,
    split_months: bool = False,
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
        if split_months:
            month_boundary = (start + pd.offsets.MonthBegin(1)).normalize()
            if month_boundary < end:
                label = (
                    f"{start:%Y-%m-%d}_to_"
                    f"{(month_boundary - pd.Timedelta(days=1)):%Y-%m-%d}"
                )
                ranges.append((start, month_boundary, label))
                label = (
                    f"{month_boundary:%Y-%m-%d}_to_"
                    f"{(end - pd.Timedelta(days=1)):%Y-%m-%d}"
                )
                ranges.append((month_boundary, end, label))
                continue
        label = f"{start:%Y-%m-%d}_to_{(end - pd.Timedelta(days=1)):%Y-%m-%d}"
        ranges.append((start, end, label))
    return ranges


def _aggregate_batch_features(
    batches_df: pd.DataFrame,
    batch_mode: Optional[str],
    monthly_weighting: bool,
    lane_changes_extra: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    if batches_df is None or batches_df.empty:
        return pd.DataFrame()

    grouped = batches_df.groupby("plate", sort=False)
    total_passes = grouped["total_passes"].sum()
    n_days_active = (
        grouped["n_days_active"].sum() if "n_days_active" in batches_df.columns else None
    )
    n_weeks_active = (
        grouped["n_weeks_active"].sum()
        if batch_mode == "week" and "n_weeks_active" in batches_df.columns
        else None
    )
    n_months_active = (
        grouped["n_months_active"].sum()
        if batch_mode == "month" and "n_months_active" in batches_df.columns
        else None
    )
    n_years_active = None
    if "batch_start" in batches_df.columns:
        batch_year = pd.to_datetime(batches_df["batch_start"], errors="coerce").dt.year
        year_df = pd.DataFrame(
            {
                "plate": batches_df["plate"],
                "batch_year": batch_year,
            }
        ).dropna(subset=["batch_year"])
        if not year_df.empty:
            n_years_active = year_df.groupby("plate", sort=False)["batch_year"].nunique()
    elif "batch_month" in batches_df.columns:
        month_start = pd.to_datetime(
            batches_df["batch_month"].astype(str) + "-01",
            errors="coerce",
        )
        year_df = pd.DataFrame(
            {
                "plate": batches_df["plate"],
                "batch_year": month_start.dt.year,
            }
        ).dropna(subset=["batch_year"])
        if not year_df.empty:
            n_years_active = year_df.groupby("plate", sort=False)["batch_year"].nunique()
    elif "n_years_active" in batches_df.columns:
        n_years_active = grouped["n_years_active"].sum()

    if monthly_weighting:
        transitions_sum = (
            (batches_df["total_passes"] - 1)
            .clip(lower=0)
            .groupby(batches_df["plate"], sort=False)
            .sum()
        )
        weighted_cols = [
            "avg_speed_kmh",
            "avg_relative_speed",
            "avg_headway_s",
            "conflict_rate",
            "lane_prop_1",
            "lane_prop_2",
            "lane_prop_3",
        ]
        weighted = batches_df.copy()
        weighted[weighted_cols] = weighted[weighted_cols].multiply(
            weighted["total_passes"], axis=0
        )
        weighted_grouped = weighted.groupby("plate", sort=False)
        weighted_sum = weighted_grouped[weighted_cols].sum()
        summary = weighted_sum.div(total_passes, axis=0)
        if "exceso_velocidad" in batches_df.columns:
            if "speed_limit_count" in batches_df.columns:
                speed_limit_count = grouped["speed_limit_count"].sum()
                speed_limit_sum = (
                    batches_df["exceso_velocidad"] * batches_df["speed_limit_count"]
                ).groupby(batches_df["plate"], sort=False).sum()
                speed_limit_den = speed_limit_count.replace(0, np.nan)
                summary["exceso_velocidad"] = (
                    speed_limit_sum / speed_limit_den
                ).fillna(0.0)
                summary["speed_limit_count"] = speed_limit_count
            else:
                summary["exceso_velocidad"] = (
                    batches_df["exceso_velocidad"] * batches_df["total_passes"]
                ).groupby(batches_df["plate"], sort=False).sum() / total_passes
        summary["total_passes"] = total_passes
        if n_days_active is not None:
            summary["n_days_active"] = n_days_active
        if n_weeks_active is not None:
            summary["n_weeks_active"] = n_weeks_active
        if n_months_active is not None:
            summary["n_months_active"] = n_months_active
        if n_years_active is not None:
            summary["n_years_active"] = n_years_active
        if "lane_changes" in batches_df.columns:
            lane_changes_sum = grouped["lane_changes"].sum()
            if lane_changes_extra:
                lane_changes_sum = lane_changes_sum.add(
                    pd.Series(lane_changes_extra), fill_value=0
                )
            summary["lane_changes"] = lane_changes_sum
            summary["lane_change_rate"] = 0.0
            valid = transitions_sum > 0
            summary.loc[valid, "lane_change_rate"] = (
                summary.loc[valid, "lane_changes"] / transitions_sum.loc[valid]
            )
        return summary.reset_index()

    speed_sum = (batches_df["avg_speed_kmh"] * batches_df["total_passes"]).groupby(
        batches_df["plate"], sort=False
    ).sum()
    lane1_sum = (batches_df["lane_prop_1"] * batches_df["total_passes"]).groupby(
        batches_df["plate"], sort=False
    ).sum()
    lane2_sum = (batches_df["lane_prop_2"] * batches_df["total_passes"]).groupby(
        batches_df["plate"], sort=False
    ).sum()
    lane3_sum = (batches_df["lane_prop_3"] * batches_df["total_passes"]).groupby(
        batches_df["plate"], sort=False
    ).sum()

    if {"rel_speed_count", "headway_count", "conflict_count"}.issubset(batches_df.columns):
        rel_count = grouped["rel_speed_count"].sum()
        rel_sum = (
            batches_df["avg_relative_speed"] * batches_df["rel_speed_count"]
        ).groupby(batches_df["plate"], sort=False).sum()
        headway_count = grouped["headway_count"].sum()
        headway_sum = (
            batches_df["avg_headway_s"] * batches_df["headway_count"]
        ).groupby(batches_df["plate"], sort=False).sum()
        conflict_count = grouped["conflict_count"].sum()
        conflict_sum = (
            batches_df["conflict_rate"] * batches_df["conflict_count"]
        ).groupby(batches_df["plate"], sort=False).sum()
        rel_den = rel_count.replace(0, pd.NA)
        headway_den = headway_count.replace(0, pd.NA)
        conflict_den = conflict_count.replace(0, pd.NA)
        avg_relative_speed = rel_sum / rel_den
        avg_headway = headway_sum / headway_den
        conflict_rate = conflict_sum / conflict_den
    else:
        avg_relative_speed = (
            batches_df["avg_relative_speed"] * batches_df["total_passes"]
        ).groupby(batches_df["plate"], sort=False).sum() / total_passes
        avg_headway = (
            batches_df["avg_headway_s"] * batches_df["total_passes"]
        ).groupby(batches_df["plate"], sort=False).sum() / total_passes
        conflict_rate = (
            batches_df["conflict_rate"] * batches_df["total_passes"]
        ).groupby(batches_df["plate"], sort=False).sum() / total_passes

    if "exceso_velocidad" in batches_df.columns:
        if "speed_limit_count" in batches_df.columns:
            speed_limit_count = grouped["speed_limit_count"].sum()
            speed_limit_sum = (
                batches_df["exceso_velocidad"] * batches_df["speed_limit_count"]
            ).groupby(batches_df["plate"], sort=False).sum()
            speed_limit_den = speed_limit_count.replace(0, np.nan)
            exceso_velocidad = (speed_limit_sum / speed_limit_den).fillna(0.0)
        else:
            speed_limit_count = None
            exceso_velocidad = (
                batches_df["exceso_velocidad"] * batches_df["total_passes"]
            ).groupby(batches_df["plate"], sort=False).sum() / total_passes
    else:
        speed_limit_count = None
        exceso_velocidad = pd.Series(0.0, index=total_passes.index)

    lane_changes_sum = grouped["lane_changes"].sum()
    if lane_changes_extra:
        lane_changes_sum = lane_changes_sum.add(
            pd.Series(lane_changes_extra), fill_value=0
        )

    summary = pd.DataFrame(index=total_passes.index)
    summary["total_passes"] = total_passes
    summary["avg_speed_kmh"] = speed_sum / total_passes
    summary["exceso_velocidad"] = exceso_velocidad
    summary["avg_relative_speed"] = avg_relative_speed
    summary["avg_headway_s"] = avg_headway
    summary["conflict_rate"] = conflict_rate
    summary["lane_prop_1"] = lane1_sum / total_passes
    summary["lane_prop_2"] = lane2_sum / total_passes
    summary["lane_prop_3"] = lane3_sum / total_passes
    summary["lane_changes"] = lane_changes_sum
    summary["lane_change_rate"] = 0.0
    valid_rate = total_passes > 1
    summary.loc[valid_rate, "lane_change_rate"] = (
        summary.loc[valid_rate, "lane_changes"] / (total_passes[valid_rate] - 1)
    )
    if n_days_active is not None:
        summary["n_days_active"] = n_days_active
    if n_weeks_active is not None:
        summary["n_weeks_active"] = n_weeks_active
    if n_months_active is not None:
        summary["n_months_active"] = n_months_active
    if n_years_active is not None:
        summary["n_years_active"] = n_years_active
    if speed_limit_count is not None:
        summary["speed_limit_count"] = speed_limit_count
    return summary.reset_index()


def _aggregate_weekly_batches_by_month(
    batches_df: pd.DataFrame,
    lane_changes_extra_by_month: Optional[Dict[str, Dict[str, int]]] = None,
) -> pd.DataFrame:
    if batches_df is None or batches_df.empty:
        return pd.DataFrame()
    if "batch_start" not in batches_df.columns:
        return pd.DataFrame()

    df = batches_df.copy()
    batch_start = pd.to_datetime(df["batch_start"], errors="coerce")
    df["batch_month"] = batch_start.dt.to_period("M").astype(str)
    df = df[df["batch_month"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    required_cols = {
        "plate",
        "total_passes",
        "avg_speed_kmh",
        "avg_relative_speed",
        "avg_headway_s",
        "conflict_rate",
        "lane_prop_1",
        "lane_prop_2",
        "lane_prop_3",
        "lane_changes",
    }
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    numeric_cols = [
        "total_passes",
        "avg_speed_kmh",
        "avg_relative_speed",
        "avg_headway_s",
        "conflict_rate",
        "lane_prop_1",
        "lane_prop_2",
        "lane_prop_3",
        "exceso_velocidad",
        "lane_changes",
        "rel_speed_count",
        "headway_count",
        "conflict_count",
        "speed_limit_count",
        "n_days_active",
        "n_weeks_active",
        "n_years_active",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    count_cols = [
        col
        for col in ["rel_speed_count", "headway_count", "conflict_count"]
        if col in df.columns
    ]
    if "speed_limit_count" in df.columns:
        count_cols.append("speed_limit_count")
    if count_cols:
        df[count_cols] = df[count_cols].fillna(0)

    group_cols = ["plate", "batch_month"]
    group_keys = [df[col] for col in group_cols]
    grouped = df.groupby(group_cols, sort=False)
    total_passes = grouped["total_passes"].sum()

    speed_sum = (df["avg_speed_kmh"] * df["total_passes"]).groupby(
        group_keys, sort=False
    ).sum()
    lane1_sum = (df["lane_prop_1"] * df["total_passes"]).groupby(
        group_keys, sort=False
    ).sum()
    lane2_sum = (df["lane_prop_2"] * df["total_passes"]).groupby(
        group_keys, sort=False
    ).sum()
    lane3_sum = (df["lane_prop_3"] * df["total_passes"]).groupby(
        group_keys, sort=False
    ).sum()

    if {"rel_speed_count", "headway_count", "conflict_count"}.issubset(df.columns):
        rel_count = grouped["rel_speed_count"].sum()
        rel_sum = (df["avg_relative_speed"] * df["rel_speed_count"]).groupby(
            group_keys, sort=False
        ).sum()
        headway_count = grouped["headway_count"].sum()
        headway_sum = (df["avg_headway_s"] * df["headway_count"]).groupby(
            group_keys, sort=False
        ).sum()
        conflict_count = grouped["conflict_count"].sum()
        conflict_sum = (df["conflict_rate"] * df["conflict_count"]).groupby(
            group_keys, sort=False
        ).sum()
        rel_den = rel_count.replace(0, pd.NA)
        headway_den = headway_count.replace(0, pd.NA)
        conflict_den = conflict_count.replace(0, pd.NA)
        avg_relative_speed = rel_sum / rel_den
        avg_headway = headway_sum / headway_den
        conflict_rate = conflict_sum / conflict_den
    else:
        avg_relative_speed = (
            df["avg_relative_speed"] * df["total_passes"]
        ).groupby(group_keys, sort=False).sum() / total_passes
        avg_headway = (
            df["avg_headway_s"] * df["total_passes"]
        ).groupby(group_keys, sort=False).sum() / total_passes
        conflict_rate = (
            df["conflict_rate"] * df["total_passes"]
        ).groupby(group_keys, sort=False).sum() / total_passes

    if "exceso_velocidad" in df.columns:
        if "speed_limit_count" in df.columns:
            speed_limit_count = grouped["speed_limit_count"].sum()
            speed_limit_sum = (df["exceso_velocidad"] * df["speed_limit_count"]).groupby(
                group_keys, sort=False
            ).sum()
            speed_limit_den = speed_limit_count.replace(0, np.nan)
            exceso_velocidad = (speed_limit_sum / speed_limit_den).fillna(0.0)
        else:
            speed_limit_count = None
            exceso_velocidad = (
                df["exceso_velocidad"] * df["total_passes"]
            ).groupby(group_keys, sort=False).sum() / total_passes
    else:
        speed_limit_count = None
        exceso_velocidad = pd.Series(0.0, index=total_passes.index)

    lane_changes_sum = grouped["lane_changes"].sum()
    if lane_changes_extra_by_month:
        extras = []
        for month, extra in lane_changes_extra_by_month.items():
            if not extra:
                continue
            extras.append(
                pd.DataFrame(
                    {
                        "plate": list(extra.keys()),
                        "batch_month": month,
                        "lane_changes_extra": list(extra.values()),
                    }
                )
            )
        if extras:
            extra_df = pd.concat(extras, ignore_index=True)
            extra_series = extra_df.set_index(group_cols)["lane_changes_extra"]
            lane_changes_sum = lane_changes_sum.add(extra_series, fill_value=0)

    summary = pd.DataFrame(index=total_passes.index)
    summary["total_passes"] = total_passes
    summary["avg_speed_kmh"] = speed_sum / total_passes
    summary["exceso_velocidad"] = exceso_velocidad
    summary["avg_relative_speed"] = avg_relative_speed
    summary["avg_headway_s"] = avg_headway
    summary["conflict_rate"] = conflict_rate
    summary["lane_prop_1"] = lane1_sum / total_passes
    summary["lane_prop_2"] = lane2_sum / total_passes
    summary["lane_prop_3"] = lane3_sum / total_passes
    summary["lane_changes"] = lane_changes_sum
    summary["lane_change_rate"] = 0.0
    valid_rate = total_passes > 1
    summary.loc[valid_rate, "lane_change_rate"] = (
        summary.loc[valid_rate, "lane_changes"] / (total_passes[valid_rate] - 1)
    )
    if "n_days_active" in df.columns:
        summary["n_days_active"] = grouped["n_days_active"].sum()
    if "n_weeks_active" in df.columns:
        summary["n_weeks_active"] = grouped["n_weeks_active"].sum()
    if speed_limit_count is not None:
        summary["speed_limit_count"] = speed_limit_count
    summary["n_months_active"] = 1
    summary["n_years_active"] = 1
    return summary.reset_index()


def _clusterize_in_batches(
    flow_cols: FlowColumns,
    ttc_max_map: Optional[Dict[int, float]],
    batch_mode: str,
    monthly_weighting: bool,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    batch_db_path: Optional[Path] = None,
    progress: Optional[object] = None,
    ttc_mode: str = "dynamic",
    fixed_ttc_s: Optional[float] = None,
    speed_limit_map: Optional[Dict[str, float]] = None,
    **clean_kwargs,
) -> Tuple[pd.DataFrame, List[Path]]:
    summary = ensure_flow_db_summary()
    if summary is None:
        return pd.DataFrame(), []

    if summary.min_timestamp is None or summary.max_timestamp is None:
        print("⚠️ No se pudo determinar el rango temporal.")
        return pd.DataFrame(), []

    filter_start = date_start
    filter_end_exclusive = None
    if date_end is not None:
        filter_end_exclusive = date_end + pd.Timedelta(nanoseconds=1)

    range_start = summary.min_timestamp
    range_end = summary.max_timestamp
    if filter_start is not None:
        range_start = max(range_start, filter_start)
    if filter_end_exclusive is not None:
        range_end = min(range_end, filter_end_exclusive)
    if range_end <= range_start:
        print("⚠️ El rango seleccionado no contiene datos.")
        return pd.DataFrame(), []

    rollup_monthly = batch_mode == "week" and monthly_weighting
    ranges = _build_batch_ranges(
        range_start, range_end, batch_mode, split_months=rollup_monthly
    )
    if not ranges:
        print("⚠️ No se encontraron rangos para procesar.")
        return pd.DataFrame(), []

    batch_dir = _ensure_clustering_results_dir() / "cluster_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_conn = None
    batch_table_created = False
    if batch_db_path is not None:
        _ensure_duckdb_available()
        batch_db_path.parent.mkdir(parents=True, exist_ok=True)
        if batch_db_path.exists():
            batch_db_path.unlink()
        batch_conn = duckdb.connect(str(batch_db_path))

    overlap_col = "__overlap"
    batch_paths: List[Path] = []
    carryover_headway = pd.DataFrame()
    last_lane_by_plate: Dict[str, int] = {}
    lane_changes_extra: Dict[str, int] = {}
    lane_changes_extra_by_month: Dict[str, Dict[str, int]] = {}
    active_month = None
    allow_carryover = (not monthly_weighting) or rollup_monthly

    total_ranges = len(ranges)
    if progress is not None and hasattr(progress, "set_description"):
        progress.set_description(f"Procesando {total_ranges} lotes")

    try:
        for idx, (start_ts, end_ts, label) in enumerate(ranges, start=1):
            if progress is not None:
                if hasattr(progress, "set_description"):
                    progress.set_description(f"Lote {idx}/{total_ranges}: {label}")
                if hasattr(progress, "update"):
                    progress.update(1)
            query_start = start_ts
            query_end = end_ts
            if filter_start is not None and query_start < filter_start:
                query_start = filter_start
            if filter_end_exclusive is not None and query_end > filter_end_exclusive:
                query_end = filter_end_exclusive
            if query_end <= query_start:
                continue
            batch_month = None
            if rollup_monthly:
                batch_month = query_start.to_period("M").strftime("%Y-%m")
                if batch_month != active_month:
                    carryover_headway = pd.DataFrame()
                    last_lane_by_plate = {}
                    active_month = batch_month
            df_batch = load_flujos_range(query_start, query_end)
            if df_batch.empty:
                continue

            if allow_carryover and not carryover_headway.empty:
                overlap_df = carryover_headway.copy()
                overlap_df[overlap_col] = True
                df_batch = df_batch.copy()
                df_batch[overlap_col] = False
                df_batch = pd.concat([overlap_df, df_batch], ignore_index=True, sort=False)
            else:
                df_batch = df_batch.copy()
                df_batch[overlap_col] = False

            df_clean = clean_flujos_for_clustering(
                df_batch, flow_cols, extra_cols=[overlap_col], **clean_kwargs
            )
            if df_clean.empty:
                continue
            valid_mask = ~df_clean[overlap_col].fillna(False)
            df_valid = df_clean.loc[valid_mask]
            if df_valid.empty:
                continue

            batch_summary = Clusterization(
                df_batch,
                flow_cols,
                ttc_max_map=ttc_max_map,
                monthly_weighting=False,
                overlap_col=overlap_col,
                include_counts=True,
                ttc_mode=ttc_mode,
                fixed_ttc_s=fixed_ttc_s,
                speed_limit_map=speed_limit_map,
                progress=None,
                **clean_kwargs,
            )
            if batch_summary.empty:
                continue

            batch_summary["batch_label"] = label
            batch_summary["batch_start"] = query_start.strftime("%Y-%m-%d")
            batch_summary["batch_end"] = (
                query_end - pd.Timedelta(seconds=1)
            ).strftime("%Y-%m-%d")
            batch_path = batch_dir / f"cluster_features_{label}.csv"
            batch_summary.to_csv(batch_path, index=False)
            batch_paths.append(batch_path)

            if batch_conn is not None:
                batch_conn.register("batch_summary_df", batch_summary)
                if not batch_table_created:
                    batch_conn.execute(
                        f"CREATE TABLE {CLUSTER_BATCH_TABLE_NAME} AS "
                        "SELECT * FROM batch_summary_df"
                    )
                    batch_table_created = True
                else:
                    batch_conn.execute(
                        f"INSERT INTO {CLUSTER_BATCH_TABLE_NAME} "
                        "SELECT * FROM batch_summary_df"
                    )
                batch_conn.unregister("batch_summary_df")
            del batch_summary

            if allow_carryover:
                ordered = df_valid.sort_values(
                    [flow_cols.timestamp, PLATE_CLEAN_COL],
                    kind="mergesort",
                )
                first_lanes = ordered.groupby(PLATE_CLEAN_COL)[LANE_CLEAN_COL].first()
                last_lanes = ordered.groupby(PLATE_CLEAN_COL)[LANE_CLEAN_COL].last()
                if last_lane_by_plate:
                    cross_changes = {
                        plate: int(last_lane_by_plate[plate] != lane)
                        for plate, lane in first_lanes.items()
                        if plate in last_lane_by_plate
                    }
                    if rollup_monthly and batch_month is not None:
                        month_map = lane_changes_extra_by_month.setdefault(batch_month, {})
                        for plate, change in cross_changes.items():
                            month_map[plate] = month_map.get(plate, 0) + change
                    else:
                        for plate, change in cross_changes.items():
                            lane_changes_extra[plate] = lane_changes_extra.get(plate, 0) + change
                for plate, lane in last_lanes.items():
                    last_lane_by_plate[plate] = int(lane)

                last_rows = (
                    ordered.groupby([flow_cols.portico, LANE_CLEAN_COL], sort=False)
                    .tail(1)
                )
                carryover_headway = last_rows[
                    [
                        flow_cols.timestamp,
                        flow_cols.speed_kmh,
                        flow_cols.portico,
                        flow_cols.lane,
                        PLATE_CLEAN_COL,
                    ]
                ].copy()
    finally:
        if batch_conn is not None:
            batch_conn.close()

    if not batch_paths:
        return pd.DataFrame(), []

    if batch_db_path is not None:
        batch_conn = duckdb.connect(str(batch_db_path), read_only=True)
        try:
            all_batches = batch_conn.execute(
                f"SELECT * FROM {CLUSTER_BATCH_TABLE_NAME}"
            ).df()
        finally:
            batch_conn.close()
    else:
        all_batches = pd.concat(
            (pd.read_csv(path) for path in batch_paths),
            ignore_index=True,
        )
    consolidated_path = _ensure_clustering_results_dir() / "cluster_features_batches.csv"
    all_batches.to_csv(consolidated_path, index=False)
    batch_paths.append(consolidated_path)

    aggregate_df = all_batches
    aggregate_mode = batch_mode
    lane_changes_for_aggregate = (
        lane_changes_extra if not monthly_weighting else None
    )
    if rollup_monthly:
        aggregate_df = _aggregate_weekly_batches_by_month(
            all_batches, lane_changes_extra_by_month
        )
        aggregate_mode = "month"
        lane_changes_for_aggregate = None

    consolidated = _aggregate_batch_features(
        aggregate_df,
        batch_mode=aggregate_mode,
        monthly_weighting=monthly_weighting,
        lane_changes_extra=lane_changes_for_aggregate,
    )
    consolidated = consolidated.sort_values(
        by=["total_passes", "plate"], ascending=[False, True]
    ).reset_index(drop=True)
    if batch_db_path is not None:
        save_cluster_features_duckdb(
            consolidated,
            db_path=batch_db_path,
            metadata=build_ttc_feature_metadata(
                ttc_mode=ttc_mode,
                fixed_ttc_s=fixed_ttc_s,
                ttc_max_map=ttc_max_map,
            ),
        )
    return consolidated, batch_paths


def handle_clusterization(session) -> None:
    fc = FlowColumns()
    features_df: Optional[pd.DataFrame] = None
    existing_dbs = list_cluster_feature_db_paths()
    if existing_dbs:
        reuse = input(
            f"\nSe encontraron {len(existing_dbs)} archivos de variables en Resultados."
            " ¿Usar uno para la clusterizacion sin recalcular? (s/n): "
        ).strip().lower()
        if reuse in {"s", "si", "y", "yes"}:
            selected_db = _prompt_select_feature_db(existing_dbs)
            if selected_db is None:
                return
            try:
                features_df = load_cluster_features_duckdb(selected_db)
            except ImportError as exc:
                print(f"❌ {exc}")
                return
            if features_df.empty:
                print("⚠️ El archivo existe pero no contiene variables validas.")
                features_df = None
            else:
                required_cols = {
                    "plate",
                    "total_passes",
                    "avg_speed_kmh",
                    "exceso_velocidad",
                    "avg_relative_speed",
                    "avg_headway_s",
                    "conflict_rate",
                    "lane_prop_1",
                    "lane_prop_2",
                    "lane_change_rate",
                }
                missing_cols = required_cols - set(features_df.columns)
                if missing_cols:
                    print(
                        "⚠️ El archivo no contiene las columnas requeridas: "
                        f"{', '.join(sorted(missing_cols))}. Se recalcularan las variables."
                    )
                    features_df = None
                else:
                    print(
                        f"📦 Variables cargadas desde DuckDB: {selected_db}"
                        f" ({len(features_df)} matriculas)."
                    )

    if features_df is None:
        if getattr(session, "flujos_df", None) is None:
            print("⚠️ No flow data is loaded. Loading now...")
            summary = ensure_flow_db_summary()
            if summary is None:
                return
            sample = prompt_flow_sample_selection(summary)
            if sample.row_limit is not None:
                print(
                    "ℹ️ El muestreo por porcentaje no es compatible con lotes. "
                    "Se cargará la muestra directamente."
                )
                flujos_df = load_flujos(sample=sample)
                if flujos_df is None:
                    print("❌ Flow data was not loaded.")
                    return
                session.flujos_df = flujos_df
            else:
                use_batches = input(
                    "¿Procesar por lotes (mes/semana) para reducir memoria? (s/n): "
                ).strip().lower() in {"s", "si", "y", "yes"}
                if use_batches:
                    mode_choice = input(
                        "Seleccione el modo de lotes: [m]es / [s]emana: "
                    ).strip().lower()
                    batch_mode = "week" if mode_choice.startswith("s") else "month"
                    monthly_weighting = input(
                        "¿Ponderar variables por mes antes de consolidar? (s/n): "
                    ).strip().lower() in {"s", "si", "y", "yes"}
                    print("\n⏳ Calculando variables por lotes...")
                    features_df, batch_paths = _clusterize_in_batches(
                        fc,
                        TTC_MAX_BY_PORTICO,
                        batch_mode,
                        monthly_weighting,
                        date_start=sample.date_start,
                        date_end=sample.date_end,
                    )
                    if features_df.empty:
                        print("⚠️ No se encontraron registros validos para calcular las variables.")
                        return
                    print(
                        f"📁 Lotes generados: {len(batch_paths)} archivos en Resultados."
                    )
                else:
                    flujos_df = load_flujos(sample=sample)
                    if flujos_df is None:
                        print("❌ Flow data was not loaded.")
                        return
                    session.flujos_df = flujos_df

        if features_df is None:
            monthly_weighting = input(
                "\n¿Ponderar variables por mes antes de consolidar por matricula? (s/n): "
            ).strip().lower() in {"s", "si", "y", "yes"}

            print("\n⏳ Calculando variables de clusterizacion...")
            progress = tqdm(total=5, desc="Preparando datos", unit="paso")
            try:
                features_df = Clusterization(
                    session.flujos_df,
                    fc,
                    monthly_weighting=monthly_weighting,
                    progress=progress,
                )
            finally:
                progress.close()
        if features_df.empty:
            print("⚠️ No se encontraron registros validos para calcular las variables.")
            return
        try:
            suffix = _prompt_cluster_feature_db_suffix()
            db_path = _build_cluster_feature_db_path(suffix)
            db_path = save_cluster_features_duckdb(
                features_df,
                db_path=db_path,
                metadata=build_ttc_feature_metadata(
                    ttc_mode="dynamic",
                    ttc_max_map=TTC_MAX_BY_PORTICO,
                ),
            )
            print(f"📦 Variables guardadas en DuckDB: {db_path}")
        except ImportError as exc:
            print(f"❌ {exc}")

    feature_cols = _prompt_feature_selection(features_df)
    if not feature_cols:
        return

    cluster_df = _prepare_cluster_features(features_df, feature_cols)
    dropped = len(features_df) - len(cluster_df)
    if dropped:
        print(f"⚠️ Se descartaron {dropped} matriculas por valores faltantes o invalidos.")
    if cluster_df.empty:
        print("⚠️ No quedan matriculas despues de filtrar valores invalidos.")
        return

    method = _prompt_cluster_method()
    if method is None:
        return

    if method == "kmeans":
        max_k_allowed = len(cluster_df) - 1
        if max_k_allowed < 2:
            print("⚠️ No hay suficientes muestras para evaluar K-means (minimo 3 matriculas).")
            return

        default_k_min = 2
        default_k_max = min(5, max_k_allowed)
        metrics_df: Optional[pd.DataFrame] = None
        X_scaled = None
        best_sil: Optional[int] = None

        calc_metrics = input(
            "\n¿Desea calcular las metricas (Silhouette/DB/CH) para un rango de K? (s/n): "
        ).strip().lower()
        if calc_metrics in {"s", "si", "y", "yes"}:
            use_minibatch = input(
                "¿Usar MiniBatchKMeans para evaluar K? (s/n): "
            ).strip().lower()
            use_minibatch = use_minibatch in {"s", "si", "y", "yes"}

            try:
                metrics_df, _, X_scaled = compute_kmeans_metrics(
                    cluster_df,
                    feature_cols,
                    k_min=default_k_min,
                    k_max=default_k_max,
                    use_minibatch=use_minibatch,
                )
            except ImportError as exc:
                print(f"❌ {exc}")
                return

            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print("\nMetricas de K-means:")
                print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

            best_sil = metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"]
            best_db = metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "k"]
            best_ch = metrics_df.loc[metrics_df["calinski_harabasz"].idxmax(), "k"]
            print(
                "\nMejores candidatos K:"
                f"\n  Silhouette max: {best_sil}"
                f"\n  Davies-Bouldin min: {best_db}"
                f"\n  Calinski-Harabasz max: {best_ch}"
            )

            custom_range = input(
                "\n¿Desea evaluar otro rango de K? (s/n): "
            ).strip().lower()
            if custom_range in {"s", "si", "y", "yes"}:
                print(f"\nK debe estar entre 2 y {max_k_allowed}.")
                k_min = _prompt_int_value(
                    f"Ingrese K minimo [Enter={default_k_min}, q para cancelar]: ",
                    default=default_k_min,
                    min_value=2,
                    max_value=max_k_allowed,
                )
                if k_min is None:
                    return
                k_max = _prompt_int_value(
                    f"Ingrese K maximo [Enter={default_k_max}, q para cancelar]: ",
                    default=default_k_max,
                    min_value=k_min,
                    max_value=max_k_allowed,
                )
                if k_max is None:
                    return

                try:
                    metrics_df, _, X_scaled = compute_kmeans_metrics(
                        cluster_df,
                        feature_cols,
                        k_min=k_min,
                        k_max=k_max,
                        use_minibatch=use_minibatch,
                    )
                except ImportError as exc:
                    print(f"❌ {exc}")
                    return

                with pd.option_context("display.max_rows", None, "display.max_columns", None):
                    print("\nMetricas de K-means:")
                    print(
                        metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}")
                    )

                best_sil = metrics_df.loc[metrics_df["silhouette"].idxmax(), "k"]
                best_db = metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "k"]
                best_ch = metrics_df.loc[metrics_df["calinski_harabasz"].idxmax(), "k"]
                print(
                    "\nMejores candidatos K:"
                    f"\n  Silhouette max: {best_sil}"
                    f"\n  Davies-Bouldin min: {best_db}"
                    f"\n  Calinski-Harabasz max: {best_ch}"
                )

        _maybe_export_cluster_inputs(features_df, metrics_df)

        apply_k = input("\n¿Aplicar K-means con un K especifico? (s/n): ").strip().lower()
        if apply_k not in {"s", "si", "y", "yes"}:
            return

        if best_sil is not None:
            k_choice = _prompt_int_value(
                f"Ingrese K para aplicar [Enter={best_sil} sugerido por silhouette]: ",
                default=best_sil,
                min_value=2,
                max_value=max_k_allowed,
            )
        else:
            k_choice = _prompt_int_value(
                "Ingrese K para aplicar (q para cancelar): ",
                default=None,
                min_value=2,
                max_value=max_k_allowed,
            )
        if k_choice is None:
            return

        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            print(f"❌ {exc}")
            return

        if X_scaled is None:
            try:
                X_scaled, _ = _scale_cluster_features(cluster_df, feature_cols)
            except ImportError as exc:
                print(f"❌ {exc}")
                return

        kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        clustered = cluster_df.copy()
        clustered["cluster_label"] = labels
        output_path = save_cluster_labels(clustered, "kmeans", k_choice)
        print(f"📁 Etiquetas de cluster guardadas en: {output_path}")

        summary_df = build_cluster_summary(clustered, feature_cols)
        descriptive_df = build_cluster_descriptive(clustered, feature_cols)
        summary_path = save_cluster_summary(summary_df, "kmeans", k_choice)
        descriptive_path = save_cluster_descriptive(descriptive_df, "kmeans", k_choice)
        print(f"📁 Resumen por cluster guardado en: {summary_path}")
        print(f"📁 Estadistica descriptiva guardada en: {descriptive_path}")

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("\nResumen por cluster (medias y tamanos):")
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return

    if method == "gmm":
        max_k_allowed = len(cluster_df) - 1
        if max_k_allowed < 2:
            print("⚠️ No hay suficientes muestras para evaluar GMM (minimo 3 matriculas).")
            return

        default_k_min = 2
        default_k_max = min(5, max_k_allowed)
        metrics_df = None
        X_scaled = None
        best_k: Optional[int] = None
        gmm_params = {
            "covariance_type": "full",
            "random_state": 42,
            "max_iter": 200,
            "n_init": 3,
        }

        calc_metrics = input(
            "\n¿Desea calcular BIC/AIC para un rango de K? (s/n): "
        ).strip().lower()
        if calc_metrics in {"s", "si", "y", "yes"}:
            print(f"\nK debe estar entre 2 y {max_k_allowed}.")
            k_min = _prompt_int_value(
                f"Ingrese K minimo [Enter={default_k_min}, q para cancelar]: ",
                default=default_k_min,
                min_value=2,
                max_value=max_k_allowed,
            )
            if k_min is None:
                return
            k_max = _prompt_int_value(
                f"Ingrese K maximo [Enter={default_k_max}, q para cancelar]: ",
                default=default_k_max,
                min_value=k_min,
                max_value=max_k_allowed,
            )
            if k_max is None:
                return

            try:
                metrics_df, _, X_scaled = compute_gmm_metrics(
                    cluster_df,
                    feature_cols,
                    k_min=k_min,
                    k_max=k_max,
                    covariance_type=gmm_params["covariance_type"],
                    random_state=gmm_params["random_state"],
                    max_iter=gmm_params["max_iter"],
                    n_init=gmm_params["n_init"],
                )
            except ImportError as exc:
                print(f"❌ {exc}")
                return

            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print("\nMetricas de GMM (BIC/AIC):")
                print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

            best_bic = metrics_df.loc[metrics_df["bic"].idxmin(), "k"]
            best_aic = metrics_df.loc[metrics_df["aic"].idxmin(), "k"]
            print(
                "\nMejores candidatos K:"
                f"\n  BIC min: {best_bic}"
                f"\n  AIC min: {best_aic}"
            )

            criterio = input("¿Usar [b]ic o [a]ic para sugerir K? [b]: ").strip().lower()
            best_k = best_aic if criterio.startswith("a") else best_bic

        _maybe_export_cluster_inputs(features_df, metrics_df)

        apply_k = input("\n¿Aplicar GMM con un K especifico? (s/n): ").strip().lower()
        if apply_k not in {"s", "si", "y", "yes"}:
            return

        if best_k is not None:
            k_choice = _prompt_int_value(
                f"Ingrese K para aplicar [Enter={best_k} sugerido por BIC/AIC]: ",
                default=best_k,
                min_value=2,
                max_value=max_k_allowed,
            )
        else:
            k_choice = _prompt_int_value(
                "Ingrese K para aplicar (q para cancelar): ",
                default=None,
                min_value=2,
                max_value=max_k_allowed,
            )
        if k_choice is None:
            return

        try:
            from sklearn.mixture import GaussianMixture
        except ImportError as exc:
            print(f"❌ {exc}")
            return

        if X_scaled is None:
            try:
                X_scaled, _ = _scale_cluster_features(cluster_df, feature_cols)
            except ImportError as exc:
                print(f"❌ {exc}")
                return

        gmm = GaussianMixture(n_components=k_choice, **gmm_params)
        labels = gmm.fit_predict(X_scaled)
        clustered = cluster_df.copy()
        clustered["cluster_label"] = labels
        output_path = save_cluster_labels(clustered, "gmm", k_choice)
        print(f"📁 Etiquetas de cluster guardadas en: {output_path}")

        summary_df = build_cluster_summary(clustered, feature_cols)
        descriptive_df = build_cluster_descriptive(clustered, feature_cols)
        summary_path = save_cluster_summary(summary_df, "gmm", k_choice)
        descriptive_path = save_cluster_descriptive(descriptive_df, "gmm", k_choice)
        print(f"📁 Resumen por cluster guardado en: {summary_path}")
        print(f"📁 Estadistica descriptiva guardada en: {descriptive_path}")

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("\nResumen por cluster (medias y tamanos):")
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ImportError:
            print("❌ hdbscan no esta instalado. Ejecute `pip install hdbscan`.")
            return

        min_cluster_size = _prompt_int_value(
            "Ingrese min_cluster_size [Enter=15, q para cancelar]: ",
            default=15,
            min_value=2,
        )
        if min_cluster_size is None:
            return

        min_samples = None
        define_min_samples = input(
            "¿Definir min_samples? (s/n): "
        ).strip().lower()
        if define_min_samples in {"s", "si", "y", "yes"}:
            min_samples = _prompt_int_value(
                f"Ingrese min_samples [Enter={min_cluster_size}, q para cancelar]: ",
                default=min_cluster_size,
                min_value=1,
            )
            if min_samples is None:
                return

        if len(cluster_df) < min_cluster_size:
            print("⚠️ No hay suficientes muestras para el min_cluster_size seleccionado.")
            return

        _maybe_export_cluster_inputs(features_df, None)

        try:
            X_scaled, _ = _scale_cluster_features(cluster_df, feature_cols)
        except ImportError as exc:
            print(f"❌ {exc}")
            return

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        labels = clusterer.fit_predict(X_scaled)
        clustered = cluster_df.copy()
        clustered["cluster_label"] = labels
        output_path = save_cluster_labels(clustered, "hdbscan")
        print(f"📁 Etiquetas de cluster guardadas en: {output_path}")

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        print(f"ℹ️ Clusters detectados: {n_clusters} | Ruido: {noise_count}")

        summary_df = build_cluster_summary(clustered, feature_cols)
        descriptive_df = build_cluster_descriptive(clustered, feature_cols)
        summary_path = save_cluster_summary(summary_df, "hdbscan")
        descriptive_path = save_cluster_descriptive(descriptive_df, "hdbscan")
        print(f"📁 Resumen por cluster guardado en: {summary_path}")
        print(f"📁 Estadistica descriptiva guardada en: {descriptive_path}")

        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("\nResumen por cluster (medias y tamanos):")
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return

    print(f"⚠️ Metodo de clustering no soportado: {method}")
