#!/usr/bin/env python3
"""
clustering.py
=============
Funciones para calcular variables de clusterizacion y ejecutar clustering.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None  # type: ignore

from utils import (
    FlowColumns,
    ensure_flow_db_summary,
    load_flujos,
    load_flujos_range,
    normalize_plate_series,
    prompt_flow_sample_selection,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PLATE_CLEAN_COL = "plate_clean"
LANE_CLEAN_COL = "lane_numeric"

CLUSTER_DB_PATH = ROOT_DIR / "Resultados" / "cluster_features.duckdb"
CLUSTER_TABLE_NAME = "cluster_features"
CLUSTER_BATCH_TABLE_NAME = "cluster_features_batches"
DEFAULT_CLUSTER_FEATURES = [
    "avg_speed_kmh",
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


def list_cluster_feature_db_paths() -> List[Path]:
    output_dir = ROOT_DIR / "Resultados"
    if not output_dir.exists():
        return []
    return sorted(output_dir.glob("cluster_features*.duckdb"))


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

    _tick("Preparando datos")

    plate_groups = df_valid.groupby(group_cols, sort=False)
    total_passes = plate_groups.size()
    sum_speed = plate_groups[flow_cols.speed_kmh].sum()
    summary = pd.DataFrame(index=total_passes.index)
    summary["total_passes"] = total_passes
    summary["avg_speed_kmh"] = sum_speed / total_passes

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
        if ttc_max_map:
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
        summary = weighted_sum.div(total_passes_sum, axis=0)
        summary["total_passes"] = total_passes_sum
        summary["lane_changes"] = lane_changes_sum
        summary["lane_change_rate"] = lane_changes_sum.div(transitions_sum).fillna(0.0)
        summary = summary.reset_index()
    else:
        summary = summary.reset_index()

    summary = summary.rename(columns={PLATE_CLEAN_COL: "plate"})
    summary["n_days_active"] = summary["plate"].map(n_days_active).fillna(0).astype(int)
    summary["n_weeks_active"] = summary["plate"].map(n_weeks_active).fillna(0).astype(int)
    summary["n_months_active"] = (
        summary["plate"].map(n_months_active).fillna(0).astype(int)
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
        rare_labels = kmeans.predict(X_rare_scaled)
        
        # Apply threshold
        # If dist > threshold -> -1
        mask_unknown = rare_min_dists > threshold
        rare_labels[mask_unknown] = -1
        
        # Build Rare result
        rare_result = rare_df.copy()
        rare_result["cluster_label"] = rare_labels
        rare_result["confidence_score"] = rare_min_dists
        rare_result["is_rare"] = True
    else:
        rare_result = pd.DataFrame()

    # 5. Build Frequent result
    freq_labels = kmeans.labels_
    freq_result = frequent_df.copy()
    freq_result["cluster_label"] = freq_labels
    freq_result["confidence_score"] = freq_min_dists
    freq_result["is_rare"] = False
    
    # Consolidate
    full_df = pd.concat([freq_result, rare_result], axis=0)
    return full_df, kmeans, threshold


def assign_clusters_gmm(
    frequent_df: pd.DataFrame,
    rare_df: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    confidence_threshold_proba: float = 0.70,
    random_state: int = 42,
    covariance_type: str = "full",
) -> Tuple[pd.DataFrame, object, float]:
    """
    Entrena GMM en frequent_df. Asigna clusters a rare_df con umbral de probabilidad.
    Retorna (df_consolidado, model, threshold_used).
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("scikit-learn required") from exc

    # 1. Scale based on frequent
    X_freq_scaled, scaler = _scale_cluster_features(frequent_df, feature_cols)
    
    # 2. Train GMM
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=3
    )
    gmm.fit(X_freq_scaled)
    
    # 3. Predict Frequent (mostly to get consistency check, though GMM is soft)
    freq_probs = gmm.predict_proba(X_freq_scaled)
    freq_max_probs = freq_probs.max(axis=1)
    freq_labels = gmm.predict(X_freq_scaled)
    
    # 4. Process Rare
    if not rare_df.empty:
        X_rare = rare_df[feature_cols].to_numpy(dtype=float)
        X_rare_scaled = scaler.transform(X_rare)
        
        rare_probs = gmm.predict_proba(X_rare_scaled)
        rare_max_probs = rare_probs.max(axis=1)
        rare_labels = gmm.predict(X_rare_scaled)
        
        # Apply threshold
        # If max_prob < threshold -> -1
        mask_unknown = rare_max_probs < confidence_threshold_proba
        rare_labels[mask_unknown] = -1
        
        rare_result = rare_df.copy()
        rare_result["cluster_label"] = rare_labels
        rare_result["confidence_score"] = rare_max_probs
        rare_result["is_rare"] = True
    else:
        rare_result = pd.DataFrame()
        
    # 5. Build Frequent request
    freq_result = frequent_df.copy()
    freq_result["cluster_label"] = freq_labels
    freq_result["confidence_score"] = freq_max_probs
    freq_result["is_rare"] = False
    
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


def save_cluster_features_duckdb(
    features_df: pd.DataFrame, db_path: Optional[Path] = None
) -> Path:
    target_path = db_path or CLUSTER_DB_PATH
    conn = _connect_cluster_duckdb(read_only=False, db_path=target_path)
    try:
        conn.register("cluster_features_df", features_df)
        conn.execute(
            f"CREATE OR REPLACE TABLE {CLUSTER_TABLE_NAME} AS "
            "SELECT * FROM cluster_features_df"
        )
    finally:
        conn.close()
    return target_path


def load_cluster_features_duckdb(
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    target_path = db_path or CLUSTER_DB_PATH
    if not target_path.exists():
        return pd.DataFrame()
    conn = _connect_cluster_duckdb(read_only=True, db_path=target_path)
    try:
        info = conn.execute(
            f"PRAGMA table_info('{CLUSTER_TABLE_NAME}')"
        ).fetchall()
        if not info:
            return pd.DataFrame()
        return conn.execute(f"SELECT * FROM {CLUSTER_TABLE_NAME}").df()
    finally:
        conn.close()


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
    output_dir = ROOT_DIR / "Resultados"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "cluster_features.csv"
    features_df.to_csv(path, index=False)
    return path


def save_cluster_metrics(metrics_df: pd.DataFrame) -> Path:
    output_dir = ROOT_DIR / "Resultados"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    output_dir = ROOT_DIR / "Resultados"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    output_dir = ROOT_DIR / "Resultados"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _cluster_summary_filename(method, k)
    summary_df.to_csv(path, index=False)
    return path


def save_cluster_descriptive(
    stats_df: pd.DataFrame, method: str, k: Optional[int] = None
) -> Path:
    output_dir = ROOT_DIR / "Resultados"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _cluster_descriptive_filename(method, k)
    stats_df.to_csv(path, index=False)
    return path


def list_cluster_summary_files() -> List[Path]:
    output_dir = ROOT_DIR / "Resultados"
    if not output_dir.exists():
        return []
    candidates = sorted(output_dir.glob("cluster_summary*.csv"))
    return [path for path in candidates if _parse_cluster_summary_file(path) is not None]


def list_cluster_label_files() -> List[Path]:
    output_dir = ROOT_DIR / "Resultados"
    if not output_dir.exists():
        return []
    candidates = sorted(output_dir.glob("cluster_*.csv"))
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

    labels_path = ROOT_DIR / "Resultados" / _cluster_label_filename(method, k_value)
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
    output_dir = ROOT_DIR / "Resultados"
    output_dir.mkdir(parents=True, exist_ok=True)
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
        summary["total_passes"] = total_passes
        if n_days_active is not None:
            summary["n_days_active"] = n_days_active
        if n_weeks_active is not None:
            summary["n_weeks_active"] = n_weeks_active
        if n_months_active is not None:
            summary["n_months_active"] = n_months_active
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

    lane_changes_sum = grouped["lane_changes"].sum()
    if lane_changes_extra:
        lane_changes_sum = lane_changes_sum.add(
            pd.Series(lane_changes_extra), fill_value=0
        )

    summary = pd.DataFrame(index=total_passes.index)
    summary["total_passes"] = total_passes
    summary["avg_speed_kmh"] = speed_sum / total_passes
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
        "lane_changes",
        "rel_speed_count",
        "headway_count",
        "conflict_count",
        "n_days_active",
        "n_weeks_active",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    count_cols = [
        col
        for col in ["rel_speed_count", "headway_count", "conflict_count"]
        if col in df.columns
    ]
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
    summary["n_months_active"] = 1
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

    batch_dir = ROOT_DIR / "Resultados" / "cluster_batches"
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
    consolidated_path = ROOT_DIR / "Resultados" / "cluster_features_batches.csv"
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
        _ensure_duckdb_available()
        batch_conn = duckdb.connect(str(batch_db_path))
        try:
            batch_conn.register("cluster_features_df", consolidated)
            batch_conn.execute(
                f"CREATE OR REPLACE TABLE {CLUSTER_TABLE_NAME} AS "
                "SELECT * FROM cluster_features_df"
            )
        finally:
            batch_conn.close()
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
            db_path = save_cluster_features_duckdb(features_df, db_path=db_path)
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
                X_scaled = _scale_cluster_features(cluster_df, feature_cols)
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
                X_scaled = _scale_cluster_features(cluster_df, feature_cols)
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
            X_scaled = _scale_cluster_features(cluster_df, feature_cols)
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
