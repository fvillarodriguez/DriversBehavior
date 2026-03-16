#!/usr/bin/env python3
"""
Streamlit app and reusable utilities to replicate the paper:
"Evaluating recalibration strategies for real-time crash prediction:
adaptive drift detection vs. cumulative retraining".

The module is self-contained by design to satisfy the requirement of hosting the
full "Drift detection" section in a single source file.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

from src.utils import (
    add_accident_target,
    compute_flow_features,
    find_candidate_porticos,
    load_porticos,
    process_accidentes_df,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
DATA_DIR = ROOT_DIR / "Datos"
FLOW_DB_PATH = DATA_DIR / "flujos.duckdb"
FLOW_TABLE_NAME = "flujos_duckdb"

PAPER_TITLE = (
    "Evaluating recalibration strategies for real-time crash prediction: "
    "adaptive drift detection vs. cumulative retraining"
)
PAPER_TIME_SPAN = "2018-01-01 to 2024-09-30"

ARTICLE_SECTIONS: List[Tuple[str, str]] = [
    ("1", "Introduction"),
    ("2", "Literature Review"),
    ("2.1", "Crash Prediction Models"),
    ("2.2", "Recalibration"),
    ("2.3", "Implementation of Drift Detection in Transport"),
    ("2.4", "Research gap"),
    ("3", "Data Description"),
    ("3.1", "AVI Data"),
    ("3.2", "Accident Data"),
    ("4", "Methodology"),
    ("4.1", "Feature engineering"),
    ("4.2", "Data preparation"),
    ("4.3", "Variable selection"),
    ("4.3.1", "Correlation Matrix"),
    ("4.3.2", "Random Forest for feature selection"),
    ("4.4", "Drift"),
    ("4.5", "Prediction Models"),
    ("4.6", "Performance Indicators"),
    ("4.7", "Experimental design"),
    ("5", "Experimental Results"),
    ("6", "Concluding Remarks"),
]

ARTICLE_FIGURES: List[Tuple[str, str]] = [
    ("Figure 1", "Autopista Central contextual map"),
    ("Figure 2", "Autopista Central gate layout"),
    ("Figure 3", "Accidents per hour"),
    ("Figure 4", "Data preparation flowchart"),
    ("Figure 5", "Density of |rho| pairwise correlations"),
    ("Figure 6", "Top-20 Random Forest feature importances"),
    ("Figure 7", "Average ROC curves by strategy and model"),
]

ARTICLE_TABLES: List[Tuple[str, str]] = [
    ("Table 1", "Related work on concept/data drift in transportation"),
    ("Table 2", "Sample of AVI vehicle records"),
    ("Table 3", "Annual summary of traffic volume/composition/speed"),
    ("Table 4", "Feature engineering variable catalog"),
    ("Table 5", "Hyperparameter grids"),
    ("Table A.6", "Static strategy results"),
    ("Table A.7", "Period-aligned strategy results"),
    ("Table A.8", "Cumulative strategy results"),
    ("Table A.9", "Adaptive ADWIN drift-iteration results"),
]

REPLICATION_ANALYSES: List[Dict[str, str]] = [
    {
        "analysis": "Traffic and accidents descriptive analysis",
        "paper_ref": "Sections 3.1-3.2, Table 3, Figure 3",
        "implementation": "compute_annual_traffic_summary + compute_accident_hour_distribution",
    },
    {
        "analysis": "Feature engineering and target alignment",
        "paper_ref": "Section 4.1, Table 4",
        "implementation": "feature_catalog_table + target-ready dataset interface",
    },
    {
        "analysis": "Data preparation and anomaly interval filtering",
        "paper_ref": "Section 4.2, Figure 4",
        "implementation": "build_data_preparation_summary",
    },
    {
        "analysis": "Correlation screening",
        "paper_ref": "Section 4.3.1, Figure 5",
        "implementation": "compute_abs_correlations + drop_highly_correlated_features",
    },
    {
        "analysis": "Feature ranking with Random Forest",
        "paper_ref": "Section 4.3.2, Figure 6",
        "implementation": "rank_features_random_forest",
    },
    {
        "analysis": "Threshold calibration with Youden criterion",
        "paper_ref": "Section 4.5",
        "implementation": "youden_threshold + train_model_with_internal_validation",
    },
    {
        "analysis": "Metric computation",
        "paper_ref": "Section 4.6",
        "implementation": "compute_classification_metrics",
    },
    {
        "analysis": "Experimental design with four recalibration strategies",
        "paper_ref": "Section 4.7",
        "implementation": "run_recalibration_experiments",
    },
    {
        "analysis": "Average ROC synthesis",
        "paper_ref": "Section 5, Figure 7",
        "implementation": "build_average_roc_curves",
    },
    {
        "analysis": "Drift-event reporting",
        "paper_ref": "Table A.9",
        "implementation": "run_adaptive_strategy + format_appendix_tables",
    },
]


def build_related_work_table() -> pd.DataFrame:
    """Returns a compact replica of paper Table 1."""
    rows = [
        ("Wibisono et al. (2016)", "Traffic flow forecasting", "FIMT-DD built-in drift"),
        ("Saadallah et al. (2018)", "Taxi demand forecasting", "Adaptive ensemble"),
        ("Pan et al. (2020)", "User targeting distribution shift", "Adversarial validation"),
        ("Laha and Verma (2021)", "Mobility mode classification", "OT-based drift detector"),
        ("Manias et al. (2021)", "Streaming adaptation", "Windowed ensemble adaptation"),
        ("Andresini et al. (2021)", "Edge/fog traffic streams", "Window-based incremental learning"),
        ("Malekghaini et al. (2022)", "Vehicular network monitoring", "Spectral-entropy drift"),
        ("Lee and Park (2022)", "Autonomous vehicle control", "OSWDD"),
        ("Hossain et al. (2024)", "Safety-critical crash risk", "Empirical data drift analysis"),
        ("This paper", "RTCP recalibration timing", "ADWIN-guided adaptive retraining"),
    ]
    return pd.DataFrame(rows, columns=["paper", "objective", "drift_method"])


def feature_catalog_table() -> pd.DataFrame:
    """Returns Table 4 feature catalog."""
    rows = [
        ("Vel_x^Y", "Average speed by category x at gate Y"),
        ("Vel_x^{Y,Z}", "Average speed between consecutive gates Y and Z"),
        ("Sd_x^Y", "Speed standard deviation by category x at gate Y"),
        ("Sd_x^{Y,Z}", "Speed standard deviation between gates Y and Z"),
        ("Flow_x^Y", "Vehicle count by category x at gate Y"),
        ("Flow_x^{Y,Z}", "Traffic volume between gates Y and Z"),
        ("Den_x^Y", "Density by category x at gate Y"),
        ("Den_x^{Y,Z}", "Density between gates Y and Z"),
        ("DeltaFlow_x^Y", "Flow change between consecutive intervals"),
        ("DeltaFlow_x^{Y,Z}", "Inter-gate flow change between intervals"),
        ("DeltaVel_x^Y", "Speed change between consecutive intervals"),
        ("DeltaDen_x^Y", "Density change between consecutive intervals"),
        ("DeltaSd_x^Y", "Speed standard deviation change"),
        ("Ft_Motorcycle^Y", "Motorcycle fraction at gate Y"),
        ("Ft_Heavy^Y", "Heavy vehicle fraction at gate Y"),
        ("CL_x^{Y,Z}", "Lane-change ratio between gates Y and Z"),
        ("Vel^{Y,Z}", "Overall speed across all categories"),
        ("Flow^{Y,Z}", "Overall flow across all categories"),
        ("Sd^{Y,Z}", "Overall speed dispersion"),
        ("Acc(t)", "Binary target: 1 if at least one accident in interval t+1"),
    ]
    return pd.DataFrame(rows, columns=["variable", "description"])


def hyperparameter_reference_table() -> pd.DataFrame:
    """Returns Table 5 grids exactly as described in the paper."""
    rows = [
        (
            "Random Forest",
            "mtry={2,4,6}; splitrule={gini,extratrees}; min.node.size={1,5,10}; 5-fold CV",
        ),
        (
            "XGBoost",
            "max_depth={3,6,9}; eta={0.01,0.1,0.3}; gamma={0,1,5}; "
            "colsample_bytree={0.5,0.8,1}; min_child_weight={1,5,10}; "
            "subsample={0.5,0.7,1}; nrounds=100; 5-fold CV",
        ),
        (
            "AdaBoost",
            "iter={50,100,150}; maxdepth={1,2,3}; nu={0.1,0.5,1}; 5-fold CV",
        ),
        (
            "Neural Network (NNet)",
            "size={6,7,10,12,13,14,15,16}; decay={0.095,0.1,0.15,0.2}; "
            "maxit={190,200,300,500}; 5-fold CV",
        ),
    ]
    return pd.DataFrame(rows, columns=["model", "grid"])


def build_article_coverage_matrix() -> pd.DataFrame:
    """
    Declares full article coverage and validates each item is mapped to a
    concrete implementation artifact in this module.
    """
    mappings: List[Dict[str, str]] = []

    for sec_num, sec_title in ARTICLE_SECTIONS:
        mappings.append(
            {
                "item_id": f"Section {sec_num}",
                "item_type": "section",
                "title": sec_title,
                "artifact": "REPLICATION_ANALYSES",
            }
        )

    figure_artifacts = {
        "Figure 1": "build_gate_layout_from_porticos",
        "Figure 2": "build_gate_layout_from_porticos",
        "Figure 3": "compute_accident_hour_distribution",
        "Figure 4": "build_data_preparation_summary",
        "Figure 5": "compute_abs_correlations",
        "Figure 6": "rank_features_random_forest",
        "Figure 7": "build_average_roc_curves",
    }
    for fig_id, fig_title in ARTICLE_FIGURES:
        mappings.append(
            {
                "item_id": fig_id,
                "item_type": "figure",
                "title": fig_title,
                "artifact": figure_artifacts[fig_id],
            }
        )

    table_artifacts = {
        "Table 1": "build_related_work_table",
        "Table 2": "sample_avi_records",
        "Table 3": "compute_annual_traffic_summary",
        "Table 4": "feature_catalog_table",
        "Table 5": "hyperparameter_reference_table",
        "Table A.6": "format_appendix_tables",
        "Table A.7": "format_appendix_tables",
        "Table A.8": "format_appendix_tables",
        "Table A.9": "format_appendix_tables",
    }
    for tab_id, tab_title in ARTICLE_TABLES:
        mappings.append(
            {
                "item_id": tab_id,
                "item_type": "table",
                "title": tab_title,
                "artifact": table_artifacts[tab_id],
            }
        )

    matrix = pd.DataFrame(mappings)
    implemented: List[bool] = []
    for artifact in matrix["artifact"]:
        if artifact == "REPLICATION_ANALYSES":
            implemented.append(len(REPLICATION_ANALYSES) > 0)
            continue
        obj = globals().get(artifact)
        implemented.append(callable(obj))
    matrix["implemented"] = implemented
    return matrix


def article_coverage_percentage(matrix: Optional[pd.DataFrame] = None) -> float:
    matrix = build_article_coverage_matrix() if matrix is None else matrix
    if matrix.empty:
        return 0.0
    return float(matrix["implemented"].mean() * 100.0)


def article_replication_blueprint() -> Dict[str, pd.DataFrame]:
    analyses_df = pd.DataFrame(REPLICATION_ANALYSES)
    figures_df = pd.DataFrame(ARTICLE_FIGURES, columns=["figure", "description"])
    tables_df = pd.DataFrame(ARTICLE_TABLES, columns=["table", "description"])
    return {
        "analyses": analyses_df,
        "figures": figures_df,
        "tables": tables_df,
    }


def sample_avi_records(flows_df: pd.DataFrame, n_rows: int = 6) -> pd.DataFrame:
    """Builds a paper-like sample table (Table 2 style)."""
    if flows_df is None or flows_df.empty:
        return pd.DataFrame()
    expected = ["FECHA", "VELOCIDAD", "CATEGORIA", "MATRICULA", "PORTICO", "CARRIL"]
    available = [c for c in expected if c in flows_df.columns]
    if not available:
        return pd.DataFrame()
    out = flows_df[available].head(max(1, int(n_rows))).copy()
    return out.reset_index(drop=True)


def compute_annual_traffic_summary(
    flows_df: pd.DataFrame,
    *,
    time_col: str = "FECHA",
    speed_col: str = "VELOCIDAD",
    category_col: str = "CATEGORIA",
    lane_col: str = "CARRIL",
    plate_col: str = "MATRICULA",
    category_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Computes annual summary aligned with paper Table 3 intent.
    """
    if flows_df is None or flows_df.empty:
        return pd.DataFrame()

    work = flows_df.copy()
    if time_col not in work.columns:
        return pd.DataFrame()

    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    if work.empty:
        return pd.DataFrame()

    work["year"] = work[time_col].dt.year.astype(int)

    if category_map is None:
        category_map = {1: "Light", 2: "Heavy", 3: "Heavy", 4: "Motorcycle"}

    if category_col in work.columns:
        work[category_col] = pd.to_numeric(work[category_col], errors="coerce")
        work["category_name"] = work[category_col].map(category_map).fillna("Other")
    else:
        work["category_name"] = "Unknown"

    if speed_col in work.columns:
        work[speed_col] = pd.to_numeric(work[speed_col], errors="coerce")

    rows: List[Dict[str, Any]] = []
    for year, grp in work.groupby("year"):
        row: Dict[str, Any] = {
            "year": int(year),
            "vehicular_flow": int(len(grp)),
        }
        if plate_col in grp.columns:
            row["distinct_plates"] = int(grp[plate_col].astype(str).nunique())
        else:
            row["distinct_plates"] = np.nan

        for cat in ["Light", "Heavy", "Motorcycle"]:
            share = float((grp["category_name"] == cat).mean() * 100.0)
            row[f"share_{cat.lower()}"] = share

        if lane_col in grp.columns:
            lane_vals = pd.to_numeric(grp[lane_col], errors="coerce")
            for lane in [1, 2, 3]:
                row[f"lane_{lane}_share"] = float((lane_vals == lane).mean() * 100.0)

        if speed_col in grp.columns:
            for cat in ["Light", "Heavy", "Motorcycle"]:
                cat_speed = pd.to_numeric(
                    grp.loc[grp["category_name"] == cat, speed_col], errors="coerce"
                )
                row[f"speed_{cat.lower()}"] = float(cat_speed.mean()) if not cat_speed.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def compute_accident_hour_distribution(
    accidents_df: pd.DataFrame,
    *,
    accident_time_col: str = "accidente_time",
) -> pd.DataFrame:
    """Replicates Figure 3 style hourly accident distribution."""
    if accidents_df is None or accidents_df.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])
    if accident_time_col not in accidents_df.columns:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    work = accidents_df.copy()
    work[accident_time_col] = pd.to_datetime(work[accident_time_col], errors="coerce")
    work = work.dropna(subset=[accident_time_col])
    if work.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    work["hour"] = work[accident_time_col].dt.hour.astype(int)
    counts = work.groupby("hour").size().reindex(range(24), fill_value=0)
    total = float(counts.sum())
    out = pd.DataFrame(
        {
            "hour": counts.index.astype(int),
            "accidents": counts.values.astype(int),
        }
    )
    out["share_pct"] = np.where(total > 0, out["accidents"] / total * 100.0, 0.0)
    return out


def compute_accident_hour_distribution_from_target(
    features_df: pd.DataFrame,
    *,
    time_col: str = "interval_start",
    target_col: str = "target",
) -> pd.DataFrame:
    """Alternative hourly distribution when only interval-level target is available."""
    if features_df is None or features_df.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])
    if time_col not in features_df.columns or target_col not in features_df.columns:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    work = features_df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    if work.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    positives = work.loc[pd.to_numeric(work[target_col], errors="coerce").fillna(0).astype(int) == 1]
    if positives.empty:
        return pd.DataFrame({"hour": list(range(24)), "accidents": [0] * 24, "share_pct": [0.0] * 24})

    positives["hour"] = positives[time_col].dt.hour.astype(int)
    counts = positives.groupby("hour").size().reindex(range(24), fill_value=0)
    total = float(counts.sum())
    out = pd.DataFrame({"hour": counts.index.astype(int), "accidents": counts.values.astype(int)})
    out["share_pct"] = np.where(total > 0, out["accidents"] / total * 100.0, 0.0)
    return out


@dataclass
class DataPreparationSummary:
    initial_rows: int
    initial_features: int
    removed_high_missing_features: int
    remaining_features_after_stage1: int
    rows_after_missing_drop: int
    removed_zero_run_rows: int
    final_rows: int
    zero_run_windows_detected: int


@dataclass
class FlowSampleSelection:
    date_start: Optional[pd.Timestamp] = None
    date_end: Optional[pd.Timestamp] = None
    row_limit: Optional[int] = None


def _infer_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str],
) -> List[str]:
    candidates = [c for c in df.columns if c not in set(exclude)]
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_candidates


def apply_missing_data_policy(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    missing_threshold: float = 0.01,
    target_col: str = "target",
    time_col: str = "interval_start",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Paper Section 4.2 Stage 1:
    - remove variables with >1% missing values
    - remove intervals with any missing predictor value
    """
    if df is None or df.empty:
        return pd.DataFrame(), {
            "removed_cols": [],
            "remaining_features": [],
            "rows_after_drop": 0,
        }

    work = df.copy()
    if feature_cols is None:
        feature_cols = _infer_feature_columns(work, exclude=[target_col])
    feature_cols = [c for c in feature_cols if c in work.columns]

    missing_ratio = work[feature_cols].isna().mean()
    removed_cols = missing_ratio[missing_ratio > float(missing_threshold)].index.tolist()
    remaining = [c for c in feature_cols if c not in removed_cols]

    if remaining:
        cleaned = work.dropna(subset=remaining).copy()
    else:
        cleaned = work.copy()

    if time_col in cleaned.columns:
        cleaned[time_col] = pd.to_datetime(cleaned[time_col], errors="coerce")

    info = {
        "removed_cols": removed_cols,
        "remaining_features": remaining,
        "rows_after_drop": int(len(cleaned)),
        "missing_ratio": missing_ratio.sort_values(ascending=False),
    }
    return cleaned.reset_index(drop=True), info


def identify_long_zero_accident_runs(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    time_col: str = "interval_start",
    min_days: int = 7,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """
    Finds windows of >= min_days with no accident records (target==0).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])
    if time_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])

    work = df[[time_col, target_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce").fillna(0).astype(int)
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])

    zero_mask = work[target_col].eq(0)
    group_id = (zero_mask != zero_mask.shift(fill_value=False)).cumsum()

    min_intervals = int(math.ceil((int(min_days) * 24 * 60) / float(interval_minutes)))

    rows: List[Dict[str, Any]] = []
    for gid, grp in work.groupby(group_id):
        del gid
        if not bool(grp[target_col].iloc[0] == 0):
            continue
        length = int(len(grp))
        if length < min_intervals:
            continue
        start_ts = pd.Timestamp(grp[time_col].iloc[0])
        end_ts = pd.Timestamp(grp[time_col].iloc[-1])
        days = float(length * float(interval_minutes) / (24.0 * 60.0))
        rows.append(
            {
                "start": start_ts,
                "end": end_ts,
                "length_intervals": length,
                "length_days": days,
            }
        )

    return pd.DataFrame(rows)


def filter_long_zero_accident_runs(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    time_col: str = "interval_start",
    min_days: int = 7,
    interval_minutes: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drops rows that belong to long no-accident windows."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    runs = identify_long_zero_accident_runs(
        work,
        target_col=target_col,
        time_col=time_col,
        min_days=min_days,
        interval_minutes=interval_minutes,
    )
    if runs.empty:
        return work.reset_index(drop=True), runs

    drop_mask = np.zeros(len(work), dtype=bool)
    for run in runs.itertuples(index=False):
        drop_mask |= (work[time_col] >= run.start) & (work[time_col] <= run.end)

    filtered = work.loc[~drop_mask].reset_index(drop=True)
    return filtered, runs


def build_data_preparation_summary(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    missing_threshold: float = 0.01,
    target_col: str = "target",
    time_col: str = "interval_start",
    min_zero_days: int = 7,
    interval_minutes: int = 5,
) -> Tuple[pd.DataFrame, DataPreparationSummary, Dict[str, Any], pd.DataFrame]:
    """
    Runs the full Section 4.2 pipeline and returns clean dataset + stage statistics.
    """
    if df is None or df.empty:
        empty_summary = DataPreparationSummary(0, 0, 0, 0, 0, 0, 0, 0)
        return pd.DataFrame(), empty_summary, {}, pd.DataFrame()

    work = df.copy()
    if feature_cols is None:
        feature_cols = _infer_feature_columns(work, exclude=[target_col])

    initial_rows = int(len(work))
    initial_features = int(len(feature_cols))

    stage1_df, stage1_info = apply_missing_data_policy(
        work,
        feature_cols=feature_cols,
        missing_threshold=missing_threshold,
        target_col=target_col,
        time_col=time_col,
    )

    filtered_df, zero_runs = filter_long_zero_accident_runs(
        stage1_df,
        target_col=target_col,
        time_col=time_col,
        min_days=min_zero_days,
        interval_minutes=interval_minutes,
    )

    summary = DataPreparationSummary(
        initial_rows=initial_rows,
        initial_features=initial_features,
        removed_high_missing_features=int(len(stage1_info.get("removed_cols", []))),
        remaining_features_after_stage1=int(len(stage1_info.get("remaining_features", []))),
        rows_after_missing_drop=int(len(stage1_df)),
        removed_zero_run_rows=int(len(stage1_df) - len(filtered_df)),
        final_rows=int(len(filtered_df)),
        zero_run_windows_detected=int(len(zero_runs)),
    )

    return filtered_df, summary, stage1_info, zero_runs


def run_configurable_preparation_pipeline(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    apply_stage1: bool = True,
    missing_threshold: float = 0.01,
    apply_stage2: bool = True,
    min_zero_days: int = 7,
    target_col: str = "target",
    time_col: str = "interval_start",
    interval_minutes: int = 5,
) -> Tuple[pd.DataFrame, DataPreparationSummary, Dict[str, Any], pd.DataFrame, List[str]]:
    """
    Applies Stage 1 / Stage 2 with explicit toggles for the Feature engineering UI.
    """
    if df is None or df.empty:
        empty_summary = DataPreparationSummary(0, 0, 0, 0, 0, 0, 0, 0)
        return pd.DataFrame(), empty_summary, {}, pd.DataFrame(), []

    work = df.copy()
    if feature_cols is None:
        feature_cols = _infer_feature_columns(work, exclude=[target_col])
    feature_cols = [c for c in feature_cols if c in work.columns]

    initial_rows = int(len(work))
    initial_features = int(len(feature_cols))
    steps: List[str] = []

    stage1_info: Dict[str, Any]
    if apply_stage1:
        stage1_df, stage1_info = apply_missing_data_policy(
            work,
            feature_cols=feature_cols,
            missing_threshold=missing_threshold,
            target_col=target_col,
            time_col=time_col,
        )
        steps.append(
            "Stage 1 aplicado: se eliminaron variables con mas de "
            f"{missing_threshold * 100:.2f}% de missing y luego intervalos incompletos."
        )
    else:
        stage1_df = work.copy()
        stage1_info = {
            "removed_cols": [],
            "remaining_features": list(feature_cols),
            "rows_after_drop": int(len(stage1_df)),
            "missing_ratio": stage1_df[feature_cols].isna().mean()
            if feature_cols
            else pd.Series(dtype=float),
        }
        steps.append("Stage 1 omitido por configuracion.")

    zero_runs = pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])
    if apply_stage2:
        final_df, zero_runs = filter_long_zero_accident_runs(
            stage1_df,
            target_col=target_col,
            time_col=time_col,
            min_days=min_zero_days,
            interval_minutes=interval_minutes,
        )
        steps.append(
            "Stage 2 aplicado: se removieron ventanas multidiarias sin accidentes "
            f"(>= {int(min_zero_days)} dias)."
        )
    else:
        final_df = stage1_df.copy()
        steps.append("Stage 2 omitido por configuracion.")

    summary = DataPreparationSummary(
        initial_rows=initial_rows,
        initial_features=initial_features,
        removed_high_missing_features=int(len(stage1_info.get("removed_cols", []))),
        remaining_features_after_stage1=int(len(stage1_info.get("remaining_features", []))),
        rows_after_missing_drop=int(len(stage1_df)),
        removed_zero_run_rows=int(len(stage1_df) - len(final_df)),
        final_rows=int(len(final_df)),
        zero_run_windows_detected=int(len(zero_runs)),
    )
    return final_df, summary, stage1_info, zero_runs, steps


def compute_abs_correlations(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.Series:
    """Returns absolute pairwise correlations (upper triangle) as a 1D series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        return pd.Series(dtype=float)

    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().astype(float)
    return pairs.sort_values(ascending=False)


def drop_highly_correlated_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    threshold: float = 0.95,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Drops features with |rho| > threshold using upper-triangle greedy strategy.
    """
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        return cols, [], pd.DataFrame()

    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [c for c in cols if c not in set(to_drop)]
    return kept, to_drop, corr


def rank_features_random_forest(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "target",
    n_estimators: int = 300,
    random_state: int = 42,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Section 4.3.2 feature importance ranking.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    cols = [c for c in feature_cols if c in df.columns]
    if not cols or target_col not in df.columns:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X, y)

    imp = pd.DataFrame(
        {
            "feature": cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp.head(max(1, int(top_n))).reset_index(drop=True)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Returns threshold maximizing Youden J = sensitivity + specificity - 1.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    if np.unique(y).size < 2:
        return {
            "threshold": 0.5,
            "youden": 0.0,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }

    fpr, tpr, thr = roc_curve(y, s)
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    threshold = float(thr[idx])
    sensitivity = float(tpr[idx])
    specificity = float(1.0 - fpr[idx])
    return {
        "threshold": threshold,
        "youden": float(youden[idx]),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def compute_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    preds = (s >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    error_rate = float((fp + fn) / max(1, (tp + tn + fp + fn)))

    return {
        "auc": _safe_auc(y, s),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "error_rate": error_rate,
        "threshold": float(threshold),
    }


MODEL_NAMES = ["XGBoost", "AdaBoost", "Random Forest", "NNet"]

FULL_HYPERPARAM_GRIDS: Dict[str, Dict[str, Sequence[Any]]] = {
    "Random Forest": {
        "mtry": [2, 4, 6],
        "splitrule": ["gini", "extratrees"],
        "min_node_size": [1, 5, 10],
    },
    "XGBoost": {
        "max_depth": [3, 6, 9],
        "eta": [0.01, 0.1, 0.3],
        "gamma": [0.0, 1.0, 5.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "min_child_weight": [1.0, 5.0, 10.0],
        "subsample": [0.5, 0.7, 1.0],
        "nrounds": [100],
    },
    "AdaBoost": {
        "iter": [50, 100, 150],
        "maxdepth": [1, 2, 3],
        "nu": [0.1, 0.5, 1.0],
    },
    "NNet": {
        "size": [6, 7, 10, 12, 13, 14, 15, 16],
        "decay": [0.095, 0.1, 0.15, 0.2],
        "maxit": [190, 200, 300, 500],
    },
}

FAST_HYPERPARAM_GRIDS: Dict[str, Dict[str, Sequence[Any]]] = {
    "Random Forest": {
        "mtry": [2, 6],
        "splitrule": ["gini", "extratrees"],
        "min_node_size": [1, 10],
    },
    "XGBoost": {
        "max_depth": [3, 6],
        "eta": [0.1],
        "gamma": [0.0, 1.0],
        "colsample_bytree": [0.8],
        "min_child_weight": [1.0, 5.0],
        "subsample": [0.7],
        "nrounds": [100],
    },
    "AdaBoost": {
        "iter": [50, 100],
        "maxdepth": [1, 2],
        "nu": [0.1, 0.5],
    },
    "NNet": {
        "size": [10, 14],
        "decay": [0.1, 0.2],
        "maxit": [200],
    },
}


def _parameter_combinations(
    model_name: str,
    *,
    fast_mode: bool,
    grid_limit: Optional[int],
    custom_grid: Optional[Dict[str, Sequence[Any]]] = None,
) -> List[Dict[str, Any]]:
    if custom_grid is not None:
        grid = custom_grid
    else:
        grid = FAST_HYPERPARAM_GRIDS[model_name] if fast_mode else FULL_HYPERPARAM_GRIDS[model_name]

    combos = list(ParameterGrid(grid))
    if grid_limit is None or grid_limit <= 0 or len(combos) <= int(grid_limit):
        return combos

    # Deterministic down-sampling to keep runtime bounded.
    idx = np.linspace(0, len(combos) - 1, int(grid_limit)).round().astype(int)
    dedup = sorted(set(idx.tolist()))
    return [combos[i] for i in dedup]


def _build_model(
    model_name: str,
    params: Dict[str, Any],
    *,
    random_state: int,
    n_features: int,
):
    if model_name == "Random Forest":
        criterion = "gini" if str(params.get("splitrule", "gini")) == "gini" else "entropy"
        mtry = int(params.get("mtry", min(6, max(1, n_features))))
        return RandomForestClassifier(
            n_estimators=400,
            criterion=criterion,
            max_features=max(1, min(mtry, n_features)),
            min_samples_leaf=int(params.get("min_node_size", 1)),
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "AdaBoost":
        base = DecisionTreeClassifier(
            max_depth=int(params.get("maxdepth", 1)),
            random_state=random_state,
        )
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=int(params.get("iter", 100)),
            learning_rate=float(params.get("nu", 0.1)),
            random_state=random_state,
        )

    if model_name == "NNet":
        return MLPClassifier(
            hidden_layer_sizes=(int(params.get("size", 10)),),
            alpha=float(params.get("decay", 0.1)),
            max_iter=int(params.get("maxit", 200)),
            random_state=random_state,
            solver="adam",
        )

    if model_name == "XGBoost":
        if xgb is None:
            raise ImportError("xgboost is not installed in this environment.")
        return xgb.XGBClassifier(
            n_estimators=int(params.get("nrounds", 100)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("eta", 0.1)),
            gamma=float(params.get("gamma", 0.0)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            subsample=float(params.get("subsample", 0.7)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _model_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _cv_auc(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    params: Dict[str, Any],
    folds: int,
    random_state: int,
) -> float:
    y_np = np.asarray(y).astype(int)
    if np.unique(y_np).size < 2:
        return float("nan")

    min_class = int(np.bincount(y_np).min())
    effective_folds = max(2, min(int(folds), min_class))
    if effective_folds < 2:
        return float("nan")

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)

    aucs: List[float] = []
    for tr_idx, va_idx in skf.split(X, y_np):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr = y_np[tr_idx]
        y_va = y_np[va_idx]

        try:
            model = _build_model(
                model_name,
                params,
                random_state=random_state,
                n_features=X.shape[1],
            )
            model.fit(X_tr, y_tr)
            scores = _model_scores(model, X_va)
            auc = _safe_auc(y_va, scores)
            if not np.isnan(auc):
                aucs.append(float(auc))
        except Exception:
            continue

    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
    grid_limit: Optional[int] = None,
    custom_grid: Optional[Dict[str, Sequence[Any]]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    combinations = _parameter_combinations(
        model_name,
        fast_mode=fast_mode,
        grid_limit=grid_limit,
        custom_grid=custom_grid,
    )

    rows: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, Any]] = None
    best_auc = -np.inf

    for params in combinations:
        auc = _cv_auc(
            X,
            y,
            model_name=model_name,
            params=params,
            folds=folds,
            random_state=random_state,
        )
        row = {"model": model_name, "cv_auc": auc, "params": json.dumps(params, sort_keys=True)}
        rows.append(row)

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_params = dict(params)

    if best_params is None:
        best_params = combinations[0] if combinations else {}

    return best_params, pd.DataFrame(rows).sort_values("cv_auc", ascending=False, na_position="last")


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    return X, y


def train_model_with_internal_validation(
    train_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    model_name: str,
    validation_size: float = 0.2,
    folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
    grid_limit: Optional[int] = None,
    custom_grid: Optional[Dict[str, Sequence[Any]]] = None,
) -> Dict[str, Any]:
    """
    Section 4.5 replication:
    - internal 20% validation split
    - threshold calibration with Youden criterion
    """
    X_all, y_all = _prepare_xy(train_df, feature_cols, target_col)
    if y_all.nunique() < 2:
        raise ValueError("Training data has a single target class.")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all,
        y_all,
        test_size=float(validation_size),
        random_state=int(random_state),
        stratify=y_all,
    )

    best_params, search_df = tune_hyperparameters(
        X_tr,
        y_tr,
        model_name=model_name,
        folds=folds,
        random_state=random_state,
        fast_mode=fast_mode,
        grid_limit=grid_limit,
        custom_grid=custom_grid,
    )

    model = _build_model(
        model_name,
        best_params,
        random_state=random_state,
        n_features=X_tr.shape[1],
    )
    model.fit(X_tr, y_tr)

    val_scores = _model_scores(model, X_va)
    youden = youden_threshold(y_va.to_numpy(), val_scores)

    val_metrics = compute_classification_metrics(
        y_va.to_numpy(),
        val_scores,
        threshold=float(youden["threshold"]),
    )

    return {
        "model": model,
        "threshold": float(youden["threshold"]),
        "youden": youden,
        "val_metrics": val_metrics,
        "best_params": best_params,
        "search_df": search_df,
        "feature_cols": list(X_tr.columns),
    }


def _evaluate_split(
    model,
    test_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    threshold: float,
) -> Dict[str, Any]:
    X_te, y_te = _prepare_xy(test_df, feature_cols, target_col)
    scores = _model_scores(model, X_te)
    metrics = compute_classification_metrics(y_te.to_numpy(), scores, threshold=float(threshold))
    return {
        "metrics": metrics,
        "y_true": y_te.to_numpy(),
        "scores": scores,
    }


def _available_prediction_years(df: pd.DataFrame, time_col: str) -> List[int]:
    if df is None or df.empty or time_col not in df.columns:
        return []
    years = pd.to_datetime(df[time_col], errors="coerce").dt.year.dropna().astype(int).unique()
    return sorted(years.tolist())


def _as_year_mask(df: pd.DataFrame, time_col: str, year: int) -> pd.Series:
    times = pd.to_datetime(df[time_col], errors="coerce")
    return times.dt.year.eq(int(year))


def run_yearly_strategy(
    df: pd.DataFrame,
    *,
    strategy: str,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    validation_size: float = 0.2,
    folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
    grid_limit: Optional[int] = None,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Runs static / period-aligned / cumulative yearly experiments.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    years = _available_prediction_years(work, time_col)
    if len(years) < 2:
        return pd.DataFrame(), []

    if base_year is None:
        base_year = years[0]

    pred_years = [y for y in years if y > int(base_year)]

    rows: List[Dict[str, Any]] = []
    roc_payload: List[Dict[str, Any]] = []

    for model_name in model_names:
        for pred_year in pred_years:
            if strategy == "static":
                train_mask = _as_year_mask(work, time_col, int(base_year))
                train_label = f"{int(base_year)}"
            elif strategy == "period_aligned":
                train_mask = _as_year_mask(work, time_col, int(pred_year) - 1)
                train_label = f"{int(pred_year) - 1}"
            elif strategy == "cumulative":
                train_mask = pd.to_datetime(work[time_col], errors="coerce").dt.year.le(int(pred_year) - 1)
                train_label = f"[<= {int(pred_year) - 1}]"
            else:
                raise ValueError(f"Unsupported yearly strategy: {strategy}")

            test_mask = _as_year_mask(work, time_col, int(pred_year))
            train_df = work.loc[train_mask].copy()
            test_df = work.loc[test_mask].copy()

            if train_df.empty or test_df.empty:
                continue
            if pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
                continue
            if pd.to_numeric(test_df[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
                continue

            t0 = time.perf_counter()
            try:
                bundle = train_model_with_internal_validation(
                    train_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    model_name=model_name,
                    validation_size=validation_size,
                    folds=folds,
                    random_state=random_state,
                    fast_mode=fast_mode,
                    grid_limit=grid_limit,
                    custom_grid=(custom_grids or {}).get(model_name),
                )
            except Exception:
                continue
            train_time = time.perf_counter() - t0

            eval_payload = _evaluate_split(
                bundle["model"],
                test_df,
                feature_cols=feature_cols,
                target_col=target_col,
                threshold=float(bundle["threshold"]),
            )

            metrics = eval_payload["metrics"]
            row = {
                "strategy": strategy,
                "iteration": int(pred_year - years[0]),
                "training_year": train_label,
                "prediction_year": int(pred_year),
                "model": model_name,
                "auc": float(metrics["auc"]),
                "sensitivity": float(metrics["sensitivity"]),
                "specificity": float(metrics["specificity"]),
                "error_rate": float(metrics["error_rate"]),
                "threshold": float(bundle["threshold"]),
                "training_time_sec": float(train_time),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "best_params": json.dumps(bundle["best_params"], sort_keys=True),
            }
            rows.append(row)

            roc_payload.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "segment": str(pred_year),
                    "y_true": eval_payload["y_true"],
                    "scores": eval_payload["scores"],
                }
            )

    return pd.DataFrame(rows), roc_payload


class SimpleADWIN:
    """
    Lightweight ADWIN-style detector for binary error streams.
    """

    def __init__(
        self,
        *,
        delta: float = 0.002,
        min_window: int = 45_000,
        min_subwindow: int = 32,
        max_splits: int = 128,
    ) -> None:
        self.delta = float(delta)
        self.min_window = int(min_window)
        self.min_subwindow = int(min_subwindow)
        self.max_splits = int(max_splits)
        self.window: deque[float] = deque()

    def __len__(self) -> int:
        return len(self.window)

    def update(self, value: float) -> Tuple[bool, Optional[Dict[str, float]]]:
        self.window.append(float(value))
        n = len(self.window)

        if n < max(self.min_window, 2 * self.min_subwindow):
            return False, None

        arr = np.fromiter(self.window, dtype=float, count=n)
        prefix = np.cumsum(arr)

        step = max(1, n // max(1, self.max_splits))
        best: Optional[Dict[str, float]] = None

        for cut in range(self.min_subwindow, n - self.min_subwindow + 1, step):
            n0 = cut
            n1 = n - cut
            sum0 = prefix[cut - 1]
            sum1 = prefix[-1] - sum0
            mu0 = float(sum0 / n0)
            mu1 = float(sum1 / n1)
            gap = abs(mu0 - mu1)

            m = 1.0 / ((1.0 / n0) + (1.0 / n1))
            log_term_inner = max(1.0000001, 4.0 * math.log(max(2.0, float(n))) / max(self.delta, 1e-9))
            eps = math.sqrt((1.0 / (2.0 * m)) * math.log(log_term_inner))

            if gap > eps:
                candidate = {
                    "cut": float(cut),
                    "gap": float(gap),
                    "eps": float(eps),
                    "n": float(n),
                }
                if best is None or candidate["gap"] > best["gap"]:
                    best = candidate

        if best is None:
            return False, None

        n0 = int(best["cut"])
        n1 = n - n0
        for _ in range(n0):
            self.window.popleft()

        return True, {
            "n": float(n),
            "n0": float(n0),
            "n1": float(n1),
            "gap": float(best["gap"]),
            "eps": float(best["eps"]),
        }


def run_adaptive_strategy(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    random_state: int = 42,
    validation_size: float = 0.2,
    folds: int = 5,
    fast_mode: bool = True,
    grid_limit: Optional[int] = None,
    adwin_delta: float = 0.002,
    min_window: int = 45_000,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Runs Section 4.7 adaptive retraining with ADWIN drift detection.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    years = _available_prediction_years(work, time_col)
    if len(years) < 2:
        return pd.DataFrame(), []

    if base_year is None:
        base_year = years[0]

    base_train = work.loc[_as_year_mask(work, time_col, int(base_year))].copy()
    stream = work.loc[pd.to_datetime(work[time_col], errors="coerce").dt.year.gt(int(base_year))].copy()

    if base_train.empty or stream.empty:
        return pd.DataFrame(), []

    rows: List[Dict[str, Any]] = []
    roc_payload: List[Dict[str, Any]] = []

    for model_name in model_names:
        if pd.to_numeric(base_train[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
            continue

        try:
            t0 = time.perf_counter()
            bundle = train_model_with_internal_validation(
                base_train,
                feature_cols=feature_cols,
                target_col=target_col,
                model_name=model_name,
                validation_size=validation_size,
                folds=folds,
                random_state=random_state,
                fast_mode=fast_mode,
                grid_limit=grid_limit,
                custom_grid=(custom_grids or {}).get(model_name),
            )
            fit_time = time.perf_counter() - t0
        except Exception:
            continue

        model = bundle["model"]
        threshold = float(bundle["threshold"])
        detector = SimpleADWIN(delta=adwin_delta, min_window=min_window)

        X_stream, y_stream = _prepare_xy(stream, feature_cols, target_col)
        ts_stream = pd.to_datetime(stream[time_col], errors="coerce").reset_index(drop=True)

        segment_y: List[int] = []
        segment_s: List[float] = []
        training_window_rows: deque[Dict[str, Any]] = deque()
        drift_idx = 0

        for i in range(len(stream)):
            x_i = X_stream.iloc[[i]]
            y_i = int(y_stream.iloc[i])
            ts_i = ts_stream.iloc[i]

            score_i = float(_model_scores(model, x_i)[0])
            pred_i = int(score_i >= threshold)
            err_i = float(int(pred_i != y_i))

            segment_y.append(y_i)
            segment_s.append(score_i)
            training_window_rows.append(stream.iloc[i].to_dict())

            is_drift, info = detector.update(err_i)
            if not is_drift or info is None:
                continue

            drift_idx += 1
            seg_metrics = compute_classification_metrics(
                np.asarray(segment_y),
                np.asarray(segment_s),
                threshold=threshold,
            )

            remaining = int(len(stream) - (i + 1))
            rows.append(
                {
                    "strategy": "adaptive_adwin",
                    "drift": drift_idx,
                    "drift_date": pd.Timestamp(ts_i),
                    "W": int(info["n"]),
                    "W0": int(info["n0"]),
                    "W1": int(info["n1"]),
                    "remaining_periods": remaining,
                    "model": model_name,
                    "auc": float(seg_metrics["auc"]),
                    "sensitivity": float(seg_metrics["sensitivity"]),
                    "specificity": float(seg_metrics["specificity"]),
                    "error_rate": float(seg_metrics["error_rate"]),
                    "training_time_sec": float(fit_time),
                }
            )

            roc_payload.append(
                {
                    "strategy": "adaptive_adwin",
                    "model": model_name,
                    "segment": f"drift_{drift_idx}",
                    "y_true": np.asarray(segment_y),
                    "scores": np.asarray(segment_s),
                }
            )

            while len(training_window_rows) > int(info["n1"]):
                training_window_rows.popleft()

            retrain_df = pd.DataFrame(list(training_window_rows))
            if (
                len(retrain_df) >= int(min_window)
                and target_col in retrain_df.columns
                and pd.to_numeric(retrain_df[target_col], errors="coerce").fillna(0).astype(int).nunique() >= 2
            ):
                try:
                    t_fit = time.perf_counter()
                    bundle = train_model_with_internal_validation(
                        retrain_df,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        model_name=model_name,
                        validation_size=validation_size,
                        folds=folds,
                        random_state=random_state,
                        fast_mode=fast_mode,
                        grid_limit=grid_limit,
                        custom_grid=(custom_grids or {}).get(model_name),
                    )
                    fit_time = time.perf_counter() - t_fit
                    model = bundle["model"]
                    threshold = float(bundle["threshold"])
                except Exception:
                    pass

            segment_y = []
            segment_s = []

        # Final segment row (matches A.9 style "remaining=0")
        if segment_y:
            drift_idx += 1
            seg_metrics = compute_classification_metrics(
                np.asarray(segment_y),
                np.asarray(segment_s),
                threshold=threshold,
            )
            rows.append(
                {
                    "strategy": "adaptive_adwin",
                    "drift": drift_idx,
                    "drift_date": pd.Timestamp(ts_stream.iloc[-1]),
                    "W": int(len(detector)),
                    "W0": np.nan,
                    "W1": np.nan,
                    "remaining_periods": 0,
                    "model": model_name,
                    "auc": float(seg_metrics["auc"]),
                    "sensitivity": float(seg_metrics["sensitivity"]),
                    "specificity": float(seg_metrics["specificity"]),
                    "error_rate": float(seg_metrics["error_rate"]),
                    "training_time_sec": float(fit_time),
                }
            )
            roc_payload.append(
                {
                    "strategy": "adaptive_adwin",
                    "model": model_name,
                    "segment": "final",
                    "y_true": np.asarray(segment_y),
                    "scores": np.asarray(segment_s),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model", "drift"]).reset_index(drop=True)
    return out, roc_payload


def run_recalibration_experiments(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    strategies: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    validation_size: float = 0.2,
    folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
    grid_limit: Optional[int] = None,
    adwin_delta: float = 0.002,
    min_window: int = 45_000,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for all four paper strategies.
    """
    if model_names is None:
        model_names = MODEL_NAMES
    if strategies is None:
        strategies = ["static", "period_aligned", "cumulative", "adaptive_adwin"]

    all_rows: List[pd.DataFrame] = []
    all_roc: List[Dict[str, Any]] = []
    adaptive_rows = pd.DataFrame()

    for strategy in strategies:
        if strategy in {"static", "period_aligned", "cumulative"}:
            rows_df, roc_data = run_yearly_strategy(
                df,
                strategy=strategy,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_names=model_names,
                base_year=base_year,
                validation_size=validation_size,
                folds=folds,
                random_state=random_state,
                fast_mode=fast_mode,
                grid_limit=grid_limit,
                custom_grids=custom_grids,
            )
            if not rows_df.empty:
                all_rows.append(rows_df)
            all_roc.extend(roc_data)

        elif strategy == "adaptive_adwin":
            adaptive_rows, roc_data = run_adaptive_strategy(
                df,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_names=model_names,
                base_year=base_year,
                random_state=random_state,
                validation_size=validation_size,
                folds=folds,
                fast_mode=fast_mode,
                grid_limit=grid_limit,
                adwin_delta=adwin_delta,
                min_window=min_window,
                custom_grids=custom_grids,
            )
            all_roc.extend(roc_data)

    yearly_results = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    roc_curves = build_average_roc_curves(all_roc)
    summary = summarize_results(yearly_results, adaptive_rows)
    appendix = format_appendix_tables(yearly_results, adaptive_rows)

    return {
        "yearly_results": yearly_results,
        "adaptive_results": adaptive_rows,
        "roc_payload": all_roc,
        "average_roc": roc_curves,
        "summary": summary,
        "appendix_tables": appendix,
    }


def build_average_roc_curves(
    roc_payload: Sequence[Dict[str, Any]],
    *,
    n_points: int = 101,
) -> pd.DataFrame:
    """
    Builds Figure 7 style average ROC curves by strategy/model.
    """
    if not roc_payload:
        return pd.DataFrame(columns=["strategy", "model", "fpr", "tpr", "label"])

    fpr_grid = np.linspace(0.0, 1.0, int(n_points))
    rows: List[Dict[str, Any]] = []

    keys = sorted({(r["strategy"], r["model"]) for r in roc_payload})
    for strategy, model in keys:
        curves: List[np.ndarray] = []
        for item in roc_payload:
            if item["strategy"] != strategy or item["model"] != model:
                continue
            y_true = np.asarray(item["y_true"]).astype(int)
            scores = np.asarray(item["scores"]).astype(float)
            if np.unique(y_true).size < 2:
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                interp_tpr = np.interp(fpr_grid, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tpr[-1] = 1.0
                curves.append(interp_tpr)
            except Exception:
                continue

        if not curves:
            continue
        mean_tpr = np.mean(np.vstack(curves), axis=0)
        label = f"{strategy} | {model}"
        for fpr_val, tpr_val in zip(fpr_grid, mean_tpr):
            rows.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "fpr": float(fpr_val),
                    "tpr": float(tpr_val),
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


def summarize_results(
    yearly_results: pd.DataFrame,
    adaptive_results: pd.DataFrame,
) -> pd.DataFrame:
    """Builds compact summary table across strategies and models."""
    rows: List[pd.DataFrame] = []
    if yearly_results is not None and not yearly_results.empty:
        y = (
            yearly_results.groupby(["strategy", "model"], dropna=False)
            [["auc", "sensitivity", "specificity", "error_rate", "training_time_sec"]]
            .mean()
            .reset_index()
        )
        rows.append(y)

    if adaptive_results is not None and not adaptive_results.empty:
        a = (
            adaptive_results.groupby(["strategy", "model"], dropna=False)
            [["auc", "sensitivity", "specificity", "error_rate", "training_time_sec"]]
            .mean()
            .reset_index()
        )
        rows.append(a)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["strategy", "model"]).reset_index(drop=True)


def format_appendix_tables(
    yearly_results: pd.DataFrame,
    adaptive_results: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Formats result outputs as paper Appendix A tables A.6-A.9."""
    tables: Dict[str, pd.DataFrame] = {
        "A.6": pd.DataFrame(),
        "A.7": pd.DataFrame(),
        "A.8": pd.DataFrame(),
        "A.9": pd.DataFrame(),
    }

    if yearly_results is not None and not yearly_results.empty:
        common_cols = [
            "iteration",
            "training_year",
            "prediction_year",
            "model",
            "auc",
            "sensitivity",
            "specificity",
            "error_rate",
            "training_time_sec",
            "threshold",
        ]
        static = yearly_results.loc[yearly_results["strategy"] == "static", common_cols]
        period = yearly_results.loc[yearly_results["strategy"] == "period_aligned", common_cols]
        cumulative = yearly_results.loc[yearly_results["strategy"] == "cumulative", common_cols]

        tables["A.6"] = static.reset_index(drop=True)
        tables["A.7"] = period.reset_index(drop=True)
        tables["A.8"] = cumulative.reset_index(drop=True)

    if adaptive_results is not None and not adaptive_results.empty:
        cols = [
            "drift",
            "drift_date",
            "W",
            "W0",
            "W1",
            "remaining_periods",
            "model",
            "auc",
            "sensitivity",
            "specificity",
            "error_rate",
            "training_time_sec",
        ]
        tables["A.9"] = adaptive_results[cols].reset_index(drop=True)

    return tables


def build_gate_layout_from_porticos(porticos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Figure 1/2 support: extracts AC-07 to AC-10 gate layout from Porticos table.
    """
    if porticos_df is None or porticos_df.empty:
        return pd.DataFrame(columns=["portico", "km", "calzada", "eje"])

    work = porticos_df.copy()
    for col in ["portico", "km", "calzada", "eje"]:
        if col not in work.columns:
            return pd.DataFrame(columns=["portico", "km", "calzada", "eje"])

    keep_codes = {"AC-07", "AC-08", "AC-09", "AC-10"}
    work["portico"] = work["portico"].astype(str).str.strip().str.upper()
    work = work.loc[work["portico"].isin(keep_codes)].copy()
    work["km"] = pd.to_numeric(work["km"], errors="coerce")
    work = work.dropna(subset=["km"])\
        .sort_values("km")
    return work[["portico", "km", "calzada", "eje"]].reset_index(drop=True)


def load_csv_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return pd.read_csv(p)


def load_duckdb_table(path: str, table_name: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    p = Path(path)
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    if not p.exists():
        raise FileNotFoundError(f"DuckDB file not found: {p}")

    con = duckdb.connect(str(p), read_only=True)
    try:
        if table_name is None:
            tables = con.execute("SHOW TABLES").fetchall()
            if not tables:
                return pd.DataFrame()
            table_name = str(tables[0][0])
        query = f"SELECT * FROM {table_name}"
        if limit is not None and int(limit) > 0:
            query += f" LIMIT {int(limit)}"
        return con.execute(query).df()
    finally:
        con.close()


def _list_event_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    out: List[Path] = []
    for path in DATA_DIR.glob("*.csv"):
        lower = path.name.lower()
        if lower.startswith("eventos") or lower.startswith("accidentes"):
            out.append(path)
    return sorted(out)


def _read_event_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


def _normalize_portico_code(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None
    try:
        num = float(text.replace(",", "."))
    except ValueError:
        return text
    if num.is_integer():
        return str(int(num))
    return str(num)


def _normalize_portico_series(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    invalid = (
        out.isna()
        | out.str.len().fillna(0).eq(0)
        | out.isin(["NAN", "NONE", "NULL"]).fillna(False)
    )
    numeric = pd.to_numeric(out.str.replace(",", ".", regex=False), errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        int_mask = numeric_mask & np.isclose(numeric, np.floor(numeric))
        float_mask = numeric_mask & ~int_mask
        if int_mask.any():
            out.loc[int_mask] = numeric.loc[int_mask].astype("Int64").astype("string")
        if float_mask.any():
            out.loc[float_mask] = numeric.loc[float_mask].astype("string")
    out.loc[invalid] = pd.NA
    return out


def _find_match_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    normalized: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "").replace(".", "")
        normalized[key] = str(col)
    for candidate in candidates:
        key = str(candidate).strip().lower().replace(" ", "").replace(".", "")
        if key in normalized:
            return normalized[key]
    return None


def _feature_columns_for_modeling(df: pd.DataFrame, target_col: str) -> List[str]:
    if df is None or df.empty:
        return []
    excluded = {
        target_col,
        "interval_start",
        "portico",
        "eje",
        "calzada",
        "portico_last",
        "portico_next",
        "portico_inicio",
        "portico_fin",
    }
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def _date_defaults(summary: Dict[str, Any]) -> Tuple[date, date]:
    today = datetime.today().date()
    min_ts = summary.get("min_ts")
    max_ts = summary.get("max_ts")
    if isinstance(min_ts, pd.Timestamp) and isinstance(max_ts, pd.Timestamp):
        return min_ts.date(), max_ts.date()
    return today, today


def _build_flow_sample_mode_selector(*, key_prefix: str) -> str:
    return st.radio(
        "Muestreo",
        ["Todo", "Rango de fechas", "Porcentaje"],
        horizontal=True,
        key=f"{key_prefix}_sample_mode",
    )


def _build_flow_sample_inputs(
    summary: Dict[str, Any],
    mode: str,
    *,
    key_prefix: str,
) -> Tuple[FlowSampleSelection, bool, bool]:
    row_limit = None
    date_start = None
    date_end = None
    range_valid = True
    percent_mode = False

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

        date_start = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
        date_end = pd.Timestamp(datetime.combine(end_date, datetime.max.time()))
        if date_end <= date_start:
            st.error("La fecha final debe ser posterior a la fecha de inicio.")
            range_valid = False

    elif mode == "Porcentaje":
        percent_mode = True
        row_count = int(summary.get("row_count", 0) or 0)
        if row_count <= 0:
            st.warning("No hay filas disponibles para muestrear.")
        else:
            percent = st.slider(
                "Porcentaje",
                min_value=1,
                max_value=100,
                value=10,
                key=f"{key_prefix}_percent",
            )
            row_limit = max(1, int(row_count * (percent / 100.0)))
            st.caption(f"Se consultaran aproximadamente {row_limit:,} filas.")

    sample = FlowSampleSelection(
        date_start=date_start,
        date_end=date_end,
        row_limit=row_limit,
    )
    return sample, percent_mode, range_valid


def _query_flujos_duckdb(
    db_path: str,
    *,
    table_name: str = "flujos_duckdb",
    row_limit: Optional[int] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")

    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"""
            SELECT FECHA, VELOCIDAD, CATEGORIA, MATRICULA, PORTICO, CARRIL
            FROM {table_name}
        """
        params: List[Any] = []
        if date_start is not None and date_end is not None:
            query += " WHERE FECHA >= ? AND FECHA <= ?"
            params.extend([pd.Timestamp(date_start), pd.Timestamp(date_end)])
        query += " ORDER BY FECHA"
        if row_limit is not None and int(row_limit) > 0:
            query += " LIMIT ?"
            params.append(int(row_limit))
        return con.execute(query, params).df()
    finally:
        con.close()


def build_dataset_from_flujos_and_eventos(
    *,
    flow_db_path: str,
    flow_table_name: str = "flujos_duckdb",
    event_file_names: Sequence[str],
    interval_minutes: int = 5,
    flow_row_limit: Optional[int] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Builds an interval-level dataset from raw flows + events following the same
    event processing logic used in Crash prediction (Eventos tab).
    """
    if not event_file_names:
        raise ValueError("No event files were selected.")

    porticos_df = load_porticos()

    event_frames: List[pd.DataFrame] = []
    resolved_event_paths: List[str] = []
    for name in event_file_names:
        path = DATA_DIR / str(name)
        if not path.exists():
            raise FileNotFoundError(f"Event file does not exist: {path}")
        event_frames.append(_read_event_csv(path))
        resolved_event_paths.append(str(path))

    raw_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    accidents_df, excluded_df = process_accidentes_df(
        raw_events, porticos_df, return_excluded=True
    )
    if accidents_df.empty:
        raise ValueError("No valid accidents remained after processing event files.")

    flows_df = _query_flujos_duckdb(
        flow_db_path,
        table_name=flow_table_name,
        row_limit=flow_row_limit,
        date_start=date_start,
        date_end=date_end,
    )
    if flows_df.empty:
        raise ValueError("Flow query returned zero rows.")

    flows_df["FECHA"] = pd.to_datetime(flows_df["FECHA"], errors="coerce")
    flows_df = flows_df.dropna(subset=["FECHA"])
    flows_df["VELOCIDAD"] = pd.to_numeric(flows_df["VELOCIDAD"], errors="coerce")
    flows_df["CATEGORIA"] = pd.to_numeric(flows_df["CATEGORIA"], errors="coerce")
    flows_df = flows_df.dropna(subset=["CATEGORIA"])
    flows_df["CATEGORIA"] = flows_df["CATEGORIA"].astype(int)

    features_df = compute_flow_features(
        flows_df,
        interval_minutes=int(interval_minutes),
        metrics=["flow", "speed", "density", "delta_speed", "delta_density"],
    )
    if features_df.empty:
        raise ValueError("Feature engineering produced an empty dataset.")

    base_df = add_accident_target(
        features_df,
        accidents_df,
        interval_minutes=int(interval_minutes),
        portico_col="portico",
        interval_col="interval_start",
    )
    if base_df.empty:
        raise ValueError("Target merge produced an empty dataset.")

    meta = {
        "steps": [
            "Load and concatenate selected event files from Datos/.",
            "Process events with Porticos.csv using Crash prediction event logic (process_accidentes_df).",
            "Query raw flow records from flujos.duckdb (table flujos_duckdb).",
            "Aggregate flow data into interval features (flow/speed/density + deltas).",
            "Align accidents to intervals and build binary target with add_accident_target.",
        ],
        "flow_db_path": str(flow_db_path),
        "flow_table_name": str(flow_table_name),
        "event_files": resolved_event_paths,
        "interval_minutes": int(interval_minutes),
        "date_start": str(date_start) if date_start is not None else None,
        "date_end": str(date_end) if date_end is not None else None,
        "counts": {
            "raw_event_rows": int(len(raw_events)),
            "processed_accidents": int(len(accidents_df)),
            "excluded_events_without_portico": int(len(excluded_df)),
            "flow_rows_queried": int(len(flows_df)),
            "interval_rows_features": int(len(features_df)),
            "final_rows_with_target": int(len(base_df)),
            "positive_intervals_target": int(
                pd.to_numeric(base_df.get("target"), errors="coerce").fillna(0).astype(int).sum()
            ),
        },
    }
    return base_df, accidents_df, excluded_df, meta


def _list_feature_duckdb_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    patterns = [
        "drift_flow_features_*.duckdb",
        "accident_flow_features_*.duckdb",
        "flow_features_*.duckdb",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(list(RESULTS_DIR.glob(pattern)))
    uniq = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    return uniq


def _load_feature_df_from_duckdb(path: Path) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        if not tables:
            return pd.DataFrame()
        priority = ["clean_features", "flow_features", "features", tables[0]]
        table_name = next((t for t in priority if t in tables), tables[0])
        return con.execute(f"SELECT * FROM {table_name}").df()
    finally:
        con.close()


def _load_feature_payload_from_duckdb(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        raw_df = pd.DataFrame()
        clean_df = pd.DataFrame()
        if "raw_features" in tables:
            raw_df = con.execute("SELECT * FROM raw_features").df()
        elif "flow_features" in tables:
            raw_df = con.execute("SELECT * FROM flow_features").df()
        elif "features" in tables:
            raw_df = con.execute("SELECT * FROM features").df()

        if "clean_features" in tables:
            clean_df = con.execute("SELECT * FROM clean_features").df()
        elif not raw_df.empty:
            clean_df = raw_df.copy()

        return raw_df, clean_df
    finally:
        con.close()


def _flow_db_summary(db_path: str, table_name: str) -> Dict[str, Any]:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(db_path, read_only=True)
    try:
        row_count = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        min_ts, max_ts = con.execute(
            f"SELECT MIN(FECHA), MAX(FECHA) FROM {table_name}"
        ).fetchone()
        return {
            "row_count": row_count,
            "min_ts": pd.Timestamp(min_ts) if min_ts is not None else None,
            "max_ts": pd.Timestamp(max_ts) if max_ts is not None else None,
        }
    finally:
        con.close()


def _count_flujos_rows(
    db_path: str,
    *,
    table_name: str,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    allowed_porticos: Optional[Sequence[str]] = None,
) -> int:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(db_path, read_only=True)
    try:
        where_parts: List[str] = []
        params: List[Any] = []
        if date_start is not None and date_end is not None:
            where_parts.append("FECHA >= ? AND FECHA <= ?")
            params.extend([pd.Timestamp(date_start), pd.Timestamp(date_end)])
        if allowed_porticos:
            placeholders = ",".join(["?"] * len(allowed_porticos))
            where_parts.append(f"CAST(PORTICO AS VARCHAR) IN ({placeholders})")
            params.extend([str(p) for p in allowed_porticos])
        query = f"SELECT COUNT(*) FROM {table_name}"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        return int(con.execute(query, params).fetchone()[0])
    finally:
        con.close()


def _query_flujos_duckdb_filtered(
    db_path: str,
    *,
    table_name: str,
    row_limit: Optional[int] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    allowed_porticos: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"""
            SELECT FECHA, VELOCIDAD, CATEGORIA, MATRICULA, PORTICO, CARRIL
            FROM {table_name}
        """
        where_parts: List[str] = []
        params: List[Any] = []
        if date_start is not None and date_end is not None:
            where_parts.append("FECHA >= ? AND FECHA <= ?")
            params.extend([pd.Timestamp(date_start), pd.Timestamp(date_end)])
        if allowed_porticos:
            placeholders = ",".join(["?"] * len(allowed_porticos))
            where_parts.append(f"CAST(PORTICO AS VARCHAR) IN ({placeholders})")
            params.extend([str(p) for p in allowed_porticos])
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY FECHA"
        if row_limit is not None and int(row_limit) > 0:
            query += " LIMIT ?"
            params.append(int(row_limit))
        return con.execute(query, params).df()
    finally:
        con.close()


def _build_batch_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    mode: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if end_ts <= start_ts:
        return []
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = pd.Timestamp(start_ts)
    mode = str(mode)
    while current < end_ts:
        if mode == "Diario":
            nxt = current + pd.Timedelta(days=1)
        elif mode == "Semanal":
            nxt = current + pd.Timedelta(days=7)
        else:
            nxt = current + pd.DateOffset(months=1)
        nxt = min(pd.Timestamp(nxt), pd.Timestamp(end_ts))
        ranges.append((current, nxt))
        current = nxt
    return ranges


def _save_feature_duckdb(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    *,
    prefix: str = "drift_flow_features",
) -> Path:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{prefix}_{stamp}.duckdb"
    con = duckdb.connect(str(out_path))
    try:
        con.register("raw_view", raw_df)
        con.execute("CREATE OR REPLACE TABLE raw_features AS SELECT * FROM raw_view")
        con.register("clean_view", clean_df)
        con.execute("CREATE OR REPLACE TABLE clean_features AS SELECT * FROM clean_view")
    finally:
        con.close()
    return out_path


def _build_tramo_selector_for_feature_tab() -> Optional[Tuple[str, str, str, str]]:
    try:
        porticos_df = load_porticos()
    except Exception:
        return None
    if porticos_df is None or porticos_df.empty:
        return None

    required = {"portico", "orden", "eje", "calzada"}
    if not required.issubset(set(porticos_df.columns)):
        return None

    porticos = porticos_df.copy()
    porticos["orden_num"] = pd.to_numeric(porticos["orden"], errors="coerce")
    porticos = porticos.dropna(subset=["orden_num"])
    if porticos.empty:
        return None

    options: List[Tuple[str, Optional[Tuple[str, str, str, str]]]] = [
        ("Toda la autopista", None)
    ]

    for (eje, calzada), grp in porticos.groupby(["eje", "calzada"]):
        grp = grp.sort_values("orden_num")
        p_codes = grp["portico"].astype(str).str.strip().tolist()
        for i in range(len(p_codes) - 1):
            p_start = p_codes[i]
            p_end = p_codes[i + 1]
            label = f"{eje} | {calzada}: {p_start} -> {p_end}"
            options.append((label, (str(eje), str(calzada), str(p_start), str(p_end))))

    labels = [x[0] for x in options]
    selected = st.selectbox("Tramo", labels, index=0, key="drift_feature_tramo_selector")
    tramo_tuple = next((tpl for lbl, tpl in options if lbl == selected), None)
    return tramo_tuple


def _porticos_in_tramo(tramo: Optional[Tuple[str, str, str, str]]) -> Optional[List[str]]:
    if tramo is None:
        return None
    eje, calzada, p_start, p_end = tramo
    try:
        porticos_df = load_porticos()
    except Exception:
        return None
    if porticos_df is None or porticos_df.empty:
        return None
    required = {"portico", "orden", "eje", "calzada"}
    if not required.issubset(set(porticos_df.columns)):
        return None
    porticos = porticos_df.copy()
    porticos["orden_num"] = pd.to_numeric(porticos["orden"], errors="coerce")
    porticos = porticos.dropna(subset=["orden_num"])
    grp = porticos[
        (porticos["eje"].astype(str) == str(eje))
        & (porticos["calzada"].astype(str) == str(calzada))
    ].sort_values("orden_num")
    if grp.empty:
        return None
    codes = grp["portico"].astype(str).str.strip().tolist()
    if p_start not in codes or p_end not in codes:
        return [p_start, p_end]
    i0 = codes.index(p_start)
    i1 = codes.index(p_end)
    lo, hi = sorted([i0, i1])
    return codes[lo : hi + 1]


def generate_synthetic_article_dataset(
    *,
    years: Sequence[int] = (2018, 2019, 2020, 2021, 2022, 2023, 2024),
    rows_per_year: int = 900,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Synthetic dataset for tests and UI demos when real data is unavailable.
    Includes mild concept drift by changing coefficients over years.
    """
    rng = np.random.default_rng(int(random_state))

    rows: List[Dict[str, Any]] = []
    porticos = ["AC-07", "AC-08", "AC-09", "AC-10"]

    for year in years:
        start_ts = pd.Timestamp(f"{int(year)}-01-01 00:00:00")
        timestamps = pd.date_range(start_ts, periods=int(rows_per_year), freq="5min")

        drift_factor = (int(year) - int(years[0])) / max(1, len(years) - 1)
        c1 = 1.8 - 1.1 * drift_factor
        c2 = 0.9 + 0.8 * drift_factor
        base_bias = -3.2 + 0.4 * drift_factor

        x1 = rng.normal(0.0, 1.0, size=len(timestamps))
        x2 = rng.normal(0.0, 1.0, size=len(timestamps))
        x3 = rng.normal(0.0, 1.0, size=len(timestamps))
        flow = rng.gamma(shape=3.0 + 0.3 * drift_factor, scale=1.2, size=len(timestamps))
        speed = 80 + 8 * x1 - 5 * x2 + rng.normal(0, 3, size=len(timestamps))
        density = np.clip(flow / np.maximum(speed, 30), 0, None)

        logits = base_bias + c1 * x1 + c2 * x2 + 0.7 * x3 + 0.3 * density
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, np.clip(prob, 0.001, 0.999)).astype(int)

        # induce long no-accident run in 2020 for filtering tests / realism
        if int(year) == 2020 and len(y) >= 2100:
            y[1200:2100] = 0

        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "interval_start": ts,
                    "portico": porticos[i % len(porticos)],
                    "flow_light": float(flow[i] * 10 + 40),
                    "flow_heavy": float(flow[i] * 2 + 6),
                    "speed_light": float(speed[i]),
                    "speed_heavy": float(speed[i] - rng.normal(4, 1.5)),
                    "density_light": float(density[i] * 3.5),
                    "density_heavy": float(density[i] * 4.2),
                    "delta_speed_light": float(rng.normal(0, 1)),
                    "delta_density_light": float(rng.normal(0, 0.25)),
                    "x1": float(x1[i]),
                    "x2": float(x2[i]),
                    "x3": float(x3[i]),
                    "target": int(y[i]),
                }
            )

    return pd.DataFrame(rows)


def _init_state() -> None:
    st.session_state.setdefault("drift_raw_df", None)
    st.session_state.setdefault("drift_clean_df", None)
    st.session_state.setdefault("drift_stage1_info", None)
    st.session_state.setdefault("drift_zero_runs", None)
    st.session_state.setdefault("drift_prep_summary", None)
    st.session_state.setdefault("drift_feature_cols", None)
    st.session_state.setdefault("drift_importance_df", None)
    st.session_state.setdefault("drift_corr_pairs", None)
    st.session_state.setdefault("drift_results", None)
    st.session_state.setdefault("drift_events_df", None)
    st.session_state.setdefault("drift_events_excluded_df", None)
    st.session_state.setdefault("drift_pipeline_meta", None)
    st.session_state.setdefault("drift_event_files", [])
    st.session_state.setdefault("drift_feature_export_path", None)


def _render_blueprint_tab() -> None:
    st.subheader("Paper Replication Blueprint")
    st.caption(PAPER_TITLE)

    blueprint = article_replication_blueprint()
    st.markdown("**Analyses to replicate**")
    st.dataframe(blueprint["analyses"], width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Figures**")
        st.dataframe(blueprint["figures"], width="stretch")
    with c2:
        st.markdown("**Tables**")
        st.dataframe(blueprint["tables"], width="stretch")

    st.markdown("**Related work (Table 1 replica)**")
    st.dataframe(build_related_work_table(), width="stretch")


def _render_events_tab() -> None:
    st.subheader("Eventos (accidentes)")
    st.markdown("Selecciona uno o varios archivos de eventos desde `Datos/`.")

    event_files = _list_event_files()
    if not event_files:
        st.warning("No se encontraron archivos de eventos en la carpeta Datos.")
        return

    selected_names = st.multiselect(
        "Archivos de eventos disponibles",
        [path.name for path in event_files],
        default=st.session_state.get("drift_event_files") or [path.name for path in event_files],
        key="drift_event_file_select",
    )

    if st.button("Procesar eventos", key="drift_process_events"):
        if not selected_names:
            st.warning("Seleccione al menos un archivo de eventos.")
            return
        try:
            porticos_df = load_porticos()
        except Exception as exc:
            st.error(f"No se pudieron cargar los pórticos: {exc}")
            return

        frames: List[pd.DataFrame] = []
        for name in selected_names:
            path = DATA_DIR / name
            try:
                frames.append(_read_event_csv(path))
            except Exception as exc:
                st.error(f"No se pudo leer {name}: {exc}")
                return

        raw_df = pd.concat(frames, ignore_index=True)
        try:
            accidents_df, excluded_df = process_accidentes_df(
                raw_df, porticos_df, return_excluded=True
            )
        except Exception as exc:
            st.error(f"No se pudieron procesar los eventos: {exc}")
            return

        st.session_state["drift_events_df"] = accidents_df
        st.session_state["drift_events_excluded_df"] = excluded_df
        st.session_state["drift_event_files"] = selected_names
        st.success(
            f"Eventos procesados: {len(accidents_df):,} | "
            f"Excluidos sin pórtico: {len(excluded_df):,}"
        )

    accidents_df = st.session_state.get("drift_events_df")
    if not isinstance(accidents_df, pd.DataFrame) or accidents_df.empty:
        st.info("No hay eventos procesados en memoria.")
        return

    preview_cols = [
        col
        for col in [
            "accidente_time",
            "ultimo_portico",
            "proximo_portico",
            "duracion_accidente",
            "severidad",
            "Km.",
            "Eje",
            "Calzada",
        ]
        if col in accidents_df.columns
    ]

    st.caption(f"Archivos: {', '.join(st.session_state.get('drift_event_files', []))}")
    st.dataframe(accidents_df[preview_cols].head(120), width="stretch")

    excluded_df = st.session_state.get("drift_events_excluded_df")
    if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty:
        st.markdown("**Eventos excluidos por asignacion de pórtico**")
        st.dataframe(excluded_df.head(80), width="stretch")

    st.markdown("**Accidentes por tramo (replica de Crash prediction)**")
    km_col = _find_match_column(accidents_df, ["Km.", "Km", "Kilometro"])
    eje_col = _find_match_column(accidents_df, ["Eje"])
    calzada_col = _find_match_column(accidents_df, ["Calzada"])
    if not km_col or not eje_col or not calzada_col:
        st.info("No se encontraron columnas Km/Eje/Calzada para construir tramos.")
        return

    try:
        porticos_df = load_porticos()
    except Exception as exc:
        st.warning(f"No se pudieron cargar pórticos para construir tramos: {exc}")
        return

    acc_seg = accidents_df[[eje_col, calzada_col, km_col]].copy()
    acc_seg = acc_seg.rename(columns={eje_col: "eje", calzada_col: "calzada", km_col: "km_acc"})
    acc_seg["km_acc"] = pd.to_numeric(
        acc_seg["km_acc"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    acc_seg = acc_seg.dropna(subset=["km_acc", "eje", "calzada"])

    keys: List[Dict[str, Any]] = []
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
        keys.append(
            {
                "Eje": posterior["eje"],
                "Calzada": posterior["calzada"],
                "portico_inicio": _normalize_portico_code(posterior["portico"]),
                "portico_fin": _normalize_portico_code(cercano["portico"]),
            }
        )

    if not keys:
        st.info("No se pudo asignar accidentes a tramos.")
        return

    tramo_df = (
        pd.DataFrame(keys)
        .groupby(["Eje", "Calzada", "portico_inicio", "portico_fin"], dropna=False)
        .size()
        .reset_index(name="accidentes")
        .sort_values(["Eje", "Calzada", "portico_inicio", "portico_fin"])
        .reset_index(drop=True)
    )
    st.dataframe(tramo_df, width="stretch")


def _render_feature_engineering_output() -> None:
    raw_df = st.session_state.get("drift_raw_df")
    clean_df = st.session_state.get("drift_clean_df")
    summary = st.session_state.get("drift_prep_summary")
    pipeline_meta = st.session_state.get("drift_pipeline_meta")

    if isinstance(pipeline_meta, dict):
        steps = pipeline_meta.get("steps") or []
        if steps:
            st.markdown("**Pasos ejecutados**")
            for idx, step in enumerate(steps, start=1):
                st.write(f"{idx}. {step}")

        counts = pipeline_meta.get("counts") or {}
        if counts:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Eventos procesados", f"{int(counts.get('processed_accidents', 0)):,}")
            c2.metric("Filas flujo consultadas", f"{int(counts.get('flow_rows_queried', 0)):,}")
            c3.metric("Filas base (target)", f"{int(counts.get('base_rows_with_target', 0)):,}")
            c4.metric("Filas finales", f"{int(counts.get('final_rows', 0)):,}")

    if isinstance(summary, DataPreparationSummary):
        st.markdown("**Resumen Stage 1 / Stage 2**")
        stage_df = pd.DataFrame(
            [
                {
                    "etapa": "Entrada",
                    "filas": summary.initial_rows,
                    "variables": summary.initial_features,
                },
                {
                    "etapa": "Post Stage 1",
                    "filas": summary.rows_after_missing_drop,
                    "variables": summary.remaining_features_after_stage1,
                },
                {
                    "etapa": "Post Stage 2",
                    "filas": summary.final_rows,
                    "variables": summary.remaining_features_after_stage1,
                },
            ]
        )
        st.dataframe(stage_df, width="stretch")

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        st.markdown("**Preview base (raw_features)**")
        st.dataframe(raw_df.head(60), width="stretch")
    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        st.markdown("**Preview limpia (clean_features)**")
        st.dataframe(clean_df.head(60), width="stretch")

    export_path = st.session_state.get("drift_feature_export_path")
    if export_path:
        st.success(f"Archivo DuckDB generado: {export_path}")


def _render_feature_engineering_tab() -> None:
    st.subheader("Feature engineering")
    st.caption(f"DuckDB fijo: {FLOW_DB_PATH} | tabla: {FLOW_TABLE_NAME}")
    st.markdown(
        "**Flujo de esta pestaña:**\n"
        "1. Cargar eventos procesados (tab Eventos).\n"
        "2. Consultar `Datos/flujos.duckdb` con muestreo y filtro de tramo.\n"
        "3. Generar variables de flujo por intervalo (por lote o en una pasada).\n"
        "4. Construir `target` con accidentes.\n"
        "5. Aplicar Stage 1 y Stage 2 con selectores.\n"
        "6. Exportar resultado en DuckDB (`raw_features` + `clean_features`)."
    )

    has_memory = isinstance(st.session_state.get("drift_clean_df"), pd.DataFrame) and not st.session_state.get(
        "drift_clean_df"
    ).empty
    source_options = ["Cargar existentes", "Calcular nuevas", "En memoria"]
    if "drift_feature_source" not in st.session_state:
        st.session_state["drift_feature_source"] = "En memoria" if has_memory else "Calcular nuevas"
    source = st.radio(
        "Fuente",
        source_options,
        horizontal=True,
        key="drift_feature_source",
    )

    if source == "En memoria":
        if not has_memory:
            st.info("No hay dataset en memoria.")
            return
        _render_feature_engineering_output()
        export_prefix = st.text_input(
            "Prefijo exportación (sin .duckdb)",
            value="drift_flow_features_manual",
            key="drift_export_prefix_memory",
        ).strip()
        if st.button("Exportar DuckDB desde memoria", key="drift_export_memory"):
            try:
                out_path = _save_feature_duckdb(
                    st.session_state["drift_raw_df"],
                    st.session_state["drift_clean_df"],
                    prefix=export_prefix or "drift_flow_features_manual",
                )
            except Exception as exc:
                st.error(f"No se pudo exportar: {exc}")
            else:
                st.session_state["drift_feature_export_path"] = str(out_path)
                st.success(f"Exportado en {out_path}")
        return

    if source == "Cargar existentes":
        files = _list_feature_duckdb_files()
        if not files:
            st.warning("No se encontraron archivos de features en Resultados.")
            return
        names = [path.name for path in files]
        selected = st.selectbox("Archivo DuckDB", options=names, key="drift_load_feature_file")
        if st.button("Cargar archivo", key="drift_load_existing_features"):
            try:
                raw_df, clean_df = _load_feature_payload_from_duckdb(RESULTS_DIR / selected)
            except Exception as exc:
                st.error(f"No se pudo cargar el archivo: {exc}")
                return
            if raw_df.empty and clean_df.empty:
                st.warning("El archivo no contiene tablas de features esperadas.")
                return
            if clean_df.empty:
                clean_df = raw_df.copy()
            for df in [raw_df, clean_df]:
                if "interval_start" in df.columns:
                    df["interval_start"] = pd.to_datetime(df["interval_start"], errors="coerce")
                if "portico" in df.columns:
                    df["portico"] = _normalize_portico_series(df["portico"])
                if "target" in df.columns:
                    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)

            feature_cols = _feature_columns_for_modeling(clean_df, target_col="target")
            st.session_state["drift_raw_df"] = raw_df
            st.session_state["drift_clean_df"] = clean_df
            st.session_state["drift_feature_cols"] = feature_cols
            st.session_state["drift_stage1_info"] = {"remaining_features": feature_cols, "removed_cols": []}
            st.session_state["drift_zero_runs"] = pd.DataFrame()
            st.session_state["drift_prep_summary"] = DataPreparationSummary(
                initial_rows=int(len(raw_df)),
                initial_features=int(len(feature_cols)),
                removed_high_missing_features=0,
                remaining_features_after_stage1=int(len(feature_cols)),
                rows_after_missing_drop=int(len(clean_df)),
                removed_zero_run_rows=0,
                final_rows=int(len(clean_df)),
                zero_run_windows_detected=0,
            )
            st.session_state["drift_pipeline_meta"] = {
                "steps": [f"Archivo existente cargado: {selected}"],
                "counts": {
                    "processed_accidents": int(
                        len(st.session_state.get("drift_events_df"))
                    )
                    if isinstance(st.session_state.get("drift_events_df"), pd.DataFrame)
                    else 0,
                    "flow_rows_queried": int(len(raw_df)),
                    "base_rows_with_target": int(len(raw_df)),
                    "final_rows": int(len(clean_df)),
                },
            }
            st.session_state["drift_feature_export_path"] = str(RESULTS_DIR / selected)
            st.success(f"Archivo cargado: {selected}")
        _render_feature_engineering_output()
        return

    accidents_df = st.session_state.get("drift_events_df")
    if not isinstance(accidents_df, pd.DataFrame) or accidents_df.empty:
        st.info("Primero procesa eventos en la pestaña Eventos.")
        return
    if duckdb is None:
        st.error("duckdb no esta instalado. Ejecuta `pip install duckdb`.")
        return
    if not FLOW_DB_PATH.exists():
        st.error(f"No existe la base de flujos: {FLOW_DB_PATH}")
        return

    try:
        summary = _flow_db_summary(str(FLOW_DB_PATH), FLOW_TABLE_NAME)
    except Exception as exc:
        st.error(f"No se pudo leer {FLOW_DB_PATH}: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{int(summary.get('row_count', 0)):,}")
    c2.metric(
        "Fecha min",
        summary["min_ts"].strftime("%Y-%m-%d %H:%M") if isinstance(summary.get("min_ts"), pd.Timestamp) else "-",
    )
    c3.metric(
        "Fecha max",
        summary["max_ts"].strftime("%Y-%m-%d %H:%M") if isinstance(summary.get("max_ts"), pd.Timestamp) else "-",
    )

    sample_mode = _build_flow_sample_mode_selector(key_prefix="drift_flow")
    sample, percent_mode, range_valid = _build_flow_sample_inputs(summary, sample_mode, key_prefix="drift_flow")

    use_batches = st.checkbox("Procesamiento por lote", value=True, key="drift_use_batches")
    if percent_mode and use_batches:
        st.warning("Muestreo por porcentaje no es compatible con lote; se ejecutará en una sola consulta.")
        use_batches = False
    batch_mode = st.selectbox(
        "Modo de lote",
        ["Diario", "Semanal", "Mensual"],
        index=2,
        disabled=not use_batches,
        key="drift_batch_mode",
    )

    tramo_tuple = _build_tramo_selector_for_feature_tab()
    allowed_porticos = _porticos_in_tramo(tramo_tuple)

    metric_options = {
        "Flow": "flow",
        "Speed": "speed",
        "Speed_std": "speed_std",
        "Density": "density",
        "Delta.Speed": "delta_speed",
        "Delta.Density": "delta_density",
    }
    metrics_selected = st.multiselect(
        "Variables a generar",
        list(metric_options.keys()),
        default=list(metric_options.keys()),
        key="drift_metrics_selected",
    )
    metrics = [metric_options[item] for item in metrics_selected]

    categories = st.multiselect(
        "Tipos de vehiculo",
        ["Light", "Heavy", "Motorcycles"],
        default=["Light", "Heavy", "Motorcycles"],
        key="drift_categories_selected",
    )
    lanes = st.number_input("Carriles para normalizar Flow", min_value=1, max_value=8, value=3, step=1)
    interval_minutes = st.number_input("Intervalo (minutos)", min_value=1, max_value=60, value=5, step=1)

    stage1_mode = st.selectbox(
        "Stage 1: missing data >1% + drop intervalos incompletos",
        ["Aplicar", "Omitir"],
        index=0,
        key="drift_stage1_selector",
    )
    missing_threshold = st.number_input("Umbral Stage 1", min_value=0.0, max_value=1.0, value=0.01, step=0.001)
    stage2_mode = st.selectbox(
        "Stage 2: filtrar ventanas multi-dia sin accidentes",
        ["Aplicar", "Omitir"],
        index=0,
        key="drift_stage2_selector",
    )
    min_zero_days = st.number_input(
        "Dias minimos sin accidentes (Stage 2)",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
    )

    if st.button("Calcular nuevas variables", key="drift_run_feature_engineering", disabled=not range_valid):
        if not metrics:
            st.warning("Selecciona al menos una variable a generar.")
            return
        if not categories:
            st.warning("Selecciona al menos un tipo de vehiculo.")
            return

        flow_date_start = sample.date_start
        flow_date_end = sample.date_end
        if flow_date_start is None and isinstance(summary.get("min_ts"), pd.Timestamp):
            flow_date_start = summary["min_ts"]
        if flow_date_end is None and isinstance(summary.get("max_ts"), pd.Timestamp):
            flow_date_end = summary["max_ts"]
        if flow_date_start is None or flow_date_end is None:
            st.error("No se pudo inferir el rango de fechas para consultar flujos.")
            return

        batch_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]
        if use_batches:
            batch_end_exclusive = pd.Timestamp(flow_date_end) + pd.Timedelta(nanoseconds=1)
            batch_ranges = _build_batch_ranges(pd.Timestamp(flow_date_start), batch_end_exclusive, str(batch_mode))
            if not batch_ranges:
                st.error("No se pudieron construir lotes con el rango seleccionado.")
                return
        else:
            batch_ranges = [(pd.Timestamp(flow_date_start), pd.Timestamp(flow_date_end) + pd.Timedelta(nanoseconds=1))]

        flow_row_limit = int(sample.row_limit) if sample.row_limit is not None else None
        frames: List[pd.DataFrame] = []
        queried_rows = 0
        progress = st.progress(0)

        for idx, (start_ts, end_exclusive) in enumerate(batch_ranges, start=1):
            end_inclusive = pd.Timestamp(end_exclusive) - pd.Timedelta(nanoseconds=1)
            per_batch_limit = flow_row_limit if len(batch_ranges) == 1 else None
            df_batch = _query_flujos_duckdb_filtered(
                str(FLOW_DB_PATH),
                table_name=FLOW_TABLE_NAME,
                row_limit=per_batch_limit,
                date_start=start_ts,
                date_end=end_inclusive,
                allowed_porticos=allowed_porticos,
            )
            queried_rows += int(len(df_batch))
            if df_batch.empty:
                progress.progress(idx / len(batch_ranges))
                continue

            df_batch["FECHA"] = pd.to_datetime(df_batch["FECHA"], errors="coerce")
            df_batch["VELOCIDAD"] = pd.to_numeric(df_batch["VELOCIDAD"], errors="coerce")
            df_batch["CATEGORIA"] = pd.to_numeric(df_batch["CATEGORIA"], errors="coerce")
            df_batch = df_batch.dropna(subset=["FECHA", "CATEGORIA"])
            df_batch["CATEGORIA"] = df_batch["CATEGORIA"].astype(int)
            if df_batch.empty:
                progress.progress(idx / len(batch_ranges))
                continue

            feat_batch = compute_flow_features(
                df_batch,
                interval_minutes=int(interval_minutes),
                lanes=int(lanes),
                metrics=metrics,
                categories=categories,
            )
            if not feat_batch.empty:
                feat_batch["portico"] = _normalize_portico_series(feat_batch["portico"])
                frames.append(feat_batch)

            progress.progress(idx / len(batch_ranges))
        progress.empty()

        if not frames:
            st.warning("No se generaron variables con la configuracion actual.")
            return

        features_df = pd.concat(frames, ignore_index=True)
        features_df = features_df.drop_duplicates(subset=["portico", "interval_start"]).sort_values(
            ["interval_start", "portico"]
        )
        features_df["interval_start"] = pd.to_datetime(features_df["interval_start"], errors="coerce")
        features_df = features_df.dropna(subset=["interval_start", "portico"]).reset_index(drop=True)

        base_df = add_accident_target(
            features_df,
            accidents_df,
            interval_minutes=int(interval_minutes),
            portico_col="portico",
            interval_col="interval_start",
            accident_time_col="accidente_time",
            accident_portico_col="ultimo_portico",
        )
        if base_df.empty:
            st.warning("No se pudo generar base con target.")
            return

        base_df["target"] = pd.to_numeric(base_df["target"], errors="coerce").fillna(0).astype(int)
        feature_cols = _feature_columns_for_modeling(base_df, target_col="target")

        clean_df, prep_summary, stage1_info, zero_runs, prep_steps = run_configurable_preparation_pipeline(
            base_df,
            feature_cols=feature_cols,
            apply_stage1=(stage1_mode == "Aplicar"),
            missing_threshold=float(missing_threshold),
            apply_stage2=(stage2_mode == "Aplicar"),
            min_zero_days=int(min_zero_days),
            target_col="target",
            time_col="interval_start",
            interval_minutes=int(interval_minutes),
        )

        if clean_df.empty:
            st.warning("Las etapas de preparacion dejaron el dataset vacio.")
            return

        try:
            out_path = _save_feature_duckdb(base_df, clean_df, prefix="drift_flow_features")
        except Exception as exc:
            st.error(f"No se pudo exportar el resultado a DuckDB: {exc}")
            return

        flow_count = _count_flujos_rows(
            str(FLOW_DB_PATH),
            table_name=FLOW_TABLE_NAME,
            date_start=flow_date_start,
            date_end=flow_date_end,
            allowed_porticos=allowed_porticos,
        )

        steps = [
            "Eventos procesados desde la pestaña Eventos (Crash prediction replica).",
            f"Consulta base de flujos fija: {FLOW_DB_PATH} / {FLOW_TABLE_NAME}.",
            f"Muestreo aplicado: {sample_mode}.",
            f"Variables generadas: {', '.join(metrics)}.",
            f"Tipos de vehiculo: {', '.join(categories)}.",
        ]
        if tramo_tuple:
            eje, calzada, p_start, p_end = tramo_tuple
            steps.append(f"Filtro tramo: {eje} | {calzada} | {p_start} -> {p_end}.")
        if use_batches:
            steps.append(f"Procesamiento por lote: {batch_mode} ({len(batch_ranges)} lotes).")
        else:
            steps.append("Procesamiento sin lotes (consulta unica).")
        steps.extend(prep_steps)
        steps.append(f"Exportacion DuckDB: {out_path}.")

        st.session_state["drift_raw_df"] = base_df
        st.session_state["drift_clean_df"] = clean_df
        st.session_state["drift_stage1_info"] = stage1_info
        st.session_state["drift_zero_runs"] = zero_runs
        st.session_state["drift_prep_summary"] = prep_summary
        st.session_state["drift_feature_cols"] = stage1_info.get("remaining_features", feature_cols)
        st.session_state["drift_feature_export_path"] = str(out_path)
        st.session_state["drift_pipeline_meta"] = {
            "steps": steps,
            "counts": {
                "processed_accidents": int(len(accidents_df)),
                "flow_rows_available_in_filter": int(flow_count),
                "flow_rows_queried": int(queried_rows),
                "base_rows_with_target": int(len(base_df)),
                "positive_intervals_target": int(base_df["target"].sum()),
                "final_rows": int(len(clean_df)),
            },
        }
        st.success(
            f"Pipeline completado. Base={len(base_df):,} filas | "
            f"Limpio={len(clean_df):,} filas | Exportado={out_path.name}"
        )

    _render_feature_engineering_output()


def _render_feature_selection_tab() -> None:
    st.subheader("Feature Selection")

    clean_df = st.session_state.get("drift_clean_df")
    feature_cols = st.session_state.get("drift_feature_cols")

    if not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
        st.info("Ejecute Feature engineering primero.")
        return
    if not feature_cols:
        st.info("No feature columns selected.")
        return

    target_col = "target" if "target" in clean_df.columns else st.selectbox("Target column", options=list(clean_df.columns))

    corr_threshold = st.number_input("Correlation threshold", min_value=0.5, max_value=0.999, value=0.95, step=0.01)
    top_n = st.number_input("Top-N features", min_value=5, max_value=100, value=20, step=1)

    if st.button("Run correlation + Random Forest selection", key="drift_run_feat_sel"):
        kept, dropped, corr = drop_highly_correlated_features(
            clean_df,
            feature_cols=feature_cols,
            threshold=float(corr_threshold),
        )
        corr_pairs = compute_abs_correlations(clean_df, feature_cols)
        importance_df = rank_features_random_forest(
            clean_df,
            kept,
            target_col=target_col,
            top_n=int(top_n),
        )

        st.session_state["drift_feature_cols"] = kept
        st.session_state["drift_corr_pairs"] = corr_pairs
        st.session_state["drift_importance_df"] = importance_df

        st.success(
            f"Dropped {len(dropped)} highly correlated features. "
            f"Kept {len(kept)} features."
        )

    corr_pairs = st.session_state.get("drift_corr_pairs")
    importance_df = st.session_state.get("drift_importance_df")

    if isinstance(corr_pairs, pd.Series) and not corr_pairs.empty:
        st.markdown("**Figure 5 replica: density of |rho|**")
        hist, edges = np.histogram(corr_pairs.values, bins=60, range=(0.0, 1.0), density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        density_df = pd.DataFrame({"|rho|": centers, "density": hist})
        st.line_chart(density_df.set_index("|rho|")["density"], width="stretch")

    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        st.markdown("**Figure 6 replica: top feature importances**")
        st.bar_chart(importance_df.set_index("feature")["importance"], width="stretch")
        st.dataframe(importance_df, width="stretch")


def _render_experiments_tab() -> None:
    st.subheader("Recalibration Experiments")

    clean_df = st.session_state.get("drift_clean_df")
    feature_cols = st.session_state.get("drift_feature_cols")

    if not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
        st.info("Ejecute Feature engineering primero.")
        return
    if not feature_cols:
        st.info("Run feature selection first.")
        return

    target_col_default = "target" if "target" in clean_df.columns else clean_df.columns[-1]
    time_col_default = "interval_start" if "interval_start" in clean_df.columns else clean_df.columns[0]

    target_col = st.selectbox("Target column", options=list(clean_df.columns), index=list(clean_df.columns).index(target_col_default))
    time_col = st.selectbox("Time column", options=list(clean_df.columns), index=list(clean_df.columns).index(time_col_default))

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_models = st.multiselect("Models", MODEL_NAMES, default=MODEL_NAMES)
        selected_strategies = st.multiselect(
            "Strategies",
            ["static", "period_aligned", "cumulative", "adaptive_adwin"],
            default=["static", "period_aligned", "cumulative", "adaptive_adwin"],
        )
    with col2:
        fast_mode = st.checkbox("Fast hyperparameter mode", value=True)
        grid_limit = st.number_input("Max grid combinations", min_value=1, max_value=2000, value=40, step=1)
        folds = st.number_input("CV folds", min_value=2, max_value=10, value=5, step=1)
    with col3:
        validation_size = st.number_input("Validation size", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        adwin_delta = st.number_input("ADWIN delta", min_value=0.0001, max_value=0.1, value=0.002, step=0.0005, format="%.4f")
        min_window = st.number_input("ADWIN min window", min_value=100, max_value=200000, value=45000, step=100)

    if st.button("Run full paper experiment set", key="drift_run_experiments"):
        with st.spinner("Running experiments. This can take time on large datasets..."):
            results = run_recalibration_experiments(
                clean_df,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_names=selected_models,
                strategies=selected_strategies,
                validation_size=float(validation_size),
                folds=int(folds),
                fast_mode=bool(fast_mode),
                grid_limit=int(grid_limit),
                adwin_delta=float(adwin_delta),
                min_window=int(min_window),
            )
        st.session_state["drift_results"] = results

    results = st.session_state.get("drift_results")
    if not isinstance(results, dict):
        return

    summary = results.get("summary")
    yearly = results.get("yearly_results")
    adaptive = results.get("adaptive_results")
    avg_roc = results.get("average_roc")
    appendix = results.get("appendix_tables")

    st.markdown("**Strategy summary**")
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        st.dataframe(summary, width="stretch")

    st.markdown("**Appendix tables (A.6-A.9) replica**")
    if isinstance(appendix, dict):
        for key in ["A.6", "A.7", "A.8", "A.9"]:
            df_tab = appendix.get(key)
            st.caption(f"Table {key}")
            if isinstance(df_tab, pd.DataFrame) and not df_tab.empty:
                st.dataframe(df_tab, width="stretch")
            else:
                st.info(f"Table {key} has no rows for current run.")

    st.markdown("**Figure 7 replica: average ROC curves**")
    if isinstance(avg_roc, pd.DataFrame) and not avg_roc.empty:
        pivot = avg_roc.pivot_table(index="fpr", columns="label", values="tpr", aggfunc="mean")
        st.line_chart(pivot, width="stretch")
    else:
        st.info("No ROC curves available.")

    if isinstance(yearly, pd.DataFrame) and not yearly.empty:
        st.markdown("**Yearly strategy results**")
        st.dataframe(yearly, width="stretch")

    if isinstance(adaptive, pd.DataFrame) and not adaptive.empty:
        st.markdown("**Adaptive ADWIN drift events**")
        st.dataframe(adaptive, width="stretch")


def _render_coverage_tab() -> None:
    st.subheader("Coverage 100% Check")
    matrix = build_article_coverage_matrix()
    coverage = article_coverage_percentage(matrix)

    c1, c2 = st.columns(2)
    c1.metric("Coverage", f"{coverage:.1f}%")
    c2.metric("Missing items", int((~matrix["implemented"]).sum()))

    if coverage >= 100.0:
        st.success("Coverage verification passed: 100% of article items mapped.")
    else:
        st.warning("Coverage below 100%. Review missing items.")

    st.dataframe(matrix, width="stretch")


def main(set_page_config: bool = False, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="Drift detection", layout="wide")

    _init_state()

    st.title("Drift detection")
    st.caption(
        "Replication workspace for the crash prediction recalibration paper "
        "(static, period-aligned, cumulative, adaptive ADWIN)."
    )

    tab_blueprint, tab_events, tab_feature_eng, tab_selection, tab_experiments, tab_coverage = st.tabs(
        [
            "Blueprint",
            "Eventos",
            "Feature engineering",
            "Feature Selection",
            "Experiments",
            "Coverage",
        ]
    )

    with tab_blueprint:
        _render_blueprint_tab()

    with tab_events:
        _render_events_tab()

    with tab_feature_eng:
        _render_feature_engineering_tab()

    with tab_selection:
        _render_feature_selection_tab()

    with tab_experiments:
        _render_experiments_tab()

    with tab_coverage:
        _render_coverage_tab()

    if show_exit_button and st.sidebar.button("Cerrar app"):
        st.sidebar.write("Cerrando...")


if __name__ == "__main__":
    main(set_page_config=True)
