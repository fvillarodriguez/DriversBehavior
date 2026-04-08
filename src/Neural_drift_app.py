#!/usr/bin/env python3
"""
Neural drift workspace for short-horizon crash prediction with adaptive backtesting.

The module is intentionally self-contained so the Drift detection view can mount it
through a minimal bridge without introducing an import cycle.
"""
from __future__ import annotations

import copy
import importlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None

try:
    from river.drift import ADWIN  # type: ignore
except Exception:
    ADWIN = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"

MODEL_XGBOOST = "XGBoost"
MODEL_TORCH_MLP = "Torch MLP"
AVAILABLE_MODELS = [MODEL_XGBOOST, MODEL_TORCH_MLP]

STRATEGY_FIXED = "fixed"
STRATEGY_RECALIBRATION = "recalibration"
STRATEGY_FINE_TUNING = "fine_tuning"
STRATEGY_RETRAIN = "retrain"
AVAILABLE_STRATEGIES = [
    STRATEGY_FIXED,
    STRATEGY_RECALIBRATION,
    STRATEGY_FINE_TUNING,
    STRATEGY_RETRAIN,
]

DRIFT_INPUT = "input drift"
DRIFT_SCORE = "score drift"
DRIFT_ERROR = "error drift"
DRIFT_EMBEDDING = "embedding drift"
AVAILABLE_DRIFT_CHANNELS = [DRIFT_INPUT, DRIFT_SCORE, DRIFT_ERROR, DRIFT_EMBEDDING]

DRIFT_MONITOR_PROFILE_MODERATE = "Moderado"
DRIFT_MONITOR_PROFILE_SENSITIVE = "Sensible"
AVAILABLE_DRIFT_MONITOR_PROFILES = [
    DRIFT_MONITOR_PROFILE_MODERATE,
    DRIFT_MONITOR_PROFILE_SENSITIVE,
]

SESSION_DEFAULTS: Dict[str, Any] = {
    "neural_drift_config": None,
    "neural_drift_dataset": None,
    "neural_drift_baseline_results": None,
    "neural_drift_stream_results": None,
    "neural_drift_drift_events": None,
    "neural_drift_download_bundle": None,
    "neural_drift_feature_source_choice": None,
    "neural_drift_monitor_profile": DRIFT_MONITOR_PROFILE_MODERATE,
    "neural_drift_last_run_signature": None,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "interval_minutes": 5,
    "lookback_steps": 12,
    "horizon_steps": 1,
    "train_fraction": 0.60,
    "validation_fraction": 0.20,
    "models": [MODEL_XGBOOST, MODEL_TORCH_MLP],
    "strategies": [
        STRATEGY_FIXED,
        STRATEGY_RECALIBRATION,
        STRATEGY_FINE_TUNING,
        STRATEGY_RETRAIN,
    ],
    "drift_channels": [DRIFT_INPUT, DRIFT_SCORE, DRIFT_ERROR, DRIFT_EMBEDDING],
    "severity_threshold": 0.50,
    "recent_window_size": 96,
    "recalibration_min_rows": 32,
    "retrain_min_rows": 64,
    "history_sample_size": 256,
    "rolling_metric_window": 48,
    "max_stream_rows": 600,
    "random_state": 42,
    "threshold_beta": 2.0,
    "xgb_estimators": 80,
    "mlp_hidden_dim": 96,
    "mlp_embedding_dim": 24,
    "mlp_dropout": 0.10,
    "mlp_epochs": 20,
    "mlp_batch_size": 64,
    "mlp_learning_rate": 1e-3,
    "fine_tune_learning_rate": 3e-4,
    "fine_tune_epochs": 6,
    "drift_monitor_hidden_dim": 16,
    "drift_monitor_bottleneck_dim": 6,
    "drift_monitor_dropout": 0.05,
    "drift_monitor_epochs": 12,
    "drift_monitor_batch_size": 32,
    "drift_monitor_learning_rate": 1e-3,
    "drift_monitor_reconstruction_weight": 0.65,
    "drift_monitor_profile": DRIFT_MONITOR_PROFILE_MODERATE,
}

DRIFT_MONITOR_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    DRIFT_MONITOR_PROFILE_MODERATE: {
        "drift_monitor_bottleneck_dim": 6,
        "drift_monitor_reconstruction_weight": 0.65,
    },
    DRIFT_MONITOR_PROFILE_SENSITIVE: {
        "drift_monitor_bottleneck_dim": 4,
        "drift_monitor_reconstruction_weight": 0.85,
    },
}


@dataclass
class WindowDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    metadata: pd.DataFrame
    augmented_df: pd.DataFrame
    base_feature_cols: List[str]
    augmented_feature_cols: List[str]


def init_state() -> None:
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default_value)


def _resolve_drift_monitor_profile(value: Any) -> str:
    profile = str(value or DRIFT_MONITOR_PROFILE_MODERATE)
    if profile not in AVAILABLE_DRIFT_MONITOR_PROFILES:
        return DRIFT_MONITOR_PROFILE_MODERATE
    return profile


def _drift_monitor_profile_preset(profile: str) -> Dict[str, Any]:
    resolved_profile = _resolve_drift_monitor_profile(profile)
    return dict(DRIFT_MONITOR_PROFILE_PRESETS.get(resolved_profile) or {})


def _drift_monitor_profile_description(profile: str) -> str:
    resolved_profile = _resolve_drift_monitor_profile(profile)
    if resolved_profile == DRIFT_MONITOR_PROFILE_SENSITIVE:
        return (
            "Mas sensible a cambios nuevos: usa un bottleneck mas pequeno y da mas peso al "
            "reconstruction error para reaccionar antes a patrones no vistos."
        )
    return (
        "Equilibrio entre sensibilidad y estabilidad: mantiene la configuracion base actual "
        "para detectar drift sin disparar falsas alarmas con demasiada facilidad."
    )


def _apply_drift_monitor_profile_to_session(profile: str) -> None:
    preset = _drift_monitor_profile_preset(profile)
    st.session_state["neural_drift_monitor_profile"] = _resolve_drift_monitor_profile(profile)
    if "drift_monitor_bottleneck_dim" in preset:
        st.session_state["neural_drift_monitor_bottleneck_dim"] = int(preset["drift_monitor_bottleneck_dim"])
    if "drift_monitor_reconstruction_weight" in preset:
        st.session_state["neural_drift_monitor_reconstruction_weight"] = float(
            preset["drift_monitor_reconstruction_weight"]
        )


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def _build_run_signature(dataset_bundle: Dict[str, Any], config: Dict[str, Any]) -> str:
    dataset_df = dataset_bundle.get("df")
    payload = {
        "source": str(dataset_bundle.get("source") or ""),
        "rows": int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0,
        "feature_export_path": str(dataset_bundle.get("feature_export_path") or ""),
        "feature_cols": list(dataset_bundle.get("feature_cols") or []),
        "selection_metadata": _to_json_safe(dataset_bundle.get("selection_metadata") or {}),
        "config": {
            "interval_minutes": int(config.get("interval_minutes", DEFAULT_CONFIG["interval_minutes"])),
            "lookback_steps": int(config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"])),
            "horizon_steps": int(config.get("horizon_steps", DEFAULT_CONFIG["horizon_steps"])),
            "train_fraction": float(config.get("train_fraction", DEFAULT_CONFIG["train_fraction"])),
            "validation_fraction": float(config.get("validation_fraction", DEFAULT_CONFIG["validation_fraction"])),
            "models": list(config.get("models") or []),
            "strategies": list(config.get("strategies") or []),
            "drift_channels": list(config.get("drift_channels") or []),
            "severity_threshold": float(config.get("severity_threshold", DEFAULT_CONFIG["severity_threshold"])),
            "recent_window_size": int(config.get("recent_window_size", DEFAULT_CONFIG["recent_window_size"])),
            "max_stream_rows": int(config.get("max_stream_rows", DEFAULT_CONFIG["max_stream_rows"])),
            "threshold_beta": float(config.get("threshold_beta", DEFAULT_CONFIG["threshold_beta"])),
            "drift_monitor_profile": str(
                config.get("drift_monitor_profile", DEFAULT_CONFIG["drift_monitor_profile"])
            ),
            "drift_monitor_bottleneck_dim": int(
                config.get("drift_monitor_bottleneck_dim", DEFAULT_CONFIG["drift_monitor_bottleneck_dim"])
            ),
            "drift_monitor_reconstruction_weight": float(
                config.get(
                    "drift_monitor_reconstruction_weight",
                    DEFAULT_CONFIG["drift_monitor_reconstruction_weight"],
                )
            ),
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def _import_external_xgboost():
    src_dir = Path(__file__).resolve().parent
    original_sys_path = list(sys.path)
    existing_module = sys.modules.get("xgboost")
    removed_local_module = None
    try:
        if existing_module is not None:
            module_file = Path(str(getattr(existing_module, "__file__", "") or "")).resolve()
            if module_file == (src_dir / "xgboost.py").resolve():
                removed_local_module = sys.modules.pop("xgboost")
        sys.path = [
            entry
            for entry in original_sys_path
            if str(Path(entry or ".").resolve()) != str(src_dir)
        ]
        xgb = importlib.import_module("xgboost")  # type: ignore
    finally:
        sys.path = original_sys_path
        if removed_local_module is not None:
            sys.modules["xgboost"] = removed_local_module

    module_path = Path(str(getattr(xgb, "__file__", "") or "")).resolve()
    if module_path == (src_dir / "xgboost.py").resolve():
        raise ImportError(
            "Se importo el modulo local `src/xgboost.py` en lugar del paquete externo `xgboost`."
        )
    if not hasattr(xgb, "XGBClassifier"):
        raise ImportError("El paquete `xgboost` cargado no expone `XGBClassifier`.")
    return xgb


def _ensure_torch_available() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError("Torch no esta disponible en el entorno activo.")


def _ensure_duckdb_available() -> None:
    if duckdb is None:
        raise ImportError("duckdb no esta disponible en el entorno activo.")


def _ensure_non_empty_dataframe(df: Optional[pd.DataFrame], *, label: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"{label} no contiene filas.")
    return df.copy()


def _load_clean_features_from_duckdb(path: str | Path) -> pd.DataFrame:
    _ensure_duckdb_available()
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "clean_features" not in tables:
            raise ValueError(f"`clean_features` no existe en {db_path.name}.")
        return con.execute("SELECT * FROM clean_features ORDER BY interval_start").df()
    finally:
        con.close()


def _load_raw_features_from_duckdb(path: str | Path) -> Optional[pd.DataFrame]:
    _ensure_duckdb_available()
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "raw_features" not in tables:
            return None
        order_clause = ""
        describe_df = con.execute("DESCRIBE raw_features").df()
        if "interval_start" in describe_df["column_name"].astype(str).tolist():
            order_clause = " ORDER BY interval_start"
        return con.execute(f"SELECT * FROM raw_features{order_clause}").df()
    finally:
        con.close()


def _load_feature_selection_payload_from_duckdb(path: str | Path) -> Optional[Dict[str, Any]]:
    _ensure_duckdb_available()
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        supported_tables = {
            "feature_selection_config",
            "feature_selection_candidates",
            "feature_selection_selected",
        }
        if not (tables & supported_tables):
            return None

        config: Dict[str, Any] = {}
        if "feature_selection_config" in tables:
            cfg_df = con.execute("SELECT * FROM feature_selection_config").df()
            for row in cfg_df.itertuples(index=False):
                try:
                    config[str(row.key)] = json.loads(str(row.value))
                except Exception:
                    config[str(row.key)] = row.value

        candidate_features: List[str] = []
        if "feature_selection_candidates" in tables:
            candidate_df = con.execute(
                "SELECT feature FROM feature_selection_candidates ORDER BY candidate_rank"
            ).df()
            candidate_features = candidate_df["feature"].astype(str).tolist()

        selected_features: List[str] = []
        if "feature_selection_selected" in tables:
            selected_df = con.execute(
                "SELECT feature FROM feature_selection_selected ORDER BY selected_rank"
            ).df()
            selected_features = selected_df["feature"].astype(str).tolist()

        return {
            "candidate_features": candidate_features,
            "selected_features": selected_features,
            "config": config,
        }
    finally:
        con.close()


def list_feature_engineering_duckdb_artifacts(results_dir: str | Path = RESULTS_DIR) -> List[Dict[str, Any]]:
    _ensure_duckdb_available()
    base_dir = Path(results_dir)
    if not base_dir.exists():
        return []

    artifacts: List[Dict[str, Any]] = []
    for path in sorted(base_dir.glob("*.duckdb"), reverse=True):
        try:
            con = duckdb.connect(str(path), read_only=True)
            try:
                tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
                if "clean_features" not in tables:
                    continue
                row_count = int(con.execute("SELECT COUNT(*) FROM clean_features").fetchone()[0] or 0)
            finally:
                con.close()
        except Exception:
            continue

        selection_payload = None
        try:
            selection_payload = _load_feature_selection_payload_from_duckdb(path)
        except Exception:
            selection_payload = None

        selected_features = list((selection_payload or {}).get("selected_features") or [])
        artifacts.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "row_count": row_count,
                "selected_feature_count": int(len(selected_features)),
            }
        )
    return artifacts


def build_dataset_context_for_source_selection(
    context: Dict[str, Any],
    *,
    selected_feature_export_path: Optional[str],
) -> Dict[str, Any]:
    selected_path = str(selected_feature_export_path or "").strip()
    if not selected_path:
        return dict(context)

    selection_payload = _load_feature_selection_payload_from_duckdb(selected_path)
    selected_features = list((selection_payload or {}).get("selected_features") or [])
    candidate_features = list((selection_payload or {}).get("candidate_features") or [])
    config_meta = dict((selection_payload or {}).get("config") or {})

    effective_context = dict(context)
    effective_context["clean_df"] = None
    effective_context["raw_df"] = _load_raw_features_from_duckdb(selected_path)
    effective_context["feature_export_path"] = selected_path
    effective_context["feature_cols"] = list(selected_features or candidate_features)
    effective_context["selection_metadata"] = {
        **dict(context.get("selection_metadata") or {}),
        **config_meta,
        "selected_features": list(selected_features),
        "candidate_features": list(candidate_features),
        "feature_export_path": selected_path,
    }
    return effective_context


def resolve_dataset_from_context(context: Dict[str, Any]) -> Dict[str, Any]:
    clean_df = context.get("clean_df")
    raw_df = context.get("raw_df")
    feature_cols = list(context.get("feature_cols") or [])
    feature_export_path = context.get("feature_export_path")
    selection_metadata = dict(context.get("selection_metadata") or {})

    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        resolved_clean = clean_df.copy()
        source = "session_state"
    elif feature_export_path:
        resolved_clean = _load_clean_features_from_duckdb(feature_export_path)
        source = "duckdb_export"
    else:
        raise ValueError(
            "Neural drift requiere `drift_clean_df` o `feature_export_path` desde Drift detection."
        )

    resolved_clean["interval_start"] = pd.to_datetime(
        resolved_clean.get("interval_start"),
        errors="coerce",
    )
    resolved_clean = resolved_clean.dropna(subset=["interval_start"]).sort_values("interval_start").reset_index(drop=True)

    if "target" not in resolved_clean.columns:
        raise ValueError("El dataset de Neural drift requiere la columna `target`.")
    resolved_clean["target"] = pd.to_numeric(resolved_clean["target"], errors="coerce").fillna(0).astype(int)

    feature_cols = [str(col) for col in feature_cols if col in resolved_clean.columns]
    if not feature_cols:
        excluded = {"interval_start", "target", "portico", "eje", "calzada"}
        feature_cols = [
            str(col)
            for col in resolved_clean.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(resolved_clean[col])
        ]
    if not feature_cols:
        raise ValueError("No se encontraron features numericas para Neural drift.")

    bundle = {
        "source": source,
        "df": resolved_clean,
        "raw_df": raw_df.copy() if isinstance(raw_df, pd.DataFrame) else None,
        "feature_cols": feature_cols,
        "selection_metadata": selection_metadata,
        "feature_export_path": None if not feature_export_path else str(feature_export_path),
    }
    return bundle


def _mean_from_available(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:
    cols = [col for col in candidates if col in df.columns]
    if not cols:
        return None
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1)


def augment_feature_frame(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    time_col: str = "interval_start",
) -> Tuple[pd.DataFrame, List[str]]:
    work = _ensure_non_empty_dataframe(df, label="Feature frame")
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    base_cols = [str(col) for col in feature_cols if col in work.columns]
    if not base_cols:
        raise ValueError("No base feature columns available for feature augmentation.")

    engineered_cols: List[str] = list(base_cols)
    derived_columns: Dict[str, Any] = {}
    for col in base_cols:
        series = pd.to_numeric(work[col], errors="coerce").astype(float)
        diff_col = f"{col}__diff1"
        rate_col = f"{col}__rate1"
        mean_col = f"{col}__roll_mean3"
        std_col = f"{col}__roll_std3"
        cv_col = f"{col}__roll_cv3"

        prev = series.shift(1).replace(0.0, np.nan)
        roll = series.rolling(3, min_periods=2)
        diff_values = series.diff()
        mean_values = roll.mean()
        std_values = roll.std()
        derived_columns[diff_col] = diff_values
        derived_columns[rate_col] = (series - prev) / prev.abs()
        derived_columns[mean_col] = mean_values
        derived_columns[std_col] = std_values
        derived_columns[cv_col] = np.where(
            pd.to_numeric(mean_values, errors="coerce").abs() > 1e-9,
            pd.to_numeric(std_values, errors="coerce") / pd.to_numeric(mean_values, errors="coerce").abs(),
            np.nan,
        )
        engineered_cols.extend([diff_col, rate_col, mean_col, std_col, cv_col])

    speed_cols = [col for col in base_cols if "speed" in col.lower() or col.lower().startswith("vel")]
    flow_cols = [col for col in base_cols if "flow" in col.lower()]
    density_cols = [col for col in base_cols if "density" in col.lower() or col.lower().startswith("den")]
    heavy_cols = [col for col in base_cols if "heavy" in col.lower()]

    speed_mean = _mean_from_available(work, speed_cols)
    flow_mean = _mean_from_available(work, flow_cols)
    density_mean = _mean_from_available(work, density_cols)
    heavy_mean = _mean_from_available(work, heavy_cols)

    if speed_mean is not None:
        derived_columns["shock_speed_drop"] = (-speed_mean.diff()).clip(lower=0.0)
        derived_columns["speed_level"] = speed_mean
        engineered_cols.extend(["shock_speed_drop", "speed_level"])
    if density_mean is not None:
        derived_columns["shock_density_jump"] = density_mean.diff().clip(lower=0.0)
        derived_columns["density_level"] = density_mean
        engineered_cols.extend(["shock_density_jump", "density_level"])
    if flow_mean is not None and density_mean is not None:
        derived_columns["flow_density_ratio"] = flow_mean / (density_mean.abs() + 1e-6)
        engineered_cols.append("flow_density_ratio")
    if speed_mean is not None and density_mean is not None:
        derived_columns["speed_density_interaction"] = speed_mean * density_mean
        derived_columns["relative_congestion"] = density_mean / (speed_mean.abs() + 1e-6)
        engineered_cols.extend(["speed_density_interaction", "relative_congestion"])
    if heavy_mean is not None and speed_mean is not None:
        derived_columns["heavy_speed_interaction"] = heavy_mean * speed_mean
        engineered_cols.append("heavy_speed_interaction")

    hour = work[time_col].dt.hour + work[time_col].dt.minute / 60.0
    derived_columns["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    derived_columns["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    derived_columns["day_of_week"] = work[time_col].dt.dayofweek.astype(float)
    derived_columns["is_weekend"] = work[time_col].dt.dayofweek.isin([5, 6]).astype(float)
    engineered_cols.extend(["hour_sin", "hour_cos", "day_of_week", "is_weekend"])

    if derived_columns:
        derived_df = pd.DataFrame(derived_columns, index=work.index)
        work = pd.concat([work, derived_df], axis=1)
        work = work.copy()

    engineered_cols = [col for col in dict.fromkeys(engineered_cols) if col in work.columns]
    for col in engineered_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    return work, engineered_cols


def _streamlit_arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_object_dtype(work[col]):
            work[col] = work[col].astype("string")
    return work


def _future_target_from_interval_target(target: pd.Series, horizon_steps: int) -> pd.Series:
    work = pd.to_numeric(target, errors="coerce").fillna(0).astype(float)
    horizon = max(1, int(horizon_steps))
    shifted_targets = [work.shift(-offset) for offset in range(horizon)]
    combined = shifted_targets[0].copy()
    for shifted in shifted_targets[1:]:
        combined = np.maximum(combined, shifted)
    return pd.Series(combined, index=work.index)


def _flatten_feature_names(feature_cols: Sequence[str], lookback_steps: int) -> List[str]:
    names: List[str] = []
    for offset in range(int(lookback_steps) - 1, -1, -1):
        for col in feature_cols:
            names.append(f"{col}[t-{offset}]")
    return names


def build_window_dataset(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    interval_minutes: int = 5,
    lookback_steps: int = 12,
    horizon_steps: int = 1,
    target_col: str = "target",
    time_col: str = "interval_start",
) -> WindowDataset:
    work = _ensure_non_empty_dataframe(df, label="Window dataset")
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if target_col not in work.columns:
        raise ValueError(f"Window dataset requires `{target_col}`.")

    future_target = _future_target_from_interval_target(work[target_col], horizon_steps=horizon_steps)
    matrix = work.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")

    lookback = max(1, int(lookback_steps))
    feature_names = _flatten_feature_names(feature_cols, lookback)
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    meta_rows: List[Dict[str, Any]] = []

    for end_idx in range(lookback - 1, len(work)):
        y_value = future_target.iloc[end_idx]
        if pd.isna(y_value):
            continue
        window = matrix.iloc[end_idx - lookback + 1 : end_idx + 1].to_numpy(dtype=float, copy=True)
        X_rows.append(window.reshape(-1))
        prediction_time = pd.Timestamp(work.loc[end_idx, time_col])
        meta_rows.append(
            {
                "sample_index": len(meta_rows),
                "prediction_time": prediction_time,
                "window_start": pd.Timestamp(work.loc[end_idx - lookback + 1, time_col]),
                "window_end": prediction_time,
                "horizon_end": prediction_time + pd.Timedelta(minutes=int(interval_minutes) * int(horizon_steps)),
                "label": int(float(y_value) >= 0.5),
            }
        )
        y_rows.append(int(float(y_value) >= 0.5))

    if not X_rows:
        raise ValueError("No fue posible construir secuencias con la configuracion actual.")

    metadata = pd.DataFrame(meta_rows)
    return WindowDataset(
        X=np.vstack(X_rows).astype(float),
        y=np.asarray(y_rows, dtype=int),
        feature_names=feature_names,
        metadata=metadata,
        augmented_df=work,
        base_feature_cols=list(feature_cols),
        augmented_feature_cols=list(feature_cols),
    )


def _split_window_dataset(
    dataset: WindowDataset,
    *,
    train_fraction: float,
    validation_fraction: float,
    max_stream_rows: Optional[int] = None,
) -> Dict[str, Any]:
    n_samples = int(len(dataset.y))
    if n_samples < 30:
        raise ValueError("Neural drift requiere al menos 30 secuencias para backtesting.")

    train_end = max(1, int(math.floor(n_samples * float(train_fraction))))
    validation_count = max(1, int(math.floor(n_samples * float(validation_fraction))))
    validation_end = min(n_samples - 1, train_end + validation_count)
    if validation_end <= train_end:
        validation_end = min(n_samples - 1, train_end + 1)

    X_train = dataset.X[:train_end]
    y_train = dataset.y[:train_end]
    X_val = dataset.X[train_end:validation_end]
    y_val = dataset.y[train_end:validation_end]
    X_stream = dataset.X[validation_end:]
    y_stream = dataset.y[validation_end:]
    metadata_train = dataset.metadata.iloc[:train_end].reset_index(drop=True)
    metadata_val = dataset.metadata.iloc[train_end:validation_end].reset_index(drop=True)
    metadata_stream = dataset.metadata.iloc[validation_end:].reset_index(drop=True)

    if max_stream_rows is not None and len(y_stream) > int(max_stream_rows):
        X_stream = X_stream[-int(max_stream_rows) :]
        y_stream = y_stream[-int(max_stream_rows) :]
        metadata_stream = metadata_stream.iloc[-int(max_stream_rows) :].reset_index(drop=True)

    if len(y_train) < 10 or len(y_val) < 5 or len(y_stream) < 5:
        raise ValueError("El split temporal no dejo suficientes muestras para train/validation/stream.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_stream": X_stream,
        "y_stream": y_stream,
        "metadata_train": metadata_train,
        "metadata_val": metadata_val,
        "metadata_stream": metadata_stream,
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def _classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    preds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    pred_array = (
        np.asarray(preds).astype(int)
        if preds is not None
        else (s >= float(threshold)).astype(int)
    )
    tn, fp, fn, tp = confusion_matrix(y, pred_array, labels=[0, 1]).ravel()
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
    brier = float(np.mean((s - y) ** 2))
    return {
        "roc_auc": _safe_auc(y, s),
        "pr_auc": _safe_pr_auc(y, s),
        "f1": float(f1_score(y, pred_array, zero_division=0)),
        "recall": recall,
        "specificity": specificity,
        "fnr": fnr,
        "brier": brier,
        "threshold": float(threshold),
        "positives": int(y.sum()),
        "rows": int(len(y)),
    }


def _optimize_decision_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    beta: float = 2.0,
) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = np.clip(s[mask], 0.0, 1.0)
    beta_sq = float(beta) ** 2

    if len(s) < 10 or len(np.unique(y)) < 2:
        return {
            "threshold": 0.5,
            "f_beta": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "beta": float(beta),
            "n_candidates": 0,
        }

    work = pd.DataFrame({"score": s, "y": y}).sort_values("score", ascending=False)
    grouped = (
        work.groupby("score", sort=False)["y"]
        .agg(total="size", positives="sum")
        .reset_index()
    )
    grouped["negatives"] = grouped["total"] - grouped["positives"]
    grouped["tp"] = grouped["positives"].cumsum()
    grouped["fp"] = grouped["negatives"].cumsum()
    total_pos = float(grouped["positives"].sum())
    total_neg = float(grouped["negatives"].sum())
    grouped["fn"] = total_pos - grouped["tp"]
    grouped["tn"] = total_neg - grouped["fp"]
    grouped["precision"] = np.where(
        (grouped["tp"] + grouped["fp"]) > 0,
        grouped["tp"] / (grouped["tp"] + grouped["fp"]),
        0.0,
    )
    grouped["recall"] = np.where(total_pos > 0, grouped["tp"] / total_pos, 0.0)
    grouped["specificity"] = np.where(total_neg > 0, grouped["tn"] / total_neg, 0.0)
    grouped["f_beta"] = np.where(
        (beta_sq * grouped["precision"] + grouped["recall"]) > 0,
        (1.0 + beta_sq) * grouped["precision"] * grouped["recall"]
        / (beta_sq * grouped["precision"] + grouped["recall"]),
        0.0,
    )
    grouped = grouped.sort_values(
        ["f_beta", "recall", "precision", "specificity", "score"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    best = grouped.iloc[0]
    return {
        "threshold": float(best["score"]),
        "f_beta": float(best["f_beta"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "specificity": float(best["specificity"]),
        "beta": float(beta),
        "n_candidates": int(len(grouped)),
    }


def _fit_platt_calibrator(y_true: np.ndarray, raw_scores: np.ndarray) -> Optional[LogisticRegression]:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(raw_scores).astype(float)
    if len(scores) < 10 or len(np.unique(y)) < 2:
        return None
    model = LogisticRegression(max_iter=500)
    try:
        model.fit(scores.reshape(-1, 1), y)
    except Exception:
        return None
    return model


def _apply_calibrator(raw_scores: np.ndarray, calibrator: Optional[LogisticRegression]) -> np.ndarray:
    scores = np.asarray(raw_scores).astype(float)
    if calibrator is None:
        return np.clip(scores, 0.0, 1.0)
    try:
        return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    except Exception:
        return np.clip(scores, 0.0, 1.0)


def _fit_imputer(X: np.ndarray) -> np.ndarray:
    work = np.asarray(X, dtype=float)
    medians = np.nanmedian(work, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return medians.astype(float)


def _apply_imputer(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    work = np.asarray(X, dtype=float).copy()
    nan_mask = ~np.isfinite(work)
    if nan_mask.any():
        work[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    return work.astype(float)


def _set_random_seed(seed: int) -> None:
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


class WindowMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward_with_embeddings(self, x):
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits, embeddings

    def forward(self, x):
        logits, _ = self.forward_with_embeddings(x)
        return logits


class EmbeddingDriftAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


def _temporal_train_val_split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) < 5:
        return X, X, y, y
    val_count = max(1, int(math.ceil(len(y) * float(validation_fraction))))
    val_start = max(1, len(y) - val_count)
    X_train = X[:val_start]
    y_train = y[:val_start]
    X_val = X[val_start:]
    y_val = y[val_start:]
    return X_train, X_val, y_train, y_val


def _train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    base_model_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ensure_torch_available()
    _set_random_seed(int(config.get("random_state", 42)))

    imputer = _fit_imputer(X_train)
    X_train_imp = _apply_imputer(X_train, imputer)
    X_val_imp = _apply_imputer(X_val, imputer)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    device = torch.device("cpu")
    model = WindowMLP(
        input_dim=int(X_train_scaled.shape[1]),
        hidden_dim=int(config.get("mlp_hidden_dim", DEFAULT_CONFIG["mlp_hidden_dim"])),
        embedding_dim=int(config.get("mlp_embedding_dim", DEFAULT_CONFIG["mlp_embedding_dim"])),
        dropout=float(config.get("mlp_dropout", DEFAULT_CONFIG["mlp_dropout"])),
    ).to(device)
    if base_model_state is not None:
        model.load_state_dict(base_model_state)

    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    pos_weight = torch.tensor([float(neg / pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate or config.get("mlp_learning_rate", DEFAULT_CONFIG["mlp_learning_rate"])),
    )

    batch_size = int(config.get("mlp_batch_size", DEFAULT_CONFIG["mlp_batch_size"]))
    max_epochs = int(epochs or config.get("mlp_epochs", DEFAULT_CONFIG["mlp_epochs"]))

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.astype(np.float32), dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=max(8, batch_size), shuffle=True)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val.astype(np.float32), dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 4
    stale_epochs = 0
    for _epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    raw_val_scores, embeddings = _predict_torch_mlp(
        {
            "kind": "torch_mlp",
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
        },
        X_val,
    )
    calibrator = _fit_platt_calibrator(y_val, raw_val_scores)
    calibrated_val_scores = _apply_calibrator(raw_val_scores, calibrator)
    threshold_info = _optimize_decision_threshold(
        y_val,
        calibrated_val_scores,
        beta=float(config.get("threshold_beta", DEFAULT_CONFIG["threshold_beta"])),
    )
    embedding_monitor, reconstruction_errors = _fit_embedding_monitor(embeddings, config=config)
    reference = _build_reference_stats(
        X_ref=_apply_imputer(X_val, imputer),
        y_ref=y_val,
        calibrated_scores=calibrated_val_scores,
        embeddings=embeddings,
        reconstruction_errors=reconstruction_errors,
    )
    return {
        "kind": "torch_mlp",
        "model_name": MODEL_TORCH_MLP,
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "calibrator": calibrator,
        "reference": reference,
        "embedding_monitor": embedding_monitor,
        "decision_threshold": float(threshold_info["threshold"]),
        "threshold_info": threshold_info,
        "base_threshold": float(threshold_info["threshold"]),
        "base_model_state": copy.deepcopy(model.state_dict()),
    }


def _predict_torch_mlp(artifact: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_torch_available()
    model = artifact["model"]
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]

    X_imp = _apply_imputer(X, imputer)
    X_scaled = scaler.transform(X_imp)
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        logits, embeddings = model.forward_with_embeddings(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        emb = embeddings.cpu().numpy()
    return probs.astype(float), emb.astype(float)


def _train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    xgb = _import_external_xgboost()
    imputer = _fit_imputer(X_train)
    X_train_imp = _apply_imputer(X_train, imputer)
    X_val_imp = _apply_imputer(X_val, imputer)

    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    estimator = xgb.XGBClassifier(
        n_estimators=int(config.get("xgb_estimators", DEFAULT_CONFIG["xgb_estimators"])),
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=int(config.get("random_state", 42)),
        scale_pos_weight=float(neg / pos),
    )
    estimator.fit(X_train_imp, y_train, verbose=False)
    raw_val_scores = estimator.predict_proba(X_val_imp)[:, 1].astype(float)
    calibrator = _fit_platt_calibrator(y_val, raw_val_scores)
    calibrated_val_scores = _apply_calibrator(raw_val_scores, calibrator)
    threshold_info = _optimize_decision_threshold(
        y_val,
        calibrated_val_scores,
        beta=float(config.get("threshold_beta", DEFAULT_CONFIG["threshold_beta"])),
    )
    reference = _build_reference_stats(
        X_ref=X_val_imp,
        y_ref=y_val,
        calibrated_scores=calibrated_val_scores,
        embeddings=None,
    )
    return {
        "kind": "xgboost",
        "model_name": MODEL_XGBOOST,
        "model": estimator,
        "imputer": imputer,
        "calibrator": calibrator,
        "reference": reference,
        "decision_threshold": float(threshold_info["threshold"]),
        "threshold_info": threshold_info,
        "base_threshold": float(threshold_info["threshold"]),
        "history_training_rows": int(len(y_train)),
    }


def _predict_xgboost(artifact: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model = artifact["model"]
    imputer = artifact["imputer"]
    X_imp = _apply_imputer(X, imputer)
    probs = model.predict_proba(X_imp)[:, 1].astype(float)
    return probs, None


def _predict_with_artifact(artifact: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kind = str(artifact.get("kind"))
    if kind == "torch_mlp":
        probs, embeddings = _predict_torch_mlp(artifact, X)
        return probs, embeddings
    probs, _ = _predict_xgboost(artifact, X)
    return probs, np.empty((len(probs), 0), dtype=float)


def _train_model_artifact(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if str(model_name) == MODEL_TORCH_MLP:
        return _train_torch_mlp(X_train, y_train, X_val, y_val, config=config)
    if str(model_name) == MODEL_XGBOOST:
        return _train_xgboost_model(X_train, y_train, X_val, y_val, config=config)
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_embedding_monitor(
    embeddings: np.ndarray,
    *,
    config: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    _ensure_torch_available()
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    if emb.size == 0 or emb.shape[0] < 8 or emb.shape[1] < 2:
        return None, None

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    input_dim = int(emb_scaled.shape[1])
    bottleneck_dim = max(
        2,
        min(
            int(config.get("drift_monitor_bottleneck_dim", DEFAULT_CONFIG["drift_monitor_bottleneck_dim"])),
            max(2, input_dim - 1),
        ),
    )
    hidden_dim = max(
        bottleneck_dim + 1,
        int(config.get("drift_monitor_hidden_dim", DEFAULT_CONFIG["drift_monitor_hidden_dim"])),
    )
    dropout = float(config.get("drift_monitor_dropout", DEFAULT_CONFIG["drift_monitor_dropout"]))
    batch_size = max(8, int(config.get("drift_monitor_batch_size", DEFAULT_CONFIG["drift_monitor_batch_size"])))
    max_epochs = max(2, int(config.get("drift_monitor_epochs", DEFAULT_CONFIG["drift_monitor_epochs"])))
    learning_rate = float(config.get("drift_monitor_learning_rate", DEFAULT_CONFIG["drift_monitor_learning_rate"]))

    X_train, X_val, _, _ = _temporal_train_val_split_arrays(
        emb_scaled,
        np.zeros(len(emb_scaled), dtype=int),
        validation_fraction=0.2,
    )

    device = torch.device("cpu")
    model = EmbeddingDriftAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 3
    stale_epochs = 0
    for _epoch in range(max_epochs):
        model.train()
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(X_batch)
            loss = criterion(reconstruction, X_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_reconstruction = model(X_val_tensor)
            val_loss = float(criterion(val_reconstruction, X_val_tensor).item())
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    reconstruction_errors = _predict_embedding_monitor(
        {
            "model": model,
            "scaler": scaler,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "dropout": dropout,
        },
        emb,
    )
    return (
        {
            "model": model,
            "scaler": scaler,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "dropout": dropout,
        },
        reconstruction_errors,
    )


def _predict_embedding_monitor(monitor: Dict[str, Any], embeddings: np.ndarray) -> np.ndarray:
    _ensure_torch_available()
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if emb.size == 0:
        return np.asarray([], dtype=float)

    scaler = monitor["scaler"]
    model = monitor["model"]
    emb_scaled = scaler.transform(emb)
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(emb_scaled, dtype=torch.float32, device=device)
        reconstruction = model(tensor).cpu().numpy()
    reconstruction_error = np.mean(np.square(reconstruction - emb_scaled), axis=1)
    return reconstruction_error.astype(float)


def _build_reference_stats(
    *,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    calibrated_scores: np.ndarray,
    embeddings: Optional[np.ndarray],
    reconstruction_errors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    X_work = np.asarray(X_ref, dtype=float)
    y_work = np.asarray(y_ref).astype(int)
    scores = np.asarray(calibrated_scores, dtype=float)

    feature_mean = np.nanmean(X_work, axis=0)
    feature_std = np.nanstd(X_work, axis=0)
    feature_std = np.where(np.isfinite(feature_std) & (feature_std > 1e-6), feature_std, 1.0)

    input_stats = np.mean(np.abs((X_work - feature_mean) / feature_std), axis=1)
    error_stats = (scores - y_work) ** 2

    reference: Dict[str, Any] = {
        "feature_mean": feature_mean.astype(float),
        "feature_std": feature_std.astype(float),
        "input_stat_mean": float(np.nanmean(input_stats)),
        "input_stat_std": float(np.nanstd(input_stats) + 1e-6),
        "score_mean": float(np.nanmean(scores)),
        "score_std": float(np.nanstd(scores) + 1e-6),
        "error_mean": float(np.nanmean(error_stats)),
        "error_std": float(np.nanstd(error_stats) + 1e-6),
    }

    if embeddings is not None and np.asarray(embeddings).size > 0:
        emb = np.asarray(embeddings, dtype=float)
        centroid = emb.mean(axis=0)
        distances = np.linalg.norm(emb - centroid, axis=1)
        reference["embedding_centroid"] = centroid.astype(float)
        reference["embedding_distance_mean"] = float(np.nanmean(distances))
        reference["embedding_distance_std"] = float(np.nanstd(distances) + 1e-6)
    if reconstruction_errors is not None and np.asarray(reconstruction_errors).size > 0:
        monitor_errors = np.asarray(reconstruction_errors, dtype=float)
        reference["embedding_reconstruction_mean"] = float(np.nanmean(monitor_errors))
        reference["embedding_reconstruction_std"] = float(np.nanstd(monitor_errors) + 1e-6)
    return reference


def _refresh_reference_from_recent(
    artifact: Dict[str, Any],
    recent_X: np.ndarray,
    recent_y: np.ndarray,
    *,
    config: Dict[str, Any],
) -> None:
    raw_scores, embeddings = _predict_with_artifact(artifact, recent_X)
    calibrated_scores = _apply_calibrator(raw_scores, artifact.get("calibrator"))
    X_ref = _apply_imputer(recent_X, artifact["imputer"])
    monitor = artifact.get("embedding_monitor")
    reconstruction_errors = None
    if embeddings.size > 0:
        monitor, reconstruction_errors = _fit_embedding_monitor(embeddings, config=config)
        artifact["embedding_monitor"] = monitor
    artifact["reference"] = _build_reference_stats(
        X_ref=X_ref,
        y_ref=recent_y,
        calibrated_scores=calibrated_scores,
        embeddings=embeddings if embeddings.size else None,
        reconstruction_errors=reconstruction_errors,
    )


def _normalize_score(value: float, mean: float, std: float) -> float:
    scale = max(1e-6, float(std))
    z_score = abs(float(value) - float(mean)) / scale
    return float(np.clip(z_score / 3.0, 0.0, 1.0))


class ClassicDriftDetector:
    def __init__(self, *, delta: float = 0.002, rolling_window: int = 32) -> None:
        self._rolling_window = max(8, int(rolling_window))
        self._history: List[float] = []
        self._river_detector = ADWIN(delta=float(delta)) if ADWIN is not None else None

    def update(self, value: float) -> bool:
        numeric_value = float(value)
        self._history.append(numeric_value)
        if len(self._history) > self._rolling_window:
            self._history = self._history[-self._rolling_window :]
        if self._river_detector is not None:
            self._river_detector.update(numeric_value)
            return bool(getattr(self._river_detector, "drift_detected", False))
        if len(self._history) < self._rolling_window:
            return False
        baseline = np.asarray(self._history[:-1], dtype=float)
        mean = float(np.mean(baseline))
        std = float(np.std(baseline) + 1e-6)
        return abs(numeric_value - mean) > 3.0 * std


def _severity_label(score: float) -> str:
    if float(score) < 0.34:
        return "leve"
    if float(score) < 0.67:
        return "moderado"
    return "severo"


def _build_channel_scores(
    *,
    artifact: Dict[str, Any],
    x_row: np.ndarray,
    calibrated_score: float,
    y_true: int,
    embeddings: np.ndarray,
    selected_channels: Sequence[str],
    detectors: Dict[str, ClassicDriftDetector],
    channel_histories: Optional[Dict[str, List[float]]] = None,
    recent_window_size: int = 96,
    embedding_reconstruction_weight: float = 0.65,
) -> Dict[str, Any]:
    reference = dict(artifact.get("reference") or {})
    available_scores: Dict[str, float] = {}
    detector_flags: Dict[str, bool] = {}
    raw_channel_values: Dict[str, float] = {}

    history_limit = max(8, int(recent_window_size))

    def _window_stat(channel_name: str, value: float) -> float:
        if channel_histories is None:
            return float(value)
        history = channel_histories.setdefault(str(channel_name), [])
        history.append(float(value))
        if len(history) > history_limit:
            del history[:-history_limit]
        return float(np.mean(history))

    if DRIFT_INPUT in selected_channels:
        feature_mean = np.asarray(reference.get("feature_mean"), dtype=float)
        feature_std = np.asarray(reference.get("feature_std"), dtype=float)
        input_value = float(np.mean(np.abs((x_row - feature_mean) / np.maximum(feature_std, 1e-6))))
        raw_channel_values[DRIFT_INPUT] = input_value
        input_window_value = _window_stat(DRIFT_INPUT, input_value)
        input_score = _normalize_score(
            input_window_value,
            float(reference.get("input_stat_mean", 0.0)),
            float(reference.get("input_stat_std", 1.0)),
        )
        input_drift = detectors[DRIFT_INPUT].update(input_value)
        available_scores[DRIFT_INPUT] = float(max(input_score, 1.0 if input_drift else 0.0))
        detector_flags[DRIFT_INPUT] = bool(input_drift)

    if DRIFT_SCORE in selected_channels:
        score_value = float(calibrated_score)
        raw_channel_values[DRIFT_SCORE] = score_value
        score_window_value = _window_stat(DRIFT_SCORE, score_value)
        score_score = _normalize_score(
            score_window_value,
            float(reference.get("score_mean", 0.5)),
            float(reference.get("score_std", 0.15)),
        )
        score_drift = detectors[DRIFT_SCORE].update(score_value)
        available_scores[DRIFT_SCORE] = float(max(score_score, 1.0 if score_drift else 0.0))
        detector_flags[DRIFT_SCORE] = bool(score_drift)

    if DRIFT_ERROR in selected_channels:
        error_value = float((float(calibrated_score) - int(y_true)) ** 2)
        raw_channel_values[DRIFT_ERROR] = error_value
        error_window_value = _window_stat(DRIFT_ERROR, error_value)
        error_score = _normalize_score(
            error_window_value,
            float(reference.get("error_mean", 0.0)),
            float(reference.get("error_std", 0.1)),
        )
        error_drift = detectors[DRIFT_ERROR].update(error_value)
        available_scores[DRIFT_ERROR] = float(max(error_score, 1.0 if error_drift else 0.0))
        detector_flags[DRIFT_ERROR] = bool(error_drift)

    if DRIFT_EMBEDDING in selected_channels and embeddings.size > 0 and "embedding_centroid" in reference:
        centroid = np.asarray(reference["embedding_centroid"], dtype=float)
        embedding_distance = float(np.linalg.norm(embeddings.reshape(-1) - centroid))
        raw_channel_values["embedding_distance"] = embedding_distance

        distance_window_value = _window_stat("embedding_distance", embedding_distance)
        distance_score = _normalize_score(
            distance_window_value,
            float(reference.get("embedding_distance_mean", 0.0)),
            float(reference.get("embedding_distance_std", 0.1)),
        )

        component_scores = [distance_score]
        weighted_sum = (1.0 - float(np.clip(embedding_reconstruction_weight, 0.0, 1.0))) * distance_score
        weight_total = (1.0 - float(np.clip(embedding_reconstruction_weight, 0.0, 1.0)))

        monitor = artifact.get("embedding_monitor")
        reconstruction_score = None
        if monitor is not None:
            reconstruction_error = float(_predict_embedding_monitor(monitor, embeddings.reshape(1, -1))[0])
            raw_channel_values["embedding_reconstruction_error"] = reconstruction_error
            reconstruction_window_value = _window_stat("embedding_reconstruction_error", reconstruction_error)
            reconstruction_score = _normalize_score(
                reconstruction_window_value,
                float(reference.get("embedding_reconstruction_mean", 0.0)),
                float(reference.get("embedding_reconstruction_std", 0.1)),
            )
            component_scores.append(reconstruction_score)
            recon_weight = float(np.clip(embedding_reconstruction_weight, 0.0, 1.0))
            weighted_sum += recon_weight * reconstruction_score
            weight_total += recon_weight

        embedding_score = float(weighted_sum / max(weight_total, 1e-6))
        raw_channel_values[DRIFT_EMBEDDING] = float(
            raw_channel_values.get("embedding_reconstruction_error", embedding_distance)
        )
        available_scores[DRIFT_EMBEDDING] = embedding_score
        detector_flags[DRIFT_EMBEDDING] = bool(
            embedding_score >= 0.80
            or any(float(score) >= 0.85 for score in component_scores)
        )

    severity = float(np.mean(list(available_scores.values()))) if available_scores else 0.0
    max_channel_score = float(max(available_scores.values())) if available_scores else 0.0
    return {
        "channel_scores": available_scores,
        "raw_channel_values": raw_channel_values,
        "detector_flags": detector_flags,
        "severity_score": severity,
        "max_channel_score": max_channel_score,
        "severity_label": _severity_label(severity),
    }


def _split_recent_for_adaptation(X_recent: np.ndarray, y_recent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _temporal_train_val_split_arrays(X_recent, y_recent, validation_fraction=0.2)


def _recalibrate_artifact(artifact: Dict[str, Any], X_recent: np.ndarray, y_recent: np.ndarray, *, config: Dict[str, Any]) -> None:
    raw_scores, _embeddings = _predict_with_artifact(artifact, X_recent)
    calibrator = _fit_platt_calibrator(y_recent, raw_scores)
    artifact["calibrator"] = calibrator
    calibrated_scores = _apply_calibrator(raw_scores, calibrator)
    threshold_info = _optimize_decision_threshold(
        y_recent,
        calibrated_scores,
        beta=float(config.get("threshold_beta", DEFAULT_CONFIG["threshold_beta"])),
    )
    artifact["decision_threshold"] = float(threshold_info["threshold"])
    artifact["threshold_info"] = threshold_info
    _refresh_reference_from_recent(artifact, X_recent, y_recent, config=config)


def _fine_tune_artifact(
    artifact: Dict[str, Any],
    X_recent: np.ndarray,
    y_recent: np.ndarray,
    *,
    config: Dict[str, Any],
) -> None:
    if str(artifact.get("kind")) != "torch_mlp":
        raise ValueError("Fine-tuning solo aplica al modelo Torch MLP.")
    X_train, X_val, y_train, y_val = _split_recent_for_adaptation(X_recent, y_recent)
    tuned = _train_torch_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,
        learning_rate=float(config.get("fine_tune_learning_rate", DEFAULT_CONFIG["fine_tune_learning_rate"])),
        epochs=int(config.get("fine_tune_epochs", DEFAULT_CONFIG["fine_tune_epochs"])),
        base_model_state=artifact.get("base_model_state"),
    )
    artifact.update(tuned)
    _refresh_reference_from_recent(artifact, X_recent, y_recent, config=config)


def _retrain_artifact(
    model_name: str,
    artifact: Dict[str, Any],
    history_X: np.ndarray,
    history_y: np.ndarray,
    recent_X: np.ndarray,
    recent_y: np.ndarray,
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    history_keep = min(len(history_y), max(len(recent_y), int(config.get("history_sample_size", 256))))
    hybrid_X = np.vstack([history_X[-history_keep:], recent_X])
    hybrid_y = np.concatenate([history_y[-history_keep:], recent_y])
    X_train, X_val, y_train, y_val = _split_recent_for_adaptation(hybrid_X, hybrid_y)
    retrained = _train_model_artifact(
        model_name,
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,
    )
    _refresh_reference_from_recent(retrained, recent_X, recent_y, config=config)
    return retrained


def _allowed_strategies_for_model(model_name: str, strategies: Sequence[str]) -> List[str]:
    if str(model_name) == MODEL_XGBOOST:
        return [strategy for strategy in strategies if strategy != STRATEGY_FINE_TUNING]
    return [str(strategy) for strategy in strategies]


def _rolling_metric_table(stream_df: pd.DataFrame, *, rolling_window: int) -> pd.DataFrame:
    if stream_df.empty:
        return pd.DataFrame()
    work = stream_df.copy().reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for idx in range(len(work)):
        start = max(0, idx - int(rolling_window) + 1)
        window = work.iloc[start : idx + 1]
        threshold_value = float(window["decision_threshold"].iloc[-1]) if "decision_threshold" in window.columns else 0.5
        metrics = _classification_metrics(
            window["y_true"].to_numpy(),
            window["score"].to_numpy(),
            threshold=threshold_value,
            preds=window["prediction"].to_numpy() if "prediction" in window.columns else None,
        )
        rows.append(
            {
                "timestamp": pd.Timestamp(work.loc[idx, "timestamp"]),
                "model": str(work.loc[idx, "model"]),
                "strategy": str(work.loc[idx, "strategy"]),
                "pr_auc": metrics["pr_auc"],
                "recall": metrics["recall"],
                "fnr": metrics["fnr"],
                "brier": metrics["brier"],
                "severity_score": float(window["severity_score"].iloc[-1]),
                "decision_threshold": threshold_value,
            }
        )
    return pd.DataFrame(rows)


def _summary_from_stream(stream_df: pd.DataFrame, drift_events: pd.DataFrame) -> pd.DataFrame:
    if stream_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    grouped = stream_df.groupby(["model", "strategy"], dropna=False)
    for (model_name, strategy), group in grouped:
        threshold_value = float(group["decision_threshold"].iloc[-1]) if "decision_threshold" in group.columns else 0.5
        metrics = _classification_metrics(
            group["y_true"].to_numpy(),
            group["score"].to_numpy(),
            threshold=threshold_value,
            preds=group["prediction"].to_numpy() if "prediction" in group.columns else None,
        )
        related_events = drift_events.loc[
            drift_events["model"].astype(str).eq(str(model_name))
            & drift_events["strategy"].astype(str).eq(str(strategy))
        ].copy()
        rows.append(
            {
                "model": str(model_name),
                "strategy": str(strategy),
                "rows": int(len(group)),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "fnr": metrics["fnr"],
                "brier": metrics["brier"],
                "n_drift_events": int(len(related_events)),
                "n_actions": int(related_events["action_taken"].ne("none").sum()) if not related_events.empty else 0,
                "mean_severity": float(group["severity_score"].mean()),
                "max_trigger_score": float(group["max_channel_score"].max()) if "max_channel_score" in group.columns else float("nan"),
                "decision_threshold_last": threshold_value,
                "decision_threshold_mean": float(group["decision_threshold"].mean()) if "decision_threshold" in group.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "strategy"]).reset_index(drop=True)


def run_backtest_pipeline(
    dataset_bundle: Dict[str, Any],
    *,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    df = _ensure_non_empty_dataframe(dataset_bundle.get("df"), label="Neural drift dataset")
    feature_cols = list(dataset_bundle.get("feature_cols") or [])
    if not feature_cols:
        raise ValueError("Neural drift dataset has no feature columns.")

    if progress_callback is not None:
        progress_callback(0.02, "Generando features derivadas...")
    augmented_df, augmented_feature_cols = augment_feature_frame(df, feature_cols)
    dataset = build_window_dataset(
        augmented_df,
        feature_cols=augmented_feature_cols,
        interval_minutes=int(config.get("interval_minutes", DEFAULT_CONFIG["interval_minutes"])),
        lookback_steps=int(config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"])),
        horizon_steps=int(config.get("horizon_steps", DEFAULT_CONFIG["horizon_steps"])),
    )
    dataset.augmented_feature_cols = list(augmented_feature_cols)

    if progress_callback is not None:
        progress_callback(0.08, "Construyendo split temporal...")
    split = _split_window_dataset(
        dataset,
        train_fraction=float(config.get("train_fraction", DEFAULT_CONFIG["train_fraction"])),
        validation_fraction=float(config.get("validation_fraction", DEFAULT_CONFIG["validation_fraction"])),
        max_stream_rows=int(config.get("max_stream_rows", DEFAULT_CONFIG["max_stream_rows"])),
    )

    baseline_rows: List[Dict[str, Any]] = []
    stream_rows: List[Dict[str, Any]] = []
    drift_rows: List[Dict[str, Any]] = []

    selected_models = [model for model in config.get("models", []) if model in AVAILABLE_MODELS]
    selected_strategies = [strategy for strategy in config.get("strategies", []) if strategy in AVAILABLE_STRATEGIES]
    selected_channels = [channel for channel in config.get("drift_channels", []) if channel in AVAILABLE_DRIFT_CHANNELS]
    if not selected_models or not selected_strategies:
        raise ValueError("Selecciona al menos un modelo y una estrategia para el backtest.")

    loop_total = sum(len(_allowed_strategies_for_model(model, selected_strategies)) for model in selected_models)
    loop_total = max(1, loop_total)
    loop_index = 0

    for model_name in selected_models:
        if progress_callback is not None:
            progress_callback(
                0.12 + 0.70 * (loop_index / loop_total),
                f"Entrenando baseline para {model_name}...",
            )
        baseline_artifact = _train_model_artifact(
            model_name,
            split["X_train"],
            split["y_train"],
            split["X_val"],
            split["y_val"],
            config=config,
        )
        baseline_raw_scores, baseline_embeddings = _predict_with_artifact(baseline_artifact, split["X_val"])
        baseline_scores = _apply_calibrator(baseline_raw_scores, baseline_artifact.get("calibrator"))
        baseline_threshold = float(baseline_artifact.get("decision_threshold", baseline_artifact.get("base_threshold", 0.5)))
        baseline_preds = (baseline_scores >= baseline_threshold).astype(int)
        baseline_metrics = _classification_metrics(
            split["y_val"],
            baseline_scores,
            threshold=baseline_threshold,
            preds=baseline_preds,
        )
        baseline_rows.append(
            {
                "model": str(model_name),
                "split": "validation",
                "rows": int(len(split["y_val"])),
                **baseline_metrics,
                "embedding_channels_available": bool(baseline_embeddings.size > 0),
            }
        )

        valid_strategies = _allowed_strategies_for_model(model_name, selected_strategies)
        for strategy in valid_strategies:
            loop_index += 1
            if progress_callback is not None:
                progress_callback(
                    0.15 + 0.75 * (loop_index / loop_total),
                    f"Simulando {model_name} | {strategy}...",
                )
            artifact = _train_model_artifact(
                model_name,
                split["X_train"],
                split["y_train"],
                split["X_val"],
                split["y_val"],
                config=config,
            )
            history_X = np.vstack([split["X_train"], split["X_val"]])
            history_y = np.concatenate([split["y_train"], split["y_val"]])

            detectors = {
                DRIFT_INPUT: ClassicDriftDetector(),
                DRIFT_SCORE: ClassicDriftDetector(),
                DRIFT_ERROR: ClassicDriftDetector(),
            }
            channel_histories: Dict[str, List[float]] = {
                DRIFT_INPUT: [],
                DRIFT_SCORE: [],
                DRIFT_ERROR: [],
                DRIFT_EMBEDDING: [],
            }

            for idx in range(len(split["y_stream"])):
                x_row = split["X_stream"][idx : idx + 1]
                y_true = int(split["y_stream"][idx])
                timestamp = pd.Timestamp(split["metadata_stream"].loc[idx, "prediction_time"])

                raw_scores, embeddings = _predict_with_artifact(artifact, x_row)
                score = float(_apply_calibrator(raw_scores, artifact.get("calibrator"))[0])
                decision_threshold = float(artifact.get("decision_threshold", artifact.get("base_threshold", 0.5)))
                pred = int(score >= decision_threshold)
                channel_payload = _build_channel_scores(
                    artifact=artifact,
                    x_row=_apply_imputer(x_row, artifact["imputer"]).reshape(-1),
                    calibrated_score=score,
                    y_true=y_true,
                    embeddings=embeddings.reshape(-1),
                    selected_channels=selected_channels,
                    detectors=detectors,
                    channel_histories=channel_histories,
                    recent_window_size=int(config.get("recent_window_size", DEFAULT_CONFIG["recent_window_size"])),
                    embedding_reconstruction_weight=float(
                        config.get(
                            "drift_monitor_reconstruction_weight",
                            DEFAULT_CONFIG["drift_monitor_reconstruction_weight"],
                        )
                    ),
                )

                severity_score = float(channel_payload["severity_score"])
                max_channel_score = float(channel_payload.get("max_channel_score", severity_score))
                action_taken = "none"
                recent_start = max(0, idx - int(config.get("recent_window_size", DEFAULT_CONFIG["recent_window_size"])) + 1)
                recent_X = split["X_stream"][recent_start : idx + 1]
                recent_y = split["y_stream"][recent_start : idx + 1]
                recent_has_two_classes = len(np.unique(recent_y)) >= 2
                severity_threshold = float(config.get("severity_threshold", DEFAULT_CONFIG["severity_threshold"]))
                severity_triggered = bool(
                    severity_score >= severity_threshold
                    or max_channel_score >= severity_threshold
                )

                if severity_triggered:
                    if strategy == STRATEGY_FIXED:
                        action_taken = "none"
                    elif strategy == STRATEGY_RECALIBRATION and len(recent_y) >= int(config.get("recalibration_min_rows", DEFAULT_CONFIG["recalibration_min_rows"])) and recent_has_two_classes:
                        _recalibrate_artifact(artifact, recent_X, recent_y, config=config)
                        action_taken = "recalibration"
                    elif strategy == STRATEGY_FINE_TUNING and len(recent_y) >= int(config.get("recalibration_min_rows", DEFAULT_CONFIG["recalibration_min_rows"])) and recent_has_two_classes:
                        _fine_tune_artifact(artifact, recent_X, recent_y, config=config)
                        action_taken = "fine_tuning"
                    elif strategy == STRATEGY_RETRAIN and len(recent_y) >= int(config.get("retrain_min_rows", DEFAULT_CONFIG["retrain_min_rows"])) and recent_has_two_classes:
                        artifact = _retrain_artifact(
                            model_name,
                            artifact,
                            history_X,
                            history_y,
                            recent_X,
                            recent_y,
                            config=config,
                        )
                        action_taken = "retrain"
                    event_threshold = float(artifact.get("decision_threshold", decision_threshold))

                    drift_rows.append(
                        {
                            "timestamp": timestamp,
                            "model": str(model_name),
                            "strategy": str(strategy),
                            "severity_score": severity_score,
                            "max_channel_score": max_channel_score,
                            "severity_label": str(channel_payload["severity_label"]),
                            "decision_threshold": event_threshold,
                            "channel_scores": json.dumps(_to_json_safe(channel_payload["channel_scores"]), ensure_ascii=True, sort_keys=True),
                            "raw_channel_values": json.dumps(_to_json_safe(channel_payload.get("raw_channel_values") or {}), ensure_ascii=True, sort_keys=True),
                            "detector_flags": json.dumps(_to_json_safe(channel_payload["detector_flags"]), ensure_ascii=True, sort_keys=True),
                            "action_taken": str(action_taken),
                            "recent_rows": int(len(recent_y)),
                            "recent_positive_rows": int(np.sum(recent_y)),
                        }
                    )

                stream_rows.append(
                    {
                        "timestamp": timestamp,
                        "model": str(model_name),
                        "strategy": str(strategy),
                        "y_true": int(y_true),
                        "prediction": int(pred),
                        "score": float(score),
                        "decision_threshold": decision_threshold,
                        "severity_score": severity_score,
                        "max_channel_score": max_channel_score,
                        "severity_label": str(channel_payload["severity_label"]),
                        "action_taken": str(action_taken),
                        "brier_component": float((score - y_true) ** 2),
                    }
                )

    baseline_df = pd.DataFrame(baseline_rows)
    if not baseline_df.empty:
        baseline_df = baseline_df.sort_values(["model"]).reset_index(drop=True)

    stream_df = pd.DataFrame(stream_rows)
    if not stream_df.empty:
        stream_df = stream_df.sort_values(["model", "strategy", "timestamp"]).reset_index(drop=True)

    drift_df = pd.DataFrame(
        drift_rows,
        columns=[
            "timestamp",
            "model",
            "strategy",
            "severity_score",
            "max_channel_score",
            "severity_label",
            "channel_scores",
            "raw_channel_values",
            "detector_flags",
            "action_taken",
            "recent_rows",
            "recent_positive_rows",
        ],
    )
    if not drift_df.empty:
        drift_df = drift_df.sort_values(["model", "strategy", "timestamp"]).reset_index(drop=True)

    summary_df = _summary_from_stream(stream_df, drift_df)
    rolling_df = _rolling_metric_table(
        stream_df,
        rolling_window=int(config.get("rolling_metric_window", DEFAULT_CONFIG["rolling_metric_window"])),
    )
    return {
        "dataset": dataset,
        "split": split,
        "baseline": baseline_df,
        "summary": summary_df,
        "stream_metrics": stream_df,
        "rolling_metrics": rolling_df,
        "drift_events": drift_df,
    }


def _download_bundle_from_results(results: Dict[str, Any]) -> Dict[str, str]:
    bundle: Dict[str, str] = {}
    for key in ["baseline", "summary", "stream_metrics", "rolling_metrics", "drift_events"]:
        df = results.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            bundle[key] = df.to_csv(index=False)
    return bundle


def generate_synthetic_neural_drift_dataset(
    *,
    rows: int = 240,
    interval_minutes: int = 5,
    drift_start: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=int(rows), freq=f"{int(interval_minutes)}min")
    idx = np.arange(int(rows), dtype=float)

    shift = np.where(idx >= int(drift_start), 1.7, 0.0)
    flow = 85.0 + 12.0 * np.sin(idx / 7.0) + 6.0 * shift + rng.normal(0.0, 3.0, size=len(idx))
    speed = 78.0 - 5.5 * shift + 4.0 * np.cos(idx / 9.0) + rng.normal(0.0, 2.0, size=len(idx))
    density = np.clip(flow / np.maximum(speed, 20.0), 0.0, None) * 4.0
    heavy = 12.0 + 2.0 * shift + rng.normal(0.0, 0.8, size=len(idx))

    logits = -2.2 + 0.035 * flow + 0.22 * density - 0.025 * speed + 0.05 * heavy + 0.40 * shift
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    target = rng.binomial(1, np.clip(probabilities, 0.01, 0.90)).astype(int)

    return pd.DataFrame(
        {
            "interval_start": timestamps,
            "portico": ["15"] * len(idx),
            "flow_light": flow,
            "flow_heavy": heavy,
            "speed_light": speed,
            "speed_heavy": speed - 5.0 + rng.normal(0.0, 1.0, size=len(idx)),
            "density_light": density,
            "density_heavy": density * 1.15,
            "target": target,
        }
    )


def _estimate_network_shapes(dataset_bundle: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, int]:
    try:
        augmented_df, engineered_cols = augment_feature_frame(
            dataset_bundle["df"],
            dataset_bundle.get("feature_cols") or [],
        )
        input_dim = int(len(engineered_cols) * int(config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"])))
        augmented_feature_count = int(len(engineered_cols))
        rows = int(len(augmented_df))
    except Exception:
        input_dim = int(len(dataset_bundle.get("feature_cols") or [])) * int(
            config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"])
        )
        augmented_feature_count = int(len(dataset_bundle.get("feature_cols") or []))
        fallback_df = dataset_bundle.get("df")
        rows = int(len(fallback_df)) if isinstance(fallback_df, pd.DataFrame) else 0

    embedding_dim = int(config.get("mlp_embedding_dim", DEFAULT_CONFIG["mlp_embedding_dim"]))
    monitor_hidden_dim = int(config.get("drift_monitor_hidden_dim", DEFAULT_CONFIG["drift_monitor_hidden_dim"]))
    monitor_bottleneck_dim = int(
        config.get("drift_monitor_bottleneck_dim", DEFAULT_CONFIG["drift_monitor_bottleneck_dim"])
    )
    return {
        "rows": rows,
        "augmented_feature_count": augmented_feature_count,
        "predictor_input_dim": input_dim,
        "predictor_embedding_dim": embedding_dim,
        "monitor_input_dim": embedding_dim,
        "monitor_hidden_dim": max(monitor_bottleneck_dim + 1, monitor_hidden_dim),
        "monitor_bottleneck_dim": max(2, min(monitor_bottleneck_dim, max(2, embedding_dim - 1))),
    }


def _build_monitor_architecture_explanation(
    shapes: Dict[str, int],
    config: Dict[str, Any],
    reconstruction_weight: float,
) -> Dict[str, Any]:
    mlp_hidden_dim = int(config.get("mlp_hidden_dim", DEFAULT_CONFIG["mlp_hidden_dim"]))
    mlp_second_hidden = max(2, mlp_hidden_dim // 2)
    distance_weight = 1.0 - float(reconstruction_weight)
    return {
        "overview": (
            "La arquitectura actual muestra dos redes conectadas. La primera predice riesgo de accidente; "
            "la segunda no predice accidentes, sino que vigila si los embeddings producidos por la primera "
            "siguen pareciendose al comportamiento historico."
        ),
        "predictor_steps": [
            (
                f"`window[{shapes['predictor_input_dim']}]`",
                "Es la entrada real del predictor. Ese numero surge de multiplicar las features disponibles "
                f"despues del feature engineering (`{shapes['augmented_feature_count']}`) por los pasos de lookback. "
                "Mientras mas grande sea, mas contexto temporal ve el modelo, pero tambien mas compleja se vuelve la red."
            ),
            (
                f"`hidden[{mlp_hidden_dim}] -> hidden[{mlp_second_hidden}]`",
                "Son capas intermedias del predictor. Transforman la ventana temporal en patrones no lineales. "
                "Si aumentan mucho, el predictor gana capacidad, pero puede volverse menos estable y mas facil de sobreajustar."
            ),
            (
                f"`embedding[{shapes['predictor_embedding_dim']}]`",
                "Es la representacion latente compacta de la situacion vial. Este embedding resume la ventana y es el punto "
                "clave para monitorear drift: si cambia su geometria, la red interpreta que el regimen operativo tambien cambio."
            ),
            (
                "`score[1]`",
                "Es la probabilidad final de accidente. Se usa para la prediccion operativa, pero no es la unica senal de drift."
            ),
        ],
        "monitor_steps": [
            (
                f"`embedding[{shapes['monitor_input_dim']}]`",
                "El monitor de drift no mira la ventana cruda, mira el embedding del predictor. Eso lo hace mas semantico: "
                "vigila el espacio donde la red ya codifico su comprension del trafico."
            ),
            (
                f"`hidden[{shapes['monitor_hidden_dim']}] -> bottleneck[{shapes['monitor_bottleneck_dim']}]`",
                "El autoencoder comprime el embedding a un cuello de botella. Si el bottleneck es pequeno, el monitor se vuelve "
                "mas estricto y sensible a patrones nuevos. Si es grande, tolera mas variabilidad y detecta menos cambios."
            ),
            (
                f"`reconstruction[{shapes['monitor_input_dim']}]`",
                "La salida intenta reconstruir el embedding original. Cuando la reconstruccion empeora, significa que el patron "
                "actual no se parece a lo que el monitor aprendio como normal."
            ),
        ],
        "score_formula": (
            f"`embedding drift score = {distance_weight:.2f} * centroid_distance + {float(reconstruction_weight):.2f} * reconstruction_error`"
        ),
        "score_interpretation": [
            (
                "`centroid_distance` alto",
                "El embedding promedio se desplazo respecto al regimen de referencia. Suele indicar cambio global de contexto: "
                "otra dinamica de velocidad, flujo o densidad."
            ),
            (
                "`reconstruction_error` alto",
                "La forma interna del embedding se volvio dificil de reconstruir. Suele indicar patrones nuevos o combinaciones "
                "que la red no habia internalizado bien, incluso si el centroide no se movio demasiado."
            ),
            (
                "`ambos` altos",
                "Es una senal mas fuerte de drift real. En la practica, suele justificar mayor atencion porque hay cambio global "
                "y, ademas, novedad estructural."
            ),
        ],
        "tuning_guidance": [
            (
                "`Monitor hidden dim`",
                "Subirlo aumenta capacidad del monitor. Util si el embedding es complejo, pero si se exagera el autoencoder "
                "reconstruira casi todo y perdera sensibilidad."
            ),
            (
                "`Monitor bottleneck dim`",
                "Es el control mas importante de sensibilidad. Menor bottleneck = monitor mas severo; mayor bottleneck = monitor mas permisivo."
            ),
            (
                "`Monitor dropout`",
                "Introduce regularizacion. Ayuda a que el monitor no memorice demasiado el embedding historico."
            ),
            (
                "`Peso reconstruction error`",
                "Si sube, el drift dependera mas de que el autoencoder falle al reconstruir. Si baja, pesara mas el desplazamiento "
                "del embedding respecto al centroide historico."
            ),
        ],
    }


def _render_feature_source_selector(context: Dict[str, Any]) -> Dict[str, Any]:
    current_export_path = str(context.get("feature_export_path") or "").strip()
    has_memory_dataset = isinstance(context.get("clean_df"), pd.DataFrame) and not context["clean_df"].empty

    artifacts: List[Dict[str, Any]] = []
    try:
        artifacts = list_feature_engineering_duckdb_artifacts()
    except Exception:
        artifacts = []

    option_keys: List[str] = []
    option_labels: Dict[str, str] = {}

    if has_memory_dataset:
        option_keys.append("__memory__")
        option_labels["__memory__"] = "Dataset actual en memoria"

    seen_paths: set[str] = set()
    if current_export_path:
        resolved_current = str(Path(current_export_path).resolve())
        seen_paths.add(resolved_current)
        option_keys.append(resolved_current)
        option_labels[resolved_current] = f"Export actual: {Path(current_export_path).name}"

    for artifact in artifacts:
        artifact_path = str(artifact["path"])
        if artifact_path in seen_paths:
            continue
        seen_paths.add(artifact_path)
        option_keys.append(artifact_path)
        feature_suffix = ""
        if int(artifact.get("selected_feature_count", 0)) > 0:
            feature_suffix = f" · {int(artifact['selected_feature_count'])} features"
        option_labels[artifact_path] = (
            f"Feature engineering: {artifact['name']} · {int(artifact.get('row_count', 0))} rows{feature_suffix}"
        )

    if not option_keys:
        return dict(context)

    if st.session_state.get("neural_drift_feature_source_choice") not in option_keys:
        if has_memory_dataset:
            st.session_state["neural_drift_feature_source_choice"] = "__memory__"
        elif current_export_path:
            st.session_state["neural_drift_feature_source_choice"] = str(Path(current_export_path).resolve())
        else:
            st.session_state["neural_drift_feature_source_choice"] = option_keys[0]

    selected_key = st.selectbox(
        "Feature source",
        option_keys,
        format_func=lambda key: option_labels.get(str(key), str(key)),
        key="neural_drift_feature_source_choice",
    )
    if str(selected_key) == "__memory__":
        return dict(context)
    return build_dataset_context_for_source_selection(
        context,
        selected_feature_export_path=str(selected_key),
    )


def _render_configuration_subtab(dataset_bundle: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown("**Dataset activo**")
    df = dataset_bundle["df"]
    metrics_col_1, metrics_col_2, metrics_col_3 = st.columns(3)
    metrics_col_1.metric("Source", str(dataset_bundle.get("source", "-")))
    metrics_col_2.metric("Rows", int(len(df)))
    metrics_col_3.metric("Features", int(len(dataset_bundle.get("feature_cols", []))))
    if dataset_bundle.get("feature_export_path"):
        st.caption(f"Export DuckDB: {dataset_bundle['feature_export_path']}")

    max_stream_min = 48
    max_stream_max = max(max_stream_min, int(len(df)))
    max_stream_default = min(int(DEFAULT_CONFIG["max_stream_rows"]), max_stream_max)
    max_stream_default = max(max_stream_min, max_stream_default)

    config_col_1, config_col_2, config_col_3 = st.columns(3)
    lookback_steps = config_col_1.number_input(
        "Lookback steps",
        min_value=4,
        max_value=36,
        value=int(DEFAULT_CONFIG["lookback_steps"]),
        step=1,
        key="neural_drift_lookback_steps",
    )
    horizon_steps = config_col_2.number_input(
        "Horizon steps",
        min_value=1,
        max_value=6,
        value=int(DEFAULT_CONFIG["horizon_steps"]),
        step=1,
        key="neural_drift_horizon_steps",
    )
    max_stream_rows = config_col_3.number_input(
        "Max stream rows",
        min_value=max_stream_min,
        max_value=max_stream_max,
        value=max_stream_default,
        step=24,
        key="neural_drift_max_stream_rows",
    )

    model_col, strategy_col, channel_col = st.columns(3)
    selected_models = model_col.multiselect(
        "Models",
        AVAILABLE_MODELS,
        default=list(DEFAULT_CONFIG["models"]),
        key="neural_drift_models",
    )
    selected_strategies = strategy_col.multiselect(
        "Strategies",
        AVAILABLE_STRATEGIES,
        default=list(DEFAULT_CONFIG["strategies"]),
        key="neural_drift_strategies",
    )
    selected_channels = channel_col.multiselect(
        "Drift channels",
        AVAILABLE_DRIFT_CHANNELS,
        default=list(DEFAULT_CONFIG["drift_channels"]),
        key="neural_drift_channels",
    )
    if MODEL_XGBOOST in selected_models and STRATEGY_FINE_TUNING in selected_strategies:
        st.caption("`fine_tuning` se excluira automaticamente para XGBoost.")

    backtest_col_1, backtest_col_2, backtest_col_3 = st.columns(3)
    recent_window_size = backtest_col_1.number_input(
        "Recent window size",
        min_value=24,
        max_value=240,
        value=int(DEFAULT_CONFIG["recent_window_size"]),
        step=8,
        key="neural_drift_recent_window_size",
    )
    recalibration_min_rows = backtest_col_2.number_input(
        "Recalibration min rows",
        min_value=16,
        max_value=160,
        value=int(DEFAULT_CONFIG["recalibration_min_rows"]),
        step=8,
        key="neural_drift_recalibration_min_rows",
    )
    retrain_min_rows = backtest_col_3.number_input(
        "Retrain min rows",
        min_value=24,
        max_value=240,
        value=int(DEFAULT_CONFIG["retrain_min_rows"]),
        step=8,
        key="neural_drift_retrain_min_rows",
    )

    severity_col_1, severity_col_2, severity_col_3 = st.columns(3)
    severity_threshold = severity_col_1.slider(
        "Severity trigger",
        min_value=0.10,
        max_value=0.95,
        value=float(DEFAULT_CONFIG["severity_threshold"]),
        step=0.05,
        key="neural_drift_severity_threshold",
    )
    rolling_metric_window = severity_col_2.number_input(
        "Rolling metric window",
        min_value=12,
        max_value=120,
        value=int(DEFAULT_CONFIG["rolling_metric_window"]),
        step=6,
        key="neural_drift_rolling_metric_window",
    )
    history_sample_size = severity_col_3.number_input(
        "Hybrid history sample",
        min_value=64,
        max_value=512,
        value=int(DEFAULT_CONFIG["history_sample_size"]),
        step=32,
        key="neural_drift_history_sample_size",
    )

    config = {
        **DEFAULT_CONFIG,
        "lookback_steps": int(lookback_steps),
        "horizon_steps": int(horizon_steps),
        "max_stream_rows": int(max_stream_rows),
        "models": list(selected_models),
        "strategies": list(selected_strategies),
        "drift_channels": list(selected_channels),
        "recent_window_size": int(recent_window_size),
        "recalibration_min_rows": int(recalibration_min_rows),
        "retrain_min_rows": int(retrain_min_rows),
        "severity_threshold": float(severity_threshold),
        "rolling_metric_window": int(rolling_metric_window),
        "history_sample_size": int(history_sample_size),
    }
    st.session_state["neural_drift_config"] = config
    return config


def _render_monitor_network_subtab(dataset_bundle: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown("**Detector neuronal de drift**")
    st.caption(
        "La senal `embedding drift` se construye con un autoencoder pequeno sobre los embeddings "
        "del predictor Torch MLP y una distancia al centroide de referencia."
    )

    if MODEL_TORCH_MLP not in list(current_config.get("models") or []):
        st.info("Para activar la red que monitorea drift debes incluir `Torch MLP` en Models.")
    if DRIFT_EMBEDDING not in list(current_config.get("drift_channels") or []):
        st.info("Para usar la red de drift debes incluir `embedding drift` en Drift channels.")

    monitor_profile = _resolve_drift_monitor_profile(
        st.session_state.get(
            "neural_drift_monitor_profile",
            current_config.get("drift_monitor_profile", DEFAULT_CONFIG["drift_monitor_profile"]),
        )
    )
    st.session_state["neural_drift_monitor_profile"] = monitor_profile

    default_monitor_values = {
        "neural_drift_monitor_hidden_dim": int(
            current_config.get("drift_monitor_hidden_dim", DEFAULT_CONFIG["drift_monitor_hidden_dim"])
        ),
        "neural_drift_monitor_bottleneck_dim": int(
            current_config.get(
                "drift_monitor_bottleneck_dim",
                _drift_monitor_profile_preset(monitor_profile).get(
                    "drift_monitor_bottleneck_dim",
                    DEFAULT_CONFIG["drift_monitor_bottleneck_dim"],
                ),
            )
        ),
        "neural_drift_monitor_dropout": float(
            current_config.get("drift_monitor_dropout", DEFAULT_CONFIG["drift_monitor_dropout"])
        ),
        "neural_drift_monitor_epochs": int(
            current_config.get("drift_monitor_epochs", DEFAULT_CONFIG["drift_monitor_epochs"])
        ),
        "neural_drift_monitor_batch_size": int(
            current_config.get("drift_monitor_batch_size", DEFAULT_CONFIG["drift_monitor_batch_size"])
        ),
        "neural_drift_monitor_learning_rate": float(
            current_config.get("drift_monitor_learning_rate", DEFAULT_CONFIG["drift_monitor_learning_rate"])
        ),
        "neural_drift_monitor_reconstruction_weight": float(
            current_config.get(
                "drift_monitor_reconstruction_weight",
                _drift_monitor_profile_preset(monitor_profile).get(
                    "drift_monitor_reconstruction_weight",
                    DEFAULT_CONFIG["drift_monitor_reconstruction_weight"],
                ),
            )
        ),
    }
    for state_key, default_value in default_monitor_values.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value

    def _on_monitor_profile_change() -> None:
        _apply_drift_monitor_profile_to_session(st.session_state.get("neural_drift_monitor_profile"))

    selector_col, selector_hint_col = st.columns([1.2, 2.2])
    monitor_profile = selector_col.selectbox(
        "Tipo de detector",
        AVAILABLE_DRIFT_MONITOR_PROFILES,
        key="neural_drift_monitor_profile",
        on_change=_on_monitor_profile_change,
    )
    selector_hint_col.info(_drift_monitor_profile_description(monitor_profile))

    arch_col_1, arch_col_2, arch_col_3 = st.columns(3)
    monitor_hidden_dim = arch_col_1.number_input(
        "Monitor hidden dim",
        min_value=4,
        max_value=128,
        value=int(st.session_state.get("neural_drift_monitor_hidden_dim", current_config.get("drift_monitor_hidden_dim", DEFAULT_CONFIG["drift_monitor_hidden_dim"]))),
        step=2,
        key="neural_drift_monitor_hidden_dim",
    )
    monitor_bottleneck_dim = arch_col_2.number_input(
        "Monitor bottleneck dim",
        min_value=2,
        max_value=64,
        value=int(st.session_state.get("neural_drift_monitor_bottleneck_dim", current_config.get("drift_monitor_bottleneck_dim", DEFAULT_CONFIG["drift_monitor_bottleneck_dim"]))),
        step=1,
        key="neural_drift_monitor_bottleneck_dim",
    )
    monitor_dropout = arch_col_3.slider(
        "Monitor dropout",
        min_value=0.0,
        max_value=0.5,
        value=float(st.session_state.get("neural_drift_monitor_dropout", current_config.get("drift_monitor_dropout", DEFAULT_CONFIG["drift_monitor_dropout"]))),
        step=0.05,
        key="neural_drift_monitor_dropout",
    )

    train_col_1, train_col_2, train_col_3 = st.columns(3)
    monitor_epochs = train_col_1.number_input(
        "Monitor epochs",
        min_value=4,
        max_value=60,
        value=int(st.session_state.get("neural_drift_monitor_epochs", current_config.get("drift_monitor_epochs", DEFAULT_CONFIG["drift_monitor_epochs"]))),
        step=2,
        key="neural_drift_monitor_epochs",
    )
    monitor_batch_size = train_col_2.number_input(
        "Monitor batch size",
        min_value=8,
        max_value=256,
        value=int(st.session_state.get("neural_drift_monitor_batch_size", current_config.get("drift_monitor_batch_size", DEFAULT_CONFIG["drift_monitor_batch_size"]))),
        step=8,
        key="neural_drift_monitor_batch_size",
    )
    monitor_learning_rate = train_col_3.number_input(
        "Monitor learning rate",
        min_value=1e-4,
        max_value=1e-2,
        value=float(st.session_state.get("neural_drift_monitor_learning_rate", current_config.get("drift_monitor_learning_rate", DEFAULT_CONFIG["drift_monitor_learning_rate"]))),
        step=1e-4,
        format="%.4f",
        key="neural_drift_monitor_learning_rate",
    )

    mix_col_1, mix_col_2 = st.columns([2, 1])
    reconstruction_weight = mix_col_1.slider(
        "Peso reconstruction error",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("neural_drift_monitor_reconstruction_weight", current_config.get("drift_monitor_reconstruction_weight", DEFAULT_CONFIG["drift_monitor_reconstruction_weight"]))),
        step=0.05,
        key="neural_drift_monitor_reconstruction_weight",
    )
    mix_col_2.metric("Peso distancia", f"{1.0 - float(reconstruction_weight):.2f}")

    monitor_config = {
        **current_config,
        "drift_monitor_profile": str(monitor_profile),
        "drift_monitor_hidden_dim": int(monitor_hidden_dim),
        "drift_monitor_bottleneck_dim": int(monitor_bottleneck_dim),
        "drift_monitor_dropout": float(monitor_dropout),
        "drift_monitor_epochs": int(monitor_epochs),
        "drift_monitor_batch_size": int(monitor_batch_size),
        "drift_monitor_learning_rate": float(monitor_learning_rate),
        "drift_monitor_reconstruction_weight": float(reconstruction_weight),
    }
    st.session_state["neural_drift_config"] = monitor_config

    shapes = _estimate_network_shapes(dataset_bundle, monitor_config)
    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    summary_col_1.metric("Predictor input", int(shapes["predictor_input_dim"]))
    summary_col_2.metric("Embedding dim", int(shapes["predictor_embedding_dim"]))
    summary_col_3.metric("AE bottleneck", int(shapes["monitor_bottleneck_dim"]))

    st.markdown("**Arquitectura actual**")
    architecture_lines = [
        f"Predictor MLP: window[{shapes['predictor_input_dim']}] -> hidden[{int(monitor_config.get('mlp_hidden_dim', DEFAULT_CONFIG['mlp_hidden_dim']))}] -> "
        f"hidden[{max(2, int(monitor_config.get('mlp_hidden_dim', DEFAULT_CONFIG['mlp_hidden_dim'])) // 2)}] -> embedding[{shapes['predictor_embedding_dim']}] -> score[1]",
        f"Drift monitor AE: embedding[{shapes['monitor_input_dim']}] -> hidden[{shapes['monitor_hidden_dim']}] -> "
        f"bottleneck[{shapes['monitor_bottleneck_dim']}] -> hidden[{shapes['monitor_hidden_dim']}] -> reconstruction[{shapes['monitor_input_dim']}]",
        f"Embedding drift score = {(1.0 - float(reconstruction_weight)):.2f} * centroid_distance + {float(reconstruction_weight):.2f} * reconstruction_error",
    ]
    st.code("\n".join(architecture_lines), language="text")

    explanation = _build_monitor_architecture_explanation(
        shapes,
        monitor_config,
        float(reconstruction_weight),
    )
    st.markdown("**Como interpretar esta arquitectura**")
    st.write(explanation["overview"])

    with st.expander("Lectura guiada de la arquitectura", expanded=True):
        st.markdown("**1. Predictor de riesgo**")
        for title, body in explanation["predictor_steps"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**2. Red que monitorea drift**")
        for title, body in explanation["monitor_steps"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**3. Como leer el score de drift**")
        st.markdown(explanation["score_formula"])
        for title, body in explanation["score_interpretation"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**4. Como ajustar los controles**")
        for title, body in explanation["tuning_guidance"]:
            st.markdown(f"- {title}: {body}")

    st.info(
        "Regla practica: si quieres un detector mas sensible a cambios nuevos, reduce `Monitor bottleneck dim` "
        "y/o sube `Peso reconstruction error`. Si quieres un detector mas estable y menos propenso a falsas alarmas, "
        "haz lo contrario."
    )
    return monitor_config


def _render_backtest_subtab(dataset_bundle: Dict[str, Any], config: Dict[str, Any]) -> None:
    run_clicked = st.button("Run Neural drift backtest", key="neural_drift_run_backtest")
    if not run_clicked:
        st.info("Configura el experimento y ejecuta el backtest temporal.")
        return

    progress_bar = st.progress(0.0)
    status = st.empty()

    def _callback(ratio: float, message: str) -> None:
        progress_bar.progress(float(np.clip(ratio, 0.0, 1.0)))
        status.caption(str(message))

    try:
        with st.spinner("Ejecutando Neural drift backtest..."):
            results = run_backtest_pipeline(
                dataset_bundle,
                config=config,
                progress_callback=_callback,
            )
    except Exception as exc:
        progress_bar.progress(1.0)
        status.caption("Backtest interrumpido.")
        st.error(f"No se pudo ejecutar Neural drift: {exc}")
        return

    progress_bar.progress(1.0)
    status.caption("Backtest completado.")
    st.session_state["neural_drift_baseline_results"] = results.get("baseline")
    st.session_state["neural_drift_stream_results"] = {
        "summary": results.get("summary"),
        "stream_metrics": results.get("stream_metrics"),
        "rolling_metrics": results.get("rolling_metrics"),
    }
    st.session_state["neural_drift_drift_events"] = results.get("drift_events")
    st.session_state["neural_drift_download_bundle"] = _download_bundle_from_results(results)
    st.session_state["neural_drift_last_run_signature"] = _build_run_signature(dataset_bundle, config)
    st.success("Neural drift finalizado.")


def _render_results_subtab(dataset_bundle: Dict[str, Any], config: Dict[str, Any]) -> None:
    baseline = st.session_state.get("neural_drift_baseline_results")
    stream_results = dict(st.session_state.get("neural_drift_stream_results") or {})
    summary = stream_results.get("summary")
    stream_metrics = stream_results.get("stream_metrics")
    rolling_metrics = stream_results.get("rolling_metrics")
    drift_events = st.session_state.get("neural_drift_drift_events")
    download_bundle = dict(st.session_state.get("neural_drift_download_bundle") or {})
    current_signature = _build_run_signature(dataset_bundle, config)
    last_run_signature = st.session_state.get("neural_drift_last_run_signature")

    if not isinstance(summary, pd.DataFrame) or summary.empty:
        st.info("No hay resultados de Neural drift para mostrar.")
        return

    if last_run_signature != current_signature:
        st.warning(
            "Los resultados visibles no corresponden a la configuracion o fuente de features actual. "
            "Vuelve a ejecutar el backtest para ver resultados consistentes con el estado actual."
        )

    if isinstance(baseline, pd.DataFrame) and not baseline.empty:
        st.markdown("**Baseline temporal**")
        st.dataframe(_streamlit_arrow_safe_df(baseline), width="stretch")

    st.markdown("**Comparativa model x strategy**")
    st.dataframe(_streamlit_arrow_safe_df(summary), width="stretch")

    if isinstance(drift_events, pd.DataFrame) and not drift_events.empty:
        st.markdown("**Drift events**")
        st.dataframe(_streamlit_arrow_safe_df(drift_events), width="stretch")

    if isinstance(rolling_metrics, pd.DataFrame) and not rolling_metrics.empty:
        st.markdown("**Rolling metrics**")
        selected_metric = st.selectbox(
            "Metric",
            ["pr_auc", "recall", "fnr", "brier", "severity_score"],
            key="neural_drift_metric_selector",
        )
        plot_df = rolling_metrics.copy()
        plot_df["series_label"] = plot_df["model"].astype(str) + " | " + plot_df["strategy"].astype(str)
        chart_df = plot_df.pivot_table(
            index="timestamp",
            columns="series_label",
            values=selected_metric,
            aggfunc="last",
        ).sort_index()
        st.line_chart(chart_df, width="stretch")
        st.dataframe(_streamlit_arrow_safe_df(plot_df), width="stretch")

    if isinstance(stream_metrics, pd.DataFrame) and not stream_metrics.empty:
        st.markdown("**Stream records**")
        st.dataframe(_streamlit_arrow_safe_df(stream_metrics), width="stretch")

    st.markdown("**Descargas**")
    for key, label in [
        ("summary", "Download summary.csv"),
        ("drift_events", "Download drift_events.csv"),
        ("stream_metrics", "Download stream_metrics.csv"),
    ]:
        payload = download_bundle.get(key)
        if not payload:
            continue
        st.download_button(
            label,
            payload,
            file_name=f"neural_drift_{key}.csv",
            mime="text/csv",
            key=f"neural_drift_download_{key}",
        )


def render_tab(context: Dict[str, Any]) -> None:
    init_state()
    st.subheader("Neural drift")
    st.caption(
        "Short-horizon crash prediction with temporal backtesting, classical drift signals, "
        "and an embedding-based neural drift monitor."
    )

    selected_context = context
    try:
        selected_context = _render_feature_source_selector(context)
    except Exception as exc:
        st.warning(f"No se pudo construir el catalogo de features DuckDB: {exc}")
        selected_context = context

    try:
        dataset_bundle = resolve_dataset_from_context(selected_context)
    except Exception as exc:
        st.info(
            "Ejecuta `Eventos -> Feature engineering -> Feature Selection` en Drift detection "
            "o carga un export DuckDB valido antes de usar Neural drift."
        )
        st.warning(str(exc))
        return

    st.session_state["neural_drift_dataset"] = {
        "source": dataset_bundle["source"],
        "rows": int(len(dataset_bundle["df"])),
        "feature_cols": list(dataset_bundle["feature_cols"]),
        "feature_export_path": dataset_bundle.get("feature_export_path"),
        "feature_source_choice": st.session_state.get("neural_drift_feature_source_choice"),
        "selection_metadata": _to_json_safe(dataset_bundle.get("selection_metadata") or {}),
    }

    config_tab, monitor_tab, backtest_tab, results_tab = st.tabs(
        ["Configuración", "Red de drift", "Backtest", "Resultados"]
    )
    with config_tab:
        config = _render_configuration_subtab(dataset_bundle)
    with monitor_tab:
        config = _render_monitor_network_subtab(dataset_bundle, config)
    with backtest_tab:
        config = dict(st.session_state.get("neural_drift_config") or DEFAULT_CONFIG)
        _render_backtest_subtab(dataset_bundle, config)
    with results_tab:
        config = dict(st.session_state.get("neural_drift_config") or DEFAULT_CONFIG)
        _render_results_subtab(dataset_bundle, config)
