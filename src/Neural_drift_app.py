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
MODEL_TORCH_MLP_ATTENTION = "Torch MLP + Attention"
AVAILABLE_MODELS = [MODEL_XGBOOST, MODEL_TORCH_MLP, MODEL_TORCH_MLP_ATTENTION]

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

DRIFT_MONITOR_ARCH_CLASSIC_AE = "Autoencoder clasico"
DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION = "Attention temporal"
AVAILABLE_DRIFT_MONITOR_ARCHITECTURES = [
    DRIFT_MONITOR_ARCH_CLASSIC_AE,
    DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
]

DETECTOR_SENSITIVITY_PRESET_CONSERVATIVE = "Conservador"
DETECTOR_SENSITIVITY_PRESET_MODERATE = "Moderado"
DETECTOR_SENSITIVITY_PRESET_SENSITIVE = "Sensible"
DETECTOR_SENSITIVITY_PRESET_VERY_SENSITIVE = "Muy sensible"
AVAILABLE_DETECTOR_SENSITIVITY_PRESETS = [
    DETECTOR_SENSITIVITY_PRESET_CONSERVATIVE,
    DETECTOR_SENSITIVITY_PRESET_MODERATE,
    DETECTOR_SENSITIVITY_PRESET_SENSITIVE,
    DETECTOR_SENSITIVITY_PRESET_VERY_SENSITIVE,
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
    "neural_drift_sensitivity_preset": DETECTOR_SENSITIVITY_PRESET_MODERATE,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "interval_minutes": 5,
    "dataset_percent": 100,
    "lookback_steps": 12,
    "horizon_steps": 1,
    "train_fraction": 0.60,
    "validation_fraction": 0.20,
    "models": [MODEL_XGBOOST, MODEL_TORCH_MLP, MODEL_TORCH_MLP_ATTENTION],
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
    "attention_feature_hidden_dim": 32,
    "attention_temporal_hidden_dim": 32,
    "attention_dropout": 0.10,
    "attention_top_k": 8,
    "fine_tune_learning_rate": 3e-4,
    "fine_tune_epochs": 6,
    "drift_monitor_hidden_dim": 16,
    "drift_monitor_bottleneck_dim": 6,
    "drift_monitor_dropout": 0.05,
    "drift_monitor_epochs": 12,
    "drift_monitor_batch_size": 32,
    "drift_monitor_learning_rate": 1e-3,
    "drift_monitor_reconstruction_weight": 0.65,
    "drift_monitor_architecture": DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
    "drift_monitor_sequence_length": 12,
    "drift_monitor_attention_hidden_dim": 32,
    "drift_monitor_attention_dropout": 0.05,
    "drift_monitor_profile": DRIFT_MONITOR_PROFILE_MODERATE,
    "detector_sensitivity_preset": DETECTOR_SENSITIVITY_PRESET_MODERATE,
    "detector_adwin_delta": 0.002,
    "drift_point_signal_weight": 1.0,
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


def _is_torch_model(model_name: str) -> bool:
    return str(model_name) in {MODEL_TORCH_MLP, MODEL_TORCH_MLP_ATTENTION}


def _has_any_torch_model(models: Sequence[str]) -> bool:
    return any(_is_torch_model(str(model_name)) for model_name in models)


def _resolve_torch_model_family(model_name: str) -> str:
    if str(model_name) == MODEL_TORCH_MLP_ATTENTION:
        return "torch_mlp_attention"
    return "torch_mlp"


def _resolve_torch_model_name(model_family: str) -> str:
    if str(model_family) == "torch_mlp_attention":
        return MODEL_TORCH_MLP_ATTENTION
    return MODEL_TORCH_MLP


def _time_step_labels(lookback_steps: int) -> List[str]:
    return [f"t-{offset}" for offset in range(int(lookback_steps) - 1, -1, -1)]


def init_state() -> None:
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default_value)


def _resolve_drift_monitor_profile(value: Any) -> str:
    profile = str(value or DRIFT_MONITOR_PROFILE_MODERATE)
    if profile not in AVAILABLE_DRIFT_MONITOR_PROFILES:
        return DRIFT_MONITOR_PROFILE_MODERATE
    return profile


def _resolve_drift_monitor_architecture(value: Any) -> str:
    architecture = str(value or DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION)
    if architecture not in AVAILABLE_DRIFT_MONITOR_ARCHITECTURES:
        return DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION
    return architecture


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


def _resolve_detector_sensitivity_preset(value: Any) -> str:
    preset = str(value or DETECTOR_SENSITIVITY_PRESET_MODERATE)
    if preset not in AVAILABLE_DETECTOR_SENSITIVITY_PRESETS:
        return DETECTOR_SENSITIVITY_PRESET_MODERATE
    return preset


def _detector_sensitivity_preset_config(preset: str) -> Dict[str, Any]:
    resolved = _resolve_detector_sensitivity_preset(preset)
    if resolved == DETECTOR_SENSITIVITY_PRESET_CONSERVATIVE:
        return {
            "severity_threshold": 0.65,
            "recent_window_size": 120,
            "detector_adwin_delta": 0.001,
            "drift_point_signal_weight": 0.35,
            "drift_monitor_profile": DRIFT_MONITOR_PROFILE_MODERATE,
            "drift_monitor_bottleneck_dim": 8,
            "drift_monitor_reconstruction_weight": 0.45,
        }
    if resolved == DETECTOR_SENSITIVITY_PRESET_SENSITIVE:
        return {
            "severity_threshold": 0.40,
            "recent_window_size": 48,
            "detector_adwin_delta": 0.004,
            "drift_point_signal_weight": 1.0,
            "drift_monitor_profile": DRIFT_MONITOR_PROFILE_SENSITIVE,
            "drift_monitor_bottleneck_dim": 4,
            "drift_monitor_reconstruction_weight": 0.85,
        }
    if resolved == DETECTOR_SENSITIVITY_PRESET_VERY_SENSITIVE:
        return {
            "severity_threshold": 0.30,
            "recent_window_size": 24,
            "detector_adwin_delta": 0.010,
            "drift_point_signal_weight": 1.0,
            "drift_monitor_profile": DRIFT_MONITOR_PROFILE_SENSITIVE,
            "drift_monitor_bottleneck_dim": 3,
            "drift_monitor_reconstruction_weight": 0.95,
        }
    return {
        "severity_threshold": float(DEFAULT_CONFIG["severity_threshold"]),
        "recent_window_size": int(DEFAULT_CONFIG["recent_window_size"]),
        "detector_adwin_delta": float(DEFAULT_CONFIG["detector_adwin_delta"]),
        "drift_point_signal_weight": float(DEFAULT_CONFIG["drift_point_signal_weight"]),
        "drift_monitor_profile": DRIFT_MONITOR_PROFILE_MODERATE,
        "drift_monitor_bottleneck_dim": int(DEFAULT_CONFIG["drift_monitor_bottleneck_dim"]),
        "drift_monitor_reconstruction_weight": float(DEFAULT_CONFIG["drift_monitor_reconstruction_weight"]),
    }


def _detector_sensitivity_preset_description(preset: str) -> str:
    resolved = _resolve_detector_sensitivity_preset(preset)
    if resolved == DETECTOR_SENSITIVITY_PRESET_CONSERVATIVE:
        return "Prioriza estabilidad y menos falsas alarmas. Usa mas suavizado y menor sensibilidad local."
    if resolved == DETECTOR_SENSITIVITY_PRESET_SENSITIVE:
        return "Prioriza reaccion temprana a cambios nuevos. Baja umbrales y sube el peso de senales puntuales."
    if resolved == DETECTOR_SENSITIVITY_PRESET_VERY_SENSITIVE:
        return "Maxima reactividad. Puede detectar cambios muy temprano, pero con mayor riesgo de falsas alarmas."
    return "Equilibrio entre sensibilidad y estabilidad. Replica la configuracion base actual del detector."


def _apply_detector_sensitivity_preset_to_session(preset: str) -> None:
    resolved = _resolve_detector_sensitivity_preset(preset)
    preset_config = _detector_sensitivity_preset_config(resolved)
    st.session_state["neural_drift_sensitivity_preset"] = resolved
    st.session_state["neural_drift_severity_threshold"] = float(preset_config["severity_threshold"])
    st.session_state["neural_drift_recent_window_size"] = int(preset_config["recent_window_size"])
    st.session_state["neural_drift_detector_adwin_delta"] = float(preset_config["detector_adwin_delta"])
    st.session_state["neural_drift_point_signal_weight"] = float(preset_config["drift_point_signal_weight"])
    st.session_state["neural_drift_monitor_profile"] = str(preset_config["drift_monitor_profile"])
    st.session_state["neural_drift_monitor_bottleneck_dim"] = int(
        preset_config["drift_monitor_bottleneck_dim"]
    )
    st.session_state["neural_drift_monitor_reconstruction_weight"] = float(
        preset_config["drift_monitor_reconstruction_weight"]
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
            "dataset_percent": int(config.get("dataset_percent", DEFAULT_CONFIG["dataset_percent"])),
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
            "detector_sensitivity_preset": str(
                config.get("detector_sensitivity_preset", DEFAULT_CONFIG["detector_sensitivity_preset"])
            ),
            "detector_adwin_delta": float(
                config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"])
            ),
            "drift_point_signal_weight": float(
                config.get("drift_point_signal_weight", DEFAULT_CONFIG["drift_point_signal_weight"])
            ),
            "drift_monitor_profile": str(
                config.get("drift_monitor_profile", DEFAULT_CONFIG["drift_monitor_profile"])
            ),
            "drift_monitor_architecture": str(
                config.get("drift_monitor_architecture", DEFAULT_CONFIG["drift_monitor_architecture"])
            ),
            "drift_monitor_bottleneck_dim": int(
                config.get("drift_monitor_bottleneck_dim", DEFAULT_CONFIG["drift_monitor_bottleneck_dim"])
            ),
            "drift_monitor_sequence_length": int(
                config.get("drift_monitor_sequence_length", DEFAULT_CONFIG["drift_monitor_sequence_length"])
            ),
            "drift_monitor_attention_hidden_dim": int(
                config.get(
                    "drift_monitor_attention_hidden_dim",
                    DEFAULT_CONFIG["drift_monitor_attention_hidden_dim"],
                )
            ),
            "drift_monitor_attention_dropout": float(
                config.get("drift_monitor_attention_dropout", DEFAULT_CONFIG["drift_monitor_attention_dropout"])
            ),
            "drift_monitor_reconstruction_weight": float(
                config.get(
                    "drift_monitor_reconstruction_weight",
                    DEFAULT_CONFIG["drift_monitor_reconstruction_weight"],
                )
            ),
            "attention_feature_hidden_dim": int(
                config.get("attention_feature_hidden_dim", DEFAULT_CONFIG["attention_feature_hidden_dim"])
            ),
            "attention_temporal_hidden_dim": int(
                config.get("attention_temporal_hidden_dim", DEFAULT_CONFIG["attention_temporal_hidden_dim"])
            ),
            "attention_dropout": float(
                config.get("attention_dropout", DEFAULT_CONFIG["attention_dropout"])
            ),
            "attention_top_k": int(
                config.get("attention_top_k", DEFAULT_CONFIG["attention_top_k"])
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


def _subset_dataset_by_percentage(
    df: pd.DataFrame,
    *,
    dataset_percent: float,
    time_col: str = "interval_start",
) -> pd.DataFrame:
    work = _ensure_non_empty_dataframe(df, label="Dataset percentage selection").copy()
    pct = float(np.clip(dataset_percent, 1.0, 100.0))

    if time_col in work.columns:
        ordered_time = pd.to_datetime(work[time_col], errors="coerce")
        work = (
            work.assign(__dataset_percent_time__=ordered_time)
            .sort_values("__dataset_percent_time__", kind="stable", na_position="first")
            .drop(columns="__dataset_percent_time__")
            .reset_index(drop=True)
        )

    if pct >= 100.0:
        return work.reset_index(drop=True).copy()

    keep_rows = max(1, int(math.ceil(len(work) * pct / 100.0)))
    return work.tail(keep_rows).reset_index(drop=True).copy()


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
        return logits, embeddings, {}

    def forward(self, x):
        logits, _, _ = self.forward_with_embeddings(x)
        return logits


class WindowAttentionMLP(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int,
        lookback_steps: int,
        feature_hidden_dim: int,
        temporal_hidden_dim: int,
        encoder_hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_count = int(feature_count)
        self.lookback_steps = int(lookback_steps)
        self.feature_hidden_dim = int(feature_hidden_dim)

        self.feature_projection = nn.Linear(1, feature_hidden_dim)
        self.feature_identity = nn.Parameter(torch.randn(self.feature_count, feature_hidden_dim) * 0.02)
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_hidden_dim, feature_hidden_dim),
            nn.Tanh(),
            nn.Linear(feature_hidden_dim, 1),
        )
        self.temporal_position = nn.Parameter(torch.randn(self.lookback_steps, feature_hidden_dim) * 0.02)
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_hidden_dim, temporal_hidden_dim),
            nn.Tanh(),
            nn.Linear(temporal_hidden_dim, 1),
        )
        encoder_hidden_dim = max(int(embedding_dim) + 2, int(encoder_hidden_dim))
        self.encoder = nn.Sequential(
            nn.Linear(feature_hidden_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def _reshape_inputs(self, x):
        if x.ndim != 2:
            raise ValueError("WindowAttentionMLP expects flat windows with shape [batch, lookback * features].")
        expected_dim = self.lookback_steps * self.feature_count
        if int(x.shape[1]) != expected_dim:
            raise ValueError(
                f"WindowAttentionMLP expected input_dim={expected_dim}, got {int(x.shape[1])}."
            )
        return x.reshape(x.shape[0], self.lookback_steps, self.feature_count)

    def forward_with_embeddings(self, x):
        window = self._reshape_inputs(x)
        feature_tokens = self.feature_projection(window.unsqueeze(-1))
        feature_tokens = feature_tokens + self.feature_identity.view(1, 1, self.feature_count, self.feature_hidden_dim)

        feature_logits = self.feature_attention(feature_tokens).squeeze(-1)
        feature_weights = torch.softmax(feature_logits, dim=-1)
        step_context = torch.sum(feature_weights.unsqueeze(-1) * feature_tokens, dim=2)

        temporal_tokens = step_context + self.temporal_position.view(1, self.lookback_steps, self.feature_hidden_dim)
        temporal_logits = self.temporal_attention(temporal_tokens).squeeze(-1)
        temporal_weights = torch.softmax(temporal_logits, dim=1)
        window_context = torch.sum(temporal_weights.unsqueeze(-1) * temporal_tokens, dim=1)

        embeddings = self.encoder(window_context)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits, embeddings, {
            "feature_attention": feature_weights,
            "temporal_attention": temporal_weights,
        }

    def forward(self, x):
        logits, _, _ = self.forward_with_embeddings(x)
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


class TemporalAttentionEmbeddingMonitor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        attention_hidden_dim: int,
        bottleneck_dim: int,
        hidden_dim: int,
        sequence_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.hidden_dim = int(hidden_dim)

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.temporal_position = nn.Parameter(torch.randn(self.sequence_length, hidden_dim) * 0.02)
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        tokens = self.input_projection(x)
        tokens = tokens + self.temporal_position.view(1, self.sequence_length, self.hidden_dim)
        attention_logits = self.temporal_attention(tokens).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = torch.sum(attention_weights.unsqueeze(-1) * tokens, dim=1)
        latent = self.bottleneck(context)
        reconstruction = self.decoder(latent)
        return reconstruction, attention_weights


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


def _resolve_feature_count(*, input_dim: int, lookback_steps: int, feature_metadata: Optional[Dict[str, Any]]) -> int:
    if feature_metadata is not None and int(feature_metadata.get("feature_count") or 0) > 0:
        return int(feature_metadata["feature_count"])
    lookback = max(1, int(lookback_steps))
    if int(input_dim) % lookback != 0:
        raise ValueError(
            f"Input dimension {int(input_dim)} is not divisible by lookback_steps={lookback}."
        )
    return int(input_dim) // lookback


def _torch_feature_metadata(
    *,
    config: Dict[str, Any],
    feature_metadata: Optional[Dict[str, Any]],
    input_dim: int,
) -> Dict[str, Any]:
    lookback_steps = int(
        (feature_metadata or {}).get("lookback_steps", config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"]))
    )
    feature_count = _resolve_feature_count(
        input_dim=int(input_dim),
        lookback_steps=lookback_steps,
        feature_metadata=feature_metadata,
    )
    augmented_feature_cols = list((feature_metadata or {}).get("augmented_feature_cols") or [])
    if len(augmented_feature_cols) != feature_count:
        augmented_feature_cols = [f"feature_{idx}" for idx in range(feature_count)]
    base_feature_cols = list((feature_metadata or {}).get("base_feature_cols") or augmented_feature_cols)
    return {
        "lookback_steps": int(lookback_steps),
        "feature_count": int(feature_count),
        "augmented_feature_cols": list(augmented_feature_cols),
        "base_feature_cols": list(base_feature_cols),
        "temporal_labels": _time_step_labels(lookback_steps),
    }


def _summarize_attention_payload(
    attention_payload: Dict[str, Any],
    *,
    feature_labels: Sequence[str],
    temporal_labels: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not attention_payload:
        return None

    feature_attention = attention_payload.get("feature_attention")
    temporal_attention = attention_payload.get("temporal_attention")
    if feature_attention is None or temporal_attention is None:
        return None

    feature_mean = feature_attention.mean(dim=(0, 1)).detach().cpu().numpy().astype(float)
    temporal_mean = temporal_attention.mean(dim=0).detach().cpu().numpy().astype(float)
    return {
        "feature_attention_mean": feature_mean,
        "temporal_attention_mean": temporal_mean,
        "feature_labels": list(feature_labels),
        "temporal_labels": list(temporal_labels),
    }


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float)


def _build_embedding_monitor_sequences(
    embeddings: np.ndarray,
    *,
    sequence_length: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)

    seq_len = max(2, int(sequence_length))
    step = max(1, int(stride))
    if emb.size == 0 or emb.shape[0] <= seq_len:
        return (
            np.empty((0, seq_len, emb.shape[1] if emb.ndim == 2 else 0), dtype=float),
            np.empty((0, emb.shape[1] if emb.ndim == 2 else 0), dtype=float),
            np.empty((0,), dtype=int),
        )

    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    target_indices: List[int] = []
    for target_idx in range(seq_len, emb.shape[0], step):
        X_rows.append(emb[target_idx - seq_len : target_idx])
        y_rows.append(emb[target_idx])
        target_indices.append(int(target_idx))

    if not X_rows:
        return (
            np.empty((0, seq_len, emb.shape[1]), dtype=float),
            np.empty((0, emb.shape[1]), dtype=float),
            np.empty((0,), dtype=int),
        )
    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(target_indices, dtype=int),
    )


def _summarize_monitor_attention_weights(
    attention_weights: np.ndarray,
    *,
    sequence_length: int,
) -> Optional[Dict[str, Any]]:
    weights = np.asarray(attention_weights, dtype=float)
    if weights.size == 0:
        return None
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    return {
        "temporal_attention_mean": weights.mean(axis=0).astype(float),
        "temporal_labels": [f"lag_{offset}" for offset in range(int(sequence_length), 0, -1)],
    }


def _train_torch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
    model_name: str = MODEL_TORCH_MLP,
    feature_metadata: Optional[Dict[str, Any]] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    base_model_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ensure_torch_available()
    _set_random_seed(int(config.get("random_state", 42)))
    model_family = _resolve_torch_model_family(model_name)

    imputer = _fit_imputer(X_train)
    X_train_imp = _apply_imputer(X_train, imputer)
    X_val_imp = _apply_imputer(X_val, imputer)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    try:
        torch_metadata = _torch_feature_metadata(
            config=config,
            feature_metadata=feature_metadata,
            input_dim=int(X_train_scaled.shape[1]),
        )
    except ValueError:
        if model_family == "torch_mlp_attention":
            raise
        fallback_feature_count = int((feature_metadata or {}).get("feature_count") or int(X_train_scaled.shape[1]))
        fallback_feature_cols = list((feature_metadata or {}).get("augmented_feature_cols") or [])
        if len(fallback_feature_cols) != fallback_feature_count:
            fallback_feature_cols = [f"feature_{idx}" for idx in range(fallback_feature_count)]
        torch_metadata = {
            "lookback_steps": int((feature_metadata or {}).get("lookback_steps") or 1),
            "feature_count": fallback_feature_count,
            "augmented_feature_cols": list(fallback_feature_cols),
            "base_feature_cols": list((feature_metadata or {}).get("base_feature_cols") or fallback_feature_cols),
            "temporal_labels": _time_step_labels(int((feature_metadata or {}).get("lookback_steps") or 1)),
        }

    device = torch.device("cpu")
    if model_family == "torch_mlp_attention":
        model = WindowAttentionMLP(
            feature_count=int(torch_metadata["feature_count"]),
            lookback_steps=int(torch_metadata["lookback_steps"]),
            feature_hidden_dim=int(
                config.get("attention_feature_hidden_dim", DEFAULT_CONFIG["attention_feature_hidden_dim"])
            ),
            temporal_hidden_dim=int(
                config.get("attention_temporal_hidden_dim", DEFAULT_CONFIG["attention_temporal_hidden_dim"])
            ),
            encoder_hidden_dim=max(
                8,
                int(config.get("mlp_hidden_dim", DEFAULT_CONFIG["mlp_hidden_dim"])) // 2,
            ),
            embedding_dim=int(config.get("mlp_embedding_dim", DEFAULT_CONFIG["mlp_embedding_dim"])),
            dropout=float(config.get("attention_dropout", DEFAULT_CONFIG["attention_dropout"])),
        ).to(device)
    else:
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
    prediction_details = _predict_torch_model_details(
        {
            "kind": "torch_mlp",
            "model_family": model_family,
            "model_name": str(model_name),
            "model": model,
            "imputer": imputer,
            "scaler": scaler,
            **torch_metadata,
        },
        X_val,
    )
    raw_val_scores = prediction_details["probs"]
    embeddings = prediction_details["embeddings"]
    calibrator = _fit_platt_calibrator(y_val, raw_val_scores)
    calibrated_val_scores = _apply_calibrator(raw_val_scores, calibrator)
    threshold_info = _optimize_decision_threshold(
        y_val,
        calibrated_val_scores,
        beta=float(config.get("threshold_beta", DEFAULT_CONFIG["threshold_beta"])),
    )
    embedding_monitor, reconstruction_errors = _fit_embedding_monitor(embeddings, config=config)
    attention_summary_reference = prediction_details.get("attention_summary")
    monitor_effective_architecture = str(
        (embedding_monitor or {}).get("monitor_effective_architecture", (embedding_monitor or {}).get("kind", "none"))
    )
    reference = _build_reference_stats(
        X_ref=_apply_imputer(X_val, imputer),
        y_ref=y_val,
        calibrated_scores=calibrated_val_scores,
        embeddings=embeddings,
        reconstruction_errors=reconstruction_errors,
    )
    return {
        "kind": "torch_mlp",
        "model_family": model_family,
        "model_name": _resolve_torch_model_name(model_family),
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "calibrator": calibrator,
        "reference": reference,
        "embedding_monitor": embedding_monitor,
        "monitor_effective_architecture": monitor_effective_architecture,
        "lookback_steps": int(torch_metadata["lookback_steps"]),
        "feature_count": int(torch_metadata["feature_count"]),
        "base_feature_cols": list(torch_metadata["base_feature_cols"]),
        "augmented_feature_cols": list(torch_metadata["augmented_feature_cols"]),
        "temporal_labels": list(torch_metadata["temporal_labels"]),
        "attention_summary_reference": attention_summary_reference,
        "decision_threshold": float(threshold_info["threshold"]),
        "threshold_info": threshold_info,
        "base_threshold": float(threshold_info["threshold"]),
        "base_model_state": copy.deepcopy(model.state_dict()),
    }


def _predict_torch_model_details(artifact: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
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
        outputs = model.forward_with_embeddings(tensor)
        if len(outputs) == 3:
            logits, embeddings, attention_payload = outputs
        else:
            logits, embeddings = outputs
            attention_payload = {}
        probs = torch.sigmoid(logits).cpu().numpy()
        emb = embeddings.cpu().numpy()
    attention_summary = _summarize_attention_payload(
        attention_payload,
        feature_labels=list(artifact.get("augmented_feature_cols") or []),
        temporal_labels=list(
            artifact.get("temporal_labels")
            or _time_step_labels(int(artifact.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"])))
        ),
    )
    return {
        "probs": probs.astype(float),
        "embeddings": emb.astype(float),
        "attention_summary": attention_summary,
    }


def _predict_torch_mlp(artifact: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    details = _predict_torch_model_details(artifact, X)
    return details["probs"], details["embeddings"]


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
        "monitor_effective_architecture": "not_available",
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


def _predict_with_artifact_details(artifact: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    kind = str(artifact.get("kind"))
    if kind == "torch_mlp":
        return _predict_torch_model_details(artifact, X)
    probs, _ = _predict_xgboost(artifact, X)
    return {
        "probs": probs,
        "embeddings": np.empty((len(probs), 0), dtype=float),
        "attention_summary": None,
    }


def _train_model_artifact(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
    feature_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if _is_torch_model(str(model_name)):
        return _train_torch_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            model_name=str(model_name),
            feature_metadata=feature_metadata,
        )
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
    if emb.size == 0 or emb.shape[1] < 2:
        return None, None

    requested_architecture = _resolve_drift_monitor_architecture(
        config.get("drift_monitor_architecture", DEFAULT_CONFIG["drift_monitor_architecture"])
    )
    requested_sequence_length = max(
        2,
        int(config.get("drift_monitor_sequence_length", DEFAULT_CONFIG["drift_monitor_sequence_length"])),
    )

    if requested_architecture == DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION:
        monitor, reconstruction_errors = _fit_temporal_attention_embedding_monitor(
            emb,
            config=config,
            sequence_length=requested_sequence_length,
        )
        if monitor is not None:
            monitor["requested_architecture"] = requested_architecture
            monitor["monitor_effective_architecture"] = DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION
            return monitor, reconstruction_errors

    if emb.shape[0] < 8:
        return None, None

    monitor, reconstruction_errors = _fit_classic_embedding_monitor(emb, config=config)
    if monitor is None:
        return None, None
    monitor["requested_architecture"] = requested_architecture
    monitor["monitor_effective_architecture"] = DRIFT_MONITOR_ARCH_CLASSIC_AE
    if requested_architecture == DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION:
        monitor["fallback_reason"] = "insufficient_embeddings_for_temporal_attention"
    return monitor, reconstruction_errors


def _fit_classic_embedding_monitor(
    embeddings: np.ndarray,
    *,
    config: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
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
            "kind": DRIFT_MONITOR_ARCH_CLASSIC_AE,
            "model": model,
            "scaler": scaler,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "dropout": dropout,
        },
        embeddings,
    )
    return (
        {
            "kind": DRIFT_MONITOR_ARCH_CLASSIC_AE,
            "model": model,
            "scaler": scaler,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "dropout": dropout,
        },
        reconstruction_errors,
    )


def _fit_temporal_attention_embedding_monitor(
    embeddings: np.ndarray,
    *,
    config: Dict[str, Any],
    sequence_length: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    _ensure_torch_available()
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    X_seq, y_target, target_indices = _build_embedding_monitor_sequences(
        emb_scaled,
        sequence_length=sequence_length,
        stride=1,
    )
    if len(target_indices) < 8:
        return None, None

    val_count = max(1, int(math.ceil(len(target_indices) * 0.2)))
    val_start = max(1, len(target_indices) - val_count)
    X_train = X_seq[:val_start]
    y_train = y_target[:val_start]
    X_val = X_seq[val_start:]
    y_val = y_target[val_start:]
    if len(X_train) == 0 or len(X_val) == 0:
        return None, None

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
    attention_hidden_dim = max(
        4,
        int(
            config.get(
                "drift_monitor_attention_hidden_dim",
                DEFAULT_CONFIG["drift_monitor_attention_hidden_dim"],
            )
        ),
    )
    dropout = float(
        config.get("drift_monitor_attention_dropout", DEFAULT_CONFIG["drift_monitor_attention_dropout"])
    )
    batch_size = max(8, int(config.get("drift_monitor_batch_size", DEFAULT_CONFIG["drift_monitor_batch_size"])))
    max_epochs = max(2, int(config.get("drift_monitor_epochs", DEFAULT_CONFIG["drift_monitor_epochs"])))
    learning_rate = float(config.get("drift_monitor_learning_rate", DEFAULT_CONFIG["drift_monitor_learning_rate"]))

    device = torch.device("cpu")
    model = TemporalAttentionEmbeddingMonitor(
        input_dim=input_dim,
        attention_hidden_dim=attention_hidden_dim,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience = 3
    stale_epochs = 0
    for _epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            reconstruction, _attention_weights = model(X_batch)
            loss = criterion(reconstruction, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_reconstruction, _val_attention = model(X_val_tensor)
            val_loss = float(criterion(val_reconstruction, y_val_tensor).item())
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    full_details = _predict_embedding_monitor_details(
        {
            "kind": DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
            "model": model,
            "scaler": scaler,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "dropout": dropout,
            "sequence_length": int(sequence_length),
            "attention_hidden_dim": attention_hidden_dim,
        },
        embeddings=emb,
        recent_embeddings=emb,
        use_internal_sequences=True,
    )
    if full_details.get("reconstruction_error") is None:
        return None, None
    return (
        {
            "kind": DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
            "model": model,
            "scaler": scaler,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "bottleneck_dim": bottleneck_dim,
            "dropout": dropout,
            "sequence_length": int(sequence_length),
            "attention_hidden_dim": attention_hidden_dim,
            "attention_reference_summary": full_details.get("attention_summary"),
        },
        _as_float_array(full_details.get("reconstruction_error")),
    )


def _predict_embedding_monitor_details(
    monitor: Dict[str, Any],
    *,
    embeddings: np.ndarray,
    recent_embeddings: Optional[np.ndarray] = None,
    use_internal_sequences: bool = False,
) -> Dict[str, Any]:
    _ensure_torch_available()
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if emb.size == 0:
        return {
            "reconstruction_error": None,
            "attention_summary": None,
            "warmup": True,
        }

    scaler = monitor["scaler"]
    model = monitor["model"]
    kind = str(monitor.get("kind") or DRIFT_MONITOR_ARCH_CLASSIC_AE)
    emb_scaled = scaler.transform(emb)
    device = torch.device("cpu")
    model.eval()
    if kind == DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION:
        sequence_length = int(monitor.get("sequence_length", DEFAULT_CONFIG["drift_monitor_sequence_length"]))
        if use_internal_sequences:
            history = np.asarray(recent_embeddings if recent_embeddings is not None else embeddings, dtype=float)
            if history.ndim == 1:
                history = history.reshape(1, -1)
            history_scaled = scaler.transform(history)
            X_seq, y_target, _ = _build_embedding_monitor_sequences(
                history_scaled,
                sequence_length=sequence_length,
                stride=1,
            )
            if len(X_seq) == 0:
                return {
                    "reconstruction_error": None,
                    "attention_summary": None,
                    "warmup": True,
                }
            with torch.no_grad():
                tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
                target_tensor = torch.tensor(y_target, dtype=torch.float32, device=device)
                reconstruction, attention_weights = model(tensor)
                reconstruction_np = reconstruction.cpu().numpy()
                target_np = target_tensor.cpu().numpy()
                attention_np = attention_weights.cpu().numpy()
            reconstruction_error = np.mean(np.square(reconstruction_np - target_np), axis=1)
            attention_summary = _summarize_monitor_attention_weights(
                attention_np,
                sequence_length=sequence_length,
            )
            return {
                "reconstruction_error": reconstruction_error.astype(float),
                "attention_summary": attention_summary,
                "warmup": False,
            }

        history = np.asarray(recent_embeddings if recent_embeddings is not None else np.empty((0, emb.shape[1])), dtype=float)
        if history.ndim == 1:
            history = history.reshape(1, -1)
        if history.shape[0] < sequence_length:
            return {
                "reconstruction_error": None,
                "attention_summary": None,
                "warmup": True,
            }
        history_scaled = scaler.transform(history[-sequence_length:])
        with torch.no_grad():
            tensor = torch.tensor(history_scaled.reshape(1, sequence_length, -1), dtype=torch.float32, device=device)
            reconstruction, attention_weights = model(tensor)
            reconstruction_np = reconstruction.cpu().numpy()
            attention_np = attention_weights.cpu().numpy()
        reconstruction_error = np.mean(np.square(reconstruction_np - emb_scaled), axis=1)
        attention_summary = _summarize_monitor_attention_weights(
            attention_np,
            sequence_length=sequence_length,
        )
        return {
            "reconstruction_error": reconstruction_error.astype(float),
            "attention_summary": attention_summary,
            "warmup": False,
        }

    with torch.no_grad():
        tensor = torch.tensor(emb_scaled, dtype=torch.float32, device=device)
        reconstruction = model(tensor).cpu().numpy()
    reconstruction_error = np.mean(np.square(reconstruction - emb_scaled), axis=1)
    return {
        "reconstruction_error": reconstruction_error.astype(float),
        "attention_summary": None,
        "warmup": False,
    }


def _predict_embedding_monitor(
    monitor: Dict[str, Any],
    embeddings: np.ndarray,
    recent_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    details = _predict_embedding_monitor_details(
        monitor,
        embeddings=embeddings,
        recent_embeddings=recent_embeddings,
    )
    reconstruction_error = details.get("reconstruction_error")
    if reconstruction_error is None:
        return np.asarray([], dtype=float)
    return _as_float_array(reconstruction_error)


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
    prediction_details = _predict_with_artifact_details(artifact, recent_X)
    raw_scores = prediction_details["probs"]
    embeddings = prediction_details["embeddings"]
    calibrated_scores = _apply_calibrator(raw_scores, artifact.get("calibrator"))
    X_ref = _apply_imputer(recent_X, artifact["imputer"])
    monitor = artifact.get("embedding_monitor")
    reconstruction_errors = None
    if embeddings.size > 0:
        monitor, reconstruction_errors = _fit_embedding_monitor(embeddings, config=config)
        artifact["embedding_monitor"] = monitor
        if monitor is not None:
            artifact["monitor_effective_architecture"] = str(
                monitor.get("monitor_effective_architecture", monitor.get("kind", "none"))
            )
    artifact["reference"] = _build_reference_stats(
        X_ref=X_ref,
        y_ref=recent_y,
        calibrated_scores=calibrated_scores,
        embeddings=embeddings if embeddings.size else None,
        reconstruction_errors=reconstruction_errors,
    )
    artifact["attention_summary_reference"] = prediction_details.get("attention_summary")


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
    recent_embedding_history: Optional[np.ndarray],
    selected_channels: Sequence[str],
    detectors: Dict[str, ClassicDriftDetector],
    channel_histories: Optional[Dict[str, List[float]]] = None,
    recent_window_size: int = 96,
    embedding_reconstruction_weight: float = 0.65,
    point_signal_weight: float = 1.0,
) -> Dict[str, Any]:
    reference = dict(artifact.get("reference") or {})
    available_scores: Dict[str, float] = {}
    detector_flags: Dict[str, bool] = {}
    raw_channel_values: Dict[str, float] = {}
    point_weight = float(np.clip(point_signal_weight, 0.0, 1.0))
    detector_attention_summary: Optional[Dict[str, Any]] = None
    monitor_warmup = False
    monitor_effective_architecture = str(artifact.get("monitor_effective_architecture", "none"))

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
        input_point_score = _normalize_score(
            input_value,
            float(reference.get("input_stat_mean", 0.0)),
            float(reference.get("input_stat_std", 1.0)),
        )
        input_window_score = _normalize_score(
            input_window_value,
            float(reference.get("input_stat_mean", 0.0)),
            float(reference.get("input_stat_std", 1.0)),
        )
        input_drift = detectors[DRIFT_INPUT].update(input_value)
        input_peak_score = float(max(input_point_score, input_window_score))
        input_score = float((1.0 - point_weight) * input_window_score + point_weight * input_peak_score)
        available_scores[DRIFT_INPUT] = float(max(input_score, 1.0 if input_drift else 0.0))
        detector_flags[DRIFT_INPUT] = bool(input_drift)

    if DRIFT_SCORE in selected_channels:
        score_value = float(calibrated_score)
        raw_channel_values[DRIFT_SCORE] = score_value
        score_window_value = _window_stat(DRIFT_SCORE, score_value)
        score_point_score = _normalize_score(
            score_value,
            float(reference.get("score_mean", 0.5)),
            float(reference.get("score_std", 0.15)),
        )
        score_window_score = _normalize_score(
            score_window_value,
            float(reference.get("score_mean", 0.5)),
            float(reference.get("score_std", 0.15)),
        )
        score_drift = detectors[DRIFT_SCORE].update(score_value)
        score_peak_score = float(max(score_point_score, score_window_score))
        score_score = float((1.0 - point_weight) * score_window_score + point_weight * score_peak_score)
        available_scores[DRIFT_SCORE] = float(max(score_score, 1.0 if score_drift else 0.0))
        detector_flags[DRIFT_SCORE] = bool(score_drift)

    if DRIFT_ERROR in selected_channels:
        error_value = float((float(calibrated_score) - int(y_true)) ** 2)
        raw_channel_values[DRIFT_ERROR] = error_value
        error_window_value = _window_stat(DRIFT_ERROR, error_value)
        error_point_score = _normalize_score(
            error_value,
            float(reference.get("error_mean", 0.0)),
            float(reference.get("error_std", 0.1)),
        )
        error_window_score = _normalize_score(
            error_window_value,
            float(reference.get("error_mean", 0.0)),
            float(reference.get("error_std", 0.1)),
        )
        error_drift = detectors[DRIFT_ERROR].update(error_value)
        error_peak_score = float(max(error_point_score, error_window_score))
        error_score = float((1.0 - point_weight) * error_window_score + point_weight * error_peak_score)
        available_scores[DRIFT_ERROR] = float(max(error_score, 1.0 if error_drift else 0.0))
        detector_flags[DRIFT_ERROR] = bool(error_drift)

    if DRIFT_EMBEDDING in selected_channels and embeddings.size > 0 and "embedding_centroid" in reference:
        centroid = np.asarray(reference["embedding_centroid"], dtype=float)
        embedding_distance = float(np.linalg.norm(embeddings.reshape(-1) - centroid))
        raw_channel_values["embedding_distance"] = embedding_distance

        distance_window_value = _window_stat("embedding_distance", embedding_distance)
        distance_point_score = _normalize_score(
            embedding_distance,
            float(reference.get("embedding_distance_mean", 0.0)),
            float(reference.get("embedding_distance_std", 0.1)),
        )
        distance_window_score = _normalize_score(
            distance_window_value,
            float(reference.get("embedding_distance_mean", 0.0)),
            float(reference.get("embedding_distance_std", 0.1)),
        )

        component_scores = [distance_point_score, distance_window_score]
        recon_weight = float(np.clip(embedding_reconstruction_weight, 0.0, 1.0))
        point_weighted_sum = (1.0 - recon_weight) * distance_point_score
        window_weighted_sum = (1.0 - recon_weight) * distance_window_score
        weight_total = (1.0 - recon_weight)

        monitor = artifact.get("embedding_monitor")
        if monitor is not None:
            monitor_details = _predict_embedding_monitor_details(
                monitor,
                embeddings=embeddings.reshape(1, -1),
                recent_embeddings=recent_embedding_history,
            )
            detector_attention_summary = monitor_details.get("attention_summary")
            monitor_warmup = bool(monitor_details.get("warmup", False))
            monitor_effective_architecture = str(
                monitor.get("monitor_effective_architecture", monitor.get("kind", monitor_effective_architecture))
            )
            reconstruction_values = _as_float_array(monitor_details.get("reconstruction_error"))
            if reconstruction_values.size > 0:
                reconstruction_error = float(reconstruction_values[0])
                raw_channel_values["embedding_reconstruction_error"] = reconstruction_error
                reconstruction_window_value = _window_stat("embedding_reconstruction_error", reconstruction_error)
                reconstruction_point_score = _normalize_score(
                    reconstruction_error,
                    float(reference.get("embedding_reconstruction_mean", 0.0)),
                    float(reference.get("embedding_reconstruction_std", 0.1)),
                )
                reconstruction_window_score = _normalize_score(
                    reconstruction_window_value,
                    float(reference.get("embedding_reconstruction_mean", 0.0)),
                    float(reference.get("embedding_reconstruction_std", 0.1)),
                )
                component_scores.extend([reconstruction_point_score, reconstruction_window_score])
                point_weighted_sum += recon_weight * reconstruction_point_score
                window_weighted_sum += recon_weight * reconstruction_window_score
                weight_total += recon_weight

        embedding_point_score = float(point_weighted_sum / max(weight_total, 1e-6))
        embedding_window_score = float(window_weighted_sum / max(weight_total, 1e-6))
        embedding_peak_score = float(max(embedding_point_score, embedding_window_score))
        embedding_score = float((1.0 - point_weight) * embedding_window_score + point_weight * embedding_peak_score)
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
        "detector_attention_summary": detector_attention_summary,
        "monitor_warmup": bool(monitor_warmup),
        "monitor_effective_architecture": monitor_effective_architecture,
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
        model_name=str(artifact.get("model_name", MODEL_TORCH_MLP)),
        feature_metadata={
            "lookback_steps": artifact.get("lookback_steps"),
            "feature_count": artifact.get("feature_count"),
            "base_feature_cols": artifact.get("base_feature_cols"),
            "augmented_feature_cols": artifact.get("augmented_feature_cols"),
        },
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
        feature_metadata={
            "lookback_steps": artifact.get("lookback_steps"),
            "feature_count": artifact.get("feature_count"),
            "base_feature_cols": artifact.get("base_feature_cols"),
            "augmented_feature_cols": artifact.get("augmented_feature_cols"),
        },
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
                "monitor_effective_architecture": str(group["monitor_effective_architecture"].iloc[-1]) if "monitor_effective_architecture" in group.columns else "none",
                "monitor_warmup_rows": int(group["monitor_warmup"].sum()) if "monitor_warmup" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "strategy"]).reset_index(drop=True)


def _build_attention_outputs(attention_rows: Sequence[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    feature_acc: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    temporal_acc: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    drift_acc: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}

    for row in attention_rows:
        feature_values = _as_float_array(row.get("feature_attention_mean"))
        temporal_values = _as_float_array(row.get("temporal_attention_mean"))
        feature_labels = list(row.get("feature_labels") or [])
        temporal_labels = list(row.get("temporal_labels") or [])
        ref_feature_values = _as_float_array(row.get("reference_feature_attention_mean"))
        ref_temporal_values = _as_float_array(row.get("reference_temporal_attention_mean"))
        is_drift_event = bool(row.get("is_drift_event"))
        model_name = str(row.get("model"))
        strategy = str(row.get("strategy"))

        for label, value in zip(feature_labels, feature_values):
            key = (model_name, strategy, str(label))
            acc = feature_acc.setdefault(key, {"sum": 0.0, "count": 0.0, "drift_rows": 0.0})
            acc["sum"] += float(value)
            acc["count"] += 1.0
            if is_drift_event:
                acc["drift_rows"] += 1.0

        for label, value in zip(temporal_labels, temporal_values):
            key = (model_name, strategy, str(label))
            acc = temporal_acc.setdefault(key, {"sum": 0.0, "count": 0.0, "drift_rows": 0.0})
            acc["sum"] += float(value)
            acc["count"] += 1.0
            if is_drift_event:
                acc["drift_rows"] += 1.0

        if not is_drift_event:
            continue

        for label, value, ref_value in zip(feature_labels, feature_values, ref_feature_values):
            key = (model_name, strategy, "feature", str(label))
            acc = drift_acc.setdefault(key, {"attention_sum": 0.0, "reference_sum": 0.0, "count": 0.0})
            acc["attention_sum"] += float(value)
            acc["reference_sum"] += float(ref_value)
            acc["count"] += 1.0

        for label, value, ref_value in zip(temporal_labels, temporal_values, ref_temporal_values):
            key = (model_name, strategy, "time", str(label))
            acc = drift_acc.setdefault(key, {"attention_sum": 0.0, "reference_sum": 0.0, "count": 0.0})
            acc["attention_sum"] += float(value)
            acc["reference_sum"] += float(ref_value)
            acc["count"] += 1.0

    feature_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "feature": label,
            "attention_mean": float(acc["sum"] / max(acc["count"], 1.0)),
            "n_rows": int(acc["count"]),
            "drift_event_rows": int(acc["drift_rows"]),
        }
        for (model_name, strategy, label), acc in feature_acc.items()
    ]
    temporal_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "time_step": label,
            "attention_mean": float(acc["sum"] / max(acc["count"], 1.0)),
            "n_rows": int(acc["count"]),
            "drift_event_rows": int(acc["drift_rows"]),
        }
        for (model_name, strategy, label), acc in temporal_acc.items()
    ]
    drift_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "scope": scope,
            "item": label,
            "reference_attention": float(acc["reference_sum"] / max(acc["count"], 1.0)),
            "drift_attention_mean": float(acc["attention_sum"] / max(acc["count"], 1.0)),
            "delta_attention": float((acc["attention_sum"] - acc["reference_sum"]) / max(acc["count"], 1.0)),
            "abs_delta_attention": float(
                abs((acc["attention_sum"] - acc["reference_sum"]) / max(acc["count"], 1.0))
            ),
            "n_drift_rows": int(acc["count"]),
        }
        for (model_name, strategy, scope, label), acc in drift_acc.items()
    ]

    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df = feature_df.sort_values(
            ["model", "strategy", "attention_mean", "feature"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    temporal_df = pd.DataFrame(temporal_rows)
    if not temporal_df.empty:
        temporal_df = temporal_df.sort_values(
            ["model", "strategy", "attention_mean", "time_step"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    drift_df = pd.DataFrame(drift_rows)
    if not drift_df.empty:
        drift_df = drift_df.sort_values(
            ["model", "strategy", "scope", "abs_delta_attention", "item"],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)

    return {
        "attention_feature_summary": feature_df,
        "attention_temporal_summary": temporal_df,
        "attention_drift_shift_summary": drift_df,
    }


def _build_detector_attention_outputs(detector_attention_rows: Sequence[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    temporal_acc: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    drift_acc: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for row in detector_attention_rows:
        temporal_values = _as_float_array(row.get("temporal_attention_mean"))
        temporal_labels = list(row.get("temporal_labels") or [])
        reference_values = _as_float_array(row.get("reference_temporal_attention_mean"))
        model_name = str(row.get("model"))
        strategy = str(row.get("strategy"))
        is_drift_event = bool(row.get("is_drift_event"))

        for label, value in zip(temporal_labels, temporal_values):
            key = (model_name, strategy, str(label))
            acc = temporal_acc.setdefault(key, {"sum": 0.0, "count": 0.0, "drift_rows": 0.0})
            acc["sum"] += float(value)
            acc["count"] += 1.0
            if is_drift_event:
                acc["drift_rows"] += 1.0

        if not is_drift_event:
            continue
        for label, value, ref_value in zip(temporal_labels, temporal_values, reference_values):
            key = (model_name, strategy, str(label))
            acc = drift_acc.setdefault(key, {"attention_sum": 0.0, "reference_sum": 0.0, "count": 0.0})
            acc["attention_sum"] += float(value)
            acc["reference_sum"] += float(ref_value)
            acc["count"] += 1.0

    temporal_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "time_step": label,
            "attention_mean": float(acc["sum"] / max(acc["count"], 1.0)),
            "n_rows": int(acc["count"]),
            "drift_event_rows": int(acc["drift_rows"]),
        }
        for (model_name, strategy, label), acc in temporal_acc.items()
    ]
    drift_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "time_step": label,
            "reference_attention": float(acc["reference_sum"] / max(acc["count"], 1.0)),
            "drift_attention_mean": float(acc["attention_sum"] / max(acc["count"], 1.0)),
            "delta_attention": float((acc["attention_sum"] - acc["reference_sum"]) / max(acc["count"], 1.0)),
            "abs_delta_attention": float(
                abs((acc["attention_sum"] - acc["reference_sum"]) / max(acc["count"], 1.0))
            ),
            "n_drift_rows": int(acc["count"]),
        }
        for (model_name, strategy, label), acc in drift_acc.items()
    ]

    temporal_df = pd.DataFrame(temporal_rows)
    if not temporal_df.empty:
        temporal_df = temporal_df.sort_values(
            ["model", "strategy", "attention_mean", "time_step"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    drift_df = pd.DataFrame(drift_rows)
    if not drift_df.empty:
        drift_df = drift_df.sort_values(
            ["model", "strategy", "abs_delta_attention", "time_step"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)

    return {
        "detector_attention_temporal_summary": temporal_df,
        "detector_attention_drift_shift_summary": drift_df,
    }


def run_backtest_pipeline(
    dataset_bundle: Dict[str, Any],
    *,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    df = _ensure_non_empty_dataframe(dataset_bundle.get("df"), label="Neural drift dataset")
    df = _subset_dataset_by_percentage(
        df,
        dataset_percent=float(config.get("dataset_percent", DEFAULT_CONFIG["dataset_percent"])),
    )
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
    attention_rows: List[Dict[str, Any]] = []
    detector_attention_rows: List[Dict[str, Any]] = []
    feature_metadata = {
        "lookback_steps": int(config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"])),
        "base_feature_cols": list(dataset.base_feature_cols),
        "augmented_feature_cols": list(dataset.augmented_feature_cols),
        "feature_count": int(len(dataset.augmented_feature_cols)),
    }

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
            feature_metadata=feature_metadata,
        )
        baseline_details = _predict_with_artifact_details(baseline_artifact, split["X_val"])
        baseline_raw_scores = baseline_details["probs"]
        baseline_embeddings = baseline_details["embeddings"]
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
                "monitor_effective_architecture": str(
                    baseline_artifact.get("monitor_effective_architecture", "not_available")
                ),
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
                feature_metadata=feature_metadata,
            )
            history_X = np.vstack([split["X_train"], split["X_val"]])
            history_y = np.concatenate([split["y_train"], split["y_val"]])

            detectors = {
                DRIFT_INPUT: ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
                DRIFT_SCORE: ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
                DRIFT_ERROR: ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
            }
            channel_histories: Dict[str, List[float]] = {
                DRIFT_INPUT: [],
                DRIFT_SCORE: [],
                DRIFT_ERROR: [],
                DRIFT_EMBEDDING: [],
            }
            embedding_buffer: List[np.ndarray] = []

            for idx in range(len(split["y_stream"])):
                x_row = split["X_stream"][idx : idx + 1]
                y_true = int(split["y_stream"][idx])
                timestamp = pd.Timestamp(split["metadata_stream"].loc[idx, "prediction_time"])

                prediction_details = _predict_with_artifact_details(artifact, x_row)
                raw_scores = prediction_details["probs"]
                embeddings = prediction_details["embeddings"]
                attention_summary = prediction_details.get("attention_summary")
                score = float(_apply_calibrator(raw_scores, artifact.get("calibrator"))[0])
                decision_threshold = float(artifact.get("decision_threshold", artifact.get("base_threshold", 0.5)))
                pred = int(score >= decision_threshold)
                pre_action_reference_attention = dict(artifact.get("attention_summary_reference") or {})
                pre_action_monitor = dict(artifact.get("embedding_monitor") or {})
                detector_attention_reference = dict(pre_action_monitor.get("attention_reference_summary") or {})
                recent_embedding_history = (
                    np.vstack(embedding_buffer).astype(float)
                    if embedding_buffer
                    else np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 and embeddings.shape[1] > 0 else 0), dtype=float)
                )
                channel_payload = _build_channel_scores(
                    artifact=artifact,
                    x_row=_apply_imputer(x_row, artifact["imputer"]).reshape(-1),
                    calibrated_score=score,
                    y_true=y_true,
                    embeddings=embeddings.reshape(-1),
                    recent_embedding_history=recent_embedding_history,
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
                    point_signal_weight=float(
                        config.get(
                            "drift_point_signal_weight",
                            DEFAULT_CONFIG["drift_point_signal_weight"],
                        )
                    ),
                )

                severity_score = float(channel_payload["severity_score"])
                max_channel_score = float(channel_payload.get("max_channel_score", severity_score))
                action_taken = "none"
                monitor_effective_architecture = str(channel_payload.get("monitor_effective_architecture", artifact.get("monitor_effective_architecture", "none")))
                monitor_warmup = bool(channel_payload.get("monitor_warmup", False))
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
                            "monitor_effective_architecture": monitor_effective_architecture,
                            "monitor_warmup": bool(monitor_warmup),
                        }
                    )

                if attention_summary is not None:
                    attention_rows.append(
                        {
                            "timestamp": timestamp,
                            "model": str(model_name),
                            "strategy": str(strategy),
                            "is_drift_event": bool(severity_triggered),
                            "feature_labels": list(attention_summary.get("feature_labels") or []),
                            "temporal_labels": list(attention_summary.get("temporal_labels") or []),
                            "feature_attention_mean": _as_float_array(attention_summary.get("feature_attention_mean")),
                            "temporal_attention_mean": _as_float_array(attention_summary.get("temporal_attention_mean")),
                            "reference_feature_attention_mean": _as_float_array(
                                pre_action_reference_attention.get("feature_attention_mean")
                            ),
                            "reference_temporal_attention_mean": _as_float_array(
                                pre_action_reference_attention.get("temporal_attention_mean")
                            ),
                        }
                    )

                detector_attention_summary = channel_payload.get("detector_attention_summary")
                if detector_attention_summary is not None:
                    detector_attention_rows.append(
                        {
                            "timestamp": timestamp,
                            "model": str(model_name),
                            "strategy": str(strategy),
                            "is_drift_event": bool(severity_triggered),
                            "temporal_labels": list(detector_attention_summary.get("temporal_labels") or []),
                            "temporal_attention_mean": _as_float_array(
                                detector_attention_summary.get("temporal_attention_mean")
                            ),
                            "reference_temporal_attention_mean": _as_float_array(
                                detector_attention_reference.get("temporal_attention_mean")
                            ),
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
                        "monitor_effective_architecture": monitor_effective_architecture,
                        "monitor_warmup": bool(monitor_warmup),
                    }
                )

                if embeddings.size > 0:
                    embedding_buffer.append(np.asarray(embeddings.reshape(-1), dtype=float))

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
            "decision_threshold",
            "channel_scores",
            "raw_channel_values",
            "detector_flags",
            "action_taken",
            "recent_rows",
            "recent_positive_rows",
            "monitor_effective_architecture",
            "monitor_warmup",
        ],
    )
    if not drift_df.empty:
        drift_df = drift_df.sort_values(["model", "strategy", "timestamp"]).reset_index(drop=True)

    summary_df = _summary_from_stream(stream_df, drift_df)
    rolling_df = _rolling_metric_table(
        stream_df,
        rolling_window=int(config.get("rolling_metric_window", DEFAULT_CONFIG["rolling_metric_window"])),
    )
    attention_outputs = _build_attention_outputs(attention_rows)
    detector_attention_outputs = _build_detector_attention_outputs(detector_attention_rows)
    return {
        "dataset": dataset,
        "split": split,
        "baseline": baseline_df,
        "summary": summary_df,
        "stream_metrics": stream_df,
        "rolling_metrics": rolling_df,
        "drift_events": drift_df,
        **attention_outputs,
        **detector_attention_outputs,
    }


def _download_bundle_from_results(results: Dict[str, Any]) -> Dict[str, str]:
    bundle: Dict[str, str] = {}
    for key in [
        "baseline",
        "summary",
        "stream_metrics",
        "rolling_metrics",
        "drift_events",
        "attention_feature_summary",
        "attention_temporal_summary",
        "attention_drift_shift_summary",
        "detector_attention_temporal_summary",
        "detector_attention_drift_shift_summary",
    ]:
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
    lookback_steps = int(config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"]))
    try:
        augmented_df, engineered_cols = augment_feature_frame(
            dataset_bundle["df"],
            dataset_bundle.get("feature_cols") or [],
        )
        input_dim = int(len(engineered_cols) * lookback_steps)
        augmented_feature_count = int(len(engineered_cols))
        rows = int(len(augmented_df))
    except Exception:
        input_dim = int(len(dataset_bundle.get("feature_cols") or [])) * lookback_steps
        augmented_feature_count = int(len(dataset_bundle.get("feature_cols") or []))
        fallback_df = dataset_bundle.get("df")
        rows = int(len(fallback_df)) if isinstance(fallback_df, pd.DataFrame) else 0

    embedding_dim = int(config.get("mlp_embedding_dim", DEFAULT_CONFIG["mlp_embedding_dim"]))
    monitor_hidden_dim = int(config.get("drift_monitor_hidden_dim", DEFAULT_CONFIG["drift_monitor_hidden_dim"]))
    monitor_bottleneck_dim = int(
        config.get("drift_monitor_bottleneck_dim", DEFAULT_CONFIG["drift_monitor_bottleneck_dim"])
    )
    monitor_sequence_length = int(
        config.get("drift_monitor_sequence_length", DEFAULT_CONFIG["drift_monitor_sequence_length"])
    )
    monitor_attention_hidden_dim = int(
        config.get(
            "drift_monitor_attention_hidden_dim",
            DEFAULT_CONFIG["drift_monitor_attention_hidden_dim"],
        )
    )
    return {
        "rows": rows,
        "augmented_feature_count": augmented_feature_count,
        "predictor_input_dim": input_dim,
        "predictor_time_steps": lookback_steps,
        "predictor_feature_count": augmented_feature_count,
        "predictor_embedding_dim": embedding_dim,
        "attention_feature_hidden_dim": int(
            config.get("attention_feature_hidden_dim", DEFAULT_CONFIG["attention_feature_hidden_dim"])
        ),
        "attention_temporal_hidden_dim": int(
            config.get("attention_temporal_hidden_dim", DEFAULT_CONFIG["attention_temporal_hidden_dim"])
        ),
        "monitor_input_dim": embedding_dim,
        "monitor_hidden_dim": max(monitor_bottleneck_dim + 1, monitor_hidden_dim),
        "monitor_bottleneck_dim": max(2, min(monitor_bottleneck_dim, max(2, embedding_dim - 1))),
        "monitor_sequence_length": monitor_sequence_length,
        "monitor_attention_hidden_dim": monitor_attention_hidden_dim,
    }


def _build_monitor_architecture_explanation(
    shapes: Dict[str, int],
    config: Dict[str, Any],
    reconstruction_weight: float,
) -> Dict[str, Any]:
    mlp_hidden_dim = int(config.get("mlp_hidden_dim", DEFAULT_CONFIG["mlp_hidden_dim"]))
    mlp_second_hidden = max(2, mlp_hidden_dim // 2)
    distance_weight = 1.0 - float(reconstruction_weight)
    predictor_time_steps = int(
        shapes.get("predictor_time_steps", config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"]))
    )
    predictor_feature_count = int(
        shapes.get("predictor_feature_count", shapes.get("augmented_feature_count", 0))
    )
    attention_feature_hidden_dim = int(
        shapes.get(
            "attention_feature_hidden_dim",
            config.get("attention_feature_hidden_dim", DEFAULT_CONFIG["attention_feature_hidden_dim"]),
        )
    )
    attention_temporal_hidden_dim = int(
        shapes.get(
            "attention_temporal_hidden_dim",
            config.get("attention_temporal_hidden_dim", DEFAULT_CONFIG["attention_temporal_hidden_dim"]),
        )
    )
    monitor_sequence_length = int(
        shapes.get(
            "monitor_sequence_length",
            config.get("drift_monitor_sequence_length", DEFAULT_CONFIG["drift_monitor_sequence_length"]),
        )
    )
    monitor_attention_hidden_dim = int(
        shapes.get(
            "monitor_attention_hidden_dim",
            config.get(
                "drift_monitor_attention_hidden_dim",
                DEFAULT_CONFIG["drift_monitor_attention_hidden_dim"],
            ),
        )
    )
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
        "attention_steps": [
            (
                f"`window[{predictor_time_steps}, {predictor_feature_count}]`",
                "Es la misma ventana del predictor, pero reordenada por tiempo y variables. Esto permite que la red "
                "aprenda que no todas las variables ni todos los pasos temporales pesan igual al construir el embedding."
            ),
            (
                f"`feature attention hidden[{attention_feature_hidden_dim}]`",
                "Primero la red asigna pesos a las variables dentro de cada paso temporal. Si una feature recibe mas peso, "
                "esa variable esta influyendo mas en la representacion latente usada para detectar drift."
            ),
            (
                f"`temporal attention hidden[{attention_temporal_hidden_dim}]`",
                "Luego la red pondera los distintos pasos del lookback. Esto le permite dar mas importancia a cambios recientes "
                "o a una secuencia especifica de eventos, no solo a un promedio plano de la ventana."
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
        "monitor_attention_steps": [
            (
                f"`sequence[{monitor_sequence_length}, embedding[{shapes['monitor_input_dim']}]]`",
                "La variante con attention del detector no ve solo un embedding aislado. Usa una secuencia reciente de embeddings "
                "para entender si el estado actual se puede explicar con el historial operativo inmediato."
            ),
            (
                f"`temporal attention hidden[{monitor_attention_hidden_dim}]`",
                "La red asigna pesos a los embeddings recientes y aprende que momentos del historial explican mejor el embedding actual."
            ),
            (
                f"`context -> bottleneck[{shapes['monitor_bottleneck_dim']}] -> reconstruction[{shapes['monitor_input_dim']}]`",
                "El contexto ponderado por attention se comprime y luego intenta reconstruir el embedding actual. Si esa reconstruccion falla, "
                "el detector interpreta que el estado reciente ya no explica bien el comportamiento actual."
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
                "`Feature attention hidden dim`",
                "Controla cuanta capacidad tiene la red para diferenciar variables relevantes dentro de cada paso temporal. "
                "Si sube demasiado, puede volverse menos interpretable y mas costosa."
            ),
            (
                "`Temporal attention hidden dim`",
                "Controla cuanta capacidad tiene la red para distinguir que momentos de la ventana son mas importantes para el embedding."
            ),
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
            (
                "`Top-k attention summary`",
                "No cambia el entrenamiento. Solo define cuantos items se muestran en la UI al resumir variables y pasos temporales mas atendidos."
            ),
            (
                "`Monitor sequence length`",
                "Define cuantos embeddings recientes usa el detector con attention. Ventanas mas largas capturan contexto, pero tardan mas en salir de warmup."
            ),
            (
                "`Monitor attention hidden dim`",
                "Controla cuanta capacidad tiene el detector para diferenciar que momentos del historial importan mas."
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
    if "neural_drift_dataset_percent" not in st.session_state:
        st.session_state["neural_drift_dataset_percent"] = int(DEFAULT_CONFIG["dataset_percent"])
    dataset_percent = st.slider(
        "Porcentaje del dataset para Neural drift",
        min_value=1,
        max_value=100,
        step=1,
        key="neural_drift_dataset_percent",
        help="Usa el tramo temporal mas reciente del dataset para acelerar experimentos manteniendo el orden temporal.",
    )
    active_df = _subset_dataset_by_percentage(df, dataset_percent=float(dataset_percent))

    metrics_col_1, metrics_col_2, metrics_col_3, metrics_col_4 = st.columns(4)
    metrics_col_1.metric("Source", str(dataset_bundle.get("source", "-")))
    metrics_col_2.metric("Rows totales", int(len(df)))
    metrics_col_3.metric("Rows usados", int(len(active_df)))
    metrics_col_4.metric("Features", int(len(dataset_bundle.get("feature_cols", []))))
    if "interval_start" in active_df.columns and not active_df.empty:
        active_start = pd.to_datetime(active_df["interval_start"], errors="coerce").min()
        active_end = pd.to_datetime(active_df["interval_start"], errors="coerce").max()
        if pd.notna(active_start) and pd.notna(active_end):
            st.caption(
                "El experimento usara el tramo mas reciente del dataset: "
                f"{pd.Timestamp(active_start)} -> {pd.Timestamp(active_end)}."
            )
    if dataset_bundle.get("feature_export_path"):
        st.caption(f"Export DuckDB: {dataset_bundle['feature_export_path']}")

    max_stream_min = 48
    max_stream_max = max(max_stream_min, int(len(df)))
    max_stream_default = min(int(DEFAULT_CONFIG["max_stream_rows"]), max_stream_max)
    max_stream_default = max(max_stream_min, max_stream_default)

    config_state_defaults = {
        "neural_drift_sensitivity_preset": str(DEFAULT_CONFIG["detector_sensitivity_preset"]),
        "neural_drift_recent_window_size": int(DEFAULT_CONFIG["recent_window_size"]),
        "neural_drift_severity_threshold": float(DEFAULT_CONFIG["severity_threshold"]),
        "neural_drift_detector_adwin_delta": float(DEFAULT_CONFIG["detector_adwin_delta"]),
        "neural_drift_point_signal_weight": float(DEFAULT_CONFIG["drift_point_signal_weight"]),
    }
    for state_key, default_value in config_state_defaults.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value

    def _on_detector_sensitivity_preset_change() -> None:
        _apply_detector_sensitivity_preset_to_session(st.session_state.get("neural_drift_sensitivity_preset"))

    st.markdown("**Sensibilidad del detector**")
    sensitivity_col_1, sensitivity_col_2 = st.columns([1.2, 2.2])
    sensitivity_preset = sensitivity_col_1.selectbox(
        "Preset de sensibilidad",
        AVAILABLE_DETECTOR_SENSITIVITY_PRESETS,
        key="neural_drift_sensitivity_preset",
        on_change=_on_detector_sensitivity_preset_change,
    )
    sensitivity_col_2.info(_detector_sensitivity_preset_description(sensitivity_preset))

    sensitivity_knob_col_1, sensitivity_knob_col_2 = st.columns(2)
    detector_adwin_delta = sensitivity_knob_col_1.number_input(
        "ADWIN delta",
        min_value=0.0005,
        max_value=0.0500,
        step=0.0005,
        format="%.4f",
        key="neural_drift_detector_adwin_delta",
    )
    point_signal_weight = sensitivity_knob_col_2.slider(
        "Point signal weight",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="neural_drift_point_signal_weight",
        help="0.0 prioriza la senal suavizada por ventana; 1.0 prioriza los picos locales y reproduce la sensibilidad actual.",
    )

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
        "dataset_percent": int(dataset_percent),
        "max_stream_rows": int(max_stream_rows),
        "models": list(selected_models),
        "strategies": list(selected_strategies),
        "drift_channels": list(selected_channels),
        "recent_window_size": int(recent_window_size),
        "recalibration_min_rows": int(recalibration_min_rows),
        "retrain_min_rows": int(retrain_min_rows),
        "severity_threshold": float(severity_threshold),
        "detector_sensitivity_preset": str(sensitivity_preset),
        "detector_adwin_delta": float(detector_adwin_delta),
        "drift_point_signal_weight": float(point_signal_weight),
        "rolling_metric_window": int(rolling_metric_window),
        "history_sample_size": int(history_sample_size),
    }
    st.session_state["neural_drift_config"] = config
    return config


def _render_monitor_network_subtab(dataset_bundle: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown("**Detector neuronal de drift**")
    st.caption(
        "La senal `embedding drift` combina distancia al centroide con un monitor neuronal sobre embeddings. "
        "Ese monitor puede ser un autoencoder clasico o una variante con attention temporal."
    )

    if not _has_any_torch_model(list(current_config.get("models") or [])):
        st.info("Para activar la red que monitorea drift debes incluir `Torch MLP` o `Torch MLP + Attention` en Models.")
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
        "neural_drift_monitor_architecture": str(
            current_config.get("drift_monitor_architecture", DEFAULT_CONFIG["drift_monitor_architecture"])
        ),
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
        "neural_drift_monitor_sequence_length": int(
            current_config.get("drift_monitor_sequence_length", DEFAULT_CONFIG["drift_monitor_sequence_length"])
        ),
        "neural_drift_monitor_attention_hidden_dim": int(
            current_config.get(
                "drift_monitor_attention_hidden_dim",
                DEFAULT_CONFIG["drift_monitor_attention_hidden_dim"],
            )
        ),
        "neural_drift_monitor_attention_dropout": float(
            current_config.get(
                "drift_monitor_attention_dropout",
                DEFAULT_CONFIG["drift_monitor_attention_dropout"],
            )
        ),
        "neural_drift_attention_feature_hidden_dim": int(
            current_config.get("attention_feature_hidden_dim", DEFAULT_CONFIG["attention_feature_hidden_dim"])
        ),
        "neural_drift_attention_temporal_hidden_dim": int(
            current_config.get("attention_temporal_hidden_dim", DEFAULT_CONFIG["attention_temporal_hidden_dim"])
        ),
        "neural_drift_attention_dropout": float(
            current_config.get("attention_dropout", DEFAULT_CONFIG["attention_dropout"])
        ),
        "neural_drift_attention_top_k": int(
            current_config.get("attention_top_k", DEFAULT_CONFIG["attention_top_k"])
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

    detector_arch_col_1, detector_arch_col_2 = st.columns([1.2, 2.2])
    monitor_architecture = detector_arch_col_1.selectbox(
        "Arquitectura del detector neuronal",
        AVAILABLE_DRIFT_MONITOR_ARCHITECTURES,
        key="neural_drift_monitor_architecture",
    )
    if str(monitor_architecture) == DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION:
        detector_arch_col_2.info(
            "El detector usara una secuencia reciente de embeddings y attention temporal para reconstruir el embedding actual."
        )
    else:
        detector_arch_col_2.info(
            "El detector reconstruye cada embedding de forma independiente con un autoencoder clasico."
        )

    arch_col_1, arch_col_2, arch_col_3 = st.columns(3)
    monitor_hidden_dim = arch_col_1.number_input(
        "Monitor hidden dim",
        min_value=4,
        max_value=128,
        step=2,
        key="neural_drift_monitor_hidden_dim",
    )
    monitor_bottleneck_dim = arch_col_2.number_input(
        "Monitor bottleneck dim",
        min_value=2,
        max_value=64,
        step=1,
        key="neural_drift_monitor_bottleneck_dim",
    )
    monitor_dropout = arch_col_3.slider(
        "Monitor dropout",
        min_value=0.0,
        max_value=0.5,
        step=0.05,
        key="neural_drift_monitor_dropout",
    )

    train_col_1, train_col_2, train_col_3 = st.columns(3)
    monitor_epochs = train_col_1.number_input(
        "Monitor epochs",
        min_value=4,
        max_value=60,
        step=2,
        key="neural_drift_monitor_epochs",
    )
    monitor_batch_size = train_col_2.number_input(
        "Monitor batch size",
        min_value=8,
        max_value=256,
        step=8,
        key="neural_drift_monitor_batch_size",
    )
    monitor_learning_rate = train_col_3.number_input(
        "Monitor learning rate",
        min_value=1e-4,
        max_value=1e-2,
        step=1e-4,
        format="%.4f",
        key="neural_drift_monitor_learning_rate",
    )

    mix_col_1, mix_col_2 = st.columns([2, 1])
    reconstruction_weight = mix_col_1.slider(
        "Peso reconstruction error",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="neural_drift_monitor_reconstruction_weight",
    )
    mix_col_2.metric("Peso distancia", f"{1.0 - float(reconstruction_weight):.2f}")

    st.markdown("**Attention en el detector neuronal**")
    st.caption(
        "Esta seccion controla la red que detecta drift sobre embeddings recientes. Es independiente de la attention del predictor."
    )
    detector_attention_col_1, detector_attention_col_2, detector_attention_col_3 = st.columns(3)
    monitor_sequence_length = detector_attention_col_1.number_input(
        "Monitor sequence length",
        min_value=4,
        max_value=48,
        step=1,
        key="neural_drift_monitor_sequence_length",
    )
    monitor_attention_hidden_dim = detector_attention_col_2.number_input(
        "Monitor attention hidden dim",
        min_value=4,
        max_value=128,
        step=4,
        key="neural_drift_monitor_attention_hidden_dim",
    )
    monitor_attention_dropout = detector_attention_col_3.slider(
        "Monitor attention dropout",
        min_value=0.0,
        max_value=0.5,
        step=0.05,
        key="neural_drift_monitor_attention_dropout",
    )

    st.markdown("**Attention en la MLP**")
    st.caption(
        "Esta variante agrega dual attention sobre la ventana temporal: primero pondera variables dentro de cada paso "
        "y luego pondera pasos temporales completos antes de construir el embedding."
    )
    if MODEL_TORCH_MLP_ATTENTION not in list(current_config.get("models") or []):
        st.info(
            "Los siguientes controles se activan cuando seleccionas `Torch MLP + Attention` en Models. "
            "La `Torch MLP` clasica sigue disponible para comparacion."
        )

    attention_col_1, attention_col_2, attention_col_3, attention_col_4 = st.columns(4)
    attention_feature_hidden_dim = attention_col_1.number_input(
        "Feature attention hidden dim",
        min_value=8,
        max_value=128,
        step=4,
        key="neural_drift_attention_feature_hidden_dim",
    )
    attention_temporal_hidden_dim = attention_col_2.number_input(
        "Temporal attention hidden dim",
        min_value=8,
        max_value=128,
        step=4,
        key="neural_drift_attention_temporal_hidden_dim",
    )
    attention_dropout = attention_col_3.slider(
        "Attention dropout",
        min_value=0.0,
        max_value=0.5,
        step=0.05,
        key="neural_drift_attention_dropout",
    )
    attention_top_k = attention_col_4.number_input(
        "Top-k attention summary",
        min_value=3,
        max_value=20,
        step=1,
        key="neural_drift_attention_top_k",
    )

    monitor_config = {
        **current_config,
        "drift_monitor_profile": str(monitor_profile),
        "drift_monitor_architecture": str(monitor_architecture),
        "drift_monitor_hidden_dim": int(monitor_hidden_dim),
        "drift_monitor_bottleneck_dim": int(monitor_bottleneck_dim),
        "drift_monitor_dropout": float(monitor_dropout),
        "drift_monitor_epochs": int(monitor_epochs),
        "drift_monitor_batch_size": int(monitor_batch_size),
        "drift_monitor_learning_rate": float(monitor_learning_rate),
        "drift_monitor_reconstruction_weight": float(reconstruction_weight),
        "drift_monitor_sequence_length": int(monitor_sequence_length),
        "drift_monitor_attention_hidden_dim": int(monitor_attention_hidden_dim),
        "drift_monitor_attention_dropout": float(monitor_attention_dropout),
        "attention_feature_hidden_dim": int(attention_feature_hidden_dim),
        "attention_temporal_hidden_dim": int(attention_temporal_hidden_dim),
        "attention_dropout": float(attention_dropout),
        "attention_top_k": int(attention_top_k),
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
        f"Predictor MLP + Attention: window[{shapes['predictor_time_steps']}, {shapes['predictor_feature_count']}] -> "
        f"feature_attention[{shapes['attention_feature_hidden_dim']}] -> temporal_attention[{shapes['attention_temporal_hidden_dim']}] -> "
        f"embedding[{shapes['predictor_embedding_dim']}] -> score[1]",
        f"Drift monitor AE: embedding[{shapes['monitor_input_dim']}] -> hidden[{shapes['monitor_hidden_dim']}] -> "
        f"bottleneck[{shapes['monitor_bottleneck_dim']}] -> hidden[{shapes['monitor_hidden_dim']}] -> reconstruction[{shapes['monitor_input_dim']}]",
        f"Drift monitor + Attention: sequence[{shapes['monitor_sequence_length']}, embedding[{shapes['monitor_input_dim']}]] -> "
        f"temporal_attention[{shapes['monitor_attention_hidden_dim']}] -> bottleneck[{shapes['monitor_bottleneck_dim']}] -> "
        f"reconstruction[{shapes['monitor_input_dim']}]",
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

        st.markdown("**3. Attention en el detector neuronal**")
        for title, body in explanation.get("monitor_attention_steps", []):
            st.markdown(f"- {title}: {body}")

        st.markdown("**4. Attention en la MLP**")
        for title, body in explanation.get("attention_steps", []):
            st.markdown(f"- {title}: {body}")

        st.markdown("**5. Como leer el score de drift**")
        st.markdown(explanation["score_formula"])
        for title, body in explanation["score_interpretation"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**6. Como ajustar los controles**")
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
        "attention_feature_summary": results.get("attention_feature_summary"),
        "attention_temporal_summary": results.get("attention_temporal_summary"),
        "attention_drift_shift_summary": results.get("attention_drift_shift_summary"),
        "detector_attention_temporal_summary": results.get("detector_attention_temporal_summary"),
        "detector_attention_drift_shift_summary": results.get("detector_attention_drift_shift_summary"),
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
    attention_feature_summary = stream_results.get("attention_feature_summary")
    attention_temporal_summary = stream_results.get("attention_temporal_summary")
    attention_drift_shift_summary = stream_results.get("attention_drift_shift_summary")
    detector_attention_temporal_summary = stream_results.get("detector_attention_temporal_summary")
    detector_attention_drift_shift_summary = stream_results.get("detector_attention_drift_shift_summary")
    drift_events = st.session_state.get("neural_drift_drift_events")
    download_bundle = dict(st.session_state.get("neural_drift_download_bundle") or {})
    current_signature = _build_run_signature(dataset_bundle, config)
    last_run_signature = st.session_state.get("neural_drift_last_run_signature")
    attention_top_k = max(1, int(config.get("attention_top_k", DEFAULT_CONFIG["attention_top_k"])))

    if not isinstance(summary, pd.DataFrame) or summary.empty:
        st.info("No hay resultados de Neural drift para mostrar.")
        return

    if last_run_signature != current_signature:
        st.warning(
            "Los resultados visibles no corresponden a la configuracion o fuente de features actual. "
            "Vuelve a ejecutar el backtest para ver resultados consistentes con el estado actual."
        )

    if (
        isinstance(summary, pd.DataFrame)
        and not summary.empty
        and str(config.get("drift_monitor_architecture", DEFAULT_CONFIG["drift_monitor_architecture"]))
        == DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION
        and summary["monitor_effective_architecture"].astype(str).eq(DRIFT_MONITOR_ARCH_CLASSIC_AE).any()
    ):
        st.warning(
            "Algunas corridas del detector neuronal cayeron automaticamente a `Autoencoder clasico` "
            "porque no habia suficientes embeddings para entrenar o usar `Attention temporal`."
        )

    if isinstance(baseline, pd.DataFrame) and not baseline.empty:
        st.markdown("**Baseline temporal**")
        st.dataframe(_streamlit_arrow_safe_df(baseline), width="stretch")

    st.markdown("**Comparativa model x strategy**")
    st.dataframe(_streamlit_arrow_safe_df(summary), width="stretch")

    if isinstance(drift_events, pd.DataFrame) and not drift_events.empty:
        st.markdown("**Drift events**")
        st.dataframe(_streamlit_arrow_safe_df(drift_events), width="stretch")

    if isinstance(attention_feature_summary, pd.DataFrame) and not attention_feature_summary.empty:
        st.markdown("**Attention sobre variables**")
        feature_display = (
            attention_feature_summary.groupby(["model", "strategy"], group_keys=False, dropna=False)
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(feature_display), width="stretch")

    if isinstance(attention_temporal_summary, pd.DataFrame) and not attention_temporal_summary.empty:
        st.markdown("**Attention sobre pasos temporales**")
        temporal_display = (
            attention_temporal_summary.groupby(["model", "strategy"], group_keys=False, dropna=False)
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(temporal_display), width="stretch")

    if isinstance(attention_drift_shift_summary, pd.DataFrame) and not attention_drift_shift_summary.empty:
        st.markdown("**Cambio de attention durante drift**")
        drift_display = (
            attention_drift_shift_summary.groupby(["model", "strategy", "scope"], group_keys=False, dropna=False)
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(drift_display), width="stretch")

    if isinstance(detector_attention_temporal_summary, pd.DataFrame) and not detector_attention_temporal_summary.empty:
        st.markdown("**Attention temporal del detector de drift**")
        detector_temporal_display = (
            detector_attention_temporal_summary.groupby(["model", "strategy"], group_keys=False, dropna=False)
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(detector_temporal_display), width="stretch")

    if isinstance(detector_attention_drift_shift_summary, pd.DataFrame) and not detector_attention_drift_shift_summary.empty:
        st.markdown("**Cambio de attention del detector durante drift**")
        detector_shift_display = (
            detector_attention_drift_shift_summary.groupby(["model", "strategy"], group_keys=False, dropna=False)
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(detector_shift_display), width="stretch")

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
        ("attention_feature_summary", "Download attention_feature_summary.csv"),
        ("attention_temporal_summary", "Download attention_temporal_summary.csv"),
        ("attention_drift_shift_summary", "Download attention_drift_shift_summary.csv"),
        ("detector_attention_temporal_summary", "Download detector_attention_temporal_summary.csv"),
        ("detector_attention_drift_shift_summary", "Download detector_attention_drift_shift_summary.csv"),
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
