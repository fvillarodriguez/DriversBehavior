#!/usr/bin/env python3
"""
Neural drift workspace for short-horizon crash prediction with adaptive backtesting.

The module is intentionally self-contained so the Drift detection view can mount it
through a minimal bridge without introducing an import cycle.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.neighbors import NearestNeighbors
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
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:
    SMOTE = None

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
NEURAL_DRIFT_RUNS_DIR = RESULTS_DIR / "neural_drift_runs"
NEURAL_DRIFT_RUN_TYPE = "neural_drift_backtest"
NEURAL_DRIFT_LIVE_STATUS_HEARTBEAT_STEPS = 25
NEURAL_DRIFT_RESULT_KEYS: Tuple[str, ...] = (
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
)

MODEL_XGBOOST = "XGBoost"
MODEL_TORCH_MLP = "Torch MLP"
MODEL_TORCH_MLP_ATTENTION = "Torch MLP + Attention"
AVAILABLE_MODELS = [MODEL_XGBOOST, MODEL_TORCH_MLP, MODEL_TORCH_MLP_ATTENTION]
XGB_PARALLEL_NEURAL_MODEL = MODEL_TORCH_MLP

BALANCE_MODE_NONE = "none"
BALANCE_MODE_SMOTE = "smote"
AVAILABLE_BALANCE_MODES = [BALANCE_MODE_NONE, BALANCE_MODE_SMOTE]

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
DRIFT_MONITOR_SOURCE_PREDICTOR_EMBEDDINGS = "predictor_embeddings"
DRIFT_MONITOR_SOURCE_XGB_PARALLEL_NEURAL_BRANCH = "xgb_parallel_neural_branch"
DRIFT_MONITOR_SOURCE_NOT_AVAILABLE = "not_available"

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

SMOTE_SAMPLING_STRATEGY_OPTIONS: Tuple[float, ...] = (0.05, 0.1, 0.3, 0.5, 1.0)
SMOTE_K_NEIGHBORS_OPTIONS: Tuple[int, ...] = (5, 10)

XGB_FINE_TUNE_SELECTION_F_BETA_RECALL = "f_beta_recall"
XGB_FINE_TUNE_SELECTION_PR_AUC = "pr_auc"
XGB_FINE_TUNE_SELECTION_BRIER = "brier"
XGB_FINE_TUNE_SELECTION_F1 = "f1"
XGB_FINE_TUNE_SELECTION_ROC_AUC = "roc_auc"
XGB_FINE_TUNE_SELECTION_BALANCED_F1 = "balanced_f1"
XGB_FINE_TUNE_SELECTION_MCC = "mcc"
AVAILABLE_XGB_FINE_TUNE_SELECTION_METRICS = [
    XGB_FINE_TUNE_SELECTION_F_BETA_RECALL,
    XGB_FINE_TUNE_SELECTION_PR_AUC,
    XGB_FINE_TUNE_SELECTION_BRIER,
    XGB_FINE_TUNE_SELECTION_F1,
    XGB_FINE_TUNE_SELECTION_ROC_AUC,
    XGB_FINE_TUNE_SELECTION_BALANCED_F1,
    XGB_FINE_TUNE_SELECTION_MCC,
]
XGB_FINE_TUNE_SELECTION_METRIC_LABELS: Dict[str, str] = {
    XGB_FINE_TUNE_SELECTION_F_BETA_RECALL: "F-beta recall",
    XGB_FINE_TUNE_SELECTION_PR_AUC: "PR-AUC",
    XGB_FINE_TUNE_SELECTION_BRIER: "Brier",
    XGB_FINE_TUNE_SELECTION_F1: "F1",
    XGB_FINE_TUNE_SELECTION_ROC_AUC: "ROC-AUC",
    XGB_FINE_TUNE_SELECTION_BALANCED_F1: "Balanced F1",
    XGB_FINE_TUNE_SELECTION_MCC: "MCC",
}

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
    "neural_drift_xgb_fine_tune_selection_metric": XGB_FINE_TUNE_SELECTION_F_BETA_RECALL,
    "neural_drift_active_run_id": None,
    "neural_drift_active_manifest_path": None,
    "neural_drift_history_selected_run_id": None,
    "neural_drift_loaded_checkpoint_run_id": None,
    "neural_drift_prepared_resume_run_id": None,
    "neural_drift_prepared_resume_manifest_path": None,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "interval_minutes": 5,
    "dataset_percent": 100,
    "split_mode": "fractions",
    "base_start": "2018-01-01",
    "base_end": "2018-12-31",
    "stream_start": "2019-01-01",
    "lookback_steps": 12,
    "horizon_steps": 1,
    "train_fraction": 0.60,
    "validation_fraction": 0.20,
    "balance_modes": [BALANCE_MODE_NONE],
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
    "xgb_parallel_neural_enabled": True,
    "xgb_fine_tune_estimators": 6,
    "xgb_fine_tune_selection_metric": XGB_FINE_TUNE_SELECTION_F_BETA_RECALL,
    "xgb_fine_tune_window_min": 32,
    "xgb_fine_tune_window_max": 160,
    "xgb_fine_tune_rounds_min": 2,
    "xgb_fine_tune_rounds_max": 24,
    "xgb_fine_tune_eta_multiplier_max": 1.75,
    "xgb_fine_tune_recent_weight_max": 4.0,
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


def _resolve_xgb_parallel_neural_enabled(value: Any) -> bool:
    if value is None:
        return bool(DEFAULT_CONFIG["xgb_parallel_neural_enabled"])
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _xgb_parallel_neural_enabled(config: Dict[str, Any]) -> bool:
    return _resolve_xgb_parallel_neural_enabled(
        config.get("xgb_parallel_neural_enabled", DEFAULT_CONFIG["xgb_parallel_neural_enabled"])
    )


def _artifact_has_xgb_parallel_neural_branch(artifact: Dict[str, Any]) -> bool:
    return (
        str(artifact.get("kind")) == "xgboost"
        and bool(artifact.get("parallel_neural_enabled", False))
        and isinstance(artifact.get("parallel_neural_branch"), dict)
    )


def _artifact_drift_monitor_source(artifact: Dict[str, Any]) -> str:
    if _artifact_has_xgb_parallel_neural_branch(artifact):
        return DRIFT_MONITOR_SOURCE_XGB_PARALLEL_NEURAL_BRANCH
    if str(artifact.get("kind")) == "xgboost":
        return DRIFT_MONITOR_SOURCE_NOT_AVAILABLE
    return DRIFT_MONITOR_SOURCE_PREDICTOR_EMBEDDINGS


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


def _load_json_file(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(dict(payload))
    except Exception:
        return []
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(_to_json_safe(payload), ensure_ascii=True, indent=2, sort_keys=True),
    )


def _atomic_write_df_csv(path: Path, df: pd.DataFrame) -> None:
    _atomic_write_text(path, df.to_csv(index=False))


def _append_jsonl_record(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json_safe(payload), ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def _resolve_balance_mode(value: Any) -> str:
    mode = str(value or BALANCE_MODE_NONE).strip().lower()
    if mode not in AVAILABLE_BALANCE_MODES:
        return BALANCE_MODE_NONE
    return mode


def _resolve_balance_modes(values: Optional[Sequence[Any]]) -> List[str]:
    modes = [
        _resolve_balance_mode(value)
        for value in list(values or [BALANCE_MODE_NONE])
    ]
    deduped = list(dict.fromkeys(mode for mode in modes if mode in AVAILABLE_BALANCE_MODES))
    return deduped or [BALANCE_MODE_NONE]


def _resolve_xgb_fine_tune_selection_metric(value: Any) -> str:
    metric = str(value or XGB_FINE_TUNE_SELECTION_F_BETA_RECALL).strip().lower()
    if metric not in AVAILABLE_XGB_FINE_TUNE_SELECTION_METRICS:
        return XGB_FINE_TUNE_SELECTION_F_BETA_RECALL
    return metric


def _xgb_fine_tune_selection_metric_label(value: Any) -> str:
    metric = _resolve_xgb_fine_tune_selection_metric(value)
    return str(XGB_FINE_TUNE_SELECTION_METRIC_LABELS.get(metric, metric))


def _balanced_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    if not labels:
        return 0.0
    per_class_f1 = np.asarray(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            average=None,
            zero_division=0,
        ),
        dtype=float,
    )
    if per_class_f1.size == 0 or np.any(per_class_f1 <= 0):
        return 0.0
    denominator = float(np.sum(1.0 / per_class_f1))
    if denominator <= 0:
        return 0.0
    return float(len(per_class_f1) / denominator)


def _normalize_smote_params(smote_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(smote_params or {})
    sampling_strategy = float(params.get("sampling_strategy", 1.0))
    k_neighbors = int(params.get("k_neighbors", 5))
    return {
        "sampling_strategy": float(np.clip(sampling_strategy, 0.001, 1.0)),
        "k_neighbors": max(1, int(k_neighbors)),
    }


def _smote_search_space(config: Dict[str, Any]) -> Dict[str, List[Any]]:
    sampling_values = [
        float(value)
        for value in list(config.get("smote_sampling_strategy_options") or SMOTE_SAMPLING_STRATEGY_OPTIONS)
        if float(value) > 0.0
    ]
    k_values = [
        int(value)
        for value in list(config.get("smote_k_neighbors_options") or SMOTE_K_NEIGHBORS_OPTIONS)
        if int(value) >= 1
    ]
    sampling_values = list(dict.fromkeys(sampling_values))
    k_values = list(dict.fromkeys(k_values))
    return {
        "sampling_strategy": sampling_values or list(SMOTE_SAMPLING_STRATEGY_OPTIONS),
        "k_neighbors": k_values or list(SMOTE_K_NEIGHBORS_OPTIONS),
    }


def _round_to_step(
    value: float,
    *,
    step: int = 8,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    step_value = max(1, int(step))
    rounded = int(step_value * round(float(value) / step_value))
    if min_value is not None:
        rounded = max(int(min_value), rounded)
    if max_value is not None:
        rounded = min(int(max_value), rounded)
    return int(rounded)


def _trigger_score(severity_score: float, max_channel_score: float) -> float:
    return float(max(float(severity_score), float(max_channel_score)))


def _severity_intensity(trigger_score: float, severity_threshold: float) -> float:
    denominator = max(1e-6, 1.0 - float(severity_threshold))
    intensity = (float(trigger_score) - float(severity_threshold)) / denominator
    return float(np.clip(intensity, 0.0, 1.0))


def _xgb_fine_tune_window_rows(severity_intensity: float, *, config: Dict[str, Any]) -> int:
    window_min = int(config.get("xgb_fine_tune_window_min", DEFAULT_CONFIG["xgb_fine_tune_window_min"]))
    window_max = int(config.get("xgb_fine_tune_window_max", DEFAULT_CONFIG["xgb_fine_tune_window_max"]))
    lower = max(8, min(window_min, window_max))
    upper = max(lower, window_max)
    target = lower + float(np.clip(severity_intensity, 0.0, 1.0)) * float(upper - lower)
    return _round_to_step(target, step=8, min_value=lower, max_value=upper)


def _sanitize_xgb_learning_rate(
    value: Any,
    *,
    fallback: float = 0.05,
    prefer_fallback_on_upper_overflow: bool = False,
) -> float:
    fallback_value = 0.05
    try:
        fallback_candidate = float(fallback)
        if np.isfinite(fallback_candidate) and fallback_candidate > 0.0:
            fallback_value = float(np.clip(fallback_candidate, 1e-6, 1.0))
    except Exception:
        fallback_value = 0.05

    try:
        learning_rate_value = float(value)
    except Exception:
        return fallback_value
    if not np.isfinite(learning_rate_value) or learning_rate_value <= 0.0:
        return fallback_value
    if learning_rate_value > 1.0:
        if prefer_fallback_on_upper_overflow:
            return fallback_value
        return 1.0
    return float(np.clip(learning_rate_value, 1e-6, 1.0))


def _xgb_fine_tune_eta_multiplier(severity_intensity: float, *, config: Dict[str, Any]) -> float:
    eta_max = max(
        1.0,
        float(
            config.get(
                "xgb_fine_tune_eta_multiplier_max",
                DEFAULT_CONFIG["xgb_fine_tune_eta_multiplier_max"],
            )
        ),
    )
    return float(1.0 + (eta_max - 1.0) * float(np.clip(severity_intensity, 0.0, 1.0)))


def _xgb_fine_tune_recent_weight_max(severity_intensity: float, *, config: Dict[str, Any]) -> float:
    weight_max = max(
        1.0,
        float(
            config.get(
                "xgb_fine_tune_recent_weight_max",
                DEFAULT_CONFIG["xgb_fine_tune_recent_weight_max"],
            )
        ),
    )
    return float(1.0 + (weight_max - 1.0) * float(np.clip(severity_intensity, 0.0, 1.0)))


def _xgb_recent_sample_weights(n_rows: int, recent_weight_max: float) -> np.ndarray:
    total_rows = max(0, int(n_rows))
    if total_rows <= 0:
        return np.asarray([], dtype=float)
    max_weight = max(1.0, float(recent_weight_max))
    if total_rows == 1:
        return np.asarray([max_weight], dtype=float)
    return np.linspace(1.0, max_weight, num=total_rows, dtype=float)


def _xgb_fine_tune_round_candidates(severity_intensity: float, *, config: Dict[str, Any]) -> List[int]:
    rounds_min = max(1, int(config.get("xgb_fine_tune_rounds_min", DEFAULT_CONFIG["xgb_fine_tune_rounds_min"])))
    rounds_max = max(
        rounds_min,
        int(config.get("xgb_fine_tune_rounds_max", DEFAULT_CONFIG["xgb_fine_tune_rounds_max"])),
    )
    anchor = int(round(rounds_min + float(np.clip(severity_intensity, 0.0, 1.0)) * float(rounds_max - rounds_min)))
    offsets = (-4, -2, 0, 2, 4)
    candidates = sorted(
        {
            max(rounds_min, min(rounds_max, anchor + int(offset)))
            for offset in offsets
        }
    )
    return candidates or [rounds_min]


def _default_xgb_fine_tune_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "severity_intensity": None,
        "xgb_adaptation_window_rows": None,
        "xgb_fine_tune_rounds_selected": None,
        "xgb_fine_tune_eta_multiplier": None,
        "xgb_fine_tune_recent_weight_max": None,
        "xgb_fine_tune_selection_metric": _resolve_xgb_fine_tune_selection_metric(
            config.get(
                "xgb_fine_tune_selection_metric",
                DEFAULT_CONFIG["xgb_fine_tune_selection_metric"],
            )
        ),
        "xgb_fine_tune_selection_score": None,
        "xgb_fine_tune_skip_reason": None,
    }


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
            "split_mode": str(config.get("split_mode", DEFAULT_CONFIG["split_mode"])),
            "base_start": str(config.get("base_start", DEFAULT_CONFIG["base_start"])),
            "base_end": str(config.get("base_end", DEFAULT_CONFIG["base_end"])),
            "stream_start": str(config.get("stream_start", DEFAULT_CONFIG["stream_start"])),
            "lookback_steps": int(config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"])),
            "horizon_steps": int(config.get("horizon_steps", DEFAULT_CONFIG["horizon_steps"])),
            "train_fraction": float(config.get("train_fraction", DEFAULT_CONFIG["train_fraction"])),
            "validation_fraction": float(config.get("validation_fraction", DEFAULT_CONFIG["validation_fraction"])),
            "balance_modes": _resolve_balance_modes(config.get("balance_modes")),
            "models": list(config.get("models") or []),
            "strategies": list(config.get("strategies") or []),
            "drift_channels": list(config.get("drift_channels") or []),
            "severity_threshold": float(config.get("severity_threshold", DEFAULT_CONFIG["severity_threshold"])),
            "recent_window_size": int(config.get("recent_window_size", DEFAULT_CONFIG["recent_window_size"])),
            "max_stream_rows": int(config.get("max_stream_rows", DEFAULT_CONFIG["max_stream_rows"])),
            "threshold_beta": float(config.get("threshold_beta", DEFAULT_CONFIG["threshold_beta"])),
            "xgb_fine_tune_estimators": int(
                config.get("xgb_fine_tune_estimators", DEFAULT_CONFIG["xgb_fine_tune_estimators"])
            ),
            "xgb_fine_tune_selection_metric": _resolve_xgb_fine_tune_selection_metric(
                config.get(
                    "xgb_fine_tune_selection_metric",
                    DEFAULT_CONFIG["xgb_fine_tune_selection_metric"],
                )
            ),
            "xgb_fine_tune_window_min": int(
                config.get("xgb_fine_tune_window_min", DEFAULT_CONFIG["xgb_fine_tune_window_min"])
            ),
            "xgb_fine_tune_window_max": int(
                config.get("xgb_fine_tune_window_max", DEFAULT_CONFIG["xgb_fine_tune_window_max"])
            ),
            "xgb_fine_tune_rounds_min": int(
                config.get("xgb_fine_tune_rounds_min", DEFAULT_CONFIG["xgb_fine_tune_rounds_min"])
            ),
            "xgb_fine_tune_rounds_max": int(
                config.get("xgb_fine_tune_rounds_max", DEFAULT_CONFIG["xgb_fine_tune_rounds_max"])
            ),
            "xgb_fine_tune_eta_multiplier_max": float(
                config.get(
                    "xgb_fine_tune_eta_multiplier_max",
                    DEFAULT_CONFIG["xgb_fine_tune_eta_multiplier_max"],
                )
            ),
            "xgb_fine_tune_recent_weight_max": float(
                config.get(
                    "xgb_fine_tune_recent_weight_max",
                    DEFAULT_CONFIG["xgb_fine_tune_recent_weight_max"],
                )
            ),
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


def _slugify_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "na"
    chars: List[str] = []
    previous_was_sep = False
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
            previous_was_sep = False
            continue
        if not previous_was_sep:
            chars.append("_")
            previous_was_sep = True
    slug = "".join(chars).strip("_")
    return slug or "na"


def _build_neural_drift_baseline_key(model_name: str, balance_mode: str) -> str:
    return f"{_slugify_token(model_name)}__{_slugify_token(balance_mode)}"


def _build_neural_drift_experiment_key(model_name: str, strategy: str, balance_mode: str) -> str:
    return (
        f"{_slugify_token(model_name)}__"
        f"{_slugify_token(strategy)}__"
        f"{_slugify_token(balance_mode)}"
    )


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


def _split_window_dataset_fixed_dates(
    dataset: WindowDataset,
    *,
    base_start: str,
    base_end: str,
    stream_start: str,
    validation_fraction: float,
    max_stream_rows: Optional[int] = None,
) -> Dict[str, Any]:
    metadata = dataset.metadata.copy()
    if metadata.empty or "prediction_time" not in metadata.columns:
        raise ValueError("Fixed-date split requires `prediction_time` metadata.")

    prediction_times = pd.to_datetime(metadata["prediction_time"], errors="coerce")
    base_start_ts = pd.Timestamp(base_start)
    base_end_ts = pd.Timestamp(base_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    stream_start_ts = pd.Timestamp(stream_start)
    if stream_start_ts <= base_end_ts:
        raise ValueError("`stream_start` must be strictly after `base_end`.")

    base_mask = prediction_times.between(base_start_ts, base_end_ts, inclusive="both")
    stream_mask = prediction_times.ge(stream_start_ts)

    base_indices = np.flatnonzero(base_mask.to_numpy(dtype=bool))
    stream_indices = np.flatnonzero(stream_mask.to_numpy(dtype=bool))
    if len(base_indices) < 15 or len(stream_indices) < 5:
        raise ValueError("The fixed-date split does not leave enough base or stream samples.")

    validation_count = max(1, int(math.floor(len(base_indices) * float(validation_fraction))))
    if len(base_indices) - validation_count < 10:
        raise ValueError("The 2018 base split does not leave enough samples for train/validation.")

    train_indices = base_indices[:-validation_count]
    val_indices = base_indices[-validation_count:]

    X_train = dataset.X[train_indices]
    y_train = dataset.y[train_indices]
    X_val = dataset.X[val_indices]
    y_val = dataset.y[val_indices]
    X_stream = dataset.X[stream_indices]
    y_stream = dataset.y[stream_indices]
    metadata_train = metadata.iloc[train_indices].reset_index(drop=True)
    metadata_val = metadata.iloc[val_indices].reset_index(drop=True)
    metadata_stream = metadata.iloc[stream_indices].reset_index(drop=True)

    if max_stream_rows is not None and int(max_stream_rows) > 0 and len(y_stream) > int(max_stream_rows):
        X_stream = X_stream[-int(max_stream_rows) :]
        y_stream = y_stream[-int(max_stream_rows) :]
        metadata_stream = metadata_stream.iloc[-int(max_stream_rows) :].reset_index(drop=True)

    if len(y_train) < 10 or len(y_val) < 5 or len(y_stream) < 5:
        raise ValueError("The fixed-date split did not leave enough samples for train/validation/stream.")

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
    try:
        mcc = float(matthews_corrcoef(y, pred_array)) if len(y) > 0 else float("nan")
    except Exception:
        mcc = float("nan")
    return {
        "roc_auc": _safe_auc(y, s),
        "pr_auc": _safe_pr_auc(y, s),
        "f1": float(f1_score(y, pred_array, zero_division=0)),
        "balanced_f1": float(_balanced_f1_score(y, pred_array)),
        "mcc": mcc,
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


def _simple_smote_balance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sampling_strategy: float,
    k_neighbors: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y).astype(int)
    classes, counts = np.unique(y_np, return_counts=True)
    if len(classes) != 2:
        return X_np, y_np

    majority_index = int(np.argmax(counts))
    minority_index = int(np.argmin(counts))
    majority_count = int(counts[majority_index])
    minority_count = int(counts[minority_index])
    minority_label = int(classes[minority_index])
    if minority_count < 2:
        return X_np, y_np

    target_minority = int(math.ceil(float(sampling_strategy) * majority_count))
    n_new = max(0, target_minority - minority_count)
    if n_new <= 0:
        return X_np, y_np

    X_minority = X_np[y_np == minority_label]
    effective_k = max(1, min(int(k_neighbors), minority_count - 1))
    neighbors = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    neighbors.fit(X_minority)
    _distances, neighbor_indices = neighbors.kneighbors(X_minority, return_distance=True)
    neighbor_indices = neighbor_indices[:, 1:]

    rng = np.random.default_rng(int(random_state))
    anchor_indices = rng.integers(0, minority_count, size=n_new)
    neighbor_choices = rng.integers(0, neighbor_indices.shape[1], size=n_new)
    chosen_neighbors = neighbor_indices[anchor_indices, neighbor_choices]
    alpha = rng.random(size=n_new)
    X_new = (
        (1.0 - alpha)[:, None] * X_minority[anchor_indices]
        + alpha[:, None] * X_minority[chosen_neighbors]
    )
    y_new = np.full(n_new, minority_label, dtype=y_np.dtype)
    return np.vstack([X_np, X_new]).astype(float), np.concatenate([y_np, y_new]).astype(int)


def _apply_smote_balance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sampling_strategy: float,
    k_neighbors: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y).astype(int)
    original_rows = int(len(y_np))
    base_info = {
        "applied": False,
        "original_rows": original_rows,
        "balanced_rows": original_rows,
        "sampling_strategy": float(sampling_strategy),
        "k_neighbors": int(k_neighbors),
    }
    if X_np.size == 0 or np.unique(y_np).size < 2:
        return X_np, y_np, dict(base_info, reason="single_class_or_empty")

    classes, counts = np.unique(y_np, return_counts=True)
    majority_count = int(counts.max())
    minority_count = int(counts.min())
    if minority_count < 2:
        return X_np, y_np, dict(base_info, reason="insufficient_minority")

    current_ratio = float(minority_count / majority_count) if majority_count > 0 else 1.0
    if current_ratio >= float(sampling_strategy):
        return X_np, y_np, dict(base_info, reason="already_at_or_above_target_ratio")

    effective_k = max(1, min(int(k_neighbors), minority_count - 1))
    if SMOTE is not None:
        sampler = SMOTE(
            sampling_strategy=float(sampling_strategy),
            k_neighbors=int(effective_k),
            random_state=int(random_state),
        )
        X_resampled, y_resampled = sampler.fit_resample(X_np, y_np)
    else:
        X_resampled, y_resampled = _simple_smote_balance(
            X_np,
            y_np,
            sampling_strategy=float(sampling_strategy),
            k_neighbors=int(effective_k),
            random_state=int(random_state),
        )
    return (
        np.asarray(X_resampled, dtype=float),
        np.asarray(y_resampled).astype(int),
        {
            "applied": True,
            "original_rows": original_rows,
            "balanced_rows": int(len(y_resampled)),
            "sampling_strategy": float(sampling_strategy),
            "k_neighbors": int(effective_k),
        },
    )


def _set_random_seed(seed: int) -> None:
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _resolve_torch_device():
    if torch is None:
        raise ImportError("Torch no esta disponible en el entorno activo.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and bool(mps_backend.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


_TorchModuleBase = nn.Module if nn is not None else object


class WindowMLP(_TorchModuleBase):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        _ensure_torch_available()
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


class WindowAttentionMLP(_TorchModuleBase):
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
        _ensure_torch_available()
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


class EmbeddingDriftAutoencoder(_TorchModuleBase):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int, dropout: float) -> None:
        _ensure_torch_available()
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


class TemporalAttentionEmbeddingMonitor(_TorchModuleBase):
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
        _ensure_torch_available()
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
    balance_mode: str = BALANCE_MODE_NONE,
    smote_params: Optional[Dict[str, Any]] = None,
    feature_metadata: Optional[Dict[str, Any]] = None,
    learning_rate: Optional[float] = None,
    epochs: Optional[int] = None,
    base_model_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _ensure_torch_available()
    _set_random_seed(int(config.get("random_state", 42)))
    model_family = _resolve_torch_model_family(model_name)
    resolved_balance_mode = _resolve_balance_mode(balance_mode)
    resolved_smote_params = _normalize_smote_params(smote_params)
    X_fit = np.asarray(X_train, dtype=float)
    y_fit = np.asarray(y_train).astype(int)
    imputer = _fit_imputer(X_fit)
    X_fit = _apply_imputer(X_fit, imputer)
    smote_fit_info = {
        "applied": False,
        "original_rows": int(len(y_fit)),
        "balanced_rows": int(len(y_fit)),
        "sampling_strategy": float(resolved_smote_params["sampling_strategy"]),
        "k_neighbors": int(resolved_smote_params["k_neighbors"]),
    }
    if resolved_balance_mode == BALANCE_MODE_SMOTE:
        X_fit, y_fit, smote_fit_info = _apply_smote_balance(
            X_fit,
            y_fit,
            sampling_strategy=float(resolved_smote_params["sampling_strategy"]),
            k_neighbors=int(resolved_smote_params["k_neighbors"]),
            random_state=int(config.get("random_state", DEFAULT_CONFIG["random_state"])),
        )

    X_train_imp = np.asarray(X_fit, dtype=float)
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

    device = _resolve_torch_device()
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

    pos = max(1, int(np.sum(y_fit == 1)))
    neg = max(1, int(np.sum(y_fit == 0)))
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
        torch.tensor(y_fit.astype(np.float32), dtype=torch.float32),
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
        "balance_mode": str(resolved_balance_mode),
        "smote_params": dict(resolved_smote_params) if resolved_balance_mode == BALANCE_MODE_SMOTE else {},
        "smote_fit_info": dict(smote_fit_info),
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
        "parallel_neural_enabled": False,
        "parallel_neural_model": "not_applicable",
        "drift_monitor_source": DRIFT_MONITOR_SOURCE_PREDICTOR_EMBEDDINGS,
    }


def _predict_torch_model_details(artifact: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    _ensure_torch_available()
    model = artifact["model"]
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]

    X_imp = _apply_imputer(X, imputer)
    X_scaled = scaler.transform(X_imp)
    device = _resolve_torch_device()
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


def _train_xgb_parallel_neural_branch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
    balance_mode: str,
    smote_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        _ensure_torch_available()
    except ImportError as exc:
        raise ImportError(
            "Torch no esta disponible; la rama neuronal paralela de `XGBoost` requiere Torch."
        ) from exc
    return _train_torch_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,
        model_name=XGB_PARALLEL_NEURAL_MODEL,
        balance_mode=balance_mode,
        smote_params=smote_params,
    )


def _merge_parallel_branch_reference_stats(
    reference: Dict[str, Any],
    parallel_branch_reference: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(reference)
    branch_reference = dict(parallel_branch_reference or {})
    aux_key_map = {
        "score_mean": "aux_score_mean",
        "score_std": "aux_score_std",
        "error_mean": "aux_error_mean",
        "error_std": "aux_error_std",
    }
    for src_key, dst_key in aux_key_map.items():
        if src_key in branch_reference:
            merged[dst_key] = float(branch_reference[src_key])
    for key in [
        "embedding_centroid",
        "embedding_distance_mean",
        "embedding_distance_std",
        "embedding_reconstruction_mean",
        "embedding_reconstruction_std",
    ]:
        if key not in branch_reference:
            continue
        value = branch_reference[key]
        if isinstance(value, np.ndarray):
            merged[key] = np.asarray(value, dtype=float).copy()
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _sync_xgb_parallel_branch_state(artifact: Dict[str, Any]) -> None:
    branch = artifact.get("parallel_neural_branch")
    enabled = (
        str(artifact.get("kind")) == "xgboost"
        and bool(artifact.get("parallel_neural_enabled", False))
        and isinstance(branch, dict)
    )
    artifact["parallel_neural_enabled"] = bool(enabled)
    artifact["parallel_neural_model"] = (
        str(branch.get("model_name", XGB_PARALLEL_NEURAL_MODEL))
        if enabled
        else DRIFT_MONITOR_SOURCE_NOT_AVAILABLE
    )
    artifact["drift_monitor_source"] = (
        DRIFT_MONITOR_SOURCE_XGB_PARALLEL_NEURAL_BRANCH
        if enabled
        else DRIFT_MONITOR_SOURCE_NOT_AVAILABLE
    )
    if enabled:
        artifact["embedding_monitor"] = branch.get("embedding_monitor")
        artifact["monitor_effective_architecture"] = str(
            branch.get("monitor_effective_architecture", branch.get("kind", "none"))
        )
        artifact["attention_summary_reference"] = branch.get("attention_summary_reference")
        return
    artifact["embedding_monitor"] = None
    artifact["monitor_effective_architecture"] = DRIFT_MONITOR_SOURCE_NOT_AVAILABLE
    artifact["attention_summary_reference"] = None


def _train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
    balance_mode: str = BALANCE_MODE_NONE,
    smote_params: Optional[Dict[str, Any]] = None,
    base_model: Optional[Any] = None,
    imputer: Optional[np.ndarray] = None,
    n_estimators_override: Optional[int] = None,
    learning_rate_override: Optional[float] = None,
    sample_weight: Optional[np.ndarray] = None,
    history_training_rows_base: int = 0,
    xgb_fine_tune_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    xgb = _import_external_xgboost()
    resolved_balance_mode = _resolve_balance_mode(balance_mode)
    resolved_smote_params = _normalize_smote_params(smote_params)
    X_fit = np.asarray(X_train, dtype=float)
    y_fit = np.asarray(y_train).astype(int)
    imputer_values = np.asarray(imputer, dtype=float) if imputer is not None else _fit_imputer(X_fit)
    X_fit = _apply_imputer(X_fit, imputer_values)
    sample_weight_values = None
    if sample_weight is not None:
        sample_weight_values = np.asarray(sample_weight, dtype=float).reshape(-1)
        if sample_weight_values.shape[0] != len(y_fit):
            raise ValueError("sample_weight must match the number of XGBoost fine-tuning rows.")
    smote_fit_info = {
        "applied": False,
        "original_rows": int(len(y_fit)),
        "balanced_rows": int(len(y_fit)),
        "sampling_strategy": float(resolved_smote_params["sampling_strategy"]),
        "k_neighbors": int(resolved_smote_params["k_neighbors"]),
    }
    if resolved_balance_mode == BALANCE_MODE_SMOTE:
        X_fit, y_fit, smote_fit_info = _apply_smote_balance(
            X_fit,
            y_fit,
            sampling_strategy=float(resolved_smote_params["sampling_strategy"]),
            k_neighbors=int(resolved_smote_params["k_neighbors"]),
            random_state=int(config.get("random_state", DEFAULT_CONFIG["random_state"])),
        )
        if sample_weight_values is not None and sample_weight_values.shape[0] != len(y_fit):
            synthetic_rows = max(0, int(len(y_fit) - sample_weight_values.shape[0]))
            if synthetic_rows > 0:
                max_recent_weight = float(sample_weight_values.max()) if sample_weight_values.size > 0 else 1.0
                sample_weight_values = np.concatenate(
                    [sample_weight_values, np.full(synthetic_rows, max_recent_weight, dtype=float)]
                )

    X_train_imp = np.asarray(X_fit, dtype=float)
    X_val_imp = _apply_imputer(X_val, imputer_values)

    pos = max(1, int(np.sum(y_fit == 1)))
    neg = max(1, int(np.sum(y_fit == 0)))
    learning_rate_value = _sanitize_xgb_learning_rate(
        learning_rate_override
        if learning_rate_override is not None
        else (base_model.get_params().get("learning_rate", 0.05) if base_model is not None else 0.05),
        fallback=_xgb_base_learning_rate({"model": base_model}) if base_model is not None else 0.05,
    )
    if base_model is None:
        estimator = xgb.XGBClassifier(
            n_estimators=int(n_estimators_override or config.get("xgb_estimators", DEFAULT_CONFIG["xgb_estimators"])),
            max_depth=3,
            learning_rate=learning_rate_value,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=int(config.get("random_state", 42)),
            scale_pos_weight=float(neg / pos),
        )
        fit_kwargs = {
            "verbose": False,
        }
        if sample_weight_values is not None:
            fit_kwargs["sample_weight"] = sample_weight_values
        estimator.fit(X_train_imp, y_fit, **fit_kwargs)
    else:
        params = dict(base_model.get_params())
        params["n_estimators"] = int(
            n_estimators_override
            or config.get("xgb_fine_tune_estimators", DEFAULT_CONFIG["xgb_fine_tune_estimators"])
        )
        params["scale_pos_weight"] = float(neg / pos)
        params["learning_rate"] = learning_rate_value
        estimator = xgb.XGBClassifier(**params)
        fit_kwargs = {
            "verbose": False,
            "xgb_model": base_model,
        }
        if sample_weight_values is not None:
            fit_kwargs["sample_weight"] = sample_weight_values
        estimator.fit(X_train_imp, y_fit, **fit_kwargs)
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
    parallel_neural_enabled = _xgb_parallel_neural_enabled(config)
    parallel_neural_branch = None
    if parallel_neural_enabled:
        parallel_neural_branch = _train_xgb_parallel_neural_branch(
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            balance_mode=str(resolved_balance_mode),
            smote_params=dict(resolved_smote_params) if resolved_balance_mode == BALANCE_MODE_SMOTE else None,
        )
        reference = _merge_parallel_branch_reference_stats(
            reference,
            parallel_neural_branch.get("reference"),
        )
    artifact = {
        "kind": "xgboost",
        "model_name": MODEL_XGBOOST,
        "balance_mode": str(resolved_balance_mode),
        "smote_params": dict(resolved_smote_params) if resolved_balance_mode == BALANCE_MODE_SMOTE else {},
        "smote_fit_info": dict(smote_fit_info),
        "model": estimator,
        "imputer": imputer_values,
        "calibrator": calibrator,
        "reference": reference,
        "decision_threshold": float(threshold_info["threshold"]),
        "threshold_info": threshold_info,
        "base_threshold": float(threshold_info["threshold"]),
        "history_training_rows": int(history_training_rows_base + len(y_train)),
        "xgb_fine_tune_metadata": dict(xgb_fine_tune_metadata or _default_xgb_fine_tune_metadata(config)),
        "parallel_neural_enabled": bool(parallel_neural_enabled),
        "parallel_neural_branch": parallel_neural_branch,
    }
    _sync_xgb_parallel_branch_state(artifact)
    return artifact


def _predict_xgboost(artifact: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model = artifact["model"]
    imputer = artifact["imputer"]
    X_imp = _apply_imputer(X, imputer)
    probs = model.predict_proba(X_imp)[:, 1].astype(float)
    return probs, None


def _predict_with_artifact(artifact: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    details = _predict_with_artifact_details(artifact, X)
    return np.asarray(details["probs"], dtype=float), np.asarray(details["embeddings"], dtype=float)


def _predict_with_artifact_details(artifact: Dict[str, Any], X: np.ndarray) -> Dict[str, Any]:
    kind = str(artifact.get("kind"))
    if kind == "torch_mlp":
        details = _predict_torch_model_details(artifact, X)
        return {
            **details,
            "auxiliary_raw_probs": np.asarray([], dtype=float),
            "auxiliary_probs": np.asarray([], dtype=float),
            "parallel_neural_enabled": False,
            "parallel_neural_model": "not_applicable",
            "drift_monitor_source": str(
                artifact.get("drift_monitor_source", DRIFT_MONITOR_SOURCE_PREDICTOR_EMBEDDINGS)
            ),
        }
    probs, _ = _predict_xgboost(artifact, X)
    auxiliary_raw_probs = np.asarray([], dtype=float)
    auxiliary_probs = np.asarray([], dtype=float)
    auxiliary_embeddings = np.empty((len(probs), 0), dtype=float)
    parallel_neural_enabled = bool(artifact.get("parallel_neural_enabled", False))
    if _artifact_has_xgb_parallel_neural_branch(artifact):
        branch_artifact = dict(artifact.get("parallel_neural_branch") or {})
        branch_details = _predict_torch_model_details(branch_artifact, X)
        auxiliary_raw_probs = np.asarray(branch_details.get("probs"), dtype=float).reshape(-1)
        auxiliary_probs = _apply_calibrator(auxiliary_raw_probs, branch_artifact.get("calibrator"))
        auxiliary_embeddings = np.asarray(branch_details.get("embeddings"), dtype=float)
    return {
        "probs": probs,
        "embeddings": auxiliary_embeddings,
        "attention_summary": None,
        "auxiliary_raw_probs": auxiliary_raw_probs,
        "auxiliary_probs": auxiliary_probs,
        "parallel_neural_enabled": parallel_neural_enabled,
        "parallel_neural_model": str(
            artifact.get(
                "parallel_neural_model",
                XGB_PARALLEL_NEURAL_MODEL if parallel_neural_enabled else DRIFT_MONITOR_SOURCE_NOT_AVAILABLE,
            )
        ),
        "drift_monitor_source": str(
            artifact.get("drift_monitor_source", _artifact_drift_monitor_source(artifact))
        ),
    }


def _train_model_artifact(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
    balance_mode: str = BALANCE_MODE_NONE,
    smote_params: Optional[Dict[str, Any]] = None,
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
            balance_mode=str(balance_mode),
            smote_params=smote_params,
            feature_metadata=feature_metadata,
        )
    if str(model_name) == MODEL_XGBOOST:
        return _train_xgboost_model(
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            balance_mode=str(balance_mode),
            smote_params=smote_params,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _artifact_selection_score(
    artifact: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, float, float, float]:
    prediction_details = _predict_with_artifact_details(artifact, X_val)
    calibrated_scores = _apply_calibrator(prediction_details["probs"], artifact.get("calibrator"))
    roc_auc = _safe_auc(y_val, calibrated_scores)
    pr_auc = _safe_pr_auc(y_val, calibrated_scores)
    smote_fit_info = dict(artifact.get("smote_fit_info") or {})
    applied = 1.0 if bool(smote_fit_info.get("applied", False)) else 0.0
    balanced_rows = float(smote_fit_info.get("balanced_rows", len(y_val)))
    return (
        -1.0 if not np.isfinite(roc_auc) else float(roc_auc),
        -1.0 if not np.isfinite(pr_auc) else float(pr_auc),
        applied,
        -balanced_rows,
    )


def _train_canonical_artifact(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    config: Dict[str, Any],
    balance_mode: str,
    feature_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_balance_mode = _resolve_balance_mode(balance_mode)
    if resolved_balance_mode != BALANCE_MODE_SMOTE:
        return _train_model_artifact(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            balance_mode=resolved_balance_mode,
            feature_metadata=feature_metadata,
        )

    search_space = _smote_search_space(config)
    best_artifact: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[float, float, float, float]] = None
    best_smote_params: Optional[Dict[str, Any]] = None
    for sampling_strategy in search_space["sampling_strategy"]:
        for k_neighbors in search_space["k_neighbors"]:
            candidate_config = dict(config)
            if str(model_name) == MODEL_XGBOOST and _xgb_parallel_neural_enabled(config):
                candidate_config["xgb_parallel_neural_enabled"] = False
            candidate_artifact = _train_model_artifact(
                model_name,
                X_train,
                y_train,
                X_val,
                y_val,
                config=candidate_config,
                balance_mode=resolved_balance_mode,
                smote_params={
                    "sampling_strategy": float(sampling_strategy),
                    "k_neighbors": int(k_neighbors),
                },
                feature_metadata=feature_metadata,
            )
            candidate_score = _artifact_selection_score(candidate_artifact, X_val, y_val)
            if best_score is None or candidate_score > best_score:
                best_artifact = candidate_artifact
                best_score = candidate_score
                best_smote_params = {
                    "sampling_strategy": float(sampling_strategy),
                    "k_neighbors": int(k_neighbors),
                }

    if best_smote_params is not None:
        return _train_model_artifact(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            balance_mode=resolved_balance_mode,
            smote_params=best_smote_params,
            feature_metadata=feature_metadata,
        )
    if best_artifact is None:
        return _train_model_artifact(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            config=config,
            balance_mode=resolved_balance_mode,
            smote_params=_normalize_smote_params(None),
            feature_metadata=feature_metadata,
        )
    return best_artifact


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

    device = _resolve_torch_device()
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

    device = _resolve_torch_device()
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
    device = _resolve_torch_device()
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


def _refresh_parallel_branch_from_recent(
    branch_artifact: Dict[str, Any],
    recent_X: np.ndarray,
    recent_y: np.ndarray,
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    branch_details = _predict_torch_model_details(branch_artifact, recent_X)
    branch_raw_scores = np.asarray(branch_details["probs"], dtype=float)
    branch_embeddings = np.asarray(branch_details["embeddings"], dtype=float)
    branch_scores = _apply_calibrator(branch_raw_scores, branch_artifact.get("calibrator"))
    branch_X_ref = _apply_imputer(recent_X, branch_artifact["imputer"])
    branch_monitor = branch_artifact.get("embedding_monitor")
    reconstruction_errors = None
    if branch_embeddings.size > 0:
        branch_monitor, reconstruction_errors = _fit_embedding_monitor(branch_embeddings, config=config)
        branch_artifact["embedding_monitor"] = branch_monitor
        if branch_monitor is not None:
            branch_artifact["monitor_effective_architecture"] = str(
                branch_monitor.get("monitor_effective_architecture", branch_monitor.get("kind", "none"))
            )
    branch_artifact["reference"] = _build_reference_stats(
        X_ref=branch_X_ref,
        y_ref=recent_y,
        calibrated_scores=branch_scores,
        embeddings=branch_embeddings if branch_embeddings.size else None,
        reconstruction_errors=reconstruction_errors,
    )
    branch_artifact["attention_summary_reference"] = branch_details.get("attention_summary")
    branch_artifact["drift_monitor_source"] = DRIFT_MONITOR_SOURCE_PREDICTOR_EMBEDDINGS
    return {
        "details": branch_details,
        "calibrated_scores": branch_scores,
        "reconstruction_errors": reconstruction_errors,
    }


def _refresh_reference_from_recent(
    artifact: Dict[str, Any],
    recent_X: np.ndarray,
    recent_y: np.ndarray,
    *,
    config: Dict[str, Any],
) -> None:
    if _artifact_has_xgb_parallel_neural_branch(artifact):
        prediction_details = _predict_with_artifact_details(artifact, recent_X)
        raw_scores = np.asarray(prediction_details["probs"], dtype=float)
        calibrated_scores = _apply_calibrator(raw_scores, artifact.get("calibrator"))
        X_ref = _apply_imputer(recent_X, artifact["imputer"])
        branch_artifact = artifact["parallel_neural_branch"]
        branch_refresh = _refresh_parallel_branch_from_recent(
            branch_artifact,
            recent_X,
            recent_y,
            config=config,
        )
        branch_embeddings = np.asarray(branch_refresh["details"].get("embeddings"), dtype=float)
        artifact["reference"] = _merge_parallel_branch_reference_stats(
            _build_reference_stats(
                X_ref=X_ref,
                y_ref=recent_y,
                calibrated_scores=calibrated_scores,
                embeddings=branch_embeddings if branch_embeddings.size else None,
                reconstruction_errors=branch_refresh.get("reconstruction_errors"),
            ),
            branch_artifact.get("reference"),
        )
        _sync_xgb_parallel_branch_state(artifact)
        return
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
    artifact["drift_monitor_source"] = _artifact_drift_monitor_source(artifact)


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
    auxiliary_calibrated_score: Optional[float] = None,
) -> Dict[str, Any]:
    reference = dict(artifact.get("reference") or {})
    available_scores: Dict[str, float] = {}
    channel_score_details: Dict[str, Any] = {}
    detector_flags: Dict[str, Any] = {}
    raw_channel_values: Dict[str, Any] = {}
    point_weight = float(np.clip(point_signal_weight, 0.0, 1.0))
    detector_attention_summary: Optional[Dict[str, Any]] = None
    monitor_warmup = False
    monitor_effective_architecture = str(artifact.get("monitor_effective_architecture", "none"))
    drift_monitor_source = str(
        artifact.get("drift_monitor_source", _artifact_drift_monitor_source(artifact))
    )

    history_limit = max(8, int(recent_window_size))

    def _window_stat(channel_name: str, value: float) -> float:
        if channel_histories is None:
            return float(value)
        history = channel_histories.setdefault(str(channel_name), [])
        history.append(float(value))
        if len(history) > history_limit:
            del history[:-history_limit]
        return float(np.mean(history))

    def _component_channel_payload(
        *,
        value: float,
        history_key: str,
        detector_key: str,
        reference_mean: float,
        reference_std: float,
    ) -> Dict[str, Any]:
        window_value = _window_stat(history_key, value)
        point_score = _normalize_score(value, reference_mean, reference_std)
        window_score = _normalize_score(window_value, reference_mean, reference_std)
        detector = detectors.get(detector_key)
        drift_flag = bool(detector.update(value)) if detector is not None else False
        peak_score = float(max(point_score, window_score))
        blended_score = float((1.0 - point_weight) * window_score + point_weight * peak_score)
        return {
            "raw_value": float(value),
            "window_value": float(window_value),
            "score": float(max(blended_score, 1.0 if drift_flag else 0.0)),
            "flag": bool(drift_flag),
        }

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
        xgb_score_payload = _component_channel_payload(
            value=float(calibrated_score),
            history_key=DRIFT_SCORE,
            detector_key=DRIFT_SCORE,
            reference_mean=float(reference.get("score_mean", 0.5)),
            reference_std=float(reference.get("score_std", 0.15)),
        )
        combined_score_payload = dict(xgb_score_payload)
        score_components = {
            "xgboost": float(xgb_score_payload["score"]),
        }
        score_raw_components = {
            "xgboost": float(xgb_score_payload["raw_value"]),
        }
        score_flag_components = {
            "xgboost": bool(xgb_score_payload["flag"]),
        }
        auxiliary_score_ready = (
            auxiliary_calibrated_score is not None
            and np.isfinite(float(auxiliary_calibrated_score))
            and "aux_score_mean" in reference
            and "aux_score_std" in reference
        )
        if auxiliary_score_ready:
            neural_score_payload = _component_channel_payload(
                value=float(auxiliary_calibrated_score),
                history_key="score_neural",
                detector_key="score_neural",
                reference_mean=float(reference.get("aux_score_mean", reference.get("score_mean", 0.5))),
                reference_std=float(reference.get("aux_score_std", reference.get("score_std", 0.15))),
            )
            combined_score_payload = (
                neural_score_payload
                if float(neural_score_payload["score"]) > float(xgb_score_payload["score"])
                else xgb_score_payload
            )
            score_components["parallel_neural"] = float(neural_score_payload["score"])
            score_raw_components["parallel_neural"] = float(neural_score_payload["raw_value"])
            score_flag_components["parallel_neural"] = bool(neural_score_payload["flag"])
        available_scores[DRIFT_SCORE] = float(combined_score_payload["score"])
        raw_channel_values[DRIFT_SCORE] = float(combined_score_payload["raw_value"])
        detector_flags[DRIFT_SCORE] = bool(any(score_flag_components.values()))
        if len(score_components) > 1:
            score_components["combined"] = float(combined_score_payload["score"])
            score_raw_components["combined"] = float(combined_score_payload["raw_value"])
            score_flag_components["combined"] = bool(any(score_flag_components.values()))
            channel_score_details["score_components"] = score_components
            raw_channel_values["score_components"] = score_raw_components
            detector_flags["score_components"] = score_flag_components

    if DRIFT_ERROR in selected_channels:
        xgb_error_payload = _component_channel_payload(
            value=float((float(calibrated_score) - int(y_true)) ** 2),
            history_key=DRIFT_ERROR,
            detector_key=DRIFT_ERROR,
            reference_mean=float(reference.get("error_mean", 0.0)),
            reference_std=float(reference.get("error_std", 0.1)),
        )
        combined_error_payload = dict(xgb_error_payload)
        error_components = {
            "xgboost": float(xgb_error_payload["score"]),
        }
        error_raw_components = {
            "xgboost": float(xgb_error_payload["raw_value"]),
        }
        error_flag_components = {
            "xgboost": bool(xgb_error_payload["flag"]),
        }
        auxiliary_error_ready = (
            auxiliary_calibrated_score is not None
            and np.isfinite(float(auxiliary_calibrated_score))
            and "aux_error_mean" in reference
            and "aux_error_std" in reference
        )
        if auxiliary_error_ready:
            neural_error_payload = _component_channel_payload(
                value=float((float(auxiliary_calibrated_score) - int(y_true)) ** 2),
                history_key="error_neural",
                detector_key="error_neural",
                reference_mean=float(reference.get("aux_error_mean", reference.get("error_mean", 0.0))),
                reference_std=float(reference.get("aux_error_std", reference.get("error_std", 0.1))),
            )
            combined_error_payload = (
                neural_error_payload
                if float(neural_error_payload["score"]) > float(xgb_error_payload["score"])
                else xgb_error_payload
            )
            error_components["parallel_neural"] = float(neural_error_payload["score"])
            error_raw_components["parallel_neural"] = float(neural_error_payload["raw_value"])
            error_flag_components["parallel_neural"] = bool(neural_error_payload["flag"])
        available_scores[DRIFT_ERROR] = float(combined_error_payload["score"])
        raw_channel_values[DRIFT_ERROR] = float(combined_error_payload["raw_value"])
        detector_flags[DRIFT_ERROR] = bool(any(error_flag_components.values()))
        if len(error_components) > 1:
            error_components["combined"] = float(combined_error_payload["score"])
            error_raw_components["combined"] = float(combined_error_payload["raw_value"])
            error_flag_components["combined"] = bool(any(error_flag_components.values()))
            channel_score_details["error_components"] = error_components
            raw_channel_values["error_components"] = error_raw_components
            detector_flags["error_components"] = error_flag_components

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
        "channel_scores": {**available_scores, **channel_score_details},
        "raw_channel_values": raw_channel_values,
        "detector_flags": detector_flags,
        "severity_score": severity,
        "max_channel_score": max_channel_score,
        "severity_label": _severity_label(severity),
        "detector_attention_summary": detector_attention_summary,
        "monitor_warmup": bool(monitor_warmup),
        "monitor_effective_architecture": monitor_effective_architecture,
        "drift_monitor_source": drift_monitor_source,
    }


def _split_recent_for_adaptation(X_recent: np.ndarray, y_recent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _temporal_train_val_split_arrays(X_recent, y_recent, validation_fraction=0.2)


def _xgb_base_learning_rate(artifact: Dict[str, Any]) -> float:
    model = artifact.get("model")
    if model is not None and hasattr(model, "get_params"):
        try:
            return _sanitize_xgb_learning_rate(
                model.get_params().get("learning_rate", 0.05),
                fallback=0.05,
                prefer_fallback_on_upper_overflow=True,
            )
        except Exception:
            return 0.05
    return 0.05


def _select_xgb_fine_tune_window(
    X_available: np.ndarray,
    y_available: np.ndarray,
    *,
    severity_intensity: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    available_rows = int(len(y_available))
    window_min = int(config.get("xgb_fine_tune_window_min", DEFAULT_CONFIG["xgb_fine_tune_window_min"]))
    window_max = int(config.get("xgb_fine_tune_window_max", DEFAULT_CONFIG["xgb_fine_tune_window_max"]))
    step = 8
    effective_window_max = max(window_min, min(window_max, available_rows))

    if available_rows < window_min:
        return {
            "applied": False,
            "skip_reason": "insufficient_available_rows",
            "window_rows": available_rows,
            "X_recent": np.asarray(X_available, dtype=float),
            "y_recent": np.asarray(y_available).astype(int),
            "X_train": np.asarray([], dtype=float),
            "X_val": np.asarray([], dtype=float),
            "y_train": np.asarray([], dtype=int),
            "y_val": np.asarray([], dtype=int),
        }

    window_rows = min(
        effective_window_max,
        _xgb_fine_tune_window_rows(severity_intensity, config=config),
    )
    if window_rows < window_min:
        window_rows = min(effective_window_max, window_min)

    last_payload: Optional[Dict[str, Any]] = None
    while True:
        start = max(0, available_rows - int(window_rows))
        X_recent = np.asarray(X_available[start:], dtype=float)
        y_recent = np.asarray(y_available[start:]).astype(int)
        X_train, X_val, y_train, y_val = _split_recent_for_adaptation(X_recent, y_recent)
        train_ok = len(y_train) > 0 and len(np.unique(y_train)) >= 2
        val_ok = len(y_val) > 0 and len(np.unique(y_val)) >= 2
        last_payload = {
            "applied": bool(train_ok and val_ok),
            "skip_reason": None if (train_ok and val_ok) else "insufficient_class_support_after_expansion",
            "window_rows": int(window_rows),
            "X_recent": X_recent,
            "y_recent": y_recent,
            "X_train": np.asarray(X_train, dtype=float),
            "X_val": np.asarray(X_val, dtype=float),
            "y_train": np.asarray(y_train).astype(int),
            "y_val": np.asarray(y_val).astype(int),
        }
        if train_ok and val_ok:
            return last_payload
        if int(window_rows) >= effective_window_max or int(window_rows) >= available_rows:
            return last_payload
        window_rows = min(effective_window_max, available_rows, int(window_rows) + step)


def _sort_metric_desc(value: float) -> float:
    return -1.0 if not np.isfinite(value) else float(value)


def _sort_metric_asc(value: float) -> float:
    return float("-inf") if not np.isfinite(value) else -float(value)


def _evaluate_xgb_fine_tune_candidate(
    artifact: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    selection_metric: str,
    rounds_selected: int,
) -> Dict[str, Any]:
    prediction_details = _predict_with_artifact_details(artifact, X_val)
    calibrated_scores = _apply_calibrator(prediction_details["probs"], artifact.get("calibrator"))
    threshold_info = dict(artifact.get("threshold_info") or {})
    threshold_value = float(
        threshold_info.get("threshold", artifact.get("decision_threshold", artifact.get("base_threshold", 0.5)))
    )
    metrics = _classification_metrics(y_val, calibrated_scores, threshold=threshold_value)
    f_beta = float(threshold_info.get("f_beta", float("nan")))
    recall = float(threshold_info.get("recall", metrics["recall"]))
    f1 = float(metrics["f1"])
    roc_auc = float(metrics["roc_auc"])
    balanced_f1 = float(metrics["balanced_f1"])
    mcc = float(metrics["mcc"])
    pr_auc = float(metrics["pr_auc"])
    brier = float(metrics["brier"])
    resolved_metric = _resolve_xgb_fine_tune_selection_metric(selection_metric)

    if resolved_metric == XGB_FINE_TUNE_SELECTION_PR_AUC:
        sort_key = (
            _sort_metric_desc(pr_auc),
            _sort_metric_desc(roc_auc),
            _sort_metric_desc(f1),
            _sort_metric_desc(balanced_f1),
            _sort_metric_desc(mcc),
            _sort_metric_desc(f_beta),
            _sort_metric_desc(recall),
            _sort_metric_asc(brier),
            -float(rounds_selected),
        )
        selection_score = pr_auc
    elif resolved_metric == XGB_FINE_TUNE_SELECTION_ROC_AUC:
        sort_key = (
            _sort_metric_desc(roc_auc),
            _sort_metric_desc(pr_auc),
            _sort_metric_desc(f1),
            _sort_metric_desc(balanced_f1),
            _sort_metric_desc(mcc),
            _sort_metric_desc(f_beta),
            _sort_metric_desc(recall),
            _sort_metric_asc(brier),
            -float(rounds_selected),
        )
        selection_score = roc_auc
    elif resolved_metric == XGB_FINE_TUNE_SELECTION_BRIER:
        sort_key = (
            _sort_metric_asc(brier),
            _sort_metric_desc(roc_auc),
            _sort_metric_desc(pr_auc),
            _sort_metric_desc(f1),
            _sort_metric_desc(balanced_f1),
            _sort_metric_desc(mcc),
            _sort_metric_desc(f_beta),
            _sort_metric_desc(recall),
            -float(rounds_selected),
        )
        selection_score = brier
    elif resolved_metric == XGB_FINE_TUNE_SELECTION_F1:
        sort_key = (
            _sort_metric_desc(f1),
            _sort_metric_desc(balanced_f1),
            _sort_metric_desc(mcc),
            _sort_metric_desc(roc_auc),
            _sort_metric_desc(pr_auc),
            _sort_metric_desc(f_beta),
            _sort_metric_desc(recall),
            _sort_metric_asc(brier),
            -float(rounds_selected),
        )
        selection_score = f1
    elif resolved_metric == XGB_FINE_TUNE_SELECTION_BALANCED_F1:
        sort_key = (
            _sort_metric_desc(balanced_f1),
            _sort_metric_desc(f1),
            _sort_metric_desc(mcc),
            _sort_metric_desc(roc_auc),
            _sort_metric_desc(pr_auc),
            _sort_metric_desc(f_beta),
            _sort_metric_desc(recall),
            _sort_metric_asc(brier),
            -float(rounds_selected),
        )
        selection_score = balanced_f1
    elif resolved_metric == XGB_FINE_TUNE_SELECTION_MCC:
        sort_key = (
            _sort_metric_desc(mcc),
            _sort_metric_desc(balanced_f1),
            _sort_metric_desc(f1),
            _sort_metric_desc(roc_auc),
            _sort_metric_desc(pr_auc),
            _sort_metric_desc(f_beta),
            _sort_metric_desc(recall),
            _sort_metric_asc(brier),
            -float(rounds_selected),
        )
        selection_score = mcc
    else:
        sort_key = (
            _sort_metric_desc(f_beta),
            _sort_metric_desc(recall),
            _sort_metric_desc(f1),
            _sort_metric_desc(balanced_f1),
            _sort_metric_desc(mcc),
            _sort_metric_desc(roc_auc),
            _sort_metric_desc(pr_auc),
            _sort_metric_asc(brier),
            -float(rounds_selected),
        )
        selection_score = f_beta

    return {
        "selection_metric": resolved_metric,
        "selection_score": float(selection_score),
        "f_beta": f_beta,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "balanced_f1": balanced_f1,
        "mcc": mcc,
        "pr_auc": pr_auc,
        "brier": brier,
        "sort_key": sort_key,
    }


def _recalibrate_artifact(artifact: Dict[str, Any], X_recent: np.ndarray, y_recent: np.ndarray, *, config: Dict[str, Any]) -> None:
    if _artifact_has_xgb_parallel_neural_branch(artifact):
        raw_scores, _embeddings = _predict_xgboost(artifact, X_recent)
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
        _recalibrate_artifact(artifact["parallel_neural_branch"], X_recent, y_recent, config=config)
        _refresh_reference_from_recent(artifact, X_recent, y_recent, config=config)
        return
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
    severity_intensity: Optional[float] = None,
) -> Dict[str, Any]:
    artifact_kind = str(artifact.get("kind"))
    if artifact_kind == "xgboost":
        resolved_intensity = float(np.clip(float(severity_intensity or 0.0), 0.0, 1.0))
        window_payload = _select_xgb_fine_tune_window(
            X_recent,
            y_recent,
            severity_intensity=resolved_intensity,
            config=config,
        )
        selection_metric = _resolve_xgb_fine_tune_selection_metric(
            config.get(
                "xgb_fine_tune_selection_metric",
                DEFAULT_CONFIG["xgb_fine_tune_selection_metric"],
            )
        )
        metadata = {
            **_default_xgb_fine_tune_metadata(config),
            "severity_intensity": resolved_intensity,
            "xgb_adaptation_window_rows": int(window_payload.get("window_rows", 0) or 0),
            "xgb_fine_tune_selection_metric": selection_metric,
        }
        if not bool(window_payload.get("applied", False)):
            metadata["xgb_fine_tune_skip_reason"] = str(
                window_payload.get("skip_reason") or "insufficient_class_support_after_expansion"
            )
            artifact["xgb_fine_tune_metadata"] = metadata
            return {
                "applied": False,
                "xgb_fine_tune_metadata": metadata,
            }

        eta_multiplier = _xgb_fine_tune_eta_multiplier(resolved_intensity, config=config)
        recent_weight_max = _xgb_fine_tune_recent_weight_max(resolved_intensity, config=config)
        learning_rate_value = _sanitize_xgb_learning_rate(
            _xgb_base_learning_rate(artifact) * eta_multiplier,
            fallback=_xgb_base_learning_rate(artifact),
        )
        sample_weight = _xgb_recent_sample_weights(len(window_payload["y_train"]), recent_weight_max)

        best_artifact: Optional[Dict[str, Any]] = None
        best_candidate: Optional[Dict[str, Any]] = None
        candidate_config = dict(config)
        if _artifact_has_xgb_parallel_neural_branch(artifact):
            candidate_config["xgb_parallel_neural_enabled"] = False
        for rounds_candidate in _xgb_fine_tune_round_candidates(resolved_intensity, config=config):
            tuned = _train_xgboost_model(
                window_payload["X_train"],
                window_payload["y_train"],
                window_payload["X_val"],
                window_payload["y_val"],
                config=candidate_config,
                balance_mode=str(artifact.get("balance_mode", BALANCE_MODE_NONE)),
                smote_params=dict(artifact.get("smote_params") or {}),
                base_model=artifact.get("model"),
                imputer=artifact.get("imputer"),
                n_estimators_override=int(rounds_candidate),
                learning_rate_override=float(learning_rate_value),
                sample_weight=sample_weight,
                history_training_rows_base=int(artifact.get("history_training_rows", 0)),
                xgb_fine_tune_metadata=metadata,
            )
            candidate_eval = _evaluate_xgb_fine_tune_candidate(
                tuned,
                window_payload["X_val"],
                window_payload["y_val"],
                selection_metric=selection_metric,
                rounds_selected=int(rounds_candidate),
            )
            if best_candidate is None or candidate_eval["sort_key"] > best_candidate["sort_key"]:
                best_artifact = tuned
                best_candidate = dict(candidate_eval, rounds_selected=int(rounds_candidate))

        if best_artifact is None or best_candidate is None:
            metadata["xgb_fine_tune_skip_reason"] = "no_candidate_selected"
            artifact["xgb_fine_tune_metadata"] = metadata
            return {
                "applied": False,
                "xgb_fine_tune_metadata": metadata,
            }

        metadata.update(
            {
                "xgb_fine_tune_rounds_selected": int(best_candidate["rounds_selected"]),
                "xgb_fine_tune_eta_multiplier": float(eta_multiplier),
                "xgb_fine_tune_recent_weight_max": float(recent_weight_max),
                "xgb_fine_tune_selection_score": float(best_candidate["selection_score"]),
                "xgb_fine_tune_skip_reason": None,
            }
        )
        best_artifact["xgb_fine_tune_metadata"] = dict(metadata)
        parallel_branch_artifact = artifact.get("parallel_neural_branch")
        parallel_enabled = _artifact_has_xgb_parallel_neural_branch(artifact)
        artifact.update(best_artifact)
        if parallel_enabled and isinstance(parallel_branch_artifact, dict):
            artifact["parallel_neural_enabled"] = True
            artifact["parallel_neural_branch"] = parallel_branch_artifact
            _fine_tune_artifact(
                parallel_branch_artifact,
                window_payload["X_recent"],
                window_payload["y_recent"],
                config=config,
            )
            _sync_xgb_parallel_branch_state(artifact)
        _recalibrate_artifact(
            artifact,
            window_payload["X_recent"],
            window_payload["y_recent"],
            config=config,
        )
        artifact["xgb_fine_tune_metadata"] = dict(metadata)
        return {
            "applied": True,
            "xgb_fine_tune_metadata": metadata,
        }
    if artifact_kind != "torch_mlp":
        raise ValueError("Fine-tuning solo aplica a Torch MLP y XGBoost.")
    X_train, X_val, y_train, y_val = _split_recent_for_adaptation(X_recent, y_recent)
    tuned = _train_torch_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        config=config,
        model_name=str(artifact.get("model_name", MODEL_TORCH_MLP)),
        balance_mode=str(artifact.get("balance_mode", BALANCE_MODE_NONE)),
        smote_params=dict(artifact.get("smote_params") or {}),
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
    return {
        "applied": True,
        "xgb_fine_tune_metadata": {},
    }


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
        balance_mode=str(artifact.get("balance_mode", BALANCE_MODE_NONE)),
        smote_params=dict(artifact.get("smote_params") or {}),
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
    return [str(strategy) for strategy in strategies]


def _rolling_metric_table(stream_df: pd.DataFrame, *, rolling_window: int) -> pd.DataFrame:
    if stream_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["model", "strategy", "balance_mode"]
    work = stream_df.copy().sort_values(group_cols + ["timestamp"]).reset_index(drop=True)
    for (model_name, strategy, balance_mode), group in work.groupby(group_cols, dropna=False, sort=False):
        group = group.reset_index(drop=True)
        for idx in range(len(group)):
            start = max(0, idx - int(rolling_window) + 1)
            window = group.iloc[start : idx + 1]
            threshold_value = (
                float(window["decision_threshold"].iloc[-1])
                if "decision_threshold" in window.columns
                else 0.5
            )
            metrics = _classification_metrics(
                window["y_true"].to_numpy(),
                window["score"].to_numpy(),
                threshold=threshold_value,
                preds=window["prediction"].to_numpy() if "prediction" in window.columns else None,
            )
            rows.append(
                {
                    "timestamp": pd.Timestamp(group.loc[idx, "timestamp"]),
                    "model": str(model_name),
                    "strategy": str(strategy),
                    "balance_mode": str(balance_mode),
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
    grouped = stream_df.groupby(["model", "strategy", "balance_mode"], dropna=False)
    for (model_name, strategy, balance_mode), group in grouped:
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
            & drift_events["balance_mode"].astype(str).eq(str(balance_mode))
        ].copy()
        rows.append(
            {
                "model": str(model_name),
                "strategy": str(strategy),
                "balance_mode": str(balance_mode),
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
                "parallel_neural_enabled": bool(group["parallel_neural_enabled"].iloc[-1]) if "parallel_neural_enabled" in group.columns else False,
                "parallel_neural_model": str(group["parallel_neural_model"].iloc[-1]) if "parallel_neural_model" in group.columns else "not_available",
                "drift_monitor_source": str(group["drift_monitor_source"].iloc[-1]) if "drift_monitor_source" in group.columns else DRIFT_MONITOR_SOURCE_NOT_AVAILABLE,
                "monitor_warmup_rows": int(group["monitor_warmup"].sum()) if "monitor_warmup" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "balance_mode", "strategy"]).reset_index(drop=True)


def _build_attention_outputs(attention_rows: Sequence[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    feature_acc: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    temporal_acc: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    drift_acc: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = {}

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
        balance_mode = str(row.get("balance_mode", BALANCE_MODE_NONE))

        for label, value in zip(feature_labels, feature_values):
            key = (model_name, strategy, balance_mode, str(label))
            acc = feature_acc.setdefault(key, {"sum": 0.0, "count": 0.0, "drift_rows": 0.0})
            acc["sum"] += float(value)
            acc["count"] += 1.0
            if is_drift_event:
                acc["drift_rows"] += 1.0

        for label, value in zip(temporal_labels, temporal_values):
            key = (model_name, strategy, balance_mode, str(label))
            acc = temporal_acc.setdefault(key, {"sum": 0.0, "count": 0.0, "drift_rows": 0.0})
            acc["sum"] += float(value)
            acc["count"] += 1.0
            if is_drift_event:
                acc["drift_rows"] += 1.0

        if not is_drift_event:
            continue

        for label, value, ref_value in zip(feature_labels, feature_values, ref_feature_values):
            key = (model_name, strategy, balance_mode, "feature", str(label))
            acc = drift_acc.setdefault(key, {"attention_sum": 0.0, "reference_sum": 0.0, "count": 0.0})
            acc["attention_sum"] += float(value)
            acc["reference_sum"] += float(ref_value)
            acc["count"] += 1.0

        for label, value, ref_value in zip(temporal_labels, temporal_values, ref_temporal_values):
            key = (model_name, strategy, balance_mode, "time", str(label))
            acc = drift_acc.setdefault(key, {"attention_sum": 0.0, "reference_sum": 0.0, "count": 0.0})
            acc["attention_sum"] += float(value)
            acc["reference_sum"] += float(ref_value)
            acc["count"] += 1.0

    feature_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "balance_mode": balance_mode,
            "feature": label,
            "attention_mean": float(acc["sum"] / max(acc["count"], 1.0)),
            "n_rows": int(acc["count"]),
            "drift_event_rows": int(acc["drift_rows"]),
        }
        for (model_name, strategy, balance_mode, label), acc in feature_acc.items()
    ]
    temporal_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "balance_mode": balance_mode,
            "time_step": label,
            "attention_mean": float(acc["sum"] / max(acc["count"], 1.0)),
            "n_rows": int(acc["count"]),
            "drift_event_rows": int(acc["drift_rows"]),
        }
        for (model_name, strategy, balance_mode, label), acc in temporal_acc.items()
    ]
    drift_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "balance_mode": balance_mode,
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
        for (model_name, strategy, balance_mode, scope, label), acc in drift_acc.items()
    ]

    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df = feature_df.sort_values(
            ["model", "balance_mode", "strategy", "attention_mean", "feature"],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)

    temporal_df = pd.DataFrame(temporal_rows)
    if not temporal_df.empty:
        temporal_df = temporal_df.sort_values(
            ["model", "balance_mode", "strategy", "attention_mean", "time_step"],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)

    drift_df = pd.DataFrame(drift_rows)
    if not drift_df.empty:
        drift_df = drift_df.sort_values(
            ["model", "balance_mode", "strategy", "scope", "abs_delta_attention", "item"],
            ascending=[True, True, True, True, False, True],
        ).reset_index(drop=True)

    return {
        "attention_feature_summary": feature_df,
        "attention_temporal_summary": temporal_df,
        "attention_drift_shift_summary": drift_df,
    }


def _build_detector_attention_outputs(detector_attention_rows: Sequence[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    temporal_acc: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    drift_acc: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}

    for row in detector_attention_rows:
        temporal_values = _as_float_array(row.get("temporal_attention_mean"))
        temporal_labels = list(row.get("temporal_labels") or [])
        reference_values = _as_float_array(row.get("reference_temporal_attention_mean"))
        model_name = str(row.get("model"))
        strategy = str(row.get("strategy"))
        balance_mode = str(row.get("balance_mode", BALANCE_MODE_NONE))
        is_drift_event = bool(row.get("is_drift_event"))

        for label, value in zip(temporal_labels, temporal_values):
            key = (model_name, strategy, balance_mode, str(label))
            acc = temporal_acc.setdefault(key, {"sum": 0.0, "count": 0.0, "drift_rows": 0.0})
            acc["sum"] += float(value)
            acc["count"] += 1.0
            if is_drift_event:
                acc["drift_rows"] += 1.0

        if not is_drift_event:
            continue
        for label, value, ref_value in zip(temporal_labels, temporal_values, reference_values):
            key = (model_name, strategy, balance_mode, str(label))
            acc = drift_acc.setdefault(key, {"attention_sum": 0.0, "reference_sum": 0.0, "count": 0.0})
            acc["attention_sum"] += float(value)
            acc["reference_sum"] += float(ref_value)
            acc["count"] += 1.0

    temporal_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "balance_mode": balance_mode,
            "time_step": label,
            "attention_mean": float(acc["sum"] / max(acc["count"], 1.0)),
            "n_rows": int(acc["count"]),
            "drift_event_rows": int(acc["drift_rows"]),
        }
        for (model_name, strategy, balance_mode, label), acc in temporal_acc.items()
    ]
    drift_rows = [
        {
            "model": model_name,
            "strategy": strategy,
            "balance_mode": balance_mode,
            "time_step": label,
            "reference_attention": float(acc["reference_sum"] / max(acc["count"], 1.0)),
            "drift_attention_mean": float(acc["attention_sum"] / max(acc["count"], 1.0)),
            "delta_attention": float((acc["attention_sum"] - acc["reference_sum"]) / max(acc["count"], 1.0)),
            "abs_delta_attention": float(
                abs((acc["attention_sum"] - acc["reference_sum"]) / max(acc["count"], 1.0))
            ),
            "n_drift_rows": int(acc["count"]),
        }
        for (model_name, strategy, balance_mode, label), acc in drift_acc.items()
    ]

    temporal_df = pd.DataFrame(temporal_rows)
    if not temporal_df.empty:
        temporal_df = temporal_df.sort_values(
            ["model", "balance_mode", "strategy", "attention_mean", "time_step"],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)

    drift_df = pd.DataFrame(drift_rows)
    if not drift_df.empty:
        drift_df = drift_df.sort_values(
            ["model", "balance_mode", "strategy", "abs_delta_attention", "time_step"],
            ascending=[True, True, True, False, True],
        ).reset_index(drop=True)

    return {
        "detector_attention_temporal_summary": temporal_df,
        "detector_attention_drift_shift_summary": drift_df,
    }


def _prepare_backtest_runtime(
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
    split_mode = str(config.get("split_mode", DEFAULT_CONFIG["split_mode"])).strip().lower()
    max_stream_rows_raw = config.get("max_stream_rows", DEFAULT_CONFIG["max_stream_rows"])
    max_stream_rows = None if max_stream_rows_raw is None else int(max_stream_rows_raw)
    if split_mode == "fixed_dates":
        split = _split_window_dataset_fixed_dates(
            dataset,
            base_start=str(config.get("base_start", DEFAULT_CONFIG["base_start"])),
            base_end=str(config.get("base_end", DEFAULT_CONFIG["base_end"])),
            stream_start=str(config.get("stream_start", DEFAULT_CONFIG["stream_start"])),
            validation_fraction=float(config.get("validation_fraction", DEFAULT_CONFIG["validation_fraction"])),
            max_stream_rows=max_stream_rows,
        )
    else:
        split = _split_window_dataset(
            dataset,
            train_fraction=float(config.get("train_fraction", DEFAULT_CONFIG["train_fraction"])),
            validation_fraction=float(config.get("validation_fraction", DEFAULT_CONFIG["validation_fraction"])),
            max_stream_rows=max_stream_rows,
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

    selected_balance_modes = _resolve_balance_modes(config.get("balance_modes"))
    selected_models = [model for model in config.get("models", []) if model in AVAILABLE_MODELS]
    selected_strategies = [strategy for strategy in config.get("strategies", []) if strategy in AVAILABLE_STRATEGIES]
    selected_channels = [channel for channel in config.get("drift_channels", []) if channel in AVAILABLE_DRIFT_CHANNELS]
    if not selected_models or not selected_strategies:
        raise ValueError("Selecciona al menos un modelo y una estrategia para el backtest.")

    baseline_specs: List[Dict[str, Any]] = []
    experiment_specs: List[Dict[str, Any]] = []
    for model_name in selected_models:
        valid_strategies = _allowed_strategies_for_model(model_name, selected_strategies)
        for balance_mode in selected_balance_modes:
            baseline_key = _build_neural_drift_baseline_key(str(model_name), str(balance_mode))
            baseline_experiment_specs: List[Dict[str, Any]] = []
            for strategy in valid_strategies:
                experiment_key = _build_neural_drift_experiment_key(
                    str(model_name),
                    str(strategy),
                    str(balance_mode),
                )
                experiment_spec = {
                    "experiment_key": experiment_key,
                    "model": str(model_name),
                    "strategy": str(strategy),
                    "balance_mode": str(balance_mode),
                    "baseline_key": baseline_key,
                }
                baseline_experiment_specs.append(experiment_spec)
                experiment_specs.append(experiment_spec)
            baseline_specs.append(
                {
                    "baseline_key": baseline_key,
                    "model": str(model_name),
                    "balance_mode": str(balance_mode),
                    "strategies": [str(item) for item in valid_strategies],
                    "experiment_specs": baseline_experiment_specs,
                }
            )
    return {
        "dataset": dataset,
        "split": split,
        "feature_metadata": feature_metadata,
        "selected_channels": selected_channels,
        "baseline_specs": baseline_specs,
        "experiment_specs": experiment_specs,
    }


def _build_baseline_result_row(
    model_name: str,
    balance_mode: str,
    canonical_artifact: Dict[str, Any],
    split: Dict[str, Any],
) -> Dict[str, Any]:
    canonical_smote_info = dict(canonical_artifact.get("smote_fit_info") or {})
    canonical_smote_params = dict(canonical_artifact.get("smote_params") or {})
    baseline_details = _predict_with_artifact_details(canonical_artifact, split["X_val"])
    baseline_raw_scores = baseline_details["probs"]
    baseline_embeddings = baseline_details["embeddings"]
    baseline_scores = _apply_calibrator(baseline_raw_scores, canonical_artifact.get("calibrator"))
    baseline_threshold = float(
        canonical_artifact.get("decision_threshold", canonical_artifact.get("base_threshold", 0.5))
    )
    baseline_preds = (baseline_scores >= baseline_threshold).astype(int)
    baseline_metrics = _classification_metrics(
        split["y_val"],
        baseline_scores,
        threshold=baseline_threshold,
        preds=baseline_preds,
    )
    return {
        "model": str(model_name),
        "balance_mode": str(balance_mode),
        "split": "validation",
        "rows": int(len(split["y_val"])),
        **baseline_metrics,
        "embedding_channels_available": bool(baseline_embeddings.size > 0),
        "monitor_effective_architecture": str(
            canonical_artifact.get("monitor_effective_architecture", "not_available")
        ),
        "parallel_neural_enabled": bool(canonical_artifact.get("parallel_neural_enabled", False)),
        "parallel_neural_model": str(
            canonical_artifact.get("parallel_neural_model", "not_available")
        ),
        "drift_monitor_source": str(
            canonical_artifact.get(
                "drift_monitor_source",
                _artifact_drift_monitor_source(canonical_artifact),
            )
        ),
        "smote_applied": bool(canonical_smote_info.get("applied", False)),
        "smote_balanced_rows": int(canonical_smote_info.get("balanced_rows", len(split["y_train"]))),
        "smote_sampling_strategy": canonical_smote_params.get("sampling_strategy"),
        "smote_k_neighbors": canonical_smote_params.get("k_neighbors"),
    }


def _execute_backtest_experiment(
    *,
    model_name: str,
    strategy: str,
    balance_mode: str,
    canonical_artifact: Dict[str, Any],
    split: Dict[str, Any],
    config: Dict[str, Any],
    selected_channels: Sequence[str],
    live_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if live_update_callback is not None:
        live_update_callback(
            {
                "event": "simulation_start",
                "model": str(model_name),
                "strategy": str(strategy),
                "balance_mode": str(balance_mode),
                "severity_threshold": float(
                    config.get("severity_threshold", DEFAULT_CONFIG["severity_threshold"])
                ),
                "rolling_metric_window": int(
                    config.get(
                        "rolling_metric_window",
                        DEFAULT_CONFIG["rolling_metric_window"],
                    )
                ),
                "stream_total_rows": int(len(split["y_stream"])),
            }
        )
    try:
        artifact = copy.deepcopy(canonical_artifact)
    except Exception:
        artifact = canonical_artifact
    history_X = np.vstack([split["X_train"], split["X_val"]])
    history_y = np.concatenate([split["y_train"], split["y_val"]])

    detectors = {
        DRIFT_INPUT: ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
        DRIFT_SCORE: ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
        DRIFT_ERROR: ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
        "score_neural": ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
        "error_neural": ClassicDriftDetector(delta=float(config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"]))),
    }
    channel_histories: Dict[str, List[float]] = {
        DRIFT_INPUT: [],
        DRIFT_SCORE: [],
        DRIFT_ERROR: [],
        DRIFT_EMBEDDING: [],
        "score_neural": [],
        "error_neural": [],
    }
    embedding_buffer: List[np.ndarray] = []
    stream_rows: List[Dict[str, Any]] = []
    drift_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    detector_attention_rows: List[Dict[str, Any]] = []

    for idx in range(len(split["y_stream"])):
        x_row = split["X_stream"][idx : idx + 1]
        y_true = int(split["y_stream"][idx])
        timestamp = pd.Timestamp(split["metadata_stream"].loc[idx, "prediction_time"])

        prediction_details = _predict_with_artifact_details(artifact, x_row)
        raw_scores = prediction_details["probs"]
        embeddings = prediction_details["embeddings"]
        attention_summary = prediction_details.get("attention_summary")
        score = float(_apply_calibrator(raw_scores, artifact.get("calibrator"))[0])
        auxiliary_probs = _as_float_array(prediction_details.get("auxiliary_probs"))
        auxiliary_score = (
            float(auxiliary_probs.reshape(-1)[0])
            if auxiliary_probs.size > 0
            else None
        )
        decision_threshold = float(
            artifact.get("decision_threshold", artifact.get("base_threshold", 0.5))
        )
        pred = int(score >= decision_threshold)
        pre_action_reference_attention = dict(artifact.get("attention_summary_reference") or {})
        pre_action_monitor = dict(artifact.get("embedding_monitor") or {})
        detector_attention_reference = dict(pre_action_monitor.get("attention_reference_summary") or {})
        recent_embedding_history = (
            np.vstack(embedding_buffer).astype(float)
            if embedding_buffer
            else np.empty(
                (0, embeddings.shape[1] if embeddings.ndim == 2 and embeddings.shape[1] > 0 else 0),
                dtype=float,
            )
        )
        channel_payload = _build_channel_scores(
            artifact=artifact,
            x_row=_apply_imputer(x_row, artifact["imputer"]).reshape(-1),
            calibrated_score=score,
            auxiliary_calibrated_score=auxiliary_score,
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
        monitor_effective_architecture = str(
            channel_payload.get(
                "monitor_effective_architecture",
                artifact.get("monitor_effective_architecture", "none"),
            )
        )
        drift_monitor_source = str(
            channel_payload.get(
                "drift_monitor_source",
                artifact.get("drift_monitor_source", _artifact_drift_monitor_source(artifact)),
            )
        )
        monitor_warmup = bool(channel_payload.get("monitor_warmup", False))
        parallel_neural_enabled = bool(
            prediction_details.get(
                "parallel_neural_enabled",
                artifact.get("parallel_neural_enabled", False),
            )
        )
        parallel_neural_model = str(
            prediction_details.get(
                "parallel_neural_model",
                artifact.get("parallel_neural_model", "not_available"),
            )
        )
        recent_start = max(
            0,
            idx - int(config.get("recent_window_size", DEFAULT_CONFIG["recent_window_size"])) + 1,
        )
        recent_X = split["X_stream"][recent_start : idx + 1]
        recent_y = split["y_stream"][recent_start : idx + 1]
        recent_has_two_classes = len(np.unique(recent_y)) >= 2
        severity_threshold = float(config.get("severity_threshold", DEFAULT_CONFIG["severity_threshold"]))
        trigger_score = _trigger_score(severity_score, max_channel_score)
        severity_intensity = _severity_intensity(trigger_score, severity_threshold)
        severity_triggered = bool(
            severity_score >= severity_threshold
            or max_channel_score >= severity_threshold
        )
        xgb_event_metadata: Dict[str, Any] = {}

        if severity_triggered:
            if strategy == STRATEGY_FIXED:
                action_taken = "none"
            elif strategy == STRATEGY_RECALIBRATION and len(recent_y) >= int(config.get("recalibration_min_rows", DEFAULT_CONFIG["recalibration_min_rows"])) and recent_has_two_classes:
                _recalibrate_artifact(artifact, recent_X, recent_y, config=config)
                action_taken = "recalibration"
            elif strategy == STRATEGY_FINE_TUNING:
                if str(model_name) == MODEL_XGBOOST:
                    fine_tune_result = _fine_tune_artifact(
                        artifact,
                        split["X_stream"][: idx + 1],
                        split["y_stream"][: idx + 1],
                        config=config,
                        severity_intensity=severity_intensity,
                    )
                    xgb_event_metadata = dict(
                        fine_tune_result.get("xgb_fine_tune_metadata") or {}
                    )
                    action_taken = "fine_tuning" if bool(fine_tune_result.get("applied", False)) else "none"
                elif len(recent_y) >= int(config.get("recalibration_min_rows", DEFAULT_CONFIG["recalibration_min_rows"])) and recent_has_two_classes:
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
            current_smote_params = dict(artifact.get("smote_params") or {})
            current_smote_info = dict(artifact.get("smote_fit_info") or {})

            drift_rows.append(
                {
                    "timestamp": timestamp,
                    "model": str(model_name),
                    "strategy": str(strategy),
                    "balance_mode": str(balance_mode),
                    "severity_score": severity_score,
                    "severity_intensity": severity_intensity,
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
                    "parallel_neural_enabled": bool(parallel_neural_enabled),
                    "parallel_neural_model": parallel_neural_model,
                    "drift_monitor_source": drift_monitor_source,
                    "monitor_warmup": bool(monitor_warmup),
                    "smote_applied": bool(current_smote_info.get("applied", False)),
                    "smote_sampling_strategy": current_smote_params.get("sampling_strategy"),
                    "smote_k_neighbors": current_smote_params.get("k_neighbors"),
                    "xgb_adaptation_window_rows": xgb_event_metadata.get("xgb_adaptation_window_rows"),
                    "xgb_fine_tune_rounds_selected": xgb_event_metadata.get("xgb_fine_tune_rounds_selected"),
                    "xgb_fine_tune_eta_multiplier": xgb_event_metadata.get("xgb_fine_tune_eta_multiplier"),
                    "xgb_fine_tune_recent_weight_max": xgb_event_metadata.get("xgb_fine_tune_recent_weight_max"),
                    "xgb_fine_tune_selection_metric": xgb_event_metadata.get("xgb_fine_tune_selection_metric"),
                    "xgb_fine_tune_selection_score": xgb_event_metadata.get("xgb_fine_tune_selection_score"),
                    "xgb_fine_tune_skip_reason": xgb_event_metadata.get("xgb_fine_tune_skip_reason"),
                }
            )

        if attention_summary is not None:
            attention_rows.append(
                {
                    "timestamp": timestamp,
                    "model": str(model_name),
                    "strategy": str(strategy),
                    "balance_mode": str(balance_mode),
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
                    "balance_mode": str(balance_mode),
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

        current_smote_params = dict(artifact.get("smote_params") or {})
        current_smote_info = dict(artifact.get("smote_fit_info") or {})
        stream_rows.append(
            {
                "timestamp": timestamp,
                "model": str(model_name),
                "strategy": str(strategy),
                "balance_mode": str(balance_mode),
                "y_true": int(y_true),
                "prediction": int(pred),
                "score": float(score),
                "decision_threshold": decision_threshold,
                "severity_score": severity_score,
                "severity_intensity": severity_intensity,
                "max_channel_score": max_channel_score,
                "severity_label": str(channel_payload["severity_label"]),
                "action_taken": str(action_taken),
                "brier_component": float((score - y_true) ** 2),
                "monitor_effective_architecture": monitor_effective_architecture,
                "parallel_neural_enabled": bool(parallel_neural_enabled),
                "parallel_neural_model": parallel_neural_model,
                "drift_monitor_source": drift_monitor_source,
                "monitor_warmup": bool(monitor_warmup),
                "smote_applied": bool(current_smote_info.get("applied", False)),
                "smote_sampling_strategy": current_smote_params.get("sampling_strategy"),
                "smote_k_neighbors": current_smote_params.get("k_neighbors"),
                "xgb_adaptation_window_rows": xgb_event_metadata.get("xgb_adaptation_window_rows"),
                "xgb_fine_tune_rounds_selected": xgb_event_metadata.get("xgb_fine_tune_rounds_selected"),
                "xgb_fine_tune_eta_multiplier": xgb_event_metadata.get("xgb_fine_tune_eta_multiplier"),
                "xgb_fine_tune_recent_weight_max": xgb_event_metadata.get("xgb_fine_tune_recent_weight_max"),
                "xgb_fine_tune_selection_metric": xgb_event_metadata.get("xgb_fine_tune_selection_metric"),
                "xgb_fine_tune_selection_score": xgb_event_metadata.get("xgb_fine_tune_selection_score"),
                "xgb_fine_tune_skip_reason": xgb_event_metadata.get("xgb_fine_tune_skip_reason"),
            }
        )
        if live_update_callback is not None:
            live_update_callback(
                {
                    "event": "stream_step",
                    "timestamp": timestamp,
                    "model": str(model_name),
                    "strategy": str(strategy),
                    "balance_mode": str(balance_mode),
                    "severity_score": severity_score,
                    "max_channel_score": max_channel_score,
                    "severity_threshold": severity_threshold,
                    "is_drift_event": bool(severity_triggered),
                    "action_taken": str(action_taken),
                    "score": float(score),
                    "decision_threshold": decision_threshold,
                    "y_true": int(y_true),
                    "prediction": int(pred),
                    "rolling_metric_window": int(
                        config.get(
                            "rolling_metric_window",
                            DEFAULT_CONFIG["rolling_metric_window"],
                        )
                    ),
                    "stream_step_index": int(idx + 1),
                    "stream_total_rows": int(len(split["y_stream"])),
                }
            )

        if embeddings.size > 0:
            embedding_buffer.append(np.asarray(embeddings.reshape(-1), dtype=float))

    return {
        "stream_rows": stream_rows,
        "drift_rows": drift_rows,
        "attention_rows": attention_rows,
        "detector_attention_rows": detector_attention_rows,
    }


def _finalize_backtest_results(
    *,
    dataset: WindowDataset,
    split: Dict[str, Any],
    baseline_rows: Sequence[Dict[str, Any]],
    stream_rows: Sequence[Dict[str, Any]],
    drift_rows: Sequence[Dict[str, Any]],
    attention_rows: Sequence[Dict[str, Any]],
    detector_attention_rows: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_df = pd.DataFrame(list(baseline_rows))
    if not baseline_df.empty:
        baseline_df = baseline_df.sort_values(["model", "balance_mode"]).reset_index(drop=True)

    stream_df = pd.DataFrame(list(stream_rows))
    if not stream_df.empty:
        stream_df = stream_df.sort_values(["model", "balance_mode", "strategy", "timestamp"]).reset_index(drop=True)

    drift_df = pd.DataFrame(
        list(drift_rows),
        columns=[
            "timestamp",
            "model",
            "strategy",
            "balance_mode",
            "severity_score",
            "severity_intensity",
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
            "parallel_neural_enabled",
            "parallel_neural_model",
            "drift_monitor_source",
            "monitor_warmup",
            "smote_applied",
            "smote_sampling_strategy",
            "smote_k_neighbors",
            "xgb_adaptation_window_rows",
            "xgb_fine_tune_rounds_selected",
            "xgb_fine_tune_eta_multiplier",
            "xgb_fine_tune_recent_weight_max",
            "xgb_fine_tune_selection_metric",
            "xgb_fine_tune_selection_score",
            "xgb_fine_tune_skip_reason",
        ],
    )
    if not drift_df.empty:
        drift_df = drift_df.sort_values(["model", "balance_mode", "strategy", "timestamp"]).reset_index(drop=True)

    summary_df = _summary_from_stream(stream_df, drift_df)
    rolling_df = _rolling_metric_table(
        stream_df,
        rolling_window=int(config.get("rolling_metric_window", DEFAULT_CONFIG["rolling_metric_window"])),
    )
    attention_outputs = _build_attention_outputs(list(attention_rows))
    detector_attention_outputs = _build_detector_attention_outputs(list(detector_attention_rows))
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


def run_backtest_pipeline(
    dataset_bundle: Dict[str, Any],
    *,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    live_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    runtime = _prepare_backtest_runtime(
        dataset_bundle,
        config=config,
        progress_callback=progress_callback,
    )
    baseline_rows: List[Dict[str, Any]] = []
    stream_rows: List[Dict[str, Any]] = []
    drift_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    detector_attention_rows: List[Dict[str, Any]] = []
    loop_total = max(1, int(len(runtime["experiment_specs"])))
    loop_index = 0

    for baseline_spec in runtime["baseline_specs"]:
        model_name = str(baseline_spec["model"])
        balance_mode = str(baseline_spec["balance_mode"])
        if progress_callback is not None:
            progress_callback(
                0.12 + 0.70 * (loop_index / loop_total),
                f"Entrenando baseline para {model_name} | {balance_mode}...",
            )
        canonical_artifact = _train_canonical_artifact(
            model_name,
            runtime["split"]["X_train"],
            runtime["split"]["y_train"],
            runtime["split"]["X_val"],
            runtime["split"]["y_val"],
            config=config,
            balance_mode=balance_mode,
            feature_metadata=runtime["feature_metadata"],
        )
        baseline_rows.append(
            _build_baseline_result_row(
                model_name,
                balance_mode,
                canonical_artifact,
                runtime["split"],
            )
        )
        for experiment_spec in baseline_spec["experiment_specs"]:
            loop_index += 1
            strategy = str(experiment_spec["strategy"])
            if progress_callback is not None:
                progress_callback(
                    0.15 + 0.75 * (loop_index / loop_total),
                    f"Simulando {model_name} | {strategy} | {balance_mode}...",
                )
            experiment_payload = _execute_backtest_experiment(
                model_name=model_name,
                strategy=strategy,
                balance_mode=balance_mode,
                canonical_artifact=canonical_artifact,
                split=runtime["split"],
                config=config,
                selected_channels=runtime["selected_channels"],
                live_update_callback=live_update_callback,
            )
            stream_rows.extend(experiment_payload["stream_rows"])
            drift_rows.extend(experiment_payload["drift_rows"])
            attention_rows.extend(experiment_payload["attention_rows"])
            detector_attention_rows.extend(experiment_payload["detector_attention_rows"])

    return _finalize_backtest_results(
        dataset=runtime["dataset"],
        split=runtime["split"],
        baseline_rows=baseline_rows,
        stream_rows=stream_rows,
        drift_rows=drift_rows,
        attention_rows=attention_rows,
        detector_attention_rows=detector_attention_rows,
        config=config,
    )


def _download_bundle_from_results(results: Dict[str, Any]) -> Dict[str, str]:
    bundle: Dict[str, str] = {}
    for key in NEURAL_DRIFT_RESULT_KEYS:
        df = results.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            bundle[key] = df.to_csv(index=False)
    return bundle


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _neural_drift_checkpoint_root(*, checkpoint_root: Optional[Path] = None) -> Path:
    return Path(checkpoint_root) if checkpoint_root is not None else NEURAL_DRIFT_RUNS_DIR


def _neural_drift_run_dir(run_id: str, *, checkpoint_root: Optional[Path] = None) -> Path:
    return _neural_drift_checkpoint_root(checkpoint_root=checkpoint_root) / str(run_id)


def _neural_drift_run_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "live_status": run_dir / "live_status.json",
        "live_events": run_dir / "live_events.jsonl",
        "artifacts_dir": run_dir / "artifacts",
        "experiments_dir": run_dir / "experiments",
        "baselines_dir": run_dir / "baselines",
    }


def _ensure_neural_drift_run_dirs(paths: Dict[str, Path]) -> None:
    for key in ["run_dir", "artifacts_dir", "experiments_dir", "baselines_dir"]:
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


def _build_neural_drift_run_id(run_signature: str) -> str:
    created_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    digest = hashlib.sha256(f"{created_token}|{run_signature}".encode("utf-8")).hexdigest()
    return f"run_{created_token}_{digest[:8]}"


def _build_neural_drift_dataset_context(
    dataset_bundle: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_df = dataset_bundle.get("df")
    active_df = (
        _subset_dataset_by_percentage(
            dataset_df,
            dataset_percent=float(config.get("dataset_percent", DEFAULT_CONFIG["dataset_percent"])),
        )
        if isinstance(dataset_df, pd.DataFrame)
        else pd.DataFrame()
    )
    return {
        "source": str(dataset_bundle.get("source") or ""),
        "rows_total": int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0,
        "rows_used": int(len(active_df)),
        "feature_cols": list(dataset_bundle.get("feature_cols") or []),
        "feature_export_path": str(dataset_bundle.get("feature_export_path") or ""),
        "selection_metadata": _to_json_safe(dataset_bundle.get("selection_metadata") or {}),
        "feature_source_choice": str(st.session_state.get("neural_drift_feature_source_choice") or ""),
    }


def _initial_neural_drift_manifest(
    *,
    run_id: str,
    run_signature: str,
    dataset_context: Dict[str, Any],
    config: Dict[str, Any],
    baseline_specs: Sequence[Dict[str, Any]],
    experiment_specs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    created_at = _now_iso()
    baseline_index = {
        str(spec["baseline_key"]): {
            "baseline_key": str(spec["baseline_key"]),
            "model": str(spec["model"]),
            "balance_mode": str(spec["balance_mode"]),
            "status": "pending",
            "artifact_paths": {},
            "error": None,
        }
        for spec in baseline_specs
    }
    experiment_index = {
        str(spec["experiment_key"]): {
            "experiment_key": str(spec["experiment_key"]),
            "baseline_key": str(spec["baseline_key"]),
            "model": str(spec["model"]),
            "strategy": str(spec["strategy"]),
            "balance_mode": str(spec["balance_mode"]),
            "status": "pending",
            "artifact_paths": {},
            "error": None,
        }
        for spec in experiment_specs
    }
    manifest = {
        "schema_version": 1,
        "run_id": str(run_id),
        "run_signature": str(run_signature),
        "run_type": NEURAL_DRIFT_RUN_TYPE,
        "status": "running",
        "result_status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "dataset_context": _to_json_safe(dataset_context),
        "config": _to_json_safe(config),
        "progress": {},
        "baseline_index": baseline_index,
        "experiment_index": experiment_index,
        "artifacts": {},
        "last_error": None,
        "resume": {
            "auto_resumed": False,
            "checkpoint_status": "fresh",
        },
    }
    _update_neural_drift_manifest_progress(manifest)
    return manifest


def _reconcile_neural_drift_manifest(
    manifest: Dict[str, Any],
    *,
    baseline_specs: Sequence[Dict[str, Any]],
    experiment_specs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    reconciled = dict(manifest or {})
    baseline_index = dict(reconciled.get("baseline_index") or {})
    experiment_index = dict(reconciled.get("experiment_index") or {})
    for spec in baseline_specs:
        key = str(spec["baseline_key"])
        current = dict(baseline_index.get(key) or {})
        status = str(current.get("status") or "pending")
        if status not in {"completed", "pending"}:
            status = "pending"
        baseline_index[key] = {
            "baseline_key": key,
            "model": str(spec["model"]),
            "balance_mode": str(spec["balance_mode"]),
            "status": status,
            "artifact_paths": dict(current.get("artifact_paths") or {}),
            "error": current.get("error"),
        }
    for spec in experiment_specs:
        key = str(spec["experiment_key"])
        current = dict(experiment_index.get(key) or {})
        status = str(current.get("status") or "pending")
        if status != "completed":
            status = "pending"
        experiment_index[key] = {
            "experiment_key": key,
            "baseline_key": str(spec["baseline_key"]),
            "model": str(spec["model"]),
            "strategy": str(spec["strategy"]),
            "balance_mode": str(spec["balance_mode"]),
            "status": status,
            "artifact_paths": dict(current.get("artifact_paths") or {}),
            "error": current.get("error"),
        }
    reconciled["baseline_index"] = baseline_index
    reconciled["experiment_index"] = experiment_index
    reconciled.setdefault("artifacts", {})
    reconciled.setdefault("last_error", None)
    reconciled.setdefault("resume", {"auto_resumed": False, "checkpoint_status": "unknown"})
    _update_neural_drift_manifest_progress(reconciled)
    return reconciled


def _update_neural_drift_manifest_progress(manifest: Dict[str, Any]) -> None:
    baseline_index = dict(manifest.get("baseline_index") or {})
    experiment_index = dict(manifest.get("experiment_index") or {})
    completed_baselines = sum(
        1 for item in baseline_index.values() if str(item.get("status") or "") == "completed"
    )
    completed_experiments = sum(
        1 for item in experiment_index.values() if str(item.get("status") or "") == "completed"
    )
    total_baselines = int(len(baseline_index))
    total_experiments = int(len(experiment_index))
    total_units = max(1, total_baselines + total_experiments)
    completed_units = completed_baselines + completed_experiments
    current_baseline_key = next(
        (
            str(key)
            for key, item in baseline_index.items()
            if str(item.get("status") or "") == "running"
        ),
        None,
    )
    current_experiment_key = next(
        (
            str(key)
            for key, item in experiment_index.items()
            if str(item.get("status") or "") == "running"
        ),
        None,
    )
    manifest["progress"] = {
        "completed_units": int(completed_units),
        "total_units": int(total_units),
        "progress_ratio": float(completed_units / total_units),
        "completed_baselines": int(completed_baselines),
        "total_baselines": int(total_baselines),
        "completed_experiments": int(completed_experiments),
        "total_experiments": int(total_experiments),
        "pending_experiments": int(total_experiments - completed_experiments),
        "current_baseline_key": current_baseline_key,
        "current_experiment_key": current_experiment_key,
    }


def _persist_neural_drift_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = _now_iso()
    _atomic_write_json(path, manifest)


def _persist_neural_drift_live_status(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def _append_neural_drift_live_event(path: Path, payload: Dict[str, Any]) -> None:
    _append_jsonl_record(path, payload)


def _build_neural_drift_live_status_payload(
    manifest: Dict[str, Any],
    *,
    label: str,
    detail: str = "",
    status: Optional[str] = None,
    result_status: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    progress = dict(manifest.get("progress") or {})
    return {
        "timestamp": _now_iso(),
        "run_id": str(manifest.get("run_id") or ""),
        "status": str(status or manifest.get("status") or "unknown"),
        "result_status": str(result_status or manifest.get("result_status") or "unknown"),
        "completed_units": int(progress.get("completed_units", 0)),
        "total_units": int(progress.get("total_units", 0)),
        "progress_ratio": float(progress.get("progress_ratio", 0.0)),
        "label": str(label),
        "detail": str(detail),
        "context": _to_json_safe(context or {}),
    }


def _baseline_artifact_path(paths: Dict[str, Path], baseline_key: str) -> Path:
    return Path(paths["baselines_dir"]) / f"{baseline_key}.csv"


def _experiment_dir(paths: Dict[str, Path], experiment_key: str) -> Path:
    return Path(paths["experiments_dir"]) / str(experiment_key)


def _persist_neural_drift_baseline_checkpoint(
    paths: Dict[str, Path],
    *,
    baseline_key: str,
    baseline_row: Dict[str, Any],
) -> Dict[str, str]:
    artifact_path = _baseline_artifact_path(paths, baseline_key)
    _atomic_write_df_csv(artifact_path, pd.DataFrame([baseline_row]))
    return {"baseline": str(artifact_path)}


def _persist_neural_drift_result_artifacts(
    base_dir: Path,
    results: Dict[str, Any],
    *,
    keys: Sequence[str],
) -> Dict[str, str]:
    artifact_paths: Dict[str, str] = {}
    for key in keys:
        df = results.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        artifact_path = base_dir / f"{key}.csv"
        _atomic_write_df_csv(artifact_path, df)
        artifact_paths[str(key)] = str(artifact_path)
    return artifact_paths


def _read_persisted_neural_drift_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in ["timestamp", "prediction_time", "window_end", "horizon_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _sort_persisted_neural_drift_results(results: Dict[str, Any]) -> Dict[str, Any]:
    ordered = dict(results)
    sort_specs = {
        "baseline": ["model", "balance_mode"],
        "summary": ["model", "balance_mode", "strategy"],
        "stream_metrics": ["model", "balance_mode", "strategy", "timestamp"],
        "rolling_metrics": ["model", "balance_mode", "strategy", "timestamp"],
        "drift_events": ["model", "balance_mode", "strategy", "timestamp"],
        "attention_feature_summary": ["model", "balance_mode", "strategy"],
        "attention_temporal_summary": ["model", "balance_mode", "strategy"],
        "attention_drift_shift_summary": ["model", "balance_mode", "strategy"],
        "detector_attention_temporal_summary": ["model", "balance_mode", "strategy"],
        "detector_attention_drift_shift_summary": ["model", "balance_mode", "strategy"],
    }
    for key, columns in sort_specs.items():
        df = ordered.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        available = [col for col in columns if col in df.columns]
        if available:
            ordered[key] = df.sort_values(available).reset_index(drop=True)
    return ordered


def _assemble_persisted_neural_drift_results(
    manifest: Dict[str, Any],
    *,
    run_dir: Path,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    artifact_paths = dict(manifest.get("artifacts") or {})
    use_run_level_artifacts = (
        str(manifest.get("status") or "") == "completed"
        and str(manifest.get("result_status") or "") == "success"
    )
    if use_run_level_artifacts:
        for key in NEURAL_DRIFT_RESULT_KEYS:
            raw_path = artifact_paths.get(key)
            if not raw_path:
                continue
            results[key] = _read_persisted_neural_drift_dataframe(Path(str(raw_path)))
    if results.get("summary") is not None and isinstance(results.get("summary"), pd.DataFrame):
        return _sort_persisted_neural_drift_results(results)

    baseline_frames: List[pd.DataFrame] = []
    experiment_frames: Dict[str, List[pd.DataFrame]] = {key: [] for key in NEURAL_DRIFT_RESULT_KEYS if key != "baseline"}
    for item in (manifest.get("baseline_index") or {}).values():
        if str(item.get("status") or "") != "completed":
            continue
        raw_path = str((item.get("artifact_paths") or {}).get("baseline") or "")
        if not raw_path:
            continue
        frame = _read_persisted_neural_drift_dataframe(Path(raw_path))
        if not frame.empty:
            baseline_frames.append(frame)
    for item in (manifest.get("experiment_index") or {}).values():
        if str(item.get("status") or "") != "completed":
            continue
        paths_map = dict(item.get("artifact_paths") or {})
        for key in experiment_frames.keys():
            raw_path = str(paths_map.get(key) or "")
            if not raw_path:
                continue
            frame = _read_persisted_neural_drift_dataframe(Path(raw_path))
            if not frame.empty:
                experiment_frames[key].append(frame)
    if baseline_frames:
        results["baseline"] = pd.concat(baseline_frames, ignore_index=True)
    for key, frames in experiment_frames.items():
        if frames:
            results[key] = pd.concat(frames, ignore_index=True)
    return _sort_persisted_neural_drift_results(results)


def _load_persisted_neural_drift_run(manifest_path: Path) -> Dict[str, Any]:
    manifest = dict(_load_json_file(manifest_path, default={}) or {})
    run_dir = manifest_path.parent
    results = _assemble_persisted_neural_drift_results(manifest, run_dir=run_dir)
    return {
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "run_signature": str(manifest.get("run_signature") or ""),
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "results": results,
        "download_bundle": _download_bundle_from_results(results),
    }


def _store_neural_drift_results_in_session_state(
    results: Dict[str, Any],
    *,
    run_signature: str,
    run_id: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> None:
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
    st.session_state["neural_drift_last_run_signature"] = str(run_signature)
    st.session_state["neural_drift_active_run_id"] = None if run_id is None else str(run_id)
    st.session_state["neural_drift_active_manifest_path"] = None if manifest_path is None else str(manifest_path)


def _apply_persisted_neural_drift_run_to_session_state(payload: Dict[str, Any]) -> None:
    _store_neural_drift_results_in_session_state(
        dict(payload.get("results") or {}),
        run_signature=str(payload.get("run_signature") or ""),
        run_id=str(payload.get("run_id") or ""),
        manifest_path=str(payload.get("manifest_path") or ""),
    )
    st.session_state["neural_drift_loaded_checkpoint_run_id"] = str(payload.get("run_id") or "")


def _list_persisted_neural_drift_runs(
    *,
    checkpoint_root: Path = NEURAL_DRIFT_RUNS_DIR,
) -> List[Dict[str, Any]]:
    if not checkpoint_root.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for manifest_path in sorted(checkpoint_root.glob("*/manifest.json")):
        manifest = dict(_load_json_file(manifest_path, default={}) or {})
        progress = dict(manifest.get("progress") or {})
        updated_at = str(manifest.get("updated_at") or manifest.get("created_at") or "")
        updated_ts = pd.to_datetime(updated_at, errors="coerce")
        entries.append(
            {
                "run_id": str(manifest.get("run_id") or manifest_path.parent.name),
                "run_signature": str(manifest.get("run_signature") or ""),
                "manifest_path": str(manifest_path),
                "run_dir": str(manifest_path.parent),
                "status": str(manifest.get("status") or "unknown"),
                "result_status": str(manifest.get("result_status") or "unknown"),
                "updated_at": updated_at,
                "sort_key": (
                    float(updated_ts.timestamp())
                    if pd.notna(updated_ts)
                    else float(manifest_path.stat().st_mtime)
                ),
                "source": str((manifest.get("dataset_context") or {}).get("source") or ""),
                "rows_used": int((manifest.get("dataset_context") or {}).get("rows_used") or 0),
                "completed_experiments": int(progress.get("completed_experiments", 0)),
                "total_experiments": int(progress.get("total_experiments", 0)),
                "can_resume": str(manifest.get("status") or "") in {"running", "failed"},
                "label": (
                    f"{manifest.get('run_id') or manifest_path.parent.name} | "
                    f"{manifest.get('status') or 'unknown'} | "
                    f"{updated_at or '-'} | "
                    f"{int(progress.get('completed_experiments', 0))}/{int(progress.get('total_experiments', 0))} experimentos"
                ),
            }
        )
    entries.sort(key=lambda item: float(item.get("sort_key", 0.0)), reverse=True)
    return entries


def run_backtest_with_checkpoints(
    dataset_bundle: Dict[str, Any],
    *,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    live_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    resume_run_id: Optional[str] = None,
    checkpoint_root: Optional[Path] = None,
) -> Dict[str, Any]:
    runtime = _prepare_backtest_runtime(
        dataset_bundle,
        config=config,
        progress_callback=progress_callback,
    )
    run_signature = _build_run_signature(dataset_bundle, config)
    dataset_context = _build_neural_drift_dataset_context(dataset_bundle, config)

    if resume_run_id:
        run_dir = _neural_drift_run_dir(str(resume_run_id), checkpoint_root=checkpoint_root)
        paths = _neural_drift_run_paths(run_dir)
        manifest = dict(_load_json_file(paths["manifest"], default={}) or {})
        if not manifest:
            raise FileNotFoundError(f"No existe un checkpoint de Neural drift para `{resume_run_id}`.")
        if str(manifest.get("run_signature") or "") != run_signature:
            raise ValueError(
                "La configuracion o la fuente de datos actual no coincide con la corrida preparada para reanudar."
            )
        manifest = _reconcile_neural_drift_manifest(
            manifest,
            baseline_specs=runtime["baseline_specs"],
            experiment_specs=runtime["experiment_specs"],
        )
        if str(manifest.get("status") or "") == "completed" and str(manifest.get("result_status") or "") == "success":
            loaded = _load_persisted_neural_drift_run(paths["manifest"])
            return {
                **dict(loaded.get("results") or {}),
                "run_id": str(loaded.get("run_id") or ""),
                "run_signature": str(loaded.get("run_signature") or ""),
                "manifest": dict(loaded.get("manifest") or {}),
                "manifest_path": str(loaded.get("manifest_path") or ""),
                "run_dir": str(loaded.get("run_dir") or ""),
                "download_bundle": dict(loaded.get("download_bundle") or {}),
            }
        previous_status = str(manifest.get("status") or "unknown")
        manifest["status"] = "running"
        manifest["result_status"] = "running"
        manifest["last_error"] = None
        manifest["config"] = _to_json_safe(config)
        manifest["dataset_context"] = _to_json_safe(dataset_context)
        manifest["resume"] = {
            "auto_resumed": True,
            "checkpoint_status": previous_status,
        }
        start_event_type = "resume"
        run_id = str(manifest.get("run_id") or resume_run_id)
    else:
        run_id = _build_neural_drift_run_id(run_signature)
        run_dir = _neural_drift_run_dir(run_id, checkpoint_root=checkpoint_root)
        paths = _neural_drift_run_paths(run_dir)
        manifest = _initial_neural_drift_manifest(
            run_id=run_id,
            run_signature=run_signature,
            dataset_context=dataset_context,
            config=config,
            baseline_specs=runtime["baseline_specs"],
            experiment_specs=runtime["experiment_specs"],
        )
        start_event_type = "run_start"

    _ensure_neural_drift_run_dirs(paths)
    _persist_neural_drift_manifest(paths["manifest"], manifest)
    _persist_neural_drift_live_status(
        paths["live_status"],
        _build_neural_drift_live_status_payload(
            manifest,
            label="Iniciando Neural drift persistente...",
            detail="Preparando runtime y checkpoints.",
            context={"event": start_event_type},
        ),
    )
    _append_neural_drift_live_event(
        paths["live_events"],
        {
            "timestamp": _now_iso(),
            "event": start_event_type,
            "run_id": run_id,
            "run_signature": run_signature,
        },
    )

    total_units = max(1, int((manifest.get("progress") or {}).get("total_units", 1)))

    def _progress_message(label: str) -> None:
        if progress_callback is None:
            return
        progress = dict(manifest.get("progress") or {})
        ratio = float(progress.get("progress_ratio", 0.0))
        progress_callback(0.10 + 0.85 * ratio, label)

    try:
        for baseline_spec in runtime["baseline_specs"]:
            baseline_key = str(baseline_spec["baseline_key"])
            baseline_entry = dict((manifest.get("baseline_index") or {}).get(baseline_key) or {})
            pending_experiment_specs = [
                dict(spec)
                for spec in baseline_spec["experiment_specs"]
                if str(
                    ((manifest.get("experiment_index") or {}).get(str(spec["experiment_key"])) or {}).get("status")
                    or "pending"
                )
                != "completed"
            ]
            if not pending_experiment_specs:
                continue

            model_name = str(baseline_spec["model"])
            balance_mode = str(baseline_spec["balance_mode"])
            baseline_entry["status"] = "running"
            baseline_entry["error"] = None
            manifest["baseline_index"][baseline_key] = baseline_entry
            _update_neural_drift_manifest_progress(manifest)
            _persist_neural_drift_manifest(paths["manifest"], manifest)
            _persist_neural_drift_live_status(
                paths["live_status"],
                _build_neural_drift_live_status_payload(
                    manifest,
                    label=f"Entrenando baseline {model_name} | {balance_mode}",
                    detail=f"{manifest['progress']['completed_units']} / {total_units} unidades completadas.",
                    context={
                        "event": "baseline_start",
                        "baseline_key": baseline_key,
                        "model": model_name,
                        "balance_mode": balance_mode,
                    },
                ),
            )
            _progress_message(f"Entrenando baseline persistente para {model_name} | {balance_mode}...")

            canonical_artifact = _train_canonical_artifact(
                model_name,
                runtime["split"]["X_train"],
                runtime["split"]["y_train"],
                runtime["split"]["X_val"],
                runtime["split"]["y_val"],
                config=config,
                balance_mode=balance_mode,
                feature_metadata=runtime["feature_metadata"],
            )
            baseline_row = _build_baseline_result_row(
                model_name,
                balance_mode,
                canonical_artifact,
                runtime["split"],
            )
            baseline_paths = _persist_neural_drift_baseline_checkpoint(
                paths,
                baseline_key=baseline_key,
                baseline_row=baseline_row,
            )
            manifest["baseline_index"][baseline_key] = {
                **baseline_entry,
                "status": "completed",
                "artifact_paths": baseline_paths,
                "error": None,
                "completed_at": _now_iso(),
            }
            _update_neural_drift_manifest_progress(manifest)
            _persist_neural_drift_manifest(paths["manifest"], manifest)
            _append_neural_drift_live_event(
                paths["live_events"],
                {
                    "timestamp": _now_iso(),
                    "event": "baseline_complete",
                    "run_id": run_id,
                    "baseline_key": baseline_key,
                    "model": model_name,
                    "balance_mode": balance_mode,
                },
            )

            for experiment_spec in pending_experiment_specs:
                experiment_key = str(experiment_spec["experiment_key"])
                strategy = str(experiment_spec["strategy"])
                manifest["experiment_index"][experiment_key]["status"] = "running"
                manifest["experiment_index"][experiment_key]["error"] = None
                _update_neural_drift_manifest_progress(manifest)
                _persist_neural_drift_manifest(paths["manifest"], manifest)
                _append_neural_drift_live_event(
                    paths["live_events"],
                    {
                        "timestamp": _now_iso(),
                        "event": "experiment_start",
                        "run_id": run_id,
                        "experiment_key": experiment_key,
                        "model": model_name,
                        "strategy": strategy,
                        "balance_mode": balance_mode,
                    },
                )
                _persist_neural_drift_live_status(
                    paths["live_status"],
                    _build_neural_drift_live_status_payload(
                        manifest,
                        label=f"Simulando {model_name} | {strategy} | {balance_mode}",
                        detail=f"{manifest['progress']['completed_units']} / {total_units} unidades completadas.",
                        context={
                            "event": "experiment_start",
                            "experiment_key": experiment_key,
                            "model": model_name,
                            "strategy": strategy,
                            "balance_mode": balance_mode,
                        },
                    ),
                )
                _progress_message(f"Simulando {model_name} | {strategy} | {balance_mode}...")

                def _persisting_live_callback(payload: Dict[str, Any]) -> None:
                    if live_update_callback is not None:
                        live_update_callback(payload)
                    if str(payload.get("event") or "") != "stream_step":
                        return
                    step_index = int(payload.get("stream_step_index", 0))
                    total_rows = max(1, int(payload.get("stream_total_rows", 0)))
                    if step_index <= 0:
                        return
                    if (
                        step_index % NEURAL_DRIFT_LIVE_STATUS_HEARTBEAT_STEPS != 0
                        and step_index != total_rows
                    ):
                        return
                    _persist_neural_drift_live_status(
                        paths["live_status"],
                        _build_neural_drift_live_status_payload(
                            manifest,
                            label=f"Streaming {model_name} | {strategy} | {balance_mode}",
                            detail=(
                                f"Paso {step_index}/{total_rows} | "
                                f"severity={float(payload.get('severity_score', 0.0)):.3f} | "
                                f"action={payload.get('action_taken', 'none')}"
                            ),
                            context={
                                "event": "stream_heartbeat",
                                "experiment_key": experiment_key,
                                "model": model_name,
                                "strategy": strategy,
                                "balance_mode": balance_mode,
                                "stream_step_index": step_index,
                                "stream_total_rows": total_rows,
                            },
                        ),
                    )

                experiment_payload = _execute_backtest_experiment(
                    model_name=model_name,
                    strategy=strategy,
                    balance_mode=balance_mode,
                    canonical_artifact=canonical_artifact,
                    split=runtime["split"],
                    config=config,
                    selected_channels=runtime["selected_channels"],
                    live_update_callback=_persisting_live_callback,
                )
                experiment_results = _finalize_backtest_results(
                    dataset=runtime["dataset"],
                    split=runtime["split"],
                    baseline_rows=[baseline_row],
                    stream_rows=experiment_payload["stream_rows"],
                    drift_rows=experiment_payload["drift_rows"],
                    attention_rows=experiment_payload["attention_rows"],
                    detector_attention_rows=experiment_payload["detector_attention_rows"],
                    config=config,
                )
                experiment_artifact_paths = _persist_neural_drift_result_artifacts(
                    _experiment_dir(paths, experiment_key),
                    experiment_results,
                    keys=[key for key in NEURAL_DRIFT_RESULT_KEYS if key != "baseline"],
                )
                summary_df = experiment_results.get("summary")
                drift_df = experiment_results.get("drift_events")
                manifest["experiment_index"][experiment_key] = {
                    **dict(manifest["experiment_index"][experiment_key]),
                    "status": "completed",
                    "artifact_paths": experiment_artifact_paths,
                    "error": None,
                    "completed_at": _now_iso(),
                    "summary_rows": (
                        int(len(summary_df))
                        if isinstance(summary_df, pd.DataFrame)
                        else 0
                    ),
                    "n_drift_events": (
                        int(len(drift_df))
                        if isinstance(drift_df, pd.DataFrame)
                        else 0
                    ),
                }
                _update_neural_drift_manifest_progress(manifest)
                _persist_neural_drift_manifest(paths["manifest"], manifest)
                _append_neural_drift_live_event(
                    paths["live_events"],
                    {
                        "timestamp": _now_iso(),
                        "event": "experiment_complete",
                        "run_id": run_id,
                        "experiment_key": experiment_key,
                        "model": model_name,
                        "strategy": strategy,
                        "balance_mode": balance_mode,
                    },
                )
                _progress_message(f"Experimento completado: {model_name} | {strategy} | {balance_mode}.")

        assembled_results = _assemble_persisted_neural_drift_results(manifest, run_dir=paths["run_dir"])
        manifest["artifacts"] = _persist_neural_drift_result_artifacts(
            paths["artifacts_dir"],
            assembled_results,
            keys=NEURAL_DRIFT_RESULT_KEYS,
        )
        manifest["status"] = "completed"
        manifest["result_status"] = "success"
        manifest["last_error"] = None
        manifest["resume"] = {
            "auto_resumed": bool(resume_run_id),
            "checkpoint_status": "completed",
        }
        _update_neural_drift_manifest_progress(manifest)
        _persist_neural_drift_manifest(paths["manifest"], manifest)
        _persist_neural_drift_live_status(
            paths["live_status"],
            _build_neural_drift_live_status_payload(
                manifest,
                label="Corrida Neural drift completada.",
                detail=f"{manifest['progress']['completed_units']} / {total_units} unidades completadas.",
                status="completed",
                result_status="success",
                context={"event": "run_complete"},
            ),
        )
        _append_neural_drift_live_event(
            paths["live_events"],
            {
                "timestamp": _now_iso(),
                "event": "run_complete",
                "run_id": run_id,
            },
        )
        _progress_message("Corrida persistente completada.")
        return {
            **assembled_results,
            "run_id": str(run_id),
            "run_signature": str(run_signature),
            "manifest": manifest,
            "manifest_path": str(paths["manifest"]),
            "run_dir": str(paths["run_dir"]),
            "download_bundle": _download_bundle_from_results(assembled_results),
        }
    except Exception as exc:
        baseline_index = dict(manifest.get("baseline_index") or {})
        experiment_index = dict(manifest.get("experiment_index") or {})
        for key, item in baseline_index.items():
            if str(item.get("status") or "") == "running":
                baseline_index[str(key)] = {
                    **dict(item),
                    "status": "failed",
                    "error": str(exc),
                }
        for key, item in experiment_index.items():
            if str(item.get("status") or "") == "running":
                experiment_index[str(key)] = {
                    **dict(item),
                    "status": "failed",
                    "error": str(exc),
                }
        manifest["baseline_index"] = baseline_index
        manifest["experiment_index"] = experiment_index
        manifest["status"] = "failed"
        manifest["result_status"] = "failed"
        manifest["last_error"] = {
            "message": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }
        _update_neural_drift_manifest_progress(manifest)
        _persist_neural_drift_manifest(paths["manifest"], manifest)
        _persist_neural_drift_live_status(
            paths["live_status"],
            _build_neural_drift_live_status_payload(
                manifest,
                label="Corrida Neural drift fallida.",
                detail=str(exc),
                status="failed",
                result_status="failed",
                context={"event": "error"},
            ),
        )
        _append_neural_drift_live_event(
            paths["live_events"],
            {
                "timestamp": _now_iso(),
                "event": "error",
                "run_id": str(manifest.get("run_id") or run_id),
                "message": str(exc),
            },
        )
        raise


def _live_backtest_chart_frames(
    stream_rows: Sequence[Dict[str, Any]],
    *,
    rolling_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not stream_rows:
        return pd.DataFrame(), pd.DataFrame()

    stream_df = pd.DataFrame(stream_rows).copy()
    if stream_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    stream_df["timestamp"] = pd.to_datetime(stream_df["timestamp"])
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)

    drift_chart = pd.DataFrame(
        {
            "timestamp": stream_df["timestamp"],
            "severity_score": pd.to_numeric(stream_df["severity_score"], errors="coerce"),
            "max_channel_score": pd.to_numeric(stream_df["max_channel_score"], errors="coerce"),
            "severity_threshold": pd.to_numeric(stream_df["severity_threshold"], errors="coerce"),
            "drift_event_flag": stream_df["is_drift_event"].astype(bool).astype(int),
            "adaptation_flag": stream_df["action_taken"].astype(str).ne("none").astype(int),
        }
    ).set_index("timestamp").sort_index()

    rolling_source = stream_df[
        [
            "timestamp",
            "model",
            "strategy",
            "balance_mode",
            "y_true",
            "prediction",
            "score",
            "decision_threshold",
            "severity_score",
        ]
    ].copy()
    rolling_df = _rolling_metric_table(
        rolling_source,
        rolling_window=max(1, int(rolling_window)),
    )
    if rolling_df.empty:
        return drift_chart, pd.DataFrame()

    metrics_chart = (
        rolling_df.set_index("timestamp")[["recall", "fnr", "brier"]]
        .sort_index()
    )
    return drift_chart, metrics_chart


def _live_backtest_status_line(
    stream_rows: Sequence[Dict[str, Any]],
    *,
    model: str,
    strategy: str,
    balance_mode: str,
    stream_total_rows: int,
) -> str:
    processed_rows = int(len(stream_rows))
    drift_events = int(sum(bool(row.get("is_drift_event", False)) for row in stream_rows))
    applied_actions = int(sum(str(row.get("action_taken", "none")) != "none" for row in stream_rows))
    return (
        f"Simulacion en vivo: {model} | {strategy} | {balance_mode} · "
        f"fila {processed_rows}/{max(1, int(stream_total_rows))} · "
        f"drifts {drift_events} · adaptaciones {applied_actions}"
    )


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


def _build_configuration_controls_explanation(config: Dict[str, Any]) -> Dict[str, Any]:
    severity_threshold = float(config.get("severity_threshold", DEFAULT_CONFIG["severity_threshold"]))
    recent_window_size = int(config.get("recent_window_size", DEFAULT_CONFIG["recent_window_size"]))
    xgb_fine_tune_selection_metric = _resolve_xgb_fine_tune_selection_metric(
        config.get(
            "xgb_fine_tune_selection_metric",
            DEFAULT_CONFIG["xgb_fine_tune_selection_metric"],
        )
    )
    xgb_fine_tune_selection_metric_label = _xgb_fine_tune_selection_metric_label(
        xgb_fine_tune_selection_metric
    )
    available_metric_labels = ", ".join(
        f"`{_xgb_fine_tune_selection_metric_label(metric)}`"
        for metric in AVAILABLE_XGB_FINE_TUNE_SELECTION_METRICS
    )
    xgb_window_min = int(config.get("xgb_fine_tune_window_min", DEFAULT_CONFIG["xgb_fine_tune_window_min"]))
    xgb_window_max = int(config.get("xgb_fine_tune_window_max", DEFAULT_CONFIG["xgb_fine_tune_window_max"]))
    xgb_rounds_min = int(config.get("xgb_fine_tune_rounds_min", DEFAULT_CONFIG["xgb_fine_tune_rounds_min"]))
    xgb_rounds_max = int(config.get("xgb_fine_tune_rounds_max", DEFAULT_CONFIG["xgb_fine_tune_rounds_max"]))
    xgb_eta_multiplier_max = float(
        config.get(
            "xgb_fine_tune_eta_multiplier_max",
            DEFAULT_CONFIG["xgb_fine_tune_eta_multiplier_max"],
        )
    )
    xgb_recent_weight_max = float(
        config.get(
            "xgb_fine_tune_recent_weight_max",
            DEFAULT_CONFIG["xgb_fine_tune_recent_weight_max"],
        )
    )
    recalibration_min_rows = int(
        config.get("recalibration_min_rows", DEFAULT_CONFIG["recalibration_min_rows"])
    )
    retrain_min_rows = int(config.get("retrain_min_rows", DEFAULT_CONFIG["retrain_min_rows"]))
    lookback_steps = int(config.get("lookback_steps", DEFAULT_CONFIG["lookback_steps"]))
    horizon_steps = int(config.get("horizon_steps", DEFAULT_CONFIG["horizon_steps"]))
    max_stream_rows = int(config.get("max_stream_rows", DEFAULT_CONFIG["max_stream_rows"]))
    dataset_percent = int(config.get("dataset_percent", DEFAULT_CONFIG["dataset_percent"]))
    rolling_metric_window = int(
        config.get("rolling_metric_window", DEFAULT_CONFIG["rolling_metric_window"])
    )
    history_sample_size = int(config.get("history_sample_size", DEFAULT_CONFIG["history_sample_size"]))
    xgb_parallel_neural_enabled = _xgb_parallel_neural_enabled(config)
    point_signal_weight = float(
        config.get("drift_point_signal_weight", DEFAULT_CONFIG["drift_point_signal_weight"])
    )
    detector_adwin_delta = float(
        config.get("detector_adwin_delta", DEFAULT_CONFIG["detector_adwin_delta"])
    )
    selected_balance_modes = [f"`{mode}`" for mode in _resolve_balance_modes(config.get("balance_modes"))]
    selected_models = [f"`{model}`" for model in config.get("models", []) if str(model).strip()]
    selected_strategies = [f"`{strategy}`" for strategy in config.get("strategies", []) if str(strategy).strip()]
    selected_channels = [
        f"`{channel}`" for channel in config.get("drift_channels", []) if str(channel).strip()
    ]
    balance_summary = ", ".join(selected_balance_modes) if selected_balance_modes else "ninguno"
    models_summary = ", ".join(selected_models) if selected_models else "ninguno"
    strategies_summary = ", ".join(selected_strategies) if selected_strategies else "ninguna"
    channels_summary = ", ".join(selected_channels) if selected_channels else "ninguno"

    return {
        "overview": (
            "Esta configuracion define que tramo temporal se evalua, como se calcula la severidad del drift, "
            "y que tipo de adaptacion se ejecuta cuando el detector decide que el cambio ya es suficientemente fuerte."
        ),
        "decision_rule": (
            f"`drift event if severity_score >= {severity_threshold:.2f} "
            f"or max_channel_score >= {severity_threshold:.2f}`"
        ),
        "scope_steps": [
            (
                f"`Porcentaje del dataset = {dataset_percent}%`",
                "Usa solo el tramo mas reciente del dataset. Bajar este valor acelera el backtest y enfoca el analisis "
                "en el regimen mas actual, pero deja menos historia disponible para ver transiciones largas."
            ),
            (
                f"`Lookback steps = {lookback_steps}` y `Horizon steps = {horizon_steps}`",
                "El lookback define cuantas observaciones pasadas ve el predictor para construir cada ventana. "
                "El horizon define cuan adelante esta la etiqueta objetivo que se quiere anticipar."
            ),
            (
                f"`Max stream rows = {max_stream_rows}`",
                "Limita cuantas filas del tramo de stream se usan en la simulacion online. Sirve para comparar estrategias "
                "sin tener que recorrer siempre todo el historial disponible."
            ),
        ],
        "sensitivity_steps": [
            (
                f"`Preset de sensibilidad`, `ADWIN delta = {detector_adwin_delta:.4f}`",
                "El preset mueve varios controles a la vez. `ADWIN delta` afecta a los canales clasicos "
                "(`input drift`, `score drift`, `error drift`): valores mas altos reaccionan antes; valores mas bajos exigen cambios mas consistentes."
            ),
            (
                f"`Point signal weight = {point_signal_weight:.2f}`",
                "Mezcla dos lecturas del drift: la senal suavizada por ventana y los picos locales. Si sube hacia 1.0, "
                "el detector reacciona mas a cambios puntuales; si baja hacia 0.0, prioriza una tendencia mas estable."
            ),
            (
                f"`Severity trigger = {severity_threshold:.2f}`",
                "Es la compuerta final. No calcula el drift por si solo: decide cuando el score ya es suficientemente alto "
                "para registrar un evento y permitir acciones de adaptacion. Bajarlo genera mas eventos; subirlo los vuelve mas exigentes."
            ),
        ],
        "execution_steps": [
            (
                f"`Balance modes = {balance_summary}`",
                "Corre cada combinacion con `none` y/o `smote`. Cuando eliges `smote`, los parametros "
                "`sampling_strategy` y `k_neighbors` se buscan sobre train/validation y el oversampling "
                "solo se aplica en entrenamiento, nunca sobre validation ni stream."
            ),
            (
                f"`Models = {models_summary}`",
                "Define que predictores se comparan en paralelo. `Torch MLP` y `Torch MLP + Attention` usan sus propios "
                "embeddings para `embedding drift`. `XGBoost` puede activar una rama neuronal auxiliar paralela para sumar "
                "senales de `score drift`, `error drift` y `embedding drift`."
            ),
            (
                f"`XGBoost parallel neural branch = {'on' if xgb_parallel_neural_enabled else 'off'}`",
                "Cuando esta activa, `XGBoost` mantiene una `Torch MLP` auxiliar sincronizada con el mismo stream para "
                "detectar drift neuronal sin cambiar el rol de `XGBoost` como predictor oficial."
            ),
            (
                f"`Strategies = {strategies_summary}`",
                "`fixed` nunca adapta el modelo; `recalibration` ajusta calibrador y threshold; `fine_tuning` actualiza "
                "el modelo con la ventana reciente; `retrain` vuelve a entrenar usando historia mas un bloque reciente."
            ),
            (
                f"`XGBoost fine-tuning metric = {xgb_fine_tune_selection_metric_label}`",
                "Cuando `XGBoost` usa `fine_tuning`, la app prueba varias cantidades de rondas nuevas sobre una ventana "
                "adaptativa y se queda con la candidata que mejor rinde segun esta metrica. "
                f"Opciones disponibles: {available_metric_labels}."
            ),
            (
                f"`Drift channels = {channels_summary}`",
                "Cada canal aporta una parte del score: cambio en covariables (`input drift`), cambio en scores "
                "del predictor (`score drift`), cambio en error (`error drift`) y cambio en embeddings (`embedding drift`)."
            ),
        ],
        "adaptation_steps": [
            (
                f"`Recent window size = {recent_window_size}`",
                "Es la ventana fija del detector: suaviza la lectura de severidad y define el bloque reciente con el "
                "que se recalibra o reentrena. En `XGBoost` + `fine_tuning`, la actualizacion usa internamente una "
                "ventana adaptativa propia guiada por `severity_intensity`, sin cambiar esta ventana del detector."
            ),
            (
                f"`Recalibration min rows = {recalibration_min_rows}` y `Retrain min rows = {retrain_min_rows}`",
                "Evitan adaptar con muy pocos datos. Si hay trigger pero la ventana reciente no alcanza estas cotas "
                "o no tiene al menos dos clases, el evento se registra y la accion queda en `none`."
            ),
            (
                "`Ventana adaptativa de XGBoost`",
                "Solo afecta a `fine_tuning` de `XGBoost`. A mayor severidad, aumenta la ventana usada para actualizar, "
                "sube el peso relativo de las observaciones mas nuevas y se habilitan mas rondas candidatas del booster."
            ),
            (
                "`Optimizador de XGBoost fine-tuning`",
                "Los controles `Window min/max`, `Rounds min/max`, `Eta multiplier max` y `Recent weight max` "
                "definen el rango que explora la actualizacion adaptativa. "
                f"Ahora mismo estan en window `{xgb_window_min}`-`{xgb_window_max}`, rounds `{xgb_rounds_min}`-`{xgb_rounds_max}`, "
                f"`eta` max `{xgb_eta_multiplier_max:.2f}` y peso reciente max `{xgb_recent_weight_max:.2f}`."
            ),
            (
                f"`Hybrid history sample = {history_sample_size}`",
                "Solo afecta a `retrain`. Controla cuanto historial previo se conserva al armar el bloque hibrido "
                "historia + reciente para reentrenar."
            ),
            (
                f"`Rolling metric window = {rolling_metric_window}`",
                "No cambia el entrenamiento ni el detector. Solo define el tamano de la ventana con la que se resumen "
                "metricas como `pr_auc`, `recall`, `fnr`, `brier` y `severity_score` en la vista de resultados."
            ),
        ],
        "tuning_guidance": [
            (
                "Si ves demasiados triggers",
                "Sube `Severity trigger`, usa un preset mas conservador, baja `Point signal weight` y/o aumenta `Recent window size`."
            ),
            (
                "Si el detector llega tarde",
                "Baja `Severity trigger`, usa un preset mas sensible y reduce `Recent window size` para que la senal responda antes."
            ),
            (
                "Si quieres saber donde nace el trigger",
                "Revisa `Drift events` y compara `severity_score`, `max_channel_score`, `channel_scores` y `action_taken`. "
                "Ahi se ve si el evento vino por consenso entre canales o por un pico fuerte en uno solo."
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
    df = dataset_bundle["df"]
    max_stream_min = 48
    max_stream_max = max(max_stream_min, int(len(df)))
    max_stream_default = min(int(DEFAULT_CONFIG["max_stream_rows"]), max_stream_max)
    max_stream_default = max(max_stream_min, max_stream_default)

    config_state_defaults = {
        "neural_drift_dataset_percent": int(DEFAULT_CONFIG["dataset_percent"]),
        "neural_drift_sensitivity_preset": str(DEFAULT_CONFIG["detector_sensitivity_preset"]),
        "neural_drift_recent_window_size": int(DEFAULT_CONFIG["recent_window_size"]),
        "neural_drift_severity_threshold": float(DEFAULT_CONFIG["severity_threshold"]),
        "neural_drift_detector_adwin_delta": float(DEFAULT_CONFIG["detector_adwin_delta"]),
        "neural_drift_point_signal_weight": float(DEFAULT_CONFIG["drift_point_signal_weight"]),
        "neural_drift_xgb_fine_tune_selection_metric": str(DEFAULT_CONFIG["xgb_fine_tune_selection_metric"]),
        "neural_drift_xgb_fine_tune_window_min": int(DEFAULT_CONFIG["xgb_fine_tune_window_min"]),
        "neural_drift_xgb_fine_tune_window_max": int(DEFAULT_CONFIG["xgb_fine_tune_window_max"]),
        "neural_drift_xgb_fine_tune_rounds_min": int(DEFAULT_CONFIG["xgb_fine_tune_rounds_min"]),
        "neural_drift_xgb_fine_tune_rounds_max": int(DEFAULT_CONFIG["xgb_fine_tune_rounds_max"]),
        "neural_drift_xgb_fine_tune_eta_multiplier_max": float(
            DEFAULT_CONFIG["xgb_fine_tune_eta_multiplier_max"]
        ),
        "neural_drift_xgb_fine_tune_recent_weight_max": float(
            DEFAULT_CONFIG["xgb_fine_tune_recent_weight_max"]
        ),
    }
    for state_key, default_value in config_state_defaults.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value

    st.session_state["neural_drift_xgb_fine_tune_window_min"] = int(
        st.session_state.get(
            "neural_drift_xgb_fine_tune_window_min",
            DEFAULT_CONFIG["xgb_fine_tune_window_min"],
        )
    )
    st.session_state["neural_drift_xgb_fine_tune_window_max"] = max(
        int(st.session_state["neural_drift_xgb_fine_tune_window_min"]),
        int(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_window_max",
                DEFAULT_CONFIG["xgb_fine_tune_window_max"],
            )
        ),
    )
    st.session_state["neural_drift_xgb_fine_tune_rounds_min"] = int(
        st.session_state.get(
            "neural_drift_xgb_fine_tune_rounds_min",
            DEFAULT_CONFIG["xgb_fine_tune_rounds_min"],
        )
    )
    st.session_state["neural_drift_xgb_fine_tune_rounds_max"] = max(
        int(st.session_state["neural_drift_xgb_fine_tune_rounds_min"]),
        int(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_rounds_max",
                DEFAULT_CONFIG["xgb_fine_tune_rounds_max"],
            )
        ),
    )

    def _on_detector_sensitivity_preset_change() -> None:
        _apply_detector_sensitivity_preset_to_session(st.session_state.get("neural_drift_sensitivity_preset"))

    selected_models: List[str] = list(DEFAULT_CONFIG["models"])
    selected_strategies: List[str] = list(DEFAULT_CONFIG["strategies"])
    selected_balance_modes: List[str] = [BALANCE_MODE_NONE, BALANCE_MODE_SMOTE]

    general_tab, adwin_tab, models_tab, adaptation_tab = st.tabs(
        ["General", "ADWIN", "Modelos", "Adaptación y XGBoost"]
    )

    with general_tab:
        st.markdown("**Dataset activo**")
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

        config_col_1, config_col_2, config_col_3 = st.columns(3)
        config_col_1.number_input(
            "Lookback steps",
            min_value=4,
            max_value=36,
            value=int(DEFAULT_CONFIG["lookback_steps"]),
            step=1,
            key="neural_drift_lookback_steps",
        )
        config_col_2.number_input(
            "Horizon steps",
            min_value=1,
            max_value=6,
            value=int(DEFAULT_CONFIG["horizon_steps"]),
            step=1,
            key="neural_drift_horizon_steps",
        )
        config_col_3.number_input(
            "Max stream rows",
            min_value=max_stream_min,
            max_value=max_stream_max,
            value=max_stream_default,
            step=24,
            key="neural_drift_max_stream_rows",
        )

    with adwin_tab:
        st.markdown("**Sensibilidad del detector**")
        sensitivity_col_1, sensitivity_col_2 = st.columns([1.2, 2.2])
        sensitivity_preset = sensitivity_col_1.selectbox(
            "Preset de sensibilidad",
            AVAILABLE_DETECTOR_SENSITIVITY_PRESETS,
            key="neural_drift_sensitivity_preset",
            on_change=_on_detector_sensitivity_preset_change,
        )
        sensitivity_description = _detector_sensitivity_preset_description(sensitivity_preset)
        if sensitivity_description:
            sensitivity_col_2.info(sensitivity_description)

        sensitivity_knob_col_1, sensitivity_knob_col_2 = st.columns(2)
        sensitivity_knob_col_1.number_input(
            "ADWIN delta",
            min_value=0.0005,
            max_value=0.0500,
            step=0.0005,
            format="%.4f",
            key="neural_drift_detector_adwin_delta",
        )
        sensitivity_knob_col_2.slider(
            "Point signal weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="neural_drift_point_signal_weight",
            help="0.0 prioriza la senal suavizada por ventana; 1.0 prioriza los picos locales y reproduce la sensibilidad actual.",
        )
        st.slider(
            "Severity trigger",
            min_value=0.10,
            max_value=0.95,
            step=0.05,
            key="neural_drift_severity_threshold",
        )

    with models_tab:
        model_col, strategy_col, balance_col, channel_col = st.columns(4)
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
        selected_balance_modes = balance_col.multiselect(
            "Balance modes",
            AVAILABLE_BALANCE_MODES,
            default=[BALANCE_MODE_NONE, BALANCE_MODE_SMOTE],
            key="neural_drift_balance_modes",
            help="`smote` se sintoniza solo sobre train/validation y nunca se aplica sobre el stream.",
        )
        channel_col.multiselect(
            "Drift channels",
            AVAILABLE_DRIFT_CHANNELS,
            default=list(DEFAULT_CONFIG["drift_channels"]),
            key="neural_drift_channels",
        )
        if MODEL_XGBOOST in selected_models and STRATEGY_FINE_TUNING in selected_strategies:
            st.caption(
                "En `XGBoost`, `fine_tuning` mantiene fija la ventana del detector (`Recent window size`), "
                "pero calcula internamente una ventana adaptativa segun la severidad del drift."
            )
        if BALANCE_MODE_SMOTE in selected_balance_modes:
            st.caption(
                "El modo `smote` busca internamente `sampling_strategy` y `k_neighbors` sobre el split de entrenamiento; "
                "la validacion y el stream se mantienen siempre sin oversampling."
            )

    with adaptation_tab:
        backtest_col_1, backtest_col_2, backtest_col_3 = st.columns(3)
        backtest_col_1.number_input(
            "Recent window size",
            min_value=24,
            max_value=240,
            step=8,
            key="neural_drift_recent_window_size",
        )
        backtest_col_2.number_input(
            "Recalibration min rows",
            min_value=16,
            max_value=160,
            value=int(DEFAULT_CONFIG["recalibration_min_rows"]),
            step=8,
            key="neural_drift_recalibration_min_rows",
        )
        backtest_col_3.number_input(
            "Retrain min rows",
            min_value=24,
            max_value=240,
            value=int(DEFAULT_CONFIG["retrain_min_rows"]),
            step=8,
            key="neural_drift_retrain_min_rows",
        )

        adaptation_col_1, adaptation_col_2 = st.columns(2)
        adaptation_col_1.number_input(
            "Rolling metric window",
            min_value=12,
            max_value=120,
            value=int(DEFAULT_CONFIG["rolling_metric_window"]),
            step=6,
            key="neural_drift_rolling_metric_window",
        )
        adaptation_col_2.number_input(
            "Hybrid history sample",
            min_value=64,
            max_value=512,
            value=int(DEFAULT_CONFIG["history_sample_size"]),
            step=32,
            key="neural_drift_history_sample_size",
        )

        st.markdown("**Optimizador de XGBoost fine-tuning**")
        xgb_policy_col_1, xgb_policy_col_2 = st.columns([1.3, 2.7])
        xgb_policy_col_1.selectbox(
            "XGBoost fine-tuning metric",
            AVAILABLE_XGB_FINE_TUNE_SELECTION_METRICS,
            key="neural_drift_xgb_fine_tune_selection_metric",
            format_func=_xgb_fine_tune_selection_metric_label,
            help="Selecciona como se elige la mejor cantidad de rondas nuevas en el `fine_tuning` adaptativo de `XGBoost`.",
        )
        xgb_policy_col_2.info(
            "Solo afecta a `XGBoost` + `fine_tuning`: el booster continua desde el modelo actual, "
            "aumenta el peso de las filas mas recientes y busca la mejor cantidad de rondas nuevas con la metrica elegida."
        )
        st.caption(
            "Estos controles ajustan el rango del optimizador adaptativo: tamano de ventana, rondas candidatas, "
            "escalado maximo de learning rate (`eta`) y peso maximo para observaciones recientes."
        )

        xgb_window_col_1, xgb_window_col_2 = st.columns(2)
        xgb_window_col_1.number_input(
            "XGBoost fine-tuning window min",
            min_value=16,
            max_value=512,
            step=8,
            key="neural_drift_xgb_fine_tune_window_min",
        )
        xgb_window_col_2.number_input(
            "XGBoost fine-tuning window max",
            min_value=int(st.session_state.get("neural_drift_xgb_fine_tune_window_min", 16)),
            max_value=1024,
            step=8,
            key="neural_drift_xgb_fine_tune_window_max",
        )

        xgb_rounds_col_1, xgb_rounds_col_2 = st.columns(2)
        xgb_rounds_col_1.number_input(
            "XGBoost fine-tuning rounds min",
            min_value=1,
            max_value=128,
            step=1,
            key="neural_drift_xgb_fine_tune_rounds_min",
        )
        xgb_rounds_col_2.number_input(
            "XGBoost fine-tuning rounds max",
            min_value=int(st.session_state.get("neural_drift_xgb_fine_tune_rounds_min", 1)),
            max_value=256,
            step=1,
            key="neural_drift_xgb_fine_tune_rounds_max",
        )

        xgb_optimizer_col_1, xgb_optimizer_col_2 = st.columns(2)
        xgb_optimizer_col_1.number_input(
            "XGBoost eta multiplier max",
            min_value=1.0,
            max_value=4.0,
            step=0.05,
            format="%.2f",
            key="neural_drift_xgb_fine_tune_eta_multiplier_max",
        )
        xgb_optimizer_col_2.number_input(
            "XGBoost recent weight max",
            min_value=1.0,
            max_value=8.0,
            step=0.10,
            format="%.2f",
            key="neural_drift_xgb_fine_tune_recent_weight_max",
        )

    base_config = {
        **DEFAULT_CONFIG,
        **dict(st.session_state.get("neural_drift_config") or {}),
    }
    config = {
        **base_config,
        "lookback_steps": int(
            st.session_state.get("neural_drift_lookback_steps", base_config["lookback_steps"])
        ),
        "horizon_steps": int(
            st.session_state.get("neural_drift_horizon_steps", base_config["horizon_steps"])
        ),
        "dataset_percent": int(
            st.session_state.get("neural_drift_dataset_percent", base_config["dataset_percent"])
        ),
        "max_stream_rows": int(
            st.session_state.get("neural_drift_max_stream_rows", max_stream_default)
        ),
        "balance_modes": _resolve_balance_modes(
            st.session_state.get("neural_drift_balance_modes", selected_balance_modes)
        ),
        "models": list(selected_models),
        "strategies": list(selected_strategies),
        "drift_channels": list(
            st.session_state.get("neural_drift_channels", base_config["drift_channels"])
        ),
        "recent_window_size": int(
            st.session_state.get("neural_drift_recent_window_size", base_config["recent_window_size"])
        ),
        "recalibration_min_rows": int(
            st.session_state.get(
                "neural_drift_recalibration_min_rows",
                base_config["recalibration_min_rows"],
            )
        ),
        "retrain_min_rows": int(
            st.session_state.get("neural_drift_retrain_min_rows", base_config["retrain_min_rows"])
        ),
        "severity_threshold": float(
            st.session_state.get("neural_drift_severity_threshold", base_config["severity_threshold"])
        ),
        "detector_sensitivity_preset": str(
            st.session_state.get(
                "neural_drift_sensitivity_preset",
                base_config["detector_sensitivity_preset"],
            )
        ),
        "detector_adwin_delta": float(
            st.session_state.get("neural_drift_detector_adwin_delta", base_config["detector_adwin_delta"])
        ),
        "drift_point_signal_weight": float(
            st.session_state.get(
                "neural_drift_point_signal_weight",
                base_config["drift_point_signal_weight"],
            )
        ),
        "rolling_metric_window": int(
            st.session_state.get("neural_drift_rolling_metric_window", base_config["rolling_metric_window"])
        ),
        "history_sample_size": int(
            st.session_state.get("neural_drift_history_sample_size", base_config["history_sample_size"])
        ),
        "xgb_fine_tune_selection_metric": _resolve_xgb_fine_tune_selection_metric(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_selection_metric",
                base_config["xgb_fine_tune_selection_metric"],
            )
        ),
        "xgb_fine_tune_window_min": int(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_window_min",
                base_config["xgb_fine_tune_window_min"],
            )
        ),
        "xgb_fine_tune_window_max": int(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_window_max",
                base_config["xgb_fine_tune_window_max"],
            )
        ),
        "xgb_fine_tune_rounds_min": int(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_rounds_min",
                base_config["xgb_fine_tune_rounds_min"],
            )
        ),
        "xgb_fine_tune_rounds_max": int(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_rounds_max",
                base_config["xgb_fine_tune_rounds_max"],
            )
        ),
        "xgb_fine_tune_eta_multiplier_max": float(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_eta_multiplier_max",
                base_config["xgb_fine_tune_eta_multiplier_max"],
            )
        ),
        "xgb_fine_tune_recent_weight_max": float(
            st.session_state.get(
                "neural_drift_xgb_fine_tune_recent_weight_max",
                base_config["xgb_fine_tune_recent_weight_max"],
            )
        ),
    }
    st.session_state["neural_drift_config"] = config

    explanation = _build_configuration_controls_explanation(config)
    st.markdown("**Como interpretar esta configuracion**")
    st.write(explanation["overview"])

    with st.expander("Lectura guiada de los controles", expanded=True):
        st.markdown("**1. Alcance temporal del experimento**")
        for title, body in explanation["scope_steps"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**2. Sensibilidad y disparo de drift**")
        st.markdown(explanation["decision_rule"])
        for title, body in explanation["sensitivity_steps"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**3. Modelos, estrategias y canales**")
        for title, body in explanation["execution_steps"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**4. Ventanas y acciones de adaptacion**")
        for title, body in explanation["adaptation_steps"]:
            st.markdown(f"- {title}: {body}")

        st.markdown("**5. Como ajustar los controles**")
        for title, body in explanation["tuning_guidance"]:
            st.markdown(f"- {title}: {body}")

    return config


def _render_monitor_network_subtab(dataset_bundle: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown("**Detector neuronal de drift**")
    st.caption(
        "La senal `embedding drift` combina distancia al centroide con un monitor neuronal sobre embeddings. "
        "Ese monitor puede ser un autoencoder clasico o una variante con attention temporal."
    )
    selected_models = list(current_config.get("models") or [])
    xgb_parallel_neural_enabled = _xgb_parallel_neural_enabled(current_config)

    if not _has_any_torch_model(selected_models) and not (
        MODEL_XGBOOST in selected_models and xgb_parallel_neural_enabled
    ):
        st.info(
            "Para activar la red que monitorea drift debes incluir `Torch MLP`, `Torch MLP + Attention` "
            "o habilitar la rama neuronal paralela de `XGBoost`."
        )
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
        "neural_drift_xgb_parallel_neural_enabled": bool(
            current_config.get("xgb_parallel_neural_enabled", DEFAULT_CONFIG["xgb_parallel_neural_enabled"])
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

    xgb_parallel_col_1, xgb_parallel_col_2 = st.columns([1.2, 2.2])
    xgb_parallel_neural_enabled = xgb_parallel_col_1.checkbox(
        "Usar rama neuronal paralela para XGBoost",
        key="neural_drift_xgb_parallel_neural_enabled",
    )
    if xgb_parallel_neural_enabled:
        xgb_parallel_col_2.info(
            "`XGBoost` mantendra una `Torch MLP` auxiliar sincronizada para aportar `embedding drift`, "
            "`score drift` y `error drift` a la deteccion/adaptacion."
        )
    else:
        xgb_parallel_col_2.info(
            "Si desactivas esta rama, `XGBoost` vuelve a usar solo canales clasicos y la arquitectura del detector "
            "quedara como `not_available`."
        )

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
        "xgb_parallel_neural_enabled": bool(xgb_parallel_neural_enabled),
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


def _render_history_subtab(dataset_bundle: Dict[str, Any], config: Dict[str, Any]) -> None:
    st.markdown("**Historial de corridas persistidas**")
    runs = _list_persisted_neural_drift_runs()
    if not runs:
        st.info("Todavia no hay corridas persistidas de Neural drift.")
        return

    run_ids = [str(entry["run_id"]) for entry in runs]
    if st.session_state.get("neural_drift_history_selected_run_id") not in run_ids:
        st.session_state["neural_drift_history_selected_run_id"] = run_ids[0]

    history_rows = [
        {
            "run_id": str(entry["run_id"]),
            "status": str(entry["status"]),
            "result_status": str(entry["result_status"]),
            "updated_at": str(entry["updated_at"]),
            "source": str(entry["source"]),
            "rows_used": int(entry["rows_used"]),
            "experiments": f"{int(entry['completed_experiments'])}/{int(entry['total_experiments'])}",
        }
        for entry in runs
    ]
    st.dataframe(_streamlit_arrow_safe_df(pd.DataFrame(history_rows)), width="stretch")

    selected_run_id = str(
        st.selectbox(
            "Corrida persistida",
            run_ids,
            key="neural_drift_history_selected_run_id",
        )
    )
    selected_entry = next(
        entry for entry in runs if str(entry.get("run_id") or "") == selected_run_id
    )
    manifest_path = Path(str(selected_entry["manifest_path"]))
    manifest = dict(_load_json_file(manifest_path, default={}) or {})
    progress = dict(manifest.get("progress") or {})
    current_signature = _build_run_signature(dataset_bundle, config)
    selected_signature = str(selected_entry.get("run_signature") or "")
    signature_matches = bool(selected_signature) and selected_signature == current_signature

    metric_cols = st.columns(4)
    metric_cols[0].metric("Estado", str(selected_entry["status"]))
    metric_cols[1].metric(
        "Experimentos",
        f"{int(progress.get('completed_experiments', 0))}/{int(progress.get('total_experiments', 0))}",
    )
    metric_cols[2].metric("Rows", int(selected_entry["rows_used"]))
    metric_cols[3].metric("Resultado", str(selected_entry["result_status"]))

    st.caption(f"Run signature: {selected_signature or '-'}")
    st.caption(f"Manifest: {manifest_path}")
    if selected_signature and not signature_matches:
        st.warning(
            "La firma de esta corrida no coincide con la configuracion o dataset activo. "
            "Puedes cargar los resultados para analizarlos, pero la reanudacion queda bloqueada."
        )

    dataset_context = dict(manifest.get("dataset_context") or {})
    if dataset_context:
        st.caption(
            "Source: "
            f"{dataset_context.get('source') or '-'} | "
            f"Rows usadas: {int(dataset_context.get('rows_used') or 0)} | "
            f"Rows totales: {int(dataset_context.get('rows_total') or 0)}"
        )

    artifacts = dict(manifest.get("artifacts") or {})
    if artifacts:
        artifact_rows = [
            {
                "artifact": key,
                "available": bool(path),
                "path": str(path),
            }
            for key, path in sorted(artifacts.items())
        ]
        st.dataframe(_streamlit_arrow_safe_df(pd.DataFrame(artifact_rows)), width="stretch")

    action_left, action_right = st.columns(2)
    load_clicked = action_left.button(
        "Cargar resultados persistidos",
        key=f"neural_drift_load_persisted_{selected_run_id}",
    )
    resume_clicked = action_right.button(
        "Preparar reanudacion",
        key=f"neural_drift_prepare_resume_{selected_run_id}",
        disabled=not bool(selected_entry.get("can_resume")) or not signature_matches,
    )

    if load_clicked:
        payload = _load_persisted_neural_drift_run(manifest_path)
        _apply_persisted_neural_drift_run_to_session_state(payload)
        st.success(f"Resultados cargados desde la corrida `{selected_run_id}`.")

    if resume_clicked:
        st.session_state["neural_drift_prepared_resume_run_id"] = str(selected_run_id)
        st.session_state["neural_drift_prepared_resume_manifest_path"] = str(manifest_path)
        st.success(
            f"Reanudacion preparada para `{selected_run_id}`. Ejecuta el backtest para continuar la corrida."
        )


def _visible_config_session_key_map() -> Dict[str, str]:
    return {
        "lookback_steps": "neural_drift_lookback_steps",
        "horizon_steps": "neural_drift_horizon_steps",
        "dataset_percent": "neural_drift_dataset_percent",
        "max_stream_rows": "neural_drift_max_stream_rows",
        "balance_modes": "neural_drift_balance_modes",
        "models": "neural_drift_models",
        "strategies": "neural_drift_strategies",
        "drift_channels": "neural_drift_channels",
        "recent_window_size": "neural_drift_recent_window_size",
        "recalibration_min_rows": "neural_drift_recalibration_min_rows",
        "retrain_min_rows": "neural_drift_retrain_min_rows",
        "severity_threshold": "neural_drift_severity_threshold",
        "detector_sensitivity_preset": "neural_drift_sensitivity_preset",
        "detector_adwin_delta": "neural_drift_detector_adwin_delta",
        "drift_point_signal_weight": "neural_drift_point_signal_weight",
        "rolling_metric_window": "neural_drift_rolling_metric_window",
        "history_sample_size": "neural_drift_history_sample_size",
        "xgb_fine_tune_selection_metric": "neural_drift_xgb_fine_tune_selection_metric",
        "xgb_fine_tune_window_min": "neural_drift_xgb_fine_tune_window_min",
        "xgb_fine_tune_window_max": "neural_drift_xgb_fine_tune_window_max",
        "xgb_fine_tune_rounds_min": "neural_drift_xgb_fine_tune_rounds_min",
        "xgb_fine_tune_rounds_max": "neural_drift_xgb_fine_tune_rounds_max",
        "xgb_fine_tune_eta_multiplier_max": "neural_drift_xgb_fine_tune_eta_multiplier_max",
        "xgb_fine_tune_recent_weight_max": "neural_drift_xgb_fine_tune_recent_weight_max",
        "drift_monitor_profile": "neural_drift_monitor_profile",
        "drift_monitor_architecture": "neural_drift_monitor_architecture",
        "drift_monitor_hidden_dim": "neural_drift_monitor_hidden_dim",
        "drift_monitor_bottleneck_dim": "neural_drift_monitor_bottleneck_dim",
        "drift_monitor_dropout": "neural_drift_monitor_dropout",
        "drift_monitor_epochs": "neural_drift_monitor_epochs",
        "drift_monitor_batch_size": "neural_drift_monitor_batch_size",
        "drift_monitor_learning_rate": "neural_drift_monitor_learning_rate",
        "drift_monitor_reconstruction_weight": "neural_drift_monitor_reconstruction_weight",
        "drift_monitor_sequence_length": "neural_drift_monitor_sequence_length",
        "drift_monitor_attention_hidden_dim": "neural_drift_monitor_attention_hidden_dim",
        "drift_monitor_attention_dropout": "neural_drift_monitor_attention_dropout",
        "xgb_parallel_neural_enabled": "neural_drift_xgb_parallel_neural_enabled",
        "attention_feature_hidden_dim": "neural_drift_attention_feature_hidden_dim",
        "attention_temporal_hidden_dim": "neural_drift_attention_temporal_hidden_dim",
        "attention_dropout": "neural_drift_attention_dropout",
        "attention_top_k": "neural_drift_attention_top_k",
    }


def _normalize_public_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {**copy.deepcopy(DEFAULT_CONFIG), **dict(config or {})}
    normalized["balance_modes"] = _resolve_balance_modes(normalized.get("balance_modes"))
    normalized["models"] = [
        str(model)
        for model in list(normalized.get("models") or [])
        if str(model) in AVAILABLE_MODELS
    ] or list(DEFAULT_CONFIG["models"])
    normalized["strategies"] = [
        str(strategy)
        for strategy in list(normalized.get("strategies") or [])
        if str(strategy) in AVAILABLE_STRATEGIES
    ] or list(DEFAULT_CONFIG["strategies"])
    normalized["drift_channels"] = [
        str(channel)
        for channel in list(normalized.get("drift_channels") or [])
        if str(channel) in AVAILABLE_DRIFT_CHANNELS
    ] or list(DEFAULT_CONFIG["drift_channels"])
    normalized["detector_sensitivity_preset"] = _resolve_detector_sensitivity_preset(
        normalized.get("detector_sensitivity_preset")
    )
    normalized["drift_monitor_profile"] = _resolve_drift_monitor_profile(
        normalized.get("drift_monitor_profile")
    )
    normalized["drift_monitor_architecture"] = _resolve_drift_monitor_architecture(
        normalized.get("drift_monitor_architecture")
    )
    normalized["xgb_fine_tune_selection_metric"] = _resolve_xgb_fine_tune_selection_metric(
        normalized.get("xgb_fine_tune_selection_metric")
    )
    normalized["xgb_parallel_neural_enabled"] = bool(
        normalized.get("xgb_parallel_neural_enabled", DEFAULT_CONFIG["xgb_parallel_neural_enabled"])
    )
    return normalized


def resolve_current_config_from_session_state(
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = {**copy.deepcopy(DEFAULT_CONFIG)}
    try:
        session_config = st.session_state.get("neural_drift_config")
        if isinstance(session_config, dict):
            config.update(session_config)
        for config_key, session_key in _visible_config_session_key_map().items():
            if session_key in st.session_state:
                config[config_key] = st.session_state[session_key]
    except Exception:
        pass
    if isinstance(overrides, dict):
        config.update(overrides)
    return _normalize_public_config(config)


def _apply_experiment_winner_config_to_session_state(winner_config: Dict[str, Any]) -> None:
    if not isinstance(winner_config, dict) or not winner_config:
        return

    visible_key_map = _visible_config_session_key_map()
    for config_key, session_key in visible_key_map.items():
        if config_key in winner_config:
            st.session_state[session_key] = winner_config[config_key]

    merged_config = {
        **DEFAULT_CONFIG,
        **dict(st.session_state.get("neural_drift_config") or {}),
        **dict(winner_config),
    }
    st.session_state["neural_drift_config"] = _normalize_public_config(merged_config)


def _render_experiments_subtab(dataset_bundle: Dict[str, Any], config: Dict[str, Any]) -> None:
    import src.neural_drift_experiments as neural_drift_experiments

    neural_drift_experiments.render_experiments_tab(
        dataset_bundle,
        config,
        apply_winner_config_callback=_apply_experiment_winner_config_to_session_state,
    )


def _render_backtest_subtab(dataset_bundle: Dict[str, Any], config: Dict[str, Any]) -> None:
    prepared_resume_run_id = st.session_state.get("neural_drift_prepared_resume_run_id")
    prepared_resume_manifest_path = st.session_state.get("neural_drift_prepared_resume_manifest_path")
    if prepared_resume_run_id:
        st.info(
            "Hay una corrida preparada para reanudacion: "
            f"`{prepared_resume_run_id}` ({prepared_resume_manifest_path or 'manifest no disponible'})."
        )
        if st.button(
            "Limpiar reanudacion preparada",
            key="neural_drift_clear_prepared_resume",
        ):
            st.session_state["neural_drift_prepared_resume_run_id"] = None
            st.session_state["neural_drift_prepared_resume_manifest_path"] = None
            prepared_resume_run_id = None

    run_clicked = st.button(
        "Resume Neural drift backtest" if prepared_resume_run_id else "Run Neural drift backtest",
        key="neural_drift_run_backtest",
    )
    if not run_clicked:
        st.info(
            "Configura el experimento y ejecuta el backtest temporal."
            if not prepared_resume_run_id
            else "La corrida preparada se reanudara cuando ejecutes el backtest."
        )
        return

    progress_bar = st.progress(0.0)
    status = st.empty()
    live_status = st.empty()
    live_left, live_right = st.columns(2)
    with live_left:
        st.markdown("**Deteccion de drift en vivo**")
        drift_chart_placeholder = st.empty()
    with live_right:
        st.markdown("**Evaluacion del modelo en vivo**")
        metrics_chart_placeholder = st.empty()
    live_status.caption("La simulacion en vivo aparecera aqui cuando comience el stream.")
    drift_chart_placeholder.info("Esperando puntos de drift...")
    metrics_chart_placeholder.info("Esperando metricas rolling...")
    live_state: Dict[str, Any] = {
        "key": None,
        "rows": [],
        "stream_total_rows": 0,
    }

    def _callback(ratio: float, message: str) -> None:
        progress_bar.progress(float(np.clip(ratio, 0.0, 1.0)))
        status.caption(str(message))

    def _live_callback(payload: Dict[str, Any]) -> None:
        event = str(payload.get("event") or "")
        if event == "simulation_start":
            live_state["key"] = (
                str(payload.get("model", "")),
                str(payload.get("strategy", "")),
                str(payload.get("balance_mode", "")),
            )
            live_state["rows"] = []
            live_state["stream_total_rows"] = int(payload.get("stream_total_rows", 0))
            live_status.caption(
                f"Simulacion en vivo: {live_state['key'][0]} | {live_state['key'][1]} | {live_state['key'][2]} · preparando stream..."
            )
            drift_chart_placeholder.info("Esperando primeros puntos de drift...")
            metrics_chart_placeholder.info("Esperando primeras metricas rolling...")
            return
        if event != "stream_step":
            return

        payload_key = (
            str(payload.get("model", "")),
            str(payload.get("strategy", "")),
            str(payload.get("balance_mode", "")),
        )
        if live_state.get("key") != payload_key:
            live_state["key"] = payload_key
            live_state["rows"] = []
        live_state["stream_total_rows"] = int(payload.get("stream_total_rows", live_state.get("stream_total_rows", 0)))
        live_state["rows"].append(dict(payload))

        drift_chart, metrics_chart = _live_backtest_chart_frames(
            live_state["rows"],
            rolling_window=int(
                payload.get(
                    "rolling_metric_window",
                    DEFAULT_CONFIG["rolling_metric_window"],
                )
            ),
        )
        live_status.caption(
            _live_backtest_status_line(
                live_state["rows"],
                model=payload_key[0],
                strategy=payload_key[1],
                balance_mode=payload_key[2],
                stream_total_rows=int(live_state["stream_total_rows"]),
            )
        )
        if drift_chart.empty:
            drift_chart_placeholder.info("Esperando primeros puntos de drift...")
        else:
            drift_chart_placeholder.line_chart(drift_chart, width="stretch")
        if metrics_chart.empty:
            metrics_chart_placeholder.info("Esperando primeras metricas rolling...")
        else:
            metrics_chart_placeholder.line_chart(metrics_chart, width="stretch")

    try:
        with st.spinner("Ejecutando Neural drift backtest..."):
            results = run_backtest_with_checkpoints(
                dataset_bundle,
                config=config,
                progress_callback=_callback,
                live_update_callback=_live_callback,
                resume_run_id=(
                    str(prepared_resume_run_id)
                    if prepared_resume_run_id is not None
                    else None
                ),
            )
    except Exception as exc:
        progress_bar.progress(1.0)
        status.caption("Backtest interrumpido.")
        live_status.caption("Backtest interrumpido.")
        st.error(f"No se pudo ejecutar Neural drift: {exc}")
        return

    progress_bar.progress(1.0)
    status.caption("Backtest completado.")
    _store_neural_drift_results_in_session_state(
        results,
        run_signature=str(results.get("run_signature") or _build_run_signature(dataset_bundle, config)),
        run_id=(
            str(results.get("run_id"))
            if results.get("run_id") is not None
            else None
        ),
        manifest_path=(
            str(results.get("manifest_path"))
            if results.get("manifest_path") is not None
            else None
        ),
    )
    st.session_state["neural_drift_loaded_checkpoint_run_id"] = None
    st.session_state["neural_drift_prepared_resume_run_id"] = None
    st.session_state["neural_drift_prepared_resume_manifest_path"] = None
    st.success(
        "Neural drift finalizado."
        if not results.get("run_id")
        else f"Neural drift finalizado. Run: {results.get('run_id')}"
    )


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
    active_run_id = st.session_state.get("neural_drift_active_run_id")
    active_manifest_path = st.session_state.get("neural_drift_active_manifest_path")
    loaded_checkpoint_run_id = st.session_state.get("neural_drift_loaded_checkpoint_run_id")
    attention_top_k = max(1, int(config.get("attention_top_k", DEFAULT_CONFIG["attention_top_k"])))

    if not isinstance(summary, pd.DataFrame) or summary.empty:
        st.info("No hay resultados de Neural drift para mostrar.")
        return

    if last_run_signature != current_signature:
        st.warning(
            "Los resultados visibles no corresponden a la configuracion o fuente de features actual. "
            "Vuelve a ejecutar el backtest para ver resultados consistentes con el estado actual."
        )

    if active_run_id:
        origin_label = (
            "checkpoint cargado"
            if str(loaded_checkpoint_run_id or "") == str(active_run_id)
            else "corrida activa"
        )
        st.caption(
            f"Mostrando {origin_label}: `{active_run_id}`"
            + (f" | manifest: {active_manifest_path}" if active_manifest_path else "")
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
            attention_feature_summary.groupby(["model", "balance_mode", "strategy"], group_keys=False, dropna=False)
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(feature_display), width="stretch")

    if isinstance(attention_temporal_summary, pd.DataFrame) and not attention_temporal_summary.empty:
        st.markdown("**Attention sobre pasos temporales**")
        temporal_display = (
            attention_temporal_summary.groupby(["model", "balance_mode", "strategy"], group_keys=False, dropna=False)
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(temporal_display), width="stretch")

    if isinstance(attention_drift_shift_summary, pd.DataFrame) and not attention_drift_shift_summary.empty:
        st.markdown("**Cambio de attention durante drift**")
        drift_display = (
            attention_drift_shift_summary.groupby(
                ["model", "balance_mode", "strategy", "scope"],
                group_keys=False,
                dropna=False,
            )
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(drift_display), width="stretch")

    if isinstance(detector_attention_temporal_summary, pd.DataFrame) and not detector_attention_temporal_summary.empty:
        st.markdown("**Attention temporal del detector de drift**")
        detector_temporal_display = (
            detector_attention_temporal_summary.groupby(
                ["model", "balance_mode", "strategy"],
                group_keys=False,
                dropna=False,
            )
            .head(attention_top_k)
            .reset_index(drop=True)
        )
        st.dataframe(_streamlit_arrow_safe_df(detector_temporal_display), width="stretch")

    if isinstance(detector_attention_drift_shift_summary, pd.DataFrame) and not detector_attention_drift_shift_summary.empty:
        st.markdown("**Cambio de attention del detector durante drift**")
        detector_shift_display = (
            detector_attention_drift_shift_summary.groupby(
                ["model", "balance_mode", "strategy"],
                group_keys=False,
                dropna=False,
            )
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
        plot_df["series_label"] = (
            plot_df["model"].astype(str)
            + " | "
            + plot_df["balance_mode"].astype(str)
            + " | "
            + plot_df["strategy"].astype(str)
        )
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

    config_tab, monitor_tab, backtest_tab, results_tab, history_tab, experiments_tab = st.tabs(
        ["Configuración", "Red de drift", "Backtest", "Resultados", "Historial", "Experimentos"]
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
    with history_tab:
        config = dict(st.session_state.get("neural_drift_config") or DEFAULT_CONFIG)
        _render_history_subtab(dataset_bundle, config)
    with experiments_tab:
        config = dict(st.session_state.get("neural_drift_config") or DEFAULT_CONFIG)
        _render_experiments_subtab(dataset_bundle, config)
