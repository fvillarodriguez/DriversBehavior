"""
Logic for running the automated feature selection and model optimization experiments.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

from src.model_training import (
    THRESHOLD_OBJECTIVE_LABELS,
    THRESHOLD_PROTOCOL_LABELS,
    build_model,
    compute_extended_metrics,
    fit_score_calibrator,
    get_model_scores,
    normalize_calibration_method,
    normalize_optuna_objective_metric,
    normalize_threshold_objective,
    normalize_threshold_protocol,
    optuna_objective_direction,
    score_optuna_objective,
    temporal_train_test_split,
    train_model_with_protocol,
)

try:
    import duckdb
except ImportError:  # pragma: no cover - optional dependency at runtime
    duckdb = None


CONTROLLED_COMPARISON_PROTOCOL_VERSION = "controlled_comparison_v3"
CONTROLLED_COMPARISON_MODELS = (
    "Random Forest",
    "Balanced Random Forest",
    "SVM",
    "XGBoost",
    "Neural Network",
)
CONTROLLED_COMPARISON_FEATURE_SETS = ("Base", "Cluster", "Base + Cluster")
CONTROLLED_COMPARISON_BALANCE_MODES = ("none", "smote")
CALIBRATION_SWEEP_PROTOCOL_VERSION = "calibration_sweep_v1"
CALIBRATION_SWEEP_PROTOCOL_FAMILY = "calibration_score_threshold"
CALIBRATION_SWEEP_BALANCE_MODES = CONTROLLED_COMPARISON_BALANCE_MODES
CALIBRATION_SWEEP_THRESHOLD_OBJECTIVES = (
    "far",
    "f1",
    "balanced_f1",
    "mcc",
    "recall_at_alerts_per_day",
    "operational_cost",
)
FROZEN_TUNING_ABLATION_PROTOCOL_FAMILY = "frozen_tuning_ablation"
FROZEN_TUNING_ABLATION_FEATURE_SETS = ("Base", "Base + Cluster")
FROZEN_TUNING_ABLATION_CONFIG = {
    "source_feature_sets": list(FROZEN_TUNING_ABLATION_FEATURE_SETS),
    "target_feature_sets": list(FROZEN_TUNING_ABLATION_FEATURE_SETS),
    "k_policy": "paired_common_k",
    "freeze_scope": ["model_params", "smote_params"],
    "threshold_policy": "recalibrate_per_target",
}
CONTROLLED_COMPARISON_OBJECTIVE_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1": "F1",
    "balanced_f1": "Balanced F1",
    "mcc": "MCC",
    "brier_score": "Brier",
    "far_sens": "FAR - Sensibilidad",
    "recall_at_alerts_per_day": "Recall@N alertas/dia",
    "operational_cost": "Costo operacional",
}
CONTROLLED_COMPARISON_MEMORY_ESTIMATOR_VERSION = "controlled_comparison_memory_v2"
_MEMORY_MB = 1024 ** 2
_CONTROLLED_MEMORY_PROCESS_OVERHEAD_BYTES = 256 * _MEMORY_MB
_CONTROLLED_MEMORY_RF_MODEL_OVERHEAD_BYTES = 128 * _MEMORY_MB
_CONTROLLED_MEMORY_XGB_MODEL_OVERHEAD_BYTES = 192 * _MEMORY_MB
_CONTROLLED_MEMORY_SVM_CACHE_BYTES = 256 * _MEMORY_MB
_CONTROLLED_MEMORY_NN_MODEL_OVERHEAD_BYTES = 128 * _MEMORY_MB
_CONTROLLED_MEMORY_PER_THREAD_MIN_BYTES = 64 * _MEMORY_MB


def _slugify(value: object) -> str:
    text = str(value or "").strip().lower()
    chars: List[str] = []
    last_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_sep = False
        elif not last_sep:
            chars.append("_")
            last_sep = True
    slug = "".join(chars).strip("_")
    return slug or "item"


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


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=True, default=_json_default) + "\n"
        )


def _file_fingerprint(path: Optional[object]) -> Dict[str, object]:
    raw = str(path or "").strip()
    if not raw:
        return {"path": None, "exists": False}
    resolved = Path(raw)
    fingerprint: Dict[str, object] = {"path": str(resolved)}
    if not resolved.exists():
        fingerprint["exists"] = False
        return fingerprint
    stat = resolved.stat()
    fingerprint.update(
        {
            "exists": True,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    return fingerprint


def _numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col not in {"target", "synthetic"} and pd.api.types.is_numeric_dtype(df[col])
    ]


def _cluster_feature_cols(df: pd.DataFrame) -> List[str]:
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
    valid_cols: List[str] = []
    for col in df.columns:
        if col.startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
        if col.startswith("last_") and col[5:].startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
        if col.startswith("next_") and col[5:].startswith(cluster_prefixes):
            valid_cols.append(col)
            continue
    return valid_cols


def _combo_id(
    model_name: str,
    feature_set: str,
    balance_mode: str,
    k: int,
    threshold_protocol: Optional[object] = None,
) -> str:
    base = (
        f"combo__{_slugify(model_name)}__{_slugify(feature_set)}__"
        f"{_slugify(balance_mode)}__k{int(k)}"
    )
    if threshold_protocol is None:
        return base
    return f"{base}__{_slugify(threshold_protocol)}"


def _ablation_combo_id(
    *,
    phase: str,
    model_name: str,
    params_source_feature_set: str,
    target_feature_set: str,
    balance_mode: str,
    k: int,
    threshold_protocol: Optional[object] = None,
) -> str:
    base = (
        f"ablation_{_slugify(phase)}__{_slugify(model_name)}__"
        f"src_{_slugify(params_source_feature_set)}__"
        f"target_{_slugify(target_feature_set)}__"
        f"{_slugify(balance_mode)}__k{int(k)}"
    )
    if threshold_protocol is None:
        return base
    return f"{base}__{_slugify(threshold_protocol)}"


def _selected_feature_family_counts(
    selected_features: Sequence[object],
    cluster_cols: Sequence[object],
) -> Dict[str, int]:
    cluster_set = {str(col) for col in cluster_cols}
    selected = [str(feature) for feature in selected_features]
    cluster_count = sum(1 for feature in selected if feature in cluster_set)
    return {
        "selected_base_feature_count": int(len(selected) - cluster_count),
        "selected_cluster_feature_count": int(cluster_count),
    }


def _json_dict(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _controlled_payload_updates_from_result(
    result: Dict[str, object],
    *,
    threshold_protocol: str,
    threshold_objective: str,
    threshold_objective_label: str,
    calibration_method: str,
    k_global: Optional[int],
    effective_k: int,
    balance_mode: str,
    optuna_n_jobs: int,
    parallel_jobs: int,
    xgb_parallel_jobs: int,
) -> Dict[str, object]:
    return {
        "status": result["status"],
        "objective_metric": result["objective_metric"],
        "objective_label": result["objective_label"],
        "objective_direction": result.get(
            "objective_direction",
            optuna_objective_direction(result["objective_metric"]),
        ),
        "optuna_objective_metric": result["objective_metric"],
        "optuna_objective_label": result["objective_label"],
        "threshold_protocol": result.get("threshold_protocol", threshold_protocol),
        "threshold_protocol_label": result.get(
            "threshold_protocol_label",
            THRESHOLD_PROTOCOL_LABELS.get(threshold_protocol, threshold_protocol),
        ),
        "threshold_objective": result.get("threshold_objective", threshold_objective),
        "threshold_objective_label": result.get(
            "threshold_objective_label",
            threshold_objective_label,
        ),
        "calibration_method": result.get("calibration_method", calibration_method),
        "k_global": k_global,
        "effective_k": int(effective_k),
        "decision_threshold": result["decision_threshold"],
        "val_objective_score": result["val_objective_score"],
        "test_objective_score": result["test_objective_score"],
        "val_accuracy": result.get("val_accuracy"),
        "test_accuracy": result.get("test_accuracy"),
        "val_recall": result.get("val_recall"),
        "test_recall": result.get("test_recall"),
        "val_sensitivity": result.get("val_sensitivity"),
        "test_sensitivity": result.get("test_sensitivity"),
        "val_roc_auc": result["val_roc_auc"],
        "test_roc_auc": result["test_roc_auc"],
        "val_pr_auc": result.get("val_pr_auc"),
        "test_pr_auc": result.get("test_pr_auc"),
        "val_brier_score": result.get("val_brier_score"),
        "test_brier_score": result.get("test_brier_score"),
        "val_f1": result["val_f1"],
        "test_f1": result["test_f1"],
        "val_f1_global": result.get("val_f1_global"),
        "test_f1_global": result.get("test_f1_global"),
        "val_balanced_f1": result.get("val_balanced_f1"),
        "test_balanced_f1": result.get("test_balanced_f1"),
        "val_f1_class_0": result.get("val_f1_class_0"),
        "test_f1_class_0": result.get("test_f1_class_0"),
        "val_f1_class_1": result.get("val_f1_class_1"),
        "test_f1_class_1": result.get("test_f1_class_1"),
        "val_mcc": result["val_mcc"],
        "test_mcc": result["test_mcc"],
        "val_alerts_per_day": result.get("val_alerts_per_day"),
        "test_alerts_per_day": result.get("test_alerts_per_day"),
        "val_false_alarms_per_day": result.get("val_false_alarms_per_day"),
        "test_false_alarms_per_day": result.get("test_false_alarms_per_day"),
        "val_event_recall_approx": result.get("val_event_recall_approx"),
        "test_event_recall_approx": result.get("test_event_recall_approx"),
        "val_operational_cost": result.get("val_operational_cost"),
        "test_operational_cost": result.get("test_operational_cost"),
        "val_cost_per_day": result.get("val_cost_per_day"),
        "test_cost_per_day": result.get("test_cost_per_day"),
        "alerts_per_day_budget": result.get("alerts_per_day_budget"),
        "fn_cost": result.get("fn_cost"),
        "fp_cost": result.get("fp_cost"),
        "val_false_negatives": result.get("val_false_negatives"),
        "test_false_negatives": result.get("test_false_negatives"),
        "val_false_positives": result.get("val_false_positives"),
        "test_false_positives": result.get("test_false_positives"),
        "val_true_negatives": result.get("val_true_negatives"),
        "test_true_negatives": result.get("test_true_negatives"),
        "val_true_positives": result.get("val_true_positives"),
        "test_true_positives": result.get("test_true_positives"),
        "val_positive_support": result.get("val_positive_support"),
        "test_positive_support": result.get("test_positive_support"),
        "val_tp_capture": result.get("val_tp_capture"),
        "test_tp_capture": result.get("test_tp_capture"),
        "val_fn_rate": result.get("val_fn_rate"),
        "test_fn_rate": result.get("test_fn_rate"),
        "val_confusion_matrix": json.dumps(
            result.get("val_confusion_matrix"),
            ensure_ascii=True,
            default=_json_default,
        ),
        "test_confusion_matrix": json.dumps(
            result.get("test_confusion_matrix"),
            ensure_ascii=True,
            default=_json_default,
        ),
        "best_params": json.dumps(
            result["best_params"],
            ensure_ascii=True,
            default=_json_default,
        ),
        "effective_model_params": json.dumps(
            result.get("effective_model_params", result["best_params"]),
            ensure_ascii=True,
            default=_json_default,
        ),
        "smote_params": json.dumps(
            result["smote_params"],
            ensure_ascii=True,
            default=_json_default,
        ),
        "smote_optimo": bool(balance_mode == "smote"),
        "train_rows": int(result["train_rows"]),
        "val_rows": int(result["val_rows"]),
        "test_rows": int(result["test_rows"]),
        "optuna_trials_completed": int(result["optuna_trials_completed"]),
        "effective_optuna_n_jobs": int(result.get("optuna_n_jobs", optuna_n_jobs)),
        "effective_parallel_jobs": int(result.get("parallel_jobs", parallel_jobs)),
        "effective_xgb_parallel_jobs": int(
            result.get("xgb_parallel_jobs", xgb_parallel_jobs)
        ),
        "effective_threshold_n_jobs": int(result.get("threshold_n_jobs", 1)),
        "optuna_jobs_cpu_cap": int(result.get("optuna_jobs_cpu_cap", 0)),
        "cpu_count": int(result.get("cpu_count", os.cpu_count() or 1)),
    }


def _build_frozen_tuning_ablation_deltas(
    completed_grid_df: pd.DataFrame,
    *,
    run_id: str,
) -> pd.DataFrame:
    if not isinstance(completed_grid_df, pd.DataFrame) or completed_grid_df.empty:
        return pd.DataFrame()
    required_cols = {
        "model_name",
        "balance_mode",
        "threshold_protocol",
        "k",
        "params_source_feature_set",
        "target_feature_set",
    }
    if not required_cols.issubset(completed_grid_df.columns):
        return pd.DataFrame()

    source_df = completed_grid_df.copy()
    if "protocol_family" in source_df.columns:
        source_df = source_df[
            source_df["protocol_family"].astype(str)
            == FROZEN_TUNING_ABLATION_PROTOCOL_FAMILY
        ].copy()
    if source_df.empty:
        return pd.DataFrame()
    source_df["k"] = pd.to_numeric(source_df["k"], errors="coerce")
    source_df = source_df.dropna(subset=["k"]).copy()
    source_df["k"] = source_df["k"].astype(int)
    metric_cols = [
        "val_objective_score",
        "test_objective_score",
        "val_roc_auc",
        "test_roc_auc",
        "val_pr_auc",
        "test_pr_auc",
        "val_brier_score",
        "test_brier_score",
        "val_f1",
        "test_f1",
        "val_mcc",
        "test_mcc",
        "val_recall",
        "test_recall",
        "val_false_positives",
        "test_false_positives",
        "val_false_alarms_per_day",
        "test_false_alarms_per_day",
        "val_cost_per_day",
        "test_cost_per_day",
    ]
    for metric_col in metric_cols:
        if metric_col in source_df.columns:
            source_df[metric_col] = pd.to_numeric(
                source_df[metric_col],
                errors="coerce",
            )

    rows: List[Dict[str, object]] = []

    def _delta_row(
        *,
        effect_type: str,
        comparison: str,
        left: pd.Series,
        right: pd.Series,
    ) -> Dict[str, object]:
        row: Dict[str, object] = {
            "run_id": run_id,
            "effect_type": effect_type,
            "comparison": comparison,
            "model_name": right.get("model_name"),
            "balance_mode": right.get("balance_mode"),
            "threshold_protocol": right.get("threshold_protocol"),
            "threshold_objective": right.get("threshold_objective"),
            "calibration_method": right.get("calibration_method"),
            "objective_metric": right.get("objective_metric"),
            "k": int(right.get("k")),
            "params_source_feature_set": right.get("params_source_feature_set"),
            "target_feature_set": right.get("target_feature_set"),
            "baseline_combo_id": left.get("combo_id"),
            "comparison_combo_id": right.get("combo_id"),
        }
        for metric_col in metric_cols:
            if metric_col not in source_df.columns:
                continue
            row[f"baseline_{metric_col}"] = left.get(metric_col)
            row[f"comparison_{metric_col}"] = right.get(metric_col)
            row[f"delta_{metric_col}"] = right.get(metric_col) - left.get(metric_col)
        return row

    feature_group_cols = [
        "model_name",
        "balance_mode",
        "threshold_protocol",
        "k",
        "params_source_feature_set",
    ]
    for _, group in source_df.groupby(feature_group_cols, dropna=False):
        base_rows = group[group["target_feature_set"].astype(str) == "Base"]
        combined_rows = group[
            group["target_feature_set"].astype(str) == "Base + Cluster"
        ]
        if base_rows.empty or combined_rows.empty:
            continue
        rows.append(
            _delta_row(
                effect_type="feature_effect",
                comparison="target Base + Cluster - target Base",
                left=base_rows.iloc[0],
                right=combined_rows.iloc[0],
            )
        )

    tuning_group_cols = [
        "model_name",
        "balance_mode",
        "threshold_protocol",
        "k",
        "target_feature_set",
    ]
    for _, group in source_df.groupby(tuning_group_cols, dropna=False):
        base_rows = group[
            group["params_source_feature_set"].astype(str) == "Base"
        ]
        combined_rows = group[
            group["params_source_feature_set"].astype(str) == "Base + Cluster"
        ]
        if base_rows.empty or combined_rows.empty:
            continue
        rows.append(
            _delta_row(
                effect_type="tuning_effect",
                comparison="params Base + Cluster - params Base",
                left=base_rows.iloc[0],
                right=combined_rows.iloc[0],
            )
        )

    return pd.DataFrame(rows)


def _maybe_roc_auc(y_true: Sequence[object], scores: Sequence[object]) -> float:
    y_arr = pd.Series(y_true).astype(int)
    if y_arr.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_arr, np.asarray(scores, dtype=float)))
    except Exception:
        return float("nan")


def _maybe_pr_auc(y_true: Sequence[object], scores: Sequence[object]) -> float:
    y_arr = pd.Series(y_true).astype(int)
    if y_arr.nunique() < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_arr, np.asarray(scores, dtype=float)))
    except Exception:
        return float("nan")


def _normalize_controlled_objective_metric(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "auc": "roc_auc",
        "rocauc": "roc_auc",
        "roc_auc": "roc_auc",
        "roc-auc": "roc_auc",
        "pr_auc": "pr_auc",
        "pr-auc": "pr_auc",
        "average_precision": "pr_auc",
        "f1": "f1",
        "balanced_f1": "balanced_f1",
        "balanced f1": "balanced_f1",
        "macro_f1": "balanced_f1",
        "f1_global": "balanced_f1",
        "mcc": "mcc",
        "brier": "brier_score",
        "brier_score": "brier_score",
        "brier score": "brier_score",
        "recall_at_alerts_per_day": "recall_at_alerts_per_day",
        "recall@n": "recall_at_alerts_per_day",
        "operational_cost": "operational_cost",
        "cost": "operational_cost",
    }
    return aliases.get(text, "roc_auc")


def _normalize_controlled_threshold_protocols(
    threshold_protocols: Optional[Sequence[object]],
) -> List[str]:
    if threshold_protocols is None:
        return ["conservative"]
    normalized: List[str] = []
    for value in threshold_protocols:
        protocol = normalize_threshold_protocol(value)
        if protocol not in normalized:
            normalized.append(protocol)
    return normalized or ["conservative"]


def _controlled_objective_score_from_metrics(
    metrics: Dict[str, object],
    objective_metric: str,
) -> float:
    metric_key = _normalize_controlled_objective_metric(objective_metric)
    if metric_key == "operational_cost":
        return float(metrics.get("operational_cost", float("inf")))
    if metric_key == "brier_score":
        return float(metrics.get("brier_score", float("nan")))
    if metric_key == "balanced_f1":
        return float(
            metrics.get(
                "balanced_f1",
                metrics.get("f1_global", float("nan")),
            )
        )
    return float(metrics.get(metric_key, float("nan")))


def _resolve_controlled_models(
    selected_models: Optional[Sequence[object]] = None,
) -> List[str]:
    allowed = list(CONTROLLED_COMPARISON_MODELS)
    if selected_models is None:
        return allowed

    normalized: List[str] = []
    for model_name in selected_models:
        text = str(model_name or "").strip()
        if text in CONTROLLED_COMPARISON_MODELS and text not in normalized:
            normalized.append(text)
    if not normalized:
        raise ValueError("Debe seleccionar al menos un modelo para la comparación controlada.")
    return normalized


def _threshold_candidates(scores: Sequence[object]) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.5], dtype=float)
    candidates = np.unique(arr)
    if candidates.size > 256:
        quantiles = np.quantile(candidates, np.linspace(0.0, 1.0, 257))
        candidates = np.unique(np.round(quantiles, 10))
    if not np.any(np.isclose(candidates, 0.5)):
        candidates = np.append(candidates, 0.5)
    return np.sort(candidates.astype(float))


def _space_upper_bound(
    space: object,
    *,
    default: float,
    caster: Callable[[object], object],
) -> object:
    if isinstance(space, dict):
        for key in ("max", "upper", "high"):
            if key in space:
                try:
                    return caster(space[key])
                except Exception:
                    break
    if isinstance(space, (list, tuple, set)) and space:
        try:
            return caster(max(space))
        except Exception:
            pass
    return caster(default)


def _matrix_bytes(
    row_count: int,
    feature_count: int,
    *,
    dtype_bytes: int = 4,
) -> int:
    rows = max(0, int(row_count))
    cols = max(0, int(feature_count))
    return int(rows * cols * max(1, int(dtype_bytes)))


def _estimate_smote_resampled_rows(
    y_train: pd.Series,
    *,
    max_sampling_strategy: float,
) -> int:
    counts = pd.Series(y_train).astype(int).value_counts(dropna=False)
    if counts.empty or len(counts) < 2:
        return int(len(y_train))
    majority = int(counts.max())
    minority = int(counts.min())
    if majority <= 0:
        return int(len(y_train))
    min_ratio = float(minority) / float(majority)
    safe_ratio = max(min_ratio, min(1.0, float(max_sampling_strategy)))
    target_minority = max(minority, int(math.ceil(float(majority) * safe_ratio)))
    return int(majority + target_minority)


def _clamp_controlled_jobs(value: object, *, cpu_count: int) -> int:
    return max(1, min(int(value), max(1, int(cpu_count))))


def _controlled_model_trial_threads(
    model_name: str,
    *,
    parallel_jobs: int,
    xgb_parallel_jobs: int,
) -> int:
    if model_name in {"Random Forest", "Balanced Random Forest"}:
        return max(1, int(parallel_jobs))
    if model_name == "XGBoost":
        return max(1, int(xgb_parallel_jobs))
    return 1


def _controlled_cpu_limited_optuna_jobs(
    model_names: Sequence[str],
    *,
    parallel_jobs: int,
    xgb_parallel_jobs: int,
    cpu_count: int,
) -> int:
    if not model_names:
        return 0
    return max(1, int(cpu_count))


def _resolve_controlled_optimization_parallelism(
    *,
    model_name: str,
    requested_optuna_n_jobs: int,
    parallel_jobs: int,
    xgb_parallel_jobs: int,
    max_cpu_count: Optional[int] = None,
) -> Dict[str, int]:
    cpu_count = max(1, int(max_cpu_count or os.cpu_count() or 1))
    resolved_parallel_jobs = _clamp_controlled_jobs(parallel_jobs, cpu_count=cpu_count)
    resolved_xgb_parallel_jobs = _clamp_controlled_jobs(
        xgb_parallel_jobs,
        cpu_count=cpu_count,
    )
    trial_threads = _controlled_model_trial_threads(
        model_name,
        parallel_jobs=resolved_parallel_jobs,
        xgb_parallel_jobs=resolved_xgb_parallel_jobs,
    )
    cpu_limited_optuna_jobs = int(cpu_count)
    resolved_optuna_n_jobs = _clamp_controlled_jobs(
        requested_optuna_n_jobs,
        cpu_count=cpu_count,
    )
    return {
        "cpu_count": int(cpu_count),
        "parallel_jobs": int(resolved_parallel_jobs),
        "xgb_parallel_jobs": int(resolved_xgb_parallel_jobs),
        "trial_threads": int(trial_threads),
        "cpu_limited_optuna_jobs": int(cpu_limited_optuna_jobs),
        "optuna_n_jobs": int(resolved_optuna_n_jobs),
    }


def estimate_controlled_comparison_parallelism(
    base_df: pd.DataFrame,
    *,
    test_size: float,
    val_size: float,
    k_min: int,
    k_max: int,
    k_step: int,
    search_space_config: Optional[Dict[str, object]],
    memory_budget_bytes: int,
    xgb_parallel_jobs: int = 1,
    selected_models: Optional[Sequence[object]] = None,
    max_cpu_count: Optional[int] = None,
) -> Dict[str, object]:
    if base_df is None or base_df.empty:
        raise ValueError("El dataset base no puede estar vacio.")
    if "interval_start" not in base_df.columns:
        raise ValueError("El dataset debe incluir interval_start para estimar memoria.")
    if pd.Series(base_df["target"]).astype(int).nunique() < 2:
        raise ValueError("El dataset debe contener ambas clases.")
    if int(memory_budget_bytes) <= 0:
        raise ValueError("El presupuesto de memoria debe ser mayor que 0.")

    search_space = dict(search_space_config or {})
    all_numeric = _numeric_feature_cols(base_df)
    cluster_cols = _cluster_feature_cols(base_df)
    base_cols = [col for col in all_numeric if col not in cluster_cols]
    if not base_cols:
        raise ValueError("No hay variables base disponibles para estimar recursos.")
    if not cluster_cols:
        raise ValueError("No hay variables de cluster disponibles para estimar recursos.")
    resolved_models = _resolve_controlled_models(selected_models)

    feature_sets = {
        "Base": list(base_cols),
        "Cluster": list(cluster_cols),
        "Base + Cluster": list(all_numeric),
    }
    k_grid_by_set = {
        feature_set_name: _k_grid_values(
            k_min=int(k_min),
            k_max=int(k_max),
            k_step=int(k_step),
            feature_count=len(cols),
        )
        for feature_set_name, cols in feature_sets.items()
    }
    if any(not values for values in k_grid_by_set.values()):
        raise ValueError("La grilla de K no es valida para al menos un conjunto.")

    train_val_df, test_df = temporal_train_test_split(
        base_df, test_size=float(test_size)
    )
    train_df, val_df = temporal_train_test_split(
        train_val_df, test_size=float(val_size)
    )
    train_rows = int(len(train_df))
    val_rows = int(len(val_df))
    test_rows = int(len(test_df))

    ranking_feature_count = int(len(feature_sets["Base + Cluster"]))
    trial_feature_count = int(
        max(max(values) for values in k_grid_by_set.values())
    )
    ranking_n_estimators = int(max(200, min(1000, ranking_feature_count * 8)))
    max_rf_estimators = int(
        _space_upper_bound(
            dict(search_space.get("rf") or {}).get("n_estimators"),
            default=300,
            caster=int,
        )
    )
    max_smote_sampling = float(
        _space_upper_bound(
            dict(search_space.get("smote") or {}).get("sampling_strategy"),
            default=1.0,
            caster=float,
        )
    )
    smote_train_rows = _estimate_smote_resampled_rows(
        train_df["target"],
        max_sampling_strategy=max_smote_sampling,
    )

    ranking_matrix_bytes = _matrix_bytes(train_rows, ranking_feature_count)
    train_matrix_bytes = _matrix_bytes(smote_train_rows, trial_feature_count)
    val_matrix_bytes = _matrix_bytes(val_rows, trial_feature_count)
    test_matrix_bytes = _matrix_bytes(test_rows, trial_feature_count)

    ranking_serial_worker_bytes = int(
        (ranking_matrix_bytes * 2.25) + (ranking_n_estimators * 2 * _MEMORY_MB)
    )
    ranking_extra_thread_bytes = max(
        _CONTROLLED_MEMORY_PER_THREAD_MIN_BYTES,
        int(ranking_matrix_bytes * 0.15),
    )
    rf_serial_worker_bytes = int(
        (train_matrix_bytes * 2.40)
        + (val_matrix_bytes * 0.80)
        + (test_matrix_bytes * 0.25)
        + (max_rf_estimators * int(1.5 * _MEMORY_MB))
        + _CONTROLLED_MEMORY_RF_MODEL_OVERHEAD_BYTES
    )
    rf_extra_thread_bytes = max(
        _CONTROLLED_MEMORY_PER_THREAD_MIN_BYTES,
        int(train_matrix_bytes * 0.18),
    )
    svm_worker_bytes = int(
        (train_matrix_bytes * 3.0)
        + val_matrix_bytes
        + max(_CONTROLLED_MEMORY_SVM_CACHE_BYTES, int(train_matrix_bytes * 0.50))
        + (128 * _MEMORY_MB)
    )
    xgb_worker_bytes = int(
        (train_matrix_bytes * 4.0)
        + val_matrix_bytes
        + _CONTROLLED_MEMORY_XGB_MODEL_OVERHEAD_BYTES
    )
    xgb_extra_thread_bytes = max(
        _CONTROLLED_MEMORY_PER_THREAD_MIN_BYTES,
        int(train_matrix_bytes * 0.22),
    )
    nn_worker_bytes = int(
        (train_matrix_bytes * 2.0)
        + val_matrix_bytes
        + _CONTROLLED_MEMORY_NN_MODEL_OVERHEAD_BYTES
    )

    cpu_count = max(1, int(max_cpu_count or os.cpu_count() or 1))
    requested_xgb_parallel_jobs = _clamp_controlled_jobs(
        xgb_parallel_jobs,
        cpu_count=cpu_count,
    )
    budget_bytes = int(memory_budget_bytes)
    available_for_workers = max(
        0,
        budget_bytes - _CONTROLLED_MEMORY_PROCESS_OVERHEAD_BYTES,
    )

    xgb_candidates = (
        list(range(1, cpu_count + 1))
        if "XGBoost" in resolved_models
        else [int(requested_xgb_parallel_jobs)]
    )
    frontier_rows: List[Dict[str, object]] = []
    for parallel_jobs in range(1, cpu_count + 1):
        ranking_worker_bytes = int(
            ranking_serial_worker_bytes
            + max(0, parallel_jobs - 1) * ranking_extra_thread_bytes
        )
        ranking_peak_bytes = int(
            _CONTROLLED_MEMORY_PROCESS_OVERHEAD_BYTES + ranking_worker_bytes
        )
        rf_worker_bytes = int(
            rf_serial_worker_bytes
            + max(0, parallel_jobs - 1) * rf_extra_thread_bytes
        )
        for candidate_xgb_jobs in xgb_candidates:
            workers_by_model = {
                "Random Forest": rf_worker_bytes,
                "Balanced Random Forest": rf_worker_bytes,
                "SVM": int(svm_worker_bytes),
                "XGBoost": int(
                    xgb_worker_bytes
                    + max(0, candidate_xgb_jobs - 1) * xgb_extra_thread_bytes
                ),
                "Neural Network": int(nn_worker_bytes),
            }
            workers_by_model = {
                model_name: workers_by_model[model_name]
                for model_name in resolved_models
            }
            dominant_model = max(workers_by_model, key=workers_by_model.get)
            worst_worker_bytes = int(workers_by_model[dominant_model])
            memory_limited_optuna_jobs = 0
            if worst_worker_bytes > 0 and available_for_workers >= worst_worker_bytes:
                memory_limited_optuna_jobs = int(
                    min(cpu_count, available_for_workers // worst_worker_bytes)
                )
            cpu_limited_optuna_jobs = _controlled_cpu_limited_optuna_jobs(
                resolved_models,
                parallel_jobs=int(parallel_jobs),
                xgb_parallel_jobs=int(candidate_xgb_jobs),
                cpu_count=int(cpu_count),
            )
            max_optuna_jobs = int(
                min(memory_limited_optuna_jobs, cpu_limited_optuna_jobs)
            )
            optimization_peak_bytes = int(
                _CONTROLLED_MEMORY_PROCESS_OVERHEAD_BYTES
                + max(1, max_optuna_jobs) * worst_worker_bytes
            )
            trial_threads_by_model = {
                model_name: _controlled_model_trial_threads(
                    model_name,
                    parallel_jobs=int(parallel_jobs),
                    xgb_parallel_jobs=int(candidate_xgb_jobs),
                )
                for model_name in resolved_models
            }
            optimization_capacity_score = int(
                sum(
                    max_optuna_jobs * int(trial_threads_by_model[model_name])
                    for model_name in resolved_models
                )
            )
            frontier_rows.append(
                {
                    "parallel_jobs": int(parallel_jobs),
                    "xgb_parallel_jobs": int(candidate_xgb_jobs),
                    "max_optuna_jobs": int(max_optuna_jobs),
                    "memory_limited_optuna_jobs": int(memory_limited_optuna_jobs),
                    "cpu_limited_optuna_jobs": int(cpu_limited_optuna_jobs),
                    "dominant_model": str(dominant_model),
                    "ranking_peak_bytes": int(ranking_peak_bytes),
                    "worst_worker_bytes": int(worst_worker_bytes),
                    "optimization_peak_bytes": int(optimization_peak_bytes),
                    "ranking_usage_fraction": float(ranking_peak_bytes) / float(budget_bytes),
                    "optimization_usage_fraction": float(optimization_peak_bytes) / float(budget_bytes),
                    "fits_ranking_budget": bool(ranking_peak_bytes <= budget_bytes),
                    "fits_single_trial_budget": bool(
                        (_CONTROLLED_MEMORY_PROCESS_OVERHEAD_BYTES + worst_worker_bytes)
                        <= budget_bytes
                    ),
                    "fits_combined_budget": bool(
                        max_optuna_jobs >= 1
                        and ranking_peak_bytes <= budget_bytes
                        and optimization_peak_bytes <= budget_bytes
                    ),
                    "throughput_score": int(
                        int(parallel_jobs) + int(optimization_capacity_score)
                    ),
                }
            )

    frontier_df = pd.DataFrame(frontier_rows)
    safe_frontier_df = frontier_df.loc[
        frontier_df["fits_combined_budget"].astype(bool)
    ].copy()

    independent_parallel_rows = frontier_df.loc[
        frontier_df["fits_ranking_budget"].astype(bool)
        & frontier_df["fits_single_trial_budget"].astype(bool)
    ].copy()
    if "XGBoost" in resolved_models and not independent_parallel_rows.empty:
        independent_parallel_rows = independent_parallel_rows.loc[
            independent_parallel_rows["xgb_parallel_jobs"].astype(int) == 1
        ].copy()
    if independent_parallel_rows.empty:
        max_parallel_jobs_when_optuna_1 = 0
    else:
        max_parallel_jobs_when_optuna_1 = int(
            independent_parallel_rows["parallel_jobs"].max()
        )

    parallel_one_row = frontier_df.loc[
        frontier_df["parallel_jobs"].astype(int) == 1
    ].copy()
    if "XGBoost" in resolved_models and not parallel_one_row.empty:
        parallel_one_row = parallel_one_row.loc[
            parallel_one_row["xgb_parallel_jobs"].astype(int) == 1
        ].copy()
    if parallel_one_row.empty:
        max_optuna_jobs_when_parallel_1 = 0
    else:
        ranked_parallel_one = parallel_one_row.sort_values(
            ["max_optuna_jobs", "memory_limited_optuna_jobs"],
            ascending=[False, False],
        )
        row = ranked_parallel_one.iloc[0]
        if bool(row.get("fits_ranking_budget")):
            max_optuna_jobs_when_parallel_1 = int(row.get("max_optuna_jobs") or 0)
        else:
            max_optuna_jobs_when_parallel_1 = 0

    if "XGBoost" in resolved_models:
        xgb_one_rows = frontier_df.loc[
            (frontier_df["parallel_jobs"].astype(int) == 1)
            & frontier_df["fits_ranking_budget"].astype(bool)
            & frontier_df["fits_single_trial_budget"].astype(bool)
        ].copy()
        if xgb_one_rows.empty:
            max_xgb_parallel_jobs_when_parallel_1_optuna_1 = 0
        else:
            max_xgb_parallel_jobs_when_parallel_1_optuna_1 = int(
                xgb_one_rows["xgb_parallel_jobs"].max()
            )
    else:
        max_xgb_parallel_jobs_when_parallel_1_optuna_1 = 0

    recommended_pair = None
    if not safe_frontier_df.empty:
        ranked_frontier = safe_frontier_df.sort_values(
            [
                "throughput_score",
                "max_optuna_jobs",
                "xgb_parallel_jobs",
                "parallel_jobs",
            ],
            ascending=[False, False, False, True],
        )
        best_row = ranked_frontier.iloc[0]
        recommended_pair = {
            "parallel_jobs": int(best_row["parallel_jobs"]),
            "optuna_n_jobs": int(best_row["max_optuna_jobs"]),
            "xgb_parallel_jobs": int(best_row["xgb_parallel_jobs"]),
            "dominant_model": str(best_row["dominant_model"]),
            "estimated_peak_bytes": int(best_row["optimization_peak_bytes"]),
            "usage_fraction": float(best_row["optimization_usage_fraction"]),
            "memory_limited_optuna_jobs": int(
                best_row["memory_limited_optuna_jobs"]
            ),
            "cpu_limited_optuna_jobs": int(best_row["cpu_limited_optuna_jobs"]),
            "throughput_score": int(best_row["throughput_score"]),
        }

    return {
        "estimator_version": CONTROLLED_COMPARISON_MEMORY_ESTIMATOR_VERSION,
        "cpu_count": int(cpu_count),
        "memory_budget_bytes": int(budget_bytes),
        "xgb_parallel_jobs": int(requested_xgb_parallel_jobs),
        "requested_xgb_parallel_jobs": int(requested_xgb_parallel_jobs),
        "selected_models": list(resolved_models),
        "process_overhead_bytes": int(_CONTROLLED_MEMORY_PROCESS_OVERHEAD_BYTES),
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "test_rows": int(test_rows),
        "smote_train_rows_estimate": int(smote_train_rows),
        "feature_counts": {
            "Base": int(len(feature_sets["Base"])),
            "Cluster": int(len(feature_sets["Cluster"])),
            "Base + Cluster": int(len(feature_sets["Base + Cluster"])),
        },
        "k_grid_by_set": {key: list(values) for key, values in k_grid_by_set.items()},
        "trial_feature_count": int(trial_feature_count),
        "ranking_feature_count": int(ranking_feature_count),
        "max_parallel_jobs_when_optuna_1": int(max_parallel_jobs_when_optuna_1),
        "max_optuna_jobs_when_parallel_1": int(max_optuna_jobs_when_parallel_1),
        "max_xgb_parallel_jobs_when_parallel_1_optuna_1": int(
            max_xgb_parallel_jobs_when_parallel_1_optuna_1
        ),
        "recommended_pair": recommended_pair,
        "frontier_df": frontier_df,
        "safe_frontier_df": safe_frontier_df,
        "components": {
            "ranking_n_estimators": int(ranking_n_estimators),
            "max_rf_n_estimators": int(max_rf_estimators),
            "max_smote_sampling_strategy": float(max_smote_sampling),
            "ranking_matrix_bytes": int(ranking_matrix_bytes),
            "train_matrix_bytes": int(train_matrix_bytes),
            "val_matrix_bytes": int(val_matrix_bytes),
            "test_matrix_bytes": int(test_matrix_bytes),
            "ranking_serial_worker_bytes": int(ranking_serial_worker_bytes),
            "ranking_extra_thread_bytes": int(ranking_extra_thread_bytes),
            "rf_serial_worker_bytes": int(rf_serial_worker_bytes),
            "rf_extra_thread_bytes": int(rf_extra_thread_bytes),
            "svm_worker_bytes": int(svm_worker_bytes),
            "xgb_worker_bytes": int(xgb_worker_bytes),
            "xgb_extra_thread_bytes": int(xgb_extra_thread_bytes),
        },
    }


def _metric_at_threshold(
    y_true: Sequence[object],
    scores: Sequence[object],
    *,
    threshold: float,
    metric: str,
) -> float:
    y_arr = pd.Series(y_true).astype(int)
    scores_arr = np.asarray(scores, dtype=float)
    preds = (scores_arr >= float(threshold)).astype(int)
    metric_name = _normalize_controlled_objective_metric(metric)
    try:
        if metric_name == "f1":
            return float(f1_score(y_arr, preds, zero_division=0))
        if metric_name == "balanced_f1":
            return float(f1_score(y_arr, preds, average="macro", zero_division=0))
        if metric_name == "mcc":
            return float(matthews_corrcoef(y_arr, preds))
        if metric_name == "brier_score":
            scored = score_optuna_objective(
                y_arr.to_numpy(),
                scores_arr,
                objective_metric="brier_score",
                threshold=float(threshold),
            )
            return -float(scored.get("score", float("nan")))
        if metric_name == "recall_at_alerts_per_day":
            return float(recall_score(y_arr, preds, zero_division=0))
        if metric_name == "operational_cost":
            tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()
            return -float((10.0 * fn) + fp)
    except Exception:
        return float("nan")
    raise ValueError(f"Metrica no soportada: {metric}")


def _best_threshold_for_metric(
    y_true: Sequence[object],
    scores: Sequence[object],
    *,
    metric: str,
) -> Tuple[float, float]:
    metric_name = _normalize_controlled_objective_metric(metric)
    if metric_name == "roc_auc":
        return _maybe_roc_auc(y_true, scores), 0.5
    if metric_name == "pr_auc":
        return _maybe_pr_auc(y_true, scores), 0.5
    if metric_name == "brier_score":
        scored = score_optuna_objective(
            np.asarray(y_true).astype(int),
            np.asarray(scores, dtype=float),
            objective_metric="brier_score",
            threshold=0.5,
        )
        return -float(scored.get("score", float("nan"))), 0.5

    best_score = float("-inf")
    best_threshold = 0.5
    for threshold in _threshold_candidates(scores):
        score = _metric_at_threshold(
            y_true,
            scores,
            threshold=float(threshold),
            metric=metric_name,
        )
        if pd.isna(score):
            continue
        if score > best_score + 1e-12:
            best_score = float(score)
            best_threshold = float(threshold)
            continue
        if abs(score - best_score) <= 1e-12 and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
            best_threshold = float(threshold)

    if best_score == float("-inf"):
        return float("nan"), float(best_threshold)
    return float(best_score), float(best_threshold)


def _classification_metrics_from_scores(
    y_true: Sequence[object],
    scores: Sequence[object],
    *,
    threshold: float,
    eval_df: Optional[pd.DataFrame] = None,
    alerts_per_day: float = 5.0,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> Dict[str, object]:
    metrics = compute_extended_metrics(
        pd.Series(y_true).astype(int).to_numpy(),
        np.asarray(scores, dtype=float),
        threshold=float(threshold),
        eval_df=eval_df,
        alerts_per_day=float(alerts_per_day),
        fn_cost=float(fn_cost),
        fp_cost=float(fp_cost),
    )
    return dict(metrics)


def _orient_scores_for_metric(
    y_true: Sequence[object],
    scores: Sequence[object],
    *,
    metric: str,
) -> Tuple[np.ndarray, float, float]:
    raw_scores = np.asarray(scores, dtype=float)
    pos_score, _ = _best_threshold_for_metric(
        y_true,
        raw_scores,
        metric=metric,
    )
    neg_score, _ = _best_threshold_for_metric(
        y_true,
        -raw_scores,
        metric=metric,
    )
    if pd.isna(pos_score) and pd.isna(neg_score):
        return raw_scores, 1.0, float("nan")
    if pd.isna(pos_score):
        return -raw_scores, -1.0, float(neg_score)
    if pd.isna(neg_score):
        return raw_scores, 1.0, float(pos_score)
    if float(neg_score) > float(pos_score) + 1e-12:
        return -raw_scores, -1.0, float(neg_score)
    return raw_scores, 1.0, float(pos_score)


def _discrete_range_values(
    config: Optional[Dict[str, object]],
    *,
    default_min: float,
    default_max: float,
    default_step: float,
    caster: Callable[[float], object],
) -> List[object]:
    cfg = dict(config or {})
    if "choices" in cfg and isinstance(cfg.get("choices"), (list, tuple)):
        values = [value for value in cfg.get("choices") if value is not None]
        if caster is int:
            return sorted({int(pd.to_numeric(value, errors="coerce")) for value in values})
        return sorted(
            {
                round(float(pd.to_numeric(value, errors="coerce")), 10)
                for value in values
            }
        )

    min_value = float(cfg.get("min", default_min))
    max_value = float(cfg.get("max", default_max))
    step = float(cfg.get("step", default_step))
    if step <= 0:
        raise ValueError("El rango discreto requiere step > 0.")
    if min_value > max_value:
        raise ValueError("El rango discreto requiere min <= max.")

    values: List[object] = []
    current = float(min_value)
    guard = 0
    while current <= (max_value + (step / 1000.0)):
        if caster is int:
            values.append(int(round(current)))
        else:
            values.append(round(float(current), 10))
        current += step
        guard += 1
        if guard > 10000:
            raise ValueError("El rango discreto genero demasiados valores.")

    if not values:
        values = [caster(min_value)]
    if caster is int:
        deduped = sorted({int(value) for value in values})
    else:
        deduped = sorted({round(float(value), 10) for value in values})
    if caster is float and round(float(max_value), 10) not in deduped:
        deduped.append(round(float(max_value), 10))
        deduped = sorted(set(deduped))
    if caster is int and int(round(max_value)) not in deduped:
        deduped.append(int(round(max_value)))
        deduped = sorted(set(deduped))
    return list(deduped)


def _k_grid_values(
    *,
    k_min: int,
    k_max: int,
    k_step: int,
    feature_count: int,
) -> List[int]:
    if feature_count <= 0:
        return []
    resolved_min = max(1, min(int(k_min), int(feature_count)))
    resolved_max = max(1, min(int(k_max), int(feature_count)))
    resolved_step = max(1, int(k_step))
    if resolved_min > resolved_max:
        resolved_min = resolved_max
    values = list(range(resolved_min, resolved_max + 1, resolved_step))
    if not values:
        values = [resolved_max]
    if values[-1] != resolved_max:
        values.append(resolved_max)
    return sorted({int(value) for value in values if value > 0})


def controlled_comparison_checkpoint_root(
    checkpoint_root: Optional[Path] = None,
) -> Path:
    return Path(checkpoint_root) if checkpoint_root is not None else Path(
        "Resultados/controlled_comparison_runs"
    )


def calibration_experiment_checkpoint_root(
    checkpoint_root: Optional[Path] = None,
) -> Path:
    return Path(checkpoint_root) if checkpoint_root is not None else Path(
        "Resultados/calibration_experiment_runs"
    )


def _controlled_comparison_run_dir(run_id: str, *, checkpoint_root: Optional[Path] = None) -> Path:
    return controlled_comparison_checkpoint_root(checkpoint_root=checkpoint_root) / str(run_id)


def _calibration_experiment_run_dir(
    run_id: str,
    *,
    checkpoint_root: Optional[Path] = None,
) -> Path:
    return calibration_experiment_checkpoint_root(checkpoint_root=checkpoint_root) / str(run_id)


def _controlled_comparison_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "live_status": run_dir / "live_status.json",
        "live_events": run_dir / "live_events.jsonl",
        "protocol": run_dir / "protocol.json",
        "splits_duckdb": run_dir / "dataset" / "splits.duckdb",
        "splits_train_csv": run_dir / "dataset" / "train.csv",
        "splits_val_csv": run_dir / "dataset" / "val.csv",
        "splits_test_csv": run_dir / "dataset" / "test.csv",
        "ranking_base": run_dir / "rankings" / "base.csv",
        "ranking_cluster": run_dir / "rankings" / "cluster.csv",
        "ranking_base_cluster": run_dir / "rankings" / "base_cluster.csv",
        "ranking_global": run_dir / "rankings" / "global_feature_selection.csv",
        "trials_dir": run_dir / "trials",
        "grid_results": run_dir / "results" / "grid_results.csv",
        "best_summary": run_dir / "results" / "best_summary.csv",
        "curves": run_dir / "results" / "curves.csv",
        "ablation_deltas": run_dir / "results" / "ablation_deltas.csv",
    }


def _calibration_experiment_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "live_status": run_dir / "live_status.json",
        "live_events": run_dir / "live_events.jsonl",
        "protocol": run_dir / "protocol.json",
        "splits_duckdb": run_dir / "dataset" / "splits.duckdb",
        "splits_train_csv": run_dir / "dataset" / "train.csv",
        "splits_val_csv": run_dir / "dataset" / "val.csv",
        "splits_test_csv": run_dir / "dataset" / "test.csv",
        "trials_dir": run_dir / "trials",
        "grid_results": run_dir / "results" / "grid_results.csv",
        "leaderboard": run_dir / "results" / "leaderboard.csv",
        "pareto_front": run_dir / "results" / "pareto_front.csv",
        "best_summary": run_dir / "results" / "best_summary.csv",
        "best_summary_json": run_dir / "results" / "best_summary.json",
    }


def _load_manifest(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _refresh_manifest_progress(manifest: Dict[str, object]) -> None:
    steps_index = manifest.setdefault("steps_index", {})
    step_sequence = manifest.setdefault("step_sequence", list(steps_index.keys()))
    completed = 0
    current_step_id = None
    for step_id in step_sequence:
        entry = steps_index.get(step_id) or {}
        if str(entry.get("status") or "") == "completed":
            completed += 1
        elif current_step_id is None and str(entry.get("status") or "") == "running":
            current_step_id = step_id
    manifest["progress"] = {
        "completed_steps": int(completed),
        "total_steps": int(len(step_sequence)),
        "current_step_id": current_step_id,
    }


def _register_step(
    manifest: Dict[str, object],
    *,
    step_id: str,
    status: str,
    message: str,
    artifact_paths: Optional[Dict[str, object]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    steps_index = manifest.setdefault("steps_index", {})
    step_sequence = manifest.setdefault("step_sequence", [])
    if step_id not in step_sequence:
        step_sequence.append(step_id)
    entry = dict(steps_index.get(step_id) or {})
    entry["status"] = str(status)
    entry["message"] = str(message)
    entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if artifact_paths is not None:
        entry["artifact_paths"] = dict(artifact_paths)
    if metadata is not None:
        entry["metadata"] = dict(metadata)
    steps_index[step_id] = entry
    _refresh_manifest_progress(manifest)
    return entry


def _persist_live_event(
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    *,
    step_id: str,
    status: str,
    message: str,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": manifest.get("run_id"),
        "status": str(manifest.get("status") or ""),
        "result_status": str(manifest.get("result_status") or ""),
        "step_id": step_id,
        "step_status": status,
        "message": message,
        "progress": manifest.get("progress") or {},
    }
    if extra:
        event.update(extra)
    _atomic_write_json(paths["live_status"], event)
    _append_jsonl(paths["live_events"], event)


def _mark_step(
    paths: Dict[str, Path],
    manifest: Dict[str, object],
    *,
    step_id: str,
    status: str,
    message: str,
    artifact_paths: Optional[Dict[str, object]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    _register_step(
        manifest,
        step_id=step_id,
        status=status,
        message=message,
        artifact_paths=artifact_paths,
        metadata=metadata,
    )
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(paths["manifest"], manifest)
    _persist_live_event(
        paths,
        manifest,
        step_id=step_id,
        status=status,
        message=message,
        extra={"artifact_paths": artifact_paths or {}, "metadata": metadata or {}},
    )


def _write_checkpoint_frame(df: pd.DataFrame, path: Path, *, table_name: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".duckdb":
        if duckdb is None:
            raise RuntimeError("duckdb no esta instalado.")
        table = table_name or "data"
        con = duckdb.connect(str(path))
        try:
            con.register("df_view", df)
            con.execute(f'DROP TABLE IF EXISTS "{table}"')
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM df_view')
        finally:
            con.close()
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _read_checkpoint_frame(path: Path, *, table_name: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".duckdb":
        if duckdb is None:
            return pd.DataFrame()
        con = duckdb.connect(str(path), read_only=True)
        try:
            if table_name:
                return con.execute(f'SELECT * FROM "{table_name}"').df()
            tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
            if not tables:
                return pd.DataFrame()
            return con.execute(f'SELECT * FROM "{tables[0]}"').df()
        finally:
            con.close()
    return pd.read_csv(path)


def _quality_from_metric_series(
    series: pd.Series,
    *,
    direction: str,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    quality = pd.Series(np.nan, index=series.index, dtype=float)
    valid_mask = values.notna()
    if not valid_mask.any():
        return quality
    valid_values = values.loc[valid_mask]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    if math.isclose(min_value, max_value, rel_tol=0.0, abs_tol=1e-12):
        quality.loc[valid_mask] = 1.0
        return quality
    span = max_value - min_value
    if str(direction) == "minimize":
        quality.loc[valid_mask] = (max_value - valid_values) / span
    else:
        quality.loc[valid_mask] = (valid_values - min_value) / span
    return quality.clip(lower=0.0, upper=1.0)


def _geometric_mean_quality(components: pd.DataFrame) -> pd.Series:
    if not isinstance(components, pd.DataFrame) or components.empty:
        return pd.Series(dtype=float)
    result = pd.Series(np.nan, index=components.index, dtype=float)
    for idx, row in components.iterrows():
        values = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            continue
        clipped = np.clip(values, 0.0, 1.0)
        if np.any(clipped <= 0.0):
            result.loc[idx] = 0.0
            continue
        result.loc[idx] = float(np.exp(np.mean(np.log(clipped))))
    return result


def _row_dominates(
    frame: pd.DataFrame,
    left_idx: object,
    right_idx: object,
    *,
    objective_directions: Dict[str, str],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for column_name, direction in objective_directions.items():
        left_value = float(frame.at[left_idx, column_name])
        right_value = float(frame.at[right_idx, column_name])
        if str(direction) == "minimize":
            if left_value > right_value + 1e-12:
                better_or_equal = False
                break
            if left_value < right_value - 1e-12:
                strictly_better = True
        else:
            if left_value < right_value - 1e-12:
                better_or_equal = False
                break
            if left_value > right_value + 1e-12:
                strictly_better = True
    return bool(better_or_equal and strictly_better)


def _pareto_front_numbers(
    frame: pd.DataFrame,
    *,
    objective_directions: Dict[str, str],
    eligible_mask: pd.Series,
) -> pd.Series:
    fronts = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    remaining = list(frame.index[eligible_mask.fillna(False)])
    front_number = 1
    while remaining:
        current_front: List[object] = []
        current_front_set: set[object] = set()
        for candidate_idx in remaining:
            dominated = False
            for other_idx in remaining:
                if other_idx == candidate_idx:
                    continue
                if _row_dominates(
                    frame,
                    other_idx,
                    candidate_idx,
                    objective_directions=objective_directions,
                ):
                    dominated = True
                    break
            if not dominated:
                current_front.append(candidate_idx)
                current_front_set.add(candidate_idx)
        if not current_front:
            break
        fronts.loc[current_front] = int(front_number)
        remaining = [idx for idx in remaining if idx not in current_front_set]
        front_number += 1
    return fronts


def _build_calibration_sweep_leaderboard(
    grid_results_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(grid_results_df, pd.DataFrame) or grid_results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    leaderboard_df = grid_results_df.copy()
    if "status" in leaderboard_df.columns:
        leaderboard_df = leaderboard_df[
            leaderboard_df["status"].astype(str).str.lower() == "completed"
        ].copy()
    if leaderboard_df.empty:
        return leaderboard_df, pd.DataFrame()

    numeric_columns = [
        "val_mcc",
        "val_brier_score",
        "val_pr_auc",
        "val_true_positives",
        "val_false_negatives",
        "val_far",
        "test_mcc",
        "test_brier_score",
        "test_pr_auc",
        "test_true_positives",
        "test_false_negatives",
        "test_far",
        "val_positive_support",
        "test_positive_support",
        "val_tp_capture",
        "test_tp_capture",
        "val_fn_rate",
        "test_fn_rate",
    ]
    for column_name in numeric_columns:
        if column_name not in leaderboard_df.columns:
            leaderboard_df[column_name] = np.nan
    for column_name in numeric_columns:
        if column_name in leaderboard_df.columns:
            leaderboard_df[column_name] = pd.to_numeric(
                leaderboard_df[column_name],
                errors="coerce",
            )

    if "val_positive_support" not in leaderboard_df.columns:
        leaderboard_df["val_positive_support"] = (
            pd.to_numeric(leaderboard_df.get("val_true_positives"), errors="coerce")
            + pd.to_numeric(leaderboard_df.get("val_false_negatives"), errors="coerce")
        )
    if "test_positive_support" not in leaderboard_df.columns:
        leaderboard_df["test_positive_support"] = (
            pd.to_numeric(leaderboard_df.get("test_true_positives"), errors="coerce")
            + pd.to_numeric(leaderboard_df.get("test_false_negatives"), errors="coerce")
        )
    if "val_tp_capture" not in leaderboard_df.columns:
        leaderboard_df["val_tp_capture"] = np.where(
            leaderboard_df["val_positive_support"] > 0,
            pd.to_numeric(leaderboard_df.get("val_true_positives"), errors="coerce")
            / leaderboard_df["val_positive_support"],
            np.nan,
        )
    if "test_tp_capture" not in leaderboard_df.columns:
        leaderboard_df["test_tp_capture"] = np.where(
            leaderboard_df["test_positive_support"] > 0,
            pd.to_numeric(leaderboard_df.get("test_true_positives"), errors="coerce")
            / leaderboard_df["test_positive_support"],
            np.nan,
        )
    if "val_fn_rate" not in leaderboard_df.columns:
        leaderboard_df["val_fn_rate"] = np.where(
            leaderboard_df["val_positive_support"] > 0,
            pd.to_numeric(leaderboard_df.get("val_false_negatives"), errors="coerce")
            / leaderboard_df["val_positive_support"],
            np.nan,
        )
    if "test_fn_rate" not in leaderboard_df.columns:
        leaderboard_df["test_fn_rate"] = np.where(
            leaderboard_df["test_positive_support"] > 0,
            pd.to_numeric(leaderboard_df.get("test_false_negatives"), errors="coerce")
            / leaderboard_df["test_positive_support"],
            np.nan,
        )

    objective_columns = {
        "val_mcc": "maximize",
        "val_brier_score": "minimize",
        "val_pr_auc": "maximize",
        "val_true_positives": "maximize",
        "val_false_negatives": "minimize",
        "val_far": "minimize",
    }
    required_columns = list(objective_columns.keys())
    leaderboard_df["rankable"] = (
        leaderboard_df[required_columns]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
        & pd.to_numeric(
            leaderboard_df["val_positive_support"],
            errors="coerce",
        ).gt(0)
        & pd.to_numeric(
            leaderboard_df["val_true_positives"],
            errors="coerce",
        ).gt(0)
    )

    leaderboard_df["capture_quality"] = np.clip(
        pd.to_numeric(leaderboard_df["val_tp_capture"], errors="coerce"),
        0.0,
        1.0,
    )
    leaderboard_df["decision_quality"] = _quality_from_metric_series(
        leaderboard_df["val_mcc"],
        direction="maximize",
    )
    leaderboard_df["ranking_quality"] = _quality_from_metric_series(
        leaderboard_df["val_pr_auc"],
        direction="maximize",
    )
    leaderboard_df["calibration_quality"] = _quality_from_metric_series(
        leaderboard_df["val_brier_score"],
        direction="minimize",
    )
    leaderboard_df["false_alarm_quality"] = _quality_from_metric_series(
        leaderboard_df["val_far"],
        direction="minimize",
    )
    leaderboard_df["stability_score"] = _geometric_mean_quality(
        leaderboard_df[
            [
                "capture_quality",
                "calibration_quality",
                "ranking_quality",
                "decision_quality",
                "false_alarm_quality",
            ]
        ]
    )

    leaderboard_df["pareto_front"] = _pareto_front_numbers(
        leaderboard_df,
        objective_directions=objective_columns,
        eligible_mask=leaderboard_df["rankable"].astype(bool),
    )
    max_front = int(
        pd.to_numeric(leaderboard_df["pareto_front"], errors="coerce").max()
        if leaderboard_df["pareto_front"].notna().any()
        else 0
    )
    leaderboard_df["__rankable_sort"] = leaderboard_df["rankable"].astype(int)
    leaderboard_df["__pareto_sort"] = (
        pd.to_numeric(leaderboard_df["pareto_front"], errors="coerce")
        .fillna(max_front + 1)
        .astype(int)
    )
    leaderboard_df = leaderboard_df.sort_values(
        [
            "__rankable_sort",
            "__pareto_sort",
            "stability_score",
            "val_mcc",
            "val_pr_auc",
            "val_brier_score",
            "val_far",
            "balance_mode",
            "optuna_objective_metric",
            "calibration_method",
            "threshold_objective",
        ],
        ascending=[
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)
    leaderboard_df["rank"] = pd.Series(
        pd.NA, index=leaderboard_df.index, dtype="Int64"
    )
    rankable_idx = leaderboard_df.index[
        leaderboard_df["rankable"].astype(bool)
    ]
    leaderboard_df.loc[rankable_idx, "rank"] = pd.Series(
        range(1, len(rankable_idx) + 1),
        index=rankable_idx,
        dtype="Int64",
    )
    leaderboard_df = leaderboard_df.drop(
        columns=["__rankable_sort", "__pareto_sort"],
        errors="ignore",
    )

    pareto_front_df = leaderboard_df[
        leaderboard_df["rankable"].astype(bool)
        & pd.to_numeric(leaderboard_df["pareto_front"], errors="coerce").eq(1)
    ].copy()
    return leaderboard_df, pareto_front_df


def _build_calibration_sweep_best_summary(
    leaderboard_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    summary_payload: Dict[str, object] = {
        "global_best": None,
        "by_balance_mode": {},
    }
    if not isinstance(leaderboard_df, pd.DataFrame) or leaderboard_df.empty:
        return pd.DataFrame(), summary_payload

    rankable_df = leaderboard_df[
        leaderboard_df["rankable"].astype(bool)
    ].copy()
    candidate_df = rankable_df if not rankable_df.empty else leaderboard_df.copy()
    if candidate_df.empty:
        return pd.DataFrame(), summary_payload

    summary_columns = [
        "rank",
        "model_name",
        "balance_mode",
        "optuna_objective_metric",
        "objective_label",
        "objective_direction",
        "calibration_method",
        "threshold_objective",
        "threshold_objective_label",
        "threshold_protocol",
        "stability_score",
        "pareto_front",
        "rankable",
        "decision_threshold",
        "val_objective_score",
        "test_objective_score",
        "val_mcc",
        "test_mcc",
        "val_brier_score",
        "test_brier_score",
        "val_pr_auc",
        "test_pr_auc",
        "val_true_positives",
        "test_true_positives",
        "val_false_negatives",
        "test_false_negatives",
        "val_far",
        "test_far",
        "val_positive_support",
        "test_positive_support",
        "val_tp_capture",
        "test_tp_capture",
        "val_fn_rate",
        "test_fn_rate",
        "best_params",
        "effective_model_params",
        "smote_params",
        "selected_feature_count",
        "selected_features",
        "trials_csv",
    ]

    def _row_payload(scope: str, row: pd.Series) -> Dict[str, object]:
        payload = {
            "scope": str(scope),
            "rank_source": "validation",
        }
        for column_name in summary_columns:
            if column_name in row.index:
                payload[column_name] = row.get(column_name)
        return payload

    global_best = candidate_df.iloc[0]
    global_payload = _row_payload("global", global_best)
    summary_rows.append(global_payload)
    summary_payload["global_best"] = dict(global_payload)

    if "balance_mode" in candidate_df.columns:
        for balance_mode, mode_df in candidate_df.groupby(
            "balance_mode",
            dropna=False,
        ):
            if mode_df.empty:
                continue
            best_row = mode_df.iloc[0]
            scope = f"balance_mode:{balance_mode}"
            payload = _row_payload(scope, best_row)
            summary_rows.append(payload)
            summary_payload["by_balance_mode"][str(balance_mode)] = dict(payload)

    return pd.DataFrame(summary_rows), summary_payload


def build_controlled_comparison_context(
    *,
    event_path: Optional[object],
    features_path: Optional[object],
    segment_info: Optional[Dict[str, object]],
    protocol: Dict[str, object],
) -> Dict[str, object]:
    context = {
        "protocol_version": CONTROLLED_COMPARISON_PROTOCOL_VERSION,
        "protocol": dict(protocol),
        "event_fingerprint": _file_fingerprint(event_path),
        "features_fingerprint": _file_fingerprint(features_path),
        "segment_info": dict(segment_info or {}),
    }
    serialized = json.dumps(context, sort_keys=True, ensure_ascii=True, default=_json_default)
    context["computed_run_id"] = hashlib.md5(serialized.encode("utf-8")).hexdigest()
    return context


def _manifest_is_compatible(
    manifest: Optional[Dict[str, object]],
    context: Dict[str, object],
) -> Tuple[bool, str]:
    if not isinstance(manifest, dict):
        return False, "Manifest inexistente."
    if str(manifest.get("protocol_version") or "") != CONTROLLED_COMPARISON_PROTOCOL_VERSION:
        return False, "Version de protocolo incompatible."
    if str(manifest.get("computed_run_id") or "") != str(context.get("computed_run_id") or ""):
        return False, "computed_run_id incompatible."
    return True, ""


def preview_controlled_comparison_checkpoint(
    context: Dict[str, object],
    *,
    checkpoint_root: Optional[Path] = None,
) -> Dict[str, object]:
    root = controlled_comparison_checkpoint_root(checkpoint_root=checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    best_manifest = None
    best_run_dir = None
    best_updated_at = ""
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _load_manifest(manifest_path)
        compatible, reason = _manifest_is_compatible(manifest, context)
        if not compatible:
            continue
        updated_at = str((manifest or {}).get("updated_at") or "")
        if updated_at >= best_updated_at:
            best_manifest = manifest
            best_run_dir = manifest_path.parent
            best_updated_at = updated_at
    if best_manifest is None or best_run_dir is None:
        return {
            "checkpoint_available": False,
            "compatible": False,
            "can_resume": False,
            "can_load_completed": False,
            "run_id": None,
            "status": "missing",
            "updated_at": None,
            "manifest_path": str(root / "missing"),
            "run_dir": None,
            "current_step_id": None,
            "completed_steps": 0,
            "total_steps": 0,
            "incompatibility_reason": "No existe checkpoint compatible.",
        }
    progress = dict(best_manifest.get("progress") or {})
    status = str(best_manifest.get("status") or "running")
    return {
        "checkpoint_available": True,
        "compatible": True,
        "can_resume": status != "completed",
        "can_load_completed": status == "completed",
        "run_id": str(best_manifest.get("run_id") or best_run_dir.name),
        "status": status,
        "updated_at": best_manifest.get("updated_at"),
        "manifest_path": str(best_run_dir / "manifest.json"),
        "run_dir": str(best_run_dir),
        "current_step_id": progress.get("current_step_id"),
        "completed_steps": int(progress.get("completed_steps") or 0),
        "total_steps": int(progress.get("total_steps") or 0),
        "incompatibility_reason": "",
    }


class ExperimentsRunner:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def calculate_feature_importance(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Calculates feature importance using a Random Forest.
        Returns a DataFrame with 'variable' and 'importance' columns, sorted by importance.
        """
        X = df[feature_cols].fillna(0)
        y = df[target_col].astype(int)

        if y.nunique() < 2:
            raise ValueError("Target must have at least 2 classes for feature importance.")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=int(n_jobs),
        )
        model.fit(X, y)

        importance_df = pd.DataFrame(
            {
                "variable": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        return importance_df

    def run_optimization_loop(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        model_choice: str,
        n_trials: int,
        timeout: int,
        far_target: float,
        search_space_config: Dict[str, Dict[str, float]],
        optuna_n_jobs: int = 1,
        objective_key: str = "far_sens",
        objective_direction: str = "minimize",
        threshold_strategy: str = "optuna",
        calibration_method: str = "none",
        progress_callback: Optional[Callable[[str], None]] = None,
        optuna_callbacks: Optional[List[Callable]] = None,
        return_model: bool = False,
    ) -> Dict[str, object]:
        """
        Runs Optuna optimization for a specific set of features and model.
        """

        X_train = train_df[feature_cols].fillna(0).astype("float32")
        y_train = train_df["target"].astype(int)
        X_val = val_df[feature_cols].fillna(0).astype("float32")
        y_val = val_df["target"].astype(int)
        X_test = test_df[feature_cols].fillna(0).astype("float32")
        y_test = test_df["target"].astype(int)

        smote_cfg = search_space_config.get("smote", {})
        model_cfg = search_space_config.get("model", {})

        objective_metric_key = normalize_optuna_objective_metric(objective_key)
        objective_direction = str(objective_direction).lower()
        expected_direction = optuna_objective_direction(objective_metric_key)
        if objective_direction not in {"minimize", "maximize"}:
            objective_direction = expected_direction
        elif objective_direction != expected_direction:
            objective_direction = expected_direction
        optuna_n_jobs = max(1, int(optuna_n_jobs))
        threshold_strategy = str(threshold_strategy).lower()
        calibration_method = normalize_calibration_method(calibration_method)
        if threshold_strategy in {"far", "calibrate", "calibrar"}:
            threshold_strategy = "far"
        elif threshold_strategy not in {"optuna", "optimize", "optimizar"}:
            threshold_strategy = "optuna"

        def objective(trial: optuna.Trial) -> float:
            k_neighbors = trial.suggest_int(
                "smote_k_neighbors",
                int(smote_cfg.get("k_neighbors", {}).get("min", 1)),
                int(smote_cfg.get("k_neighbors", {}).get("max", 5)),
            )
            sampling_strategy = trial.suggest_float(
                "smote_sampling_strategy",
                float(smote_cfg.get("sampling_strategy", {}).get("min", 0.1)),
                float(smote_cfg.get("sampling_strategy", {}).get("max", 1.0)),
            )

            try:
                smote = SMOTE(
                    k_neighbors=k_neighbors,
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                )
                X_res, y_res = smote.fit_resample(X_train, y_train)
            except ValueError as exc:
                raise optuna.TrialPruned(f"SMOTE failed: {exc}")

            params: Dict[str, object] = {}
            if model_choice == "Random Forest":
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    int(model_cfg.get("n_estimators", {}).get("min", 50)),
                    int(model_cfg.get("n_estimators", {}).get("max", 200)),
                )
                d_min = int(model_cfg.get("max_depth", {}).get("min", 0))
                d_max = int(model_cfg.get("max_depth", {}).get("max", 20))
                if d_max > 0:
                    depth = trial.suggest_int("max_depth", d_min, d_max)
                    params["max_depth"] = depth if depth > 0 else None
                else:
                    params["max_depth"] = None
            elif model_choice == "XGBoost":
                params["n_estimators"] = trial.suggest_int(
                    "n_estimators",
                    int(model_cfg.get("n_estimators", {}).get("min", 50)),
                    int(model_cfg.get("n_estimators", {}).get("max", 200)),
                )
                params["max_depth"] = trial.suggest_int(
                    "max_depth",
                    int(model_cfg.get("max_depth", {}).get("min", 2)),
                    int(model_cfg.get("max_depth", {}).get("max", 10)),
                )
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate",
                    float(model_cfg.get("learning_rate", {}).get("min", 0.01)),
                    float(model_cfg.get("learning_rate", {}).get("max", 0.3)),
                )
                params["subsample"] = trial.suggest_float(
                    "subsample",
                    float(model_cfg.get("subsample", {}).get("min", 0.5)),
                    float(model_cfg.get("subsample", {}).get("max", 1.0)),
                )
                params["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree",
                    float(model_cfg.get("colsample_bytree", {}).get("min", 0.5)),
                    float(model_cfg.get("colsample_bytree", {}).get("max", 1.0)),
                )
            elif model_choice == "SVM":
                c_min = float(model_cfg.get("C", {}).get("min", 0.1))
                c_max = float(model_cfg.get("C", {}).get("max", 10.0))
                params["C"] = trial.suggest_float("C", c_min, c_max)
                params["kernel"] = str(model_cfg.get("kernel", "rbf"))

            try:
                model = build_model(model_choice, params, self.random_state)
                model.fit(X_res, y_res)
                raw_scores_val = get_model_scores(model, X_val)
                calibrator = fit_score_calibrator(
                    y_val.to_numpy(),
                    raw_scores_val,
                    method=calibration_method,
                )
                scores_val = calibrator.transform(raw_scores_val)
                if threshold_strategy == "far":
                    scored = score_optuna_objective(
                        y_val.to_numpy(),
                        scores_val,
                        objective_metric=objective_metric_key,
                        threshold_objective="far",
                        eval_df=val_df,
                        far_target=float(far_target),
                    )
                else:
                    threshold = trial.suggest_float("threshold", 0.0, 1.0)
                    scored = score_optuna_objective(
                        y_val.to_numpy(),
                        scores_val,
                        objective_metric=objective_metric_key,
                        threshold=threshold,
                        threshold_objective="far",
                        eval_df=val_df,
                        far_target=float(far_target),
                    )
                score = float(scored.get("score", float("nan")))
            except Exception as exc:
                raise optuna.TrialPruned(f"Training failed: {exc}")

            if pd.isna(score):
                raise optuna.TrialPruned(f"{objective_metric_key} invalido en validacion.")
            return float(score)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=objective_direction, sampler=sampler)

        if progress_callback:
            progress_callback(f"Starting Optuna for {len(feature_cols)} features...")

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=optuna_n_jobs,
            callbacks=optuna_callbacks,
        )

        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        if model_choice == "SVM":
            best_params["kernel"] = str(model_cfg.get("kernel", "rbf"))

        smote_best = SMOTE(
            k_neighbors=best_params.get("smote_k_neighbors", 5),
            sampling_strategy=best_params.get("smote_sampling_strategy", 1.0),
            random_state=self.random_state,
        )
        try:
            X_train_res, y_train_res = smote_best.fit_resample(X_train, y_train)
        except Exception:
            X_train_res, y_train_res = X_train, y_train

        model_params_final = {
            k: v
            for k, v in best_params.items()
            if not k.startswith("smote_") and k != "threshold"
        }
        final_model = build_model(model_choice, model_params_final, self.random_state)
        final_model.fit(X_train_res, y_train_res)

        raw_scores_val = get_model_scores(final_model, X_val)
        calibrator = fit_score_calibrator(
            y_val.to_numpy(),
            raw_scores_val,
            method=calibration_method,
        )
        scores_val = calibrator.transform(raw_scores_val)
        final_threshold = float(best_params.get("threshold", 0.5))
        if "threshold" not in best_params:
            final_scored = score_optuna_objective(
                y_val.to_numpy(),
                scores_val,
                objective_metric=objective_metric_key,
                threshold_objective="far",
                eval_df=val_df,
                far_target=float(far_target),
            )
            final_threshold = float(final_scored.get("threshold", 0.5))

        scores_test = calibrator.transform(get_model_scores(final_model, X_test))
        test_scored = score_optuna_objective(
            y_test.to_numpy(),
            scores_test,
            objective_metric=objective_metric_key,
            threshold=float(final_threshold),
            threshold_objective="far",
            eval_df=test_df,
            far_target=float(far_target),
        )
        test_metrics = dict(test_scored.get("metrics") or {})
        tn = int(test_metrics.get("true_negatives", 0))
        fp = int(test_metrics.get("false_positives", 0))
        fn = int(test_metrics.get("false_negatives", 0))
        tp = int(test_metrics.get("true_positives", 0))
        final_fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        result = {
            "n_features": len(feature_cols),
            "best_f1": float(test_metrics.get("f1", float("nan"))),
            "f1": float(test_metrics.get("f1", float("nan"))),
            "best_params": best_params,
            "feature_cols": feature_cols,
            "accuracy": float(test_metrics.get("accuracy", float("nan"))),
            "recall": float(test_metrics.get("recall", float("nan"))),
            "precision": float(test_metrics.get("precision", float("nan"))),
            "roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
            "pr_auc": float(test_metrics.get("pr_auc", float("nan"))),
            "balanced_f1": float(test_metrics.get("balanced_f1", float("nan"))),
            "mcc": float(test_metrics.get("mcc", float("nan"))),
            "brier_score": float(test_metrics.get("brier_score", float("nan"))),
            "fnr": final_fnr,
            "far": float(test_metrics.get("far", float("nan"))),
            "sensitivity": float(test_metrics.get("sensitivity", float("nan"))),
            "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
            "threshold": float(final_threshold),
            "threshold_strategy": threshold_strategy,
            "calibration_method": calibration_method,
            "objective_metric": objective_metric_key,
            "objective_label": test_scored.get("objective_label"),
            "objective_direction": objective_direction,
            "objective_score": float(test_scored.get("score", float("nan"))),
            "optuna_n_jobs": int(optuna_n_jobs),
            "dataset_rows": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
        }
        if return_model:
            result["model"] = final_model
        return result

    def run_iterative_experiment(
        self,
        base_df: pd.DataFrame,
        base_features_ordered: List[str],
        cluster_features: List[str],
        model_choice: str,
        n_trials: int,
        timeout: int,
        far_target: float,
        search_space_config: Dict[str, Dict[str, float]],
        optuna_n_jobs: int = 1,
        step_size: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.2,
        objective_key: str = "far_sens",
        objective_direction: str = "minimize",
        objective_label: Optional[str] = None,
        cluster_feature_names: Optional[List[str]] = None,
        threshold_strategy: str = "optuna",
        calibration_method: str = "none",
        progress_bar: Any = None,
        dataset_name: str = "unknown",
        features_name: str = "unknown",
        min_k: int = 5,
        max_k_limit: int = 1000,
        result_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> List[Dict[str, object]]:
        """
        Runs the iterative experiment:
        1. Base only (Flow) - Incremental K
        2. Base + Cluster - Incremental K

        Returns list of results dicts.
        """
        results: List[Dict[str, object]] = []

        train_df, test_df = temporal_train_test_split(base_df, test_size=test_size)
        train_full_df = train_df.copy()
        train_opt_df, val_opt_df = temporal_train_test_split(
            train_full_df, test_size=val_size / (1 - test_size)
        )

        combined_features_ordered = cluster_features
        base_limit = min(len(base_features_ordered), max_k_limit)
        combined_limit = min(len(combined_features_ordered), max_k_limit)
        use_combined = combined_limit > 0
        limit_total = combined_limit if use_combined else base_limit

        k_values = list(range(min_k, limit_total + 1, step_size))
        if not k_values:
            if limit_total >= 1:
                k_values = [limit_total]
            else:
                k_values = []

        cluster_set = set(cluster_feature_names or [])
        total_steps = max(1, len(k_values))

        if progress_bar:
            progress_bar.progress(0, text="Starting optimization loop...")

        for step_counter, k in enumerate(k_values, start=1):
            if progress_bar:
                progress_bar.progress(
                    int(step_counter / total_steps * 100),
                    text=f"Optimizing K={k}...",
                )

            base_k = k
            if use_combined and combined_features_ordered:
                top_combined = combined_features_ordered[:k]
                cluster_in_top = sum(1 for col in top_combined if col in cluster_set)
                base_k = k - cluster_in_top
            if base_k > 0:
                features_k = base_features_ordered[:base_k]
                try:
                    res = self.run_optimization_loop(
                        train_df=train_opt_df,
                        val_df=val_opt_df,
                        test_df=test_df,
                        feature_cols=features_k,
                        model_choice=model_choice,
                        n_trials=n_trials,
                        timeout=timeout,
                        optuna_n_jobs=optuna_n_jobs,
                        far_target=far_target,
                        search_space_config=search_space_config,
                        objective_key=objective_key,
                        objective_direction=objective_direction,
                        threshold_strategy=threshold_strategy,
                        calibration_method=calibration_method,
                    )
                    res["type"] = "Base"
                    res["k"] = k
                    res["dataset_name"] = dataset_name
                    res["features_name"] = features_name
                    res["objective_metric"] = objective_key
                    res["objective_direction"] = objective_direction
                    res["threshold_strategy"] = threshold_strategy
                    res["calibration_method"] = calibration_method
                    if objective_label:
                        res["objective_label"] = objective_label
                    results.append(res)
                    if result_callback:
                        result_callback(dict(res))
                except Exception as exc:
                    print(f"Error in Base K={k}: {exc}")
            else:
                res = {
                    "type": "Base",
                    "k": k,
                    "dataset_name": dataset_name,
                    "features_name": features_name,
                    "objective_metric": objective_key,
                    "objective_direction": objective_direction,
                    "threshold_strategy": threshold_strategy,
                    "calibration_method": calibration_method,
                    "optuna_n_jobs": int(optuna_n_jobs),
                    "error": "K total sin variables base disponibles.",
                }
                if objective_label:
                    res["objective_label"] = objective_label
                results.append(res)
                if result_callback:
                    result_callback(dict(res))

            if use_combined:
                features_k_comb = combined_features_ordered[:k]
                try:
                    res_c = self.run_optimization_loop(
                        train_df=train_opt_df,
                        val_df=val_opt_df,
                        test_df=test_df,
                        feature_cols=features_k_comb,
                        model_choice=model_choice,
                        n_trials=n_trials,
                        timeout=timeout,
                        optuna_n_jobs=optuna_n_jobs,
                        far_target=far_target,
                        search_space_config=search_space_config,
                        objective_key=objective_key,
                        objective_direction=objective_direction,
                        threshold_strategy=threshold_strategy,
                        calibration_method=calibration_method,
                    )
                    res_c["type"] = "Base+Cluster"
                    res_c["k"] = k
                    res_c["dataset_name"] = dataset_name
                    res_c["features_name"] = features_name
                    res_c["objective_metric"] = objective_key
                    res_c["objective_direction"] = objective_direction
                    res_c["threshold_strategy"] = threshold_strategy
                    res_c["calibration_method"] = calibration_method
                    if objective_label:
                        res_c["objective_label"] = objective_label
                    results.append(res_c)
                    if result_callback:
                        result_callback(dict(res_c))
                except Exception as exc:
                    print(f"Error in Combined K={k}: {exc}")

        return results

    def _controlled_comparison_feature_sets(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        all_numeric = _numeric_feature_cols(df)
        cluster_cols = _cluster_feature_cols(df)
        base_cols = [col for col in all_numeric if col not in cluster_cols]
        return {
            "Base": base_cols,
            "Cluster": cluster_cols,
            "Base + Cluster": all_numeric,
        }

    def _controlled_comparison_search_space(
        self,
        *,
        model_name: str,
        balance_mode: str,
        search_space_config: Dict[str, object],
        y_train: pd.Series,
    ) -> Dict[str, List[object]]:
        if model_name in {"Random Forest", "Balanced Random Forest"}:
            rf_cfg_key = "balanced_rf" if model_name == "Balanced Random Forest" else "rf"
            model_space = {
                "n_estimators": _discrete_range_values(
                    search_space_config.get(rf_cfg_key, {}).get(
                        "n_estimators",
                        search_space_config.get("rf", {}).get("n_estimators"),
                    ),
                    default_min=50,
                    default_max=300,
                    default_step=10,
                    caster=int,
                ),
                "max_depth": _discrete_range_values(
                    search_space_config.get(rf_cfg_key, {}).get(
                        "max_depth",
                        search_space_config.get("rf", {}).get("max_depth"),
                    ),
                    default_min=3,
                    default_max=15,
                    default_step=1,
                    caster=int,
                ),
                "min_samples_split": _discrete_range_values(
                    search_space_config.get(rf_cfg_key, {}).get(
                        "min_samples_split",
                        search_space_config.get("rf", {}).get("min_samples_split"),
                    ),
                    default_min=2,
                    default_max=10,
                    default_step=1,
                    caster=int,
                ),
                "min_samples_leaf": _discrete_range_values(
                    search_space_config.get(rf_cfg_key, {}).get(
                        "min_samples_leaf",
                        search_space_config.get("rf", {}).get("min_samples_leaf"),
                    ),
                    default_min=1,
                    default_max=5,
                    default_step=1,
                    caster=int,
                ),
                "max_features": list(
                    search_space_config.get(rf_cfg_key, {}).get(
                        "max_features",
                        search_space_config.get("rf", {}).get(
                            "max_features", ["sqrt", "log2", None]
                        ),
                    )
                ),
            }
            if model_name == "Random Forest":
                model_space["class_weight"] = list(
                    search_space_config.get("rf", {}).get(
                        "class_weight", [None, "balanced"]
                    )
                )
            else:
                model_space["replacement"] = list(
                    search_space_config.get("balanced_rf", {}).get(
                        "replacement", [False]
                    )
                )
        elif model_name == "SVM":
            model_space = {
                "C": _discrete_range_values(
                    search_space_config.get("svm", {}).get("C"),
                    default_min=0.1,
                    default_max=10.0,
                    default_step=0.5,
                    caster=float,
                ),
                "kernel": list(
                    search_space_config.get("svm", {}).get("kernel", ["rbf", "linear"])
                ),
                "gamma": list(
                    search_space_config.get("svm", {}).get("gamma", ["scale"])
                ),
                "degree": _discrete_range_values(
                    search_space_config.get("svm", {}).get("degree"),
                    default_min=2,
                    default_max=5,
                    default_step=1,
                    caster=int,
                ),
                "coef0": _discrete_range_values(
                    search_space_config.get("svm", {}).get("coef0"),
                    default_min=0.0,
                    default_max=1.0,
                    default_step=0.2,
                    caster=float,
                ),
                "class_weight": list(
                    search_space_config.get("svm", {}).get(
                        "class_weight", [None, "balanced"]
                    )
                ),
            }
        elif model_name == "XGBoost":
            class_counts = pd.Series(y_train).astype(int).value_counts()
            pos = int(class_counts.get(1, 0))
            neg = int(class_counts.get(0, 0))
            base_spw = float(neg / pos) if pos > 0 else 1.0
            spw_multipliers = list(
                search_space_config.get("xgb", {}).get(
                    "scale_pos_weight_multipliers", [0.5, 1.0, 2.0, 5.0, 10.0]
                )
            )
            model_space = {
                "n_estimators": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("n_estimators"),
                    default_min=50,
                    default_max=300,
                    default_step=10,
                    caster=int,
                ),
                "max_depth": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("max_depth"),
                    default_min=3,
                    default_max=15,
                    default_step=1,
                    caster=int,
                ),
                "learning_rate": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("learning_rate"),
                    default_min=0.01,
                    default_max=0.3,
                    default_step=0.01,
                    caster=float,
                ),
                "subsample": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("subsample"),
                    default_min=0.5,
                    default_max=1.0,
                    default_step=0.1,
                    caster=float,
                ),
                "colsample_bytree": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("colsample_bytree"),
                    default_min=0.5,
                    default_max=1.0,
                    default_step=0.1,
                    caster=float,
                ),
                "min_child_weight": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("min_child_weight"),
                    default_min=1,
                    default_max=10,
                    default_step=1,
                    caster=float,
                ),
                "reg_alpha": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("reg_alpha"),
                    default_min=0.0,
                    default_max=5.0,
                    default_step=0.1,
                    caster=float,
                ),
                "reg_lambda": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("reg_lambda"),
                    default_min=0.0,
                    default_max=10.0,
                    default_step=0.1,
                    caster=float,
                ),
                "gamma": _discrete_range_values(
                    search_space_config.get("xgb", {}).get("gamma"),
                    default_min=0.0,
                    default_max=5.0,
                    default_step=0.1,
                    caster=float,
                ),
                "scale_pos_weight": [
                    round(float(base_spw) * float(multiplier), 10)
                    for multiplier in spw_multipliers
                ],
                "max_delta_step": list(
                    search_space_config.get("xgb", {}).get(
                        "max_delta_step", [0.0, 1.0]
                    )
                ),
            }
        elif model_name == "Neural Network":
            class_counts = pd.Series(y_train).astype(int).value_counts()
            pos = int(class_counts.get(1, 0))
            neg = int(class_counts.get(0, 0))
            base_pw = float(neg / pos) if pos > 0 else 1.0
            pw_multipliers = list(
                search_space_config.get("nn", {}).get(
                    "pos_weight_multipliers", [0.5, 1.0, 2.0, 5.0, 10.0]
                )
            )
            model_space = {
                "hidden_dim": _discrete_range_values(
                    search_space_config.get("nn", {}).get("hidden_dim"),
                    default_min=64,
                    default_max=512,
                    default_step=64,
                    caster=int,
                ),
                "num_layers": _discrete_range_values(
                    search_space_config.get("nn", {}).get("num_layers"),
                    default_min=1,
                    default_max=4,
                    default_step=1,
                    caster=int,
                ),
                "dropout": _discrete_range_values(
                    search_space_config.get("nn", {}).get("dropout"),
                    default_min=0.0,
                    default_max=0.5,
                    default_step=0.1,
                    caster=float,
                ),
                "learning_rate": list(
                    search_space_config.get("nn", {}).get(
                        "learning_rate",
                        [0.0001, 0.0003, 0.001, 0.003, 0.01],
                    )
                ),
                "weight_decay": list(
                    search_space_config.get("nn", {}).get(
                        "weight_decay", [1e-6, 1e-5, 1e-4, 1e-3]
                    )
                ),
                "batch_size": list(
                    search_space_config.get("nn", {}).get(
                        "batch_size", [256, 512, 1024, 2048]
                    )
                ),
                # epochs no se optimiza: se fija a un maximo + early stopping
                # (patience=5) para que cada trial entrene hasta convergencia
                # sin sobreajuste. El maximo real se configura en Modelos.
                "pos_weight": [
                    round(float(base_pw) * float(m), 10)
                    for m in pw_multipliers
                ],
            }
        else:
            raise ValueError(f"Modelo no soportado: {model_name}")

        if balance_mode != "smote":
            return {"model": model_space, "smote": {}}

        class_counts = pd.Series(y_train).value_counts(dropna=False)
        min_class = int(class_counts.min()) if not class_counts.empty else 0
        if min_class < 2 or int(class_counts.nunique()) < 1:
            return {"model": model_space, "smote": {}}
        max_valid_k = max(1, min_class - 1)
        smote_k = [
            int(value)
            for value in _discrete_range_values(
                search_space_config.get("smote", {}).get("k_neighbors"),
                default_min=1,
                default_max=10,
                default_step=1,
                caster=int,
            )
            if int(value) <= max_valid_k
        ]
        class_ratio = float(min_class / int(class_counts.max())) if int(class_counts.max()) > 0 else 0.0
        sampling = [
            float(value)
            for value in _discrete_range_values(
                search_space_config.get("smote", {}).get("sampling_strategy"),
                default_min=max(class_ratio, 0.1),
                default_max=1.0,
                default_step=0.1,
                caster=float,
            )
            if float(value) + 1e-12 >= class_ratio
        ]
        return {
            "model": model_space,
            "smote": {
                "k_neighbors": smote_k,
                "sampling_strategy": sampling,
            },
        }

    def _controlled_comparison_trial_params(
        self,
        trial: optuna.Trial,
        *,
        model_name: str,
        model_space: Dict[str, List[object]],
        smote_space: Dict[str, List[object]],
        balance_mode: str,
        parallel_jobs: int,
        xgb_parallel_jobs: int,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        if balance_mode == "smote":
            smote_params = {
                "k_neighbors": trial.suggest_categorical(
                    "smote_k_neighbors", list(smote_space["k_neighbors"])
                ),
                "sampling_strategy": trial.suggest_categorical(
                    "smote_sampling_strategy", list(smote_space["sampling_strategy"])
                ),
            }
        else:
            smote_params = {}

        model_params: Dict[str, object] = {}
        if model_name == "SVM":
            kernels = list(model_space.get("kernel") or ["rbf"])
            model_params["kernel"] = trial.suggest_categorical(
                "kernel", kernels
            )
            c_values = list(model_space.get("C") or [1.0])
            model_params["C"] = trial.suggest_categorical("C", c_values)
            kernel_name = str(model_params["kernel"])
            if kernel_name in {"rbf", "poly", "sigmoid"}:
                gamma_values = list(model_space.get("gamma") or ["scale"])
                model_params["gamma"] = trial.suggest_categorical(
                    "gamma", gamma_values
                )
            if kernel_name == "poly":
                degree_values = list(model_space.get("degree") or [3])
                model_params["degree"] = trial.suggest_categorical(
                    "degree", degree_values
                )
            if kernel_name in {"poly", "sigmoid"}:
                coef0_values = list(model_space.get("coef0") or [0.0])
                model_params["coef0"] = trial.suggest_categorical(
                    "coef0", coef0_values
                )
            class_weight_values = list(model_space.get("class_weight") or [None])
            if len(class_weight_values) > 1:
                class_weight_value = trial.suggest_categorical(
                    "class_weight", class_weight_values
                )
            else:
                class_weight_value = class_weight_values[0]
            if class_weight_value is not None:
                model_params["class_weight"] = class_weight_value
            model_params["probability"] = False
            return model_params, smote_params

        for key, values in model_space.items():
            if not values:
                continue
            model_params[key] = trial.suggest_categorical(str(key), list(values))

        if model_name in {"Random Forest", "Balanced Random Forest"}:
            model_params["n_jobs"] = int(parallel_jobs)
            if model_params.get("max_depth") in {0, "0"}:
                model_params["max_depth"] = None
        elif model_name == "XGBoost":
            model_params["n_jobs"] = int(xgb_parallel_jobs)
        elif model_name == "Neural Network":
            # epochs no se optimiza: se fija a un maximo interno y el early
            # stopping (patience=5) decide cuando cortar cada trial.
            model_params.setdefault("epochs", 100)
            model_params.setdefault("early_stopping_patience", 5)
        return model_params, smote_params

    def _controlled_model_scores(
        self,
        model: object,
        X: pd.DataFrame,
        *,
        model_name: str,
        objective_metric: str,
        y_ref: Optional[Sequence[object]] = None,
        orientation: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        scores = np.asarray(get_model_scores(model, X), dtype=float)
        if model_name != "SVM":
            return scores, 1.0
        if orientation is not None:
            return scores * float(orientation), float(orientation)
        if y_ref is None:
            return scores, 1.0
        oriented_scores, score_orientation, _ = _orient_scores_for_metric(
            y_ref,
            scores,
            metric=objective_metric,
        )
        return oriented_scores, float(score_orientation)

    def _apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        smote_params: Dict[str, object],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if not smote_params:
            return X_train, y_train
        smote = SMOTE(
            k_neighbors=int(smote_params["k_neighbors"]),
            sampling_strategy=float(smote_params["sampling_strategy"]),
            random_state=self.random_state,
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return (
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res, name=y_train.name),
        )

    def _optimize_controlled_combo(
        self,
        *,
        model_name: str,
        feature_set: str,
        balance_mode: str,
        objective_metric: str,
        threshold_protocol: str = "conservative",
        threshold_objective: Optional[str] = None,
        calibration_method: str = "none",
        far_target: float = 0.20,
        alerts_per_day: float = 5.0,
        fn_cost: float = 10.0,
        fp_cost: float = 1.0,
        robust_folds: int = 3,
        selected_features: List[str],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_trials: int,
        timeout: int,
        optuna_n_jobs: int,
        search_space_config: Dict[str, object],
        parallel_jobs: int,
        xgb_parallel_jobs: int = 1,
    ) -> Dict[str, object]:
        X_train = train_df[selected_features].fillna(0).astype("float32")
        y_train = train_df["target"].astype(int)
        X_val = val_df[selected_features].fillna(0).astype("float32")
        y_val = val_df["target"].astype(int)
        X_test = test_df[selected_features].fillna(0).astype("float32")
        y_test = test_df["target"].astype(int)

        if y_train.nunique() < 2:
            raise ValueError("Solo existe una clase en train.")
        if y_val.nunique() < 2:
            raise ValueError("Solo existe una clase en val.")
        if y_test.nunique() < 2:
            raise ValueError("Solo existe una clase en test.")

        objective_metric = _normalize_controlled_objective_metric(objective_metric)
        threshold_protocol = normalize_threshold_protocol(threshold_protocol)
        threshold_objective = normalize_threshold_objective(
            threshold_objective or objective_metric
        )
        calibration_method = normalize_calibration_method(calibration_method)
        objective_label = CONTROLLED_COMPARISON_OBJECTIVE_LABELS.get(
            objective_metric, str(objective_metric).upper()
        )
        objective_direction = optuna_objective_direction(objective_metric)
        threshold_objective_label = THRESHOLD_OBJECTIVE_LABELS.get(
            threshold_objective, str(threshold_objective).upper()
        )
        effective_parallelism = _resolve_controlled_optimization_parallelism(
            model_name=model_name,
            requested_optuna_n_jobs=int(optuna_n_jobs),
            parallel_jobs=int(parallel_jobs),
            xgb_parallel_jobs=int(xgb_parallel_jobs),
        )
        resolved_parallel_jobs = int(effective_parallelism["parallel_jobs"])
        resolved_xgb_parallel_jobs = int(effective_parallelism["xgb_parallel_jobs"])
        effective_optuna_n_jobs = int(effective_parallelism["optuna_n_jobs"])
        optuna_jobs_cpu_cap = int(effective_parallelism["cpu_limited_optuna_jobs"])
        threshold_parallel_jobs = int(effective_parallelism["trial_threads"])
        cpu_count = int(effective_parallelism["cpu_count"])

        search_space = self._controlled_comparison_search_space(
            model_name=model_name,
            balance_mode=balance_mode,
            search_space_config=search_space_config,
            y_train=y_train,
        )
        model_space = dict(search_space.get("model") or {})
        smote_space = dict(search_space.get("smote") or {})
        if balance_mode == "smote" and (
            not smote_space.get("k_neighbors") or not smote_space.get("sampling_strategy")
        ):
            raise ValueError("SMOTE no es valido para el split actual.")

        sampler_kwargs: Dict[str, object] = {"seed": self.random_state}
        if int(effective_optuna_n_jobs) > 1:
            sampler_kwargs["constant_liar"] = True
        try:
            sampler = optuna.samplers.TPESampler(**sampler_kwargs)
        except TypeError:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction=objective_direction,
            sampler=sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            model_params, smote_params = self._controlled_comparison_trial_params(
                trial,
                model_name=model_name,
                model_space=model_space,
                smote_space=smote_space,
                balance_mode=balance_mode,
                parallel_jobs=resolved_parallel_jobs,
                xgb_parallel_jobs=resolved_xgb_parallel_jobs,
            )
            try:
                X_fit, y_fit = self._apply_smote(
                    X_train,
                    y_train,
                    smote_params=smote_params,
                )
                model = build_model(model_name, model_params, self.random_state)
                model.fit(X_fit, y_fit)
                scores_val, _ = self._controlled_model_scores(
                    model,
                    X_val,
                    model_name=model_name,
                    objective_metric=objective_metric,
                    y_ref=y_val,
                )
                calibrator = fit_score_calibrator(
                    y_val.to_numpy(),
                    scores_val,
                    method=calibration_method,
                )
                scores_val = calibrator.transform(scores_val)
                scored = score_optuna_objective(
                    y_val.to_numpy(),
                    scores_val,
                    objective_metric=objective_metric,
                    threshold_objective=threshold_objective,
                    eval_df=val_df,
                    far_target=float(far_target),
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                    threshold_n_jobs=int(threshold_parallel_jobs),
                )
                score = float(scored.get("score", float("nan")))
            except Exception as exc:
                raise optuna.TrialPruned(str(exc)) from exc
            if pd.isna(score):
                raise optuna.TrialPruned(
                    f"{objective_label} invalido en validacion."
                )
            return float(score)

        study.optimize(
            objective,
            n_trials=max(1, int(n_trials)),
            timeout=max(1, int(timeout)),
            n_jobs=max(1, int(effective_optuna_n_jobs)),
        )

        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
        ]
        if not completed_trials:
            raise ValueError("Optuna no genero trials completos.")
        if objective_direction == "minimize":
            best_trial = min(completed_trials, key=lambda trial: float(trial.value))
        else:
            best_trial = max(completed_trials, key=lambda trial: float(trial.value))
        best_raw_params = dict(best_trial.params)
        best_model_params = {
            key: value
            for key, value in best_raw_params.items()
            if not str(key).startswith("smote_")
        }
        best_smote_params = {}
        if balance_mode == "smote":
            best_smote_params = {
                "k_neighbors": int(best_raw_params["smote_k_neighbors"]),
                "sampling_strategy": float(best_raw_params["smote_sampling_strategy"]),
            }
        effective_model_params = dict(best_model_params)
        if model_name in {"Random Forest", "Balanced Random Forest"}:
            if effective_model_params.get("max_depth") in {0, "0"}:
                effective_model_params["max_depth"] = None
            if best_model_params.get("max_depth") in {0, "0"}:
                best_model_params["max_depth"] = None
            effective_model_params["n_jobs"] = int(resolved_parallel_jobs)
        elif model_name == "XGBoost":
            effective_model_params["n_jobs"] = int(resolved_xgb_parallel_jobs)
        elif model_name == "SVM":
            effective_model_params["probability"] = False

        protocol_result = train_model_with_protocol(
            train_df,
            val_df,
            test_df,
            selected_features,
            model_name,
            effective_model_params,
            threshold_protocol=threshold_protocol,
            threshold_objective=threshold_objective,
            calibration_method=calibration_method,
            far_target=float(far_target),
            alerts_per_day=float(alerts_per_day),
            fn_cost=float(fn_cost),
            fp_cost=float(fp_cost),
            robust_folds=int(robust_folds),
            balance_strategy="smote" if balance_mode == "smote" else "none",
            smote_params=best_smote_params,
            threshold_n_jobs=int(threshold_parallel_jobs),
            random_state=self.random_state,
        )
        val_metrics = dict(protocol_result.get("validation_metrics") or {})
        test_metrics = dict(protocol_result.get("metrics") or {})
        decision_threshold = float(test_metrics.get("threshold", 0.5))
        val_objective_score = _controlled_objective_score_from_metrics(
            val_metrics,
            objective_metric,
        )
        test_objective_score = _controlled_objective_score_from_metrics(
            test_metrics,
            objective_metric,
        )

        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        if not trials_df.empty:
            trials_df = trials_df.sort_values(
                "value",
                ascending=objective_direction == "minimize",
            ).reset_index(drop=True)

        return {
            "status": "completed",
            "model_name": model_name,
            "feature_set": feature_set,
            "balance_mode": balance_mode,
            "threshold_protocol": threshold_protocol,
            "threshold_protocol_label": THRESHOLD_PROTOCOL_LABELS.get(
                threshold_protocol, threshold_protocol
            ),
            "threshold_objective": threshold_objective,
            "threshold_objective_label": threshold_objective_label,
            "calibration_method": calibration_method,
            "objective_metric": objective_metric,
            "objective_label": objective_label,
            "objective_direction": objective_direction,
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": float(decision_threshold),
            "val_objective_score": float(val_objective_score),
            "test_objective_score": float(test_objective_score),
            "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
            "test_accuracy": float(test_metrics.get("accuracy", float("nan"))),
            "val_recall": float(val_metrics.get("recall", float("nan"))),
            "test_recall": float(test_metrics.get("recall", float("nan"))),
            "val_sensitivity": float(val_metrics.get("sensitivity", float("nan"))),
            "test_sensitivity": float(test_metrics.get("sensitivity", float("nan"))),
            "val_roc_auc": float(val_metrics.get("roc_auc", float("nan"))),
            "test_roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
            "val_pr_auc": float(val_metrics.get("pr_auc", float("nan"))),
            "test_pr_auc": float(test_metrics.get("pr_auc", float("nan"))),
            "val_brier_score": float(val_metrics.get("brier_score", float("nan"))),
            "test_brier_score": float(test_metrics.get("brier_score", float("nan"))),
            "val_f1": float(val_metrics.get("f1", float("nan"))),
            "test_f1": float(test_metrics.get("f1", float("nan"))),
            "val_f1_global": float(val_metrics.get("f1_global", float("nan"))),
            "test_f1_global": float(test_metrics.get("f1_global", float("nan"))),
            "val_balanced_f1": float(
                val_metrics.get(
                    "balanced_f1",
                    val_metrics.get("f1_global", float("nan")),
                )
            ),
            "test_balanced_f1": float(
                test_metrics.get(
                    "balanced_f1",
                    test_metrics.get("f1_global", float("nan")),
                )
            ),
            "val_f1_class_0": float(val_metrics.get("f1_class_0", float("nan"))),
            "test_f1_class_0": float(test_metrics.get("f1_class_0", float("nan"))),
            "val_f1_class_1": float(val_metrics.get("f1_class_1", float("nan"))),
            "test_f1_class_1": float(test_metrics.get("f1_class_1", float("nan"))),
            "val_mcc": float(val_metrics.get("mcc", float("nan"))),
            "test_mcc": float(test_metrics.get("mcc", float("nan"))),
            "val_alerts_per_day": float(val_metrics.get("alerts_per_day", float("nan"))),
            "test_alerts_per_day": float(test_metrics.get("alerts_per_day", float("nan"))),
            "val_false_alarms_per_day": float(val_metrics.get("false_alarms_per_day", float("nan"))),
            "test_false_alarms_per_day": float(test_metrics.get("false_alarms_per_day", float("nan"))),
            "val_event_recall_approx": float(val_metrics.get("event_recall_approx", float("nan"))),
            "test_event_recall_approx": float(test_metrics.get("event_recall_approx", float("nan"))),
            "val_operational_cost": float(val_metrics.get("operational_cost", float("nan"))),
            "test_operational_cost": float(test_metrics.get("operational_cost", float("nan"))),
            "val_cost_per_day": float(val_metrics.get("cost_per_day", float("nan"))),
            "test_cost_per_day": float(test_metrics.get("cost_per_day", float("nan"))),
            "alerts_per_day_budget": float(alerts_per_day),
            "fn_cost": float(fn_cost),
            "fp_cost": float(fp_cost),
            "val_false_negatives": int(val_metrics.get("false_negatives", 0)),
            "test_false_negatives": int(test_metrics.get("false_negatives", 0)),
            "val_false_positives": int(val_metrics.get("false_positives", 0)),
            "test_false_positives": int(test_metrics.get("false_positives", 0)),
            "val_true_negatives": int(val_metrics.get("true_negatives", 0)),
            "test_true_negatives": int(test_metrics.get("true_negatives", 0)),
            "val_true_positives": int(val_metrics.get("true_positives", 0)),
            "test_true_positives": int(test_metrics.get("true_positives", 0)),
            "val_positive_support": int(val_metrics.get("positive_support", 0)),
            "test_positive_support": int(test_metrics.get("positive_support", 0)),
            "val_tp_capture": float(val_metrics.get("tp_capture", float("nan"))),
            "test_tp_capture": float(test_metrics.get("tp_capture", float("nan"))),
            "val_fn_rate": float(val_metrics.get("fn_rate", float("nan"))),
            "test_fn_rate": float(test_metrics.get("fn_rate", float("nan"))),
            "val_confusion_matrix": val_metrics.get("confusion_matrix"),
            "test_confusion_matrix": test_metrics.get("confusion_matrix"),
            "best_params": best_model_params,
            "effective_model_params": effective_model_params,
            "smote_params": best_smote_params,
            "optuna_trials_completed": int(len(completed_trials)),
            "optuna_n_jobs": int(effective_optuna_n_jobs),
            "parallel_jobs": int(resolved_parallel_jobs),
            "xgb_parallel_jobs": int(resolved_xgb_parallel_jobs),
            "threshold_n_jobs": int(threshold_parallel_jobs),
            "requested_optuna_n_jobs": int(optuna_n_jobs),
            "requested_parallel_jobs": int(parallel_jobs),
            "requested_xgb_parallel_jobs": int(xgb_parallel_jobs),
            "optuna_jobs_cpu_cap": int(optuna_jobs_cpu_cap),
            "cpu_count": int(cpu_count),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": trials_df,
        }

    def _evaluate_controlled_combo_with_frozen_params(
        self,
        *,
        model_name: str,
        feature_set: str,
        balance_mode: str,
        objective_metric: str,
        threshold_protocol: str = "conservative",
        threshold_objective: Optional[str] = None,
        calibration_method: str = "none",
        far_target: float = 0.20,
        alerts_per_day: float = 5.0,
        fn_cost: float = 10.0,
        fp_cost: float = 1.0,
        robust_folds: int = 3,
        selected_features: List[str],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        frozen_model_params: Optional[Dict[str, object]],
        frozen_smote_params: Optional[Dict[str, object]],
        optuna_n_jobs: int,
        parallel_jobs: int,
        xgb_parallel_jobs: int = 1,
    ) -> Dict[str, object]:
        if train_df["target"].astype(int).nunique() < 2:
            raise ValueError("Solo existe una clase en train.")
        if val_df["target"].astype(int).nunique() < 2:
            raise ValueError("Solo existe una clase en val.")
        if test_df["target"].astype(int).nunique() < 2:
            raise ValueError("Solo existe una clase en test.")

        objective_metric = _normalize_controlled_objective_metric(objective_metric)
        threshold_protocol = normalize_threshold_protocol(threshold_protocol)
        threshold_objective = normalize_threshold_objective(
            threshold_objective or objective_metric
        )
        calibration_method = normalize_calibration_method(calibration_method)
        objective_label = CONTROLLED_COMPARISON_OBJECTIVE_LABELS.get(
            objective_metric, str(objective_metric).upper()
        )
        objective_direction = optuna_objective_direction(objective_metric)
        threshold_objective_label = THRESHOLD_OBJECTIVE_LABELS.get(
            threshold_objective, str(threshold_objective).upper()
        )
        effective_parallelism = _resolve_controlled_optimization_parallelism(
            model_name=model_name,
            requested_optuna_n_jobs=int(optuna_n_jobs),
            parallel_jobs=int(parallel_jobs),
            xgb_parallel_jobs=int(xgb_parallel_jobs),
        )
        resolved_parallel_jobs = int(effective_parallelism["parallel_jobs"])
        resolved_xgb_parallel_jobs = int(effective_parallelism["xgb_parallel_jobs"])
        effective_optuna_n_jobs = int(effective_parallelism["optuna_n_jobs"])
        optuna_jobs_cpu_cap = int(effective_parallelism["cpu_limited_optuna_jobs"])
        threshold_parallel_jobs = int(effective_parallelism["trial_threads"])
        cpu_count = int(effective_parallelism["cpu_count"])

        best_model_params = dict(frozen_model_params or {})
        best_smote_params = (
            dict(frozen_smote_params or {}) if balance_mode == "smote" else {}
        )
        effective_model_params = dict(best_model_params)
        if model_name in {"Random Forest", "Balanced Random Forest"}:
            if effective_model_params.get("max_depth") in {0, "0"}:
                effective_model_params["max_depth"] = None
            if best_model_params.get("max_depth") in {0, "0"}:
                best_model_params["max_depth"] = None
            effective_model_params["n_jobs"] = int(resolved_parallel_jobs)
        elif model_name == "XGBoost":
            effective_model_params["n_jobs"] = int(resolved_xgb_parallel_jobs)
        elif model_name == "SVM":
            effective_model_params["probability"] = False

        protocol_result = train_model_with_protocol(
            train_df,
            val_df,
            test_df,
            selected_features,
            model_name,
            effective_model_params,
            threshold_protocol=threshold_protocol,
            threshold_objective=threshold_objective,
            calibration_method=calibration_method,
            far_target=float(far_target),
            alerts_per_day=float(alerts_per_day),
            fn_cost=float(fn_cost),
            fp_cost=float(fp_cost),
            robust_folds=int(robust_folds),
            balance_strategy="smote" if balance_mode == "smote" else "none",
            smote_params=best_smote_params,
            threshold_n_jobs=int(threshold_parallel_jobs),
            random_state=self.random_state,
        )
        val_metrics = dict(protocol_result.get("validation_metrics") or {})
        test_metrics = dict(protocol_result.get("metrics") or {})
        decision_threshold = float(test_metrics.get("threshold", 0.5))
        val_objective_score = _controlled_objective_score_from_metrics(
            val_metrics,
            objective_metric,
        )
        test_objective_score = _controlled_objective_score_from_metrics(
            test_metrics,
            objective_metric,
        )

        return {
            "status": "completed",
            "model_name": model_name,
            "feature_set": feature_set,
            "balance_mode": balance_mode,
            "threshold_protocol": threshold_protocol,
            "threshold_protocol_label": THRESHOLD_PROTOCOL_LABELS.get(
                threshold_protocol, threshold_protocol
            ),
            "threshold_objective": threshold_objective,
            "threshold_objective_label": threshold_objective_label,
            "calibration_method": calibration_method,
            "objective_metric": objective_metric,
            "objective_label": objective_label,
            "objective_direction": objective_direction,
            "k": int(len(selected_features)),
            "selected_features": list(selected_features),
            "selected_feature_count": int(len(selected_features)),
            "decision_threshold": float(decision_threshold),
            "val_objective_score": float(val_objective_score),
            "test_objective_score": float(test_objective_score),
            "val_accuracy": float(val_metrics.get("accuracy", float("nan"))),
            "test_accuracy": float(test_metrics.get("accuracy", float("nan"))),
            "val_recall": float(val_metrics.get("recall", float("nan"))),
            "test_recall": float(test_metrics.get("recall", float("nan"))),
            "val_sensitivity": float(val_metrics.get("sensitivity", float("nan"))),
            "test_sensitivity": float(test_metrics.get("sensitivity", float("nan"))),
            "val_roc_auc": float(val_metrics.get("roc_auc", float("nan"))),
            "test_roc_auc": float(test_metrics.get("roc_auc", float("nan"))),
            "val_pr_auc": float(val_metrics.get("pr_auc", float("nan"))),
            "test_pr_auc": float(test_metrics.get("pr_auc", float("nan"))),
            "val_brier_score": float(val_metrics.get("brier_score", float("nan"))),
            "test_brier_score": float(test_metrics.get("brier_score", float("nan"))),
            "val_f1": float(val_metrics.get("f1", float("nan"))),
            "test_f1": float(test_metrics.get("f1", float("nan"))),
            "val_f1_global": float(val_metrics.get("f1_global", float("nan"))),
            "test_f1_global": float(test_metrics.get("f1_global", float("nan"))),
            "val_balanced_f1": float(
                val_metrics.get(
                    "balanced_f1",
                    val_metrics.get("f1_global", float("nan")),
                )
            ),
            "test_balanced_f1": float(
                test_metrics.get(
                    "balanced_f1",
                    test_metrics.get("f1_global", float("nan")),
                )
            ),
            "val_f1_class_0": float(val_metrics.get("f1_class_0", float("nan"))),
            "test_f1_class_0": float(test_metrics.get("f1_class_0", float("nan"))),
            "val_f1_class_1": float(val_metrics.get("f1_class_1", float("nan"))),
            "test_f1_class_1": float(test_metrics.get("f1_class_1", float("nan"))),
            "val_mcc": float(val_metrics.get("mcc", float("nan"))),
            "test_mcc": float(test_metrics.get("mcc", float("nan"))),
            "val_alerts_per_day": float(val_metrics.get("alerts_per_day", float("nan"))),
            "test_alerts_per_day": float(test_metrics.get("alerts_per_day", float("nan"))),
            "val_false_alarms_per_day": float(val_metrics.get("false_alarms_per_day", float("nan"))),
            "test_false_alarms_per_day": float(test_metrics.get("false_alarms_per_day", float("nan"))),
            "val_event_recall_approx": float(val_metrics.get("event_recall_approx", float("nan"))),
            "test_event_recall_approx": float(test_metrics.get("event_recall_approx", float("nan"))),
            "val_operational_cost": float(val_metrics.get("operational_cost", float("nan"))),
            "test_operational_cost": float(test_metrics.get("operational_cost", float("nan"))),
            "val_cost_per_day": float(val_metrics.get("cost_per_day", float("nan"))),
            "test_cost_per_day": float(test_metrics.get("cost_per_day", float("nan"))),
            "alerts_per_day_budget": float(alerts_per_day),
            "fn_cost": float(fn_cost),
            "fp_cost": float(fp_cost),
            "val_false_negatives": int(val_metrics.get("false_negatives", 0)),
            "test_false_negatives": int(test_metrics.get("false_negatives", 0)),
            "val_false_positives": int(val_metrics.get("false_positives", 0)),
            "test_false_positives": int(test_metrics.get("false_positives", 0)),
            "val_true_negatives": int(val_metrics.get("true_negatives", 0)),
            "test_true_negatives": int(test_metrics.get("true_negatives", 0)),
            "val_true_positives": int(val_metrics.get("true_positives", 0)),
            "test_true_positives": int(test_metrics.get("true_positives", 0)),
            "val_positive_support": int(val_metrics.get("positive_support", 0)),
            "test_positive_support": int(test_metrics.get("positive_support", 0)),
            "val_tp_capture": float(val_metrics.get("tp_capture", float("nan"))),
            "test_tp_capture": float(test_metrics.get("tp_capture", float("nan"))),
            "val_fn_rate": float(val_metrics.get("fn_rate", float("nan"))),
            "test_fn_rate": float(test_metrics.get("fn_rate", float("nan"))),
            "val_confusion_matrix": val_metrics.get("confusion_matrix"),
            "test_confusion_matrix": test_metrics.get("confusion_matrix"),
            "best_params": best_model_params,
            "effective_model_params": effective_model_params,
            "smote_params": best_smote_params,
            "optuna_trials_completed": 0,
            "optuna_n_jobs": int(effective_optuna_n_jobs),
            "parallel_jobs": int(resolved_parallel_jobs),
            "xgb_parallel_jobs": int(resolved_xgb_parallel_jobs),
            "threshold_n_jobs": int(threshold_parallel_jobs),
            "requested_optuna_n_jobs": int(optuna_n_jobs),
            "requested_parallel_jobs": int(parallel_jobs),
            "requested_xgb_parallel_jobs": int(xgb_parallel_jobs),
            "optuna_jobs_cpu_cap": int(optuna_jobs_cpu_cap),
            "cpu_count": int(cpu_count),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "trials_df": pd.DataFrame(),
        }

    def run_calibration_sweep(
        self,
        base_df: pd.DataFrame,
        *,
        model_name: str,
        selected_features: Optional[Sequence[object]] = None,
        objective_metrics: Optional[Sequence[object]] = None,
        calibration_methods: Optional[Sequence[object]] = None,
        threshold_objectives: Optional[Sequence[object]] = None,
        event_path: Optional[object] = None,
        features_path: Optional[object] = None,
        feature_source: str = "feature_selection",
        test_size: float = 0.2,
        val_size: float = 0.2,
        n_trials: int = 25,
        timeout: int = 1800,
        optuna_n_jobs: int = 1,
        parallel_jobs: int = 1,
        xgb_parallel_jobs: int = 1,
        search_space_config: Optional[Dict[str, object]] = None,
        far_target: float = 0.20,
        alerts_per_day: float = 5.0,
        fn_cost: float = 10.0,
        fp_cost: float = 1.0,
        robust_folds: int = 3,
        checkpoint_root: Optional[Path] = None,
        result_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        if base_df is None or base_df.empty:
            raise ValueError("El dataset base no puede estar vacío.")
        if "interval_start" not in base_df.columns:
            raise ValueError("El dataset debe incluir interval_start para split temporal.")
        if pd.Series(base_df["target"]).astype(int).nunique() < 2:
            raise ValueError("El target debe contener ambas clases.")

        resolved_model_name = _resolve_controlled_models([model_name])[0]
        numeric_feature_cols = _numeric_feature_cols(base_df)
        if not numeric_feature_cols:
            raise ValueError("No hay variables numéricas disponibles para el experimento.")

        if selected_features is None:
            feature_cols = list(numeric_feature_cols)
        else:
            feature_cols = [
                str(feature)
                for feature in selected_features
                if str(feature) in numeric_feature_cols
            ]
        if not feature_cols:
            raise ValueError("No hay variables efectivas para ejecutar el experimento.")

        resolved_objective_metrics: List[str] = []
        for metric in list(
            objective_metrics
            or [
                "pr_auc",
                "mcc",
                "brier_score",
                "balanced_f1",
                "recall_at_alerts_per_day",
                "operational_cost",
                "far_sens",
            ]
        ):
            normalized_metric = _normalize_controlled_objective_metric(metric)
            if normalized_metric not in resolved_objective_metrics:
                resolved_objective_metrics.append(normalized_metric)
        resolved_calibration_methods: List[str] = []
        for method in list(calibration_methods or ["sigmoid", "isotonic", "none"]):
            normalized_method = normalize_calibration_method(method)
            if normalized_method not in resolved_calibration_methods:
                resolved_calibration_methods.append(normalized_method)
        resolved_threshold_objectives: List[str] = []
        for objective in list(
            threshold_objectives or list(CALIBRATION_SWEEP_THRESHOLD_OBJECTIVES)
        ):
            normalized_objective = normalize_threshold_objective(objective)
            if (
                normalized_objective in CALIBRATION_SWEEP_THRESHOLD_OBJECTIVES
                and normalized_objective not in resolved_threshold_objectives
            ):
                resolved_threshold_objectives.append(normalized_objective)
        if not resolved_objective_metrics:
            raise ValueError("Debe seleccionar al menos una métrica objetivo de Optuna.")
        if not resolved_calibration_methods:
            raise ValueError("Debe seleccionar al menos un método de calibración.")
        if not resolved_threshold_objectives:
            raise ValueError("Debe seleccionar al menos un objetivo de threshold.")

        search_space = dict(search_space_config or {})

        train_val_df, test_df = temporal_train_test_split(
            base_df,
            test_size=float(test_size),
        )
        train_df, val_df = temporal_train_test_split(
            train_val_df,
            test_size=float(val_size),
        )
        if train_df["target"].astype(int).nunique() < 2:
            raise ValueError("El split temporal dejó una sola clase en train.")
        if val_df["target"].astype(int).nunique() < 2:
            raise ValueError("El split temporal dejó una sola clase en validación.")
        if test_df["target"].astype(int).nunique() < 2:
            raise ValueError("El split temporal dejó una sola clase en test.")

        protocol = {
            "protocol_family": CALIBRATION_SWEEP_PROTOCOL_FAMILY,
            "protocol_version": CALIBRATION_SWEEP_PROTOCOL_VERSION,
            "model_name": resolved_model_name,
            "threshold_protocol": "robust",
            "feature_source": str(feature_source),
            "selected_features": list(feature_cols),
            "selected_feature_count": int(len(feature_cols)),
            "objective_metrics": list(resolved_objective_metrics),
            "calibration_methods": list(resolved_calibration_methods),
            "threshold_objectives": list(resolved_threshold_objectives),
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
        }
        context = {
            "protocol_version": CALIBRATION_SWEEP_PROTOCOL_VERSION,
            "protocol": dict(protocol),
            "event_fingerprint": _file_fingerprint(event_path),
            "features_fingerprint": _file_fingerprint(features_path),
        }
        serialized_context = json.dumps(
            context,
            sort_keys=True,
            ensure_ascii=True,
            default=_json_default,
        )
        computed_run_id = hashlib.md5(
            serialized_context.encode("utf-8")
        ).hexdigest()
        run_id = (
            "calibration_sweep_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{computed_run_id[:8]}"
        )
        paths = _calibration_experiment_paths(
            _calibration_experiment_run_dir(
                run_id,
                checkpoint_root=checkpoint_root,
            )
        )

        combos = list(
            itertools.product(
                resolved_objective_metrics,
                resolved_calibration_methods,
                resolved_threshold_objectives,
                CALIBRATION_SWEEP_BALANCE_MODES,
            )
        )

        manifest: Dict[str, object] = {
            "run_id": run_id,
            "computed_run_id": computed_run_id,
            "protocol_version": CALIBRATION_SWEEP_PROTOCOL_VERSION,
            "protocol_family": CALIBRATION_SWEEP_PROTOCOL_FAMILY,
            "protocol": dict(protocol),
            "status": "running",
            "result_status": "running",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": {},
            "steps_index": {},
            "step_sequence": [],
            "last_error": None,
        }
        _register_step(
            manifest,
            step_id="split_freeze",
            status="pending",
            message="Congelar split temporal.",
        )
        for objective_metric, calibration_method, threshold_objective, balance_mode in combos:
            combo_step_id = (
                "combo__"
                f"{_slugify(objective_metric)}__{_slugify(calibration_method)}__"
                f"{_slugify(threshold_objective)}__{_slugify(balance_mode)}"
            )
            _register_step(
                manifest,
                step_id=combo_step_id,
                status="pending",
                message=(
                    f"{objective_metric} | {calibration_method} | "
                    f"{threshold_objective} | {balance_mode}"
                ),
            )
        _register_step(
            manifest,
            step_id="leaderboard",
            status="pending",
            message="Construir leaderboard y resúmenes.",
        )
        _refresh_manifest_progress(manifest)
        _atomic_write_json(paths["manifest"], manifest)
        _atomic_write_json(paths["protocol"], protocol)
        _persist_live_event(
            paths,
            manifest,
            step_id="init",
            status="running",
            message="Experimento de calibración iniciado.",
        )

        split_artifacts = self._persist_controlled_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            paths=paths,
        )
        _mark_step(
            paths,
            manifest,
            step_id="split_freeze",
            status="completed",
            message="Split temporal congelado.",
            artifact_paths=split_artifacts,
            metadata={
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
            },
        )

        grid_records: List[Dict[str, object]] = []
        for combo_index, (
            objective_metric,
            calibration_method,
            threshold_objective,
            balance_mode,
        ) in enumerate(combos, start=1):
            combo_step_id = (
                "combo__"
                f"{_slugify(objective_metric)}__{_slugify(calibration_method)}__"
                f"{_slugify(threshold_objective)}__{_slugify(balance_mode)}"
            )
            payload: Dict[str, object] = {
                "experiment": "Calibration sweep",
                "protocol_family": CALIBRATION_SWEEP_PROTOCOL_FAMILY,
                "run_id": run_id,
                "computed_run_id": computed_run_id,
                "combo_id": combo_step_id,
                "model_name": resolved_model_name,
                "feature_source": str(feature_source),
                "selected_feature_count": int(len(feature_cols)),
                "selected_features": json.dumps(
                    list(feature_cols),
                    ensure_ascii=True,
                ),
                "balance_mode": str(balance_mode),
                "threshold_protocol": "robust",
                "threshold_protocol_label": THRESHOLD_PROTOCOL_LABELS.get(
                    "robust",
                    "robust",
                ),
                "threshold_objective": str(threshold_objective),
                "threshold_objective_label": THRESHOLD_OBJECTIVE_LABELS.get(
                    threshold_objective,
                    str(threshold_objective).upper(),
                ),
                "calibration_method": str(calibration_method),
                "objective_metric": str(objective_metric),
                "optuna_objective_metric": str(objective_metric),
                "objective_label": CONTROLLED_COMPARISON_OBJECTIVE_LABELS.get(
                    objective_metric,
                    str(objective_metric).upper(),
                ),
                "objective_direction": optuna_objective_direction(objective_metric),
                "status": "pending",
                "error": None,
                "event_path": str(event_path or ""),
                "features_path": str(features_path or ""),
            }
            _mark_step(
                paths,
                manifest,
                step_id=combo_step_id,
                status="running",
                message=(
                    f"Evaluando {combo_index}/{len(combos)}: "
                    f"{objective_metric} | {calibration_method} | "
                    f"{threshold_objective} | {balance_mode}"
                ),
                metadata={
                    "model_name": resolved_model_name,
                    "objective_metric": objective_metric,
                    "calibration_method": calibration_method,
                    "threshold_objective": threshold_objective,
                    "balance_mode": balance_mode,
                },
            )
            try:
                result = self._optimize_controlled_combo(
                    model_name=resolved_model_name,
                    feature_set="Frozen selection",
                    balance_mode=str(balance_mode),
                    objective_metric=str(objective_metric),
                    threshold_protocol="robust",
                    threshold_objective=str(threshold_objective),
                    calibration_method=str(calibration_method),
                    far_target=float(far_target),
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                    robust_folds=int(robust_folds),
                    selected_features=list(feature_cols),
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    n_trials=int(n_trials),
                    timeout=int(timeout),
                    optuna_n_jobs=int(optuna_n_jobs),
                    search_space_config=dict(search_space),
                    parallel_jobs=int(parallel_jobs),
                    xgb_parallel_jobs=int(xgb_parallel_jobs),
                )
                payload.update(
                    _controlled_payload_updates_from_result(
                        result,
                        threshold_protocol="robust",
                        threshold_objective=str(threshold_objective),
                        threshold_objective_label=THRESHOLD_OBJECTIVE_LABELS.get(
                            threshold_objective,
                            str(threshold_objective).upper(),
                        ),
                        calibration_method=str(calibration_method),
                        k_global=None,
                        effective_k=int(len(feature_cols)),
                        balance_mode=str(balance_mode),
                        optuna_n_jobs=int(optuna_n_jobs),
                        parallel_jobs=int(parallel_jobs),
                        xgb_parallel_jobs=int(xgb_parallel_jobs),
                    )
                )
                trials_path = paths["trials_dir"] / f"{combo_step_id}.csv"
                if (
                    isinstance(result.get("trials_df"), pd.DataFrame)
                    and not result["trials_df"].empty
                ):
                    _write_checkpoint_frame(result["trials_df"], trials_path)
                    payload["trials_csv"] = str(trials_path)
                else:
                    payload["trials_csv"] = None
                _mark_step(
                    paths,
                    manifest,
                    step_id=combo_step_id,
                    status="completed",
                    message="Combinación evaluada.",
                    artifact_paths={
                        "trials_csv": str(trials_path),
                    }
                    if payload.get("trials_csv")
                    else {},
                    metadata={
                        "objective_metric": objective_metric,
                        "calibration_method": calibration_method,
                        "threshold_objective": threshold_objective,
                        "balance_mode": balance_mode,
                        "val_objective_score": payload.get("val_objective_score"),
                    },
                )
            except Exception as exc:
                payload["status"] = "failed"
                payload["error"] = str(exc)
                _mark_step(
                    paths,
                    manifest,
                    step_id=combo_step_id,
                    status="failed",
                    message=f"Combinación falló: {exc}",
                    metadata={
                        "objective_metric": objective_metric,
                        "calibration_method": calibration_method,
                        "threshold_objective": threshold_objective,
                        "balance_mode": balance_mode,
                    },
                )
            grid_records.append(dict(payload))
            grid_results_df = pd.DataFrame(grid_records)
            _write_checkpoint_frame(grid_results_df, paths["grid_results"])
            if result_callback:
                result_callback(dict(payload))

        _mark_step(
            paths,
            manifest,
            step_id="leaderboard",
            status="running",
            message="Construyendo leaderboard robusto.",
        )
        grid_results_df = _read_checkpoint_frame(paths["grid_results"])
        leaderboard_df, pareto_front_df = _build_calibration_sweep_leaderboard(
            grid_results_df
        )
        best_summary_df, best_summary_payload = _build_calibration_sweep_best_summary(
            leaderboard_df
        )
        _write_checkpoint_frame(leaderboard_df, paths["leaderboard"])
        _write_checkpoint_frame(pareto_front_df, paths["pareto_front"])
        _write_checkpoint_frame(best_summary_df, paths["best_summary"])
        _atomic_write_json(paths["best_summary_json"], best_summary_payload)
        _mark_step(
            paths,
            manifest,
            step_id="leaderboard",
            status="completed",
            message="Leaderboard persistido.",
            artifact_paths={
                "leaderboard": str(paths["leaderboard"]),
                "pareto_front": str(paths["pareto_front"]),
                "best_summary": str(paths["best_summary"]),
                "best_summary_json": str(paths["best_summary_json"]),
            },
            metadata={
                "completed_rows": int(len(leaderboard_df)),
                "pareto_rows": int(len(pareto_front_df)),
            },
        )
        manifest["status"] = "completed"
        manifest["result_status"] = "completed"
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["artifacts"] = {
            "protocol": str(paths["protocol"]),
            "grid_results": str(paths["grid_results"]),
            "leaderboard": str(paths["leaderboard"]),
            "pareto_front": str(paths["pareto_front"]),
            "best_summary": str(paths["best_summary"]),
            "best_summary_json": str(paths["best_summary_json"]),
        }
        _refresh_manifest_progress(manifest)
        _atomic_write_json(paths["manifest"], manifest)
        _persist_live_event(
            paths,
            manifest,
            step_id="complete",
            status="completed",
            message="Experimento de calibración finalizado.",
            extra={"artifact_paths": manifest.get("artifacts") or {}},
        )
        return {
            "run_id": run_id,
            "computed_run_id": computed_run_id,
            "protocol": protocol,
            "grid_results_df": grid_results_df,
            "leaderboard_df": leaderboard_df,
            "pareto_front_df": pareto_front_df,
            "best_summary_df": best_summary_df,
            "best_summary_payload": best_summary_payload,
            "checkpoint_manifest": manifest,
            "checkpoint_manifest_path": str(paths["manifest"]),
            "checkpoint_run_dir": str(paths["run_dir"]),
            "result_status": str(manifest.get("result_status") or ""),
        }

    def _load_controlled_splits(
        self,
        paths: Dict[str, Path],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if paths["splits_duckdb"].exists():
            train_df = _read_checkpoint_frame(paths["splits_duckdb"], table_name="train")
            val_df = _read_checkpoint_frame(paths["splits_duckdb"], table_name="val")
            test_df = _read_checkpoint_frame(paths["splits_duckdb"], table_name="test")
            if not train_df.empty and not val_df.empty and not test_df.empty:
                for frame in (train_df, val_df, test_df):
                    if "interval_start" in frame.columns:
                        frame["interval_start"] = pd.to_datetime(frame["interval_start"], errors="coerce")
                return train_df, val_df, test_df
        train_df = _read_checkpoint_frame(paths["splits_train_csv"])
        val_df = _read_checkpoint_frame(paths["splits_val_csv"])
        test_df = _read_checkpoint_frame(paths["splits_test_csv"])
        for frame in (train_df, val_df, test_df):
            if "interval_start" in frame.columns:
                frame["interval_start"] = pd.to_datetime(frame["interval_start"], errors="coerce")
        return train_df, val_df, test_df

    def _persist_controlled_splits(
        self,
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        paths: Dict[str, Path],
    ) -> Dict[str, str]:
        artifact_paths: Dict[str, str] = {}
        if duckdb is not None:
            _write_checkpoint_frame(train_df, paths["splits_duckdb"], table_name="train")
            _write_checkpoint_frame(val_df, paths["splits_duckdb"], table_name="val")
            _write_checkpoint_frame(test_df, paths["splits_duckdb"], table_name="test")
            artifact_paths["splits_duckdb"] = str(paths["splits_duckdb"])
            return artifact_paths
        _write_checkpoint_frame(train_df, paths["splits_train_csv"])
        _write_checkpoint_frame(val_df, paths["splits_val_csv"])
        _write_checkpoint_frame(test_df, paths["splits_test_csv"])
        artifact_paths["train_csv"] = str(paths["splits_train_csv"])
        artifact_paths["val_csv"] = str(paths["splits_val_csv"])
        artifact_paths["test_csv"] = str(paths["splits_test_csv"])
        return artifact_paths

    def _assemble_controlled_payload(
        self,
        *,
        run_id: str,
        protocol: Dict[str, object],
        manifest: Dict[str, object],
        paths: Dict[str, Path],
        grid_results_df: pd.DataFrame,
        best_summary_df: pd.DataFrame,
        curves_df: pd.DataFrame,
        auto_resumed: bool,
        loaded_from_checkpoint: bool,
        ablation_deltas_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, object]:
        if ablation_deltas_df is None:
            ablation_deltas_df = pd.DataFrame()
        return {
            "run_id": run_id,
            "protocol": protocol,
            "grid_results_df": grid_results_df,
            "best_summary_df": best_summary_df,
            "curves_df": curves_df,
            "ablation_deltas_df": ablation_deltas_df,
            "checkpoint_manifest": manifest,
            "checkpoint_manifest_path": str(paths["manifest"]),
            "checkpoint_run_dir": str(paths["run_dir"]),
            "auto_resumed": bool(auto_resumed),
            "loaded_from_checkpoint": bool(loaded_from_checkpoint),
            "computed_run_id": str(manifest.get("computed_run_id") or ""),
            "result_status": str(manifest.get("result_status") or manifest.get("status") or ""),
        }

    def run_controlled_comparison(
        self,
        base_df: pd.DataFrame,
        *,
        event_path: Optional[object] = None,
        features_path: Optional[object] = None,
        segment_info: Optional[Dict[str, object]] = None,
        dataset_date_start: Optional[object] = None,
        dataset_date_end: Optional[object] = None,
        test_size: float,
        val_size: float,
        k_min: int,
        k_max: int,
        k_step: int,
        n_trials: int,
        timeout: int,
        optuna_n_jobs: int,
        parallel_jobs: int,
        search_space_config: Dict[str, object],
        xgb_parallel_jobs: int = 1,
        selected_models: Optional[Sequence[object]] = None,
        objective_metric: str = "roc_auc",
        threshold_protocols: Optional[Sequence[object]] = None,
        threshold_objective: str = "recall_at_alerts_per_day",
        calibration_method: str = "sigmoid",
        far_target: float = 0.20,
        alerts_per_day: float = 5.0,
        fn_cost: float = 10.0,
        fp_cost: float = 1.0,
        robust_folds: int = 3,
        feature_ranking_mode: str = "controlled",
        experimental_protocol: Optional[str] = None,
        feature_selection_n_estimators: int = 200,
        feature_selection_max_depth: Optional[int] = None,
        feature_selection_n_jobs: int = -1,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        result_callback: Optional[Callable[[Dict[str, object]], None]] = None,
        checkpoint_root: Optional[Path] = None,
        auto_resume: bool = True,
        start_fresh: bool = False,
        checkpoint_run_id_override: Optional[str] = None,
    ) -> Dict[str, object]:
        feature_sets = self._controlled_comparison_feature_sets(base_df)
        base_cols = list(feature_sets["Base"])
        cluster_cols = list(feature_sets["Cluster"])
        all_cols = list(feature_sets["Base + Cluster"])

        if not base_cols:
            raise ValueError("No hay variables base disponibles para la comparación controlada.")
        if not cluster_cols:
            raise ValueError("El archivo seleccionado no contiene variables de cluster.")
        if "interval_start" not in base_df.columns:
            raise ValueError("El dataset debe incluir interval_start para split temporal.")
        if pd.Series(base_df["target"]).astype(int).nunique() < 2:
            raise ValueError("El target debe tener al menos dos clases.")
        objective_metric = _normalize_controlled_objective_metric(objective_metric)
        resolved_models = _resolve_controlled_models(selected_models)
        resolved_threshold_protocols = _normalize_controlled_threshold_protocols(
            threshold_protocols
        )
        threshold_objective = normalize_threshold_objective(threshold_objective)
        calibration_method = normalize_calibration_method(calibration_method)
        objective_label = CONTROLLED_COMPARISON_OBJECTIVE_LABELS.get(
            objective_metric, str(objective_metric).upper()
        )
        objective_direction = optuna_objective_direction(objective_metric)
        threshold_objective_label = THRESHOLD_OBJECTIVE_LABELS.get(
            threshold_objective,
            str(threshold_objective).upper(),
        )
        ranking_mode = str(feature_ranking_mode or "controlled").strip().lower()
        global_ranking_aliases = {
            "modelos_por_k",
            "modelos_by_k",
            "modelos_global",
            "feature_selection_global",
            "global_feature_selection",
        }
        if ranking_mode in {"controlled", "per_feature_set", "controlled_per_set"}:
            use_global_feature_selection_ranking = False
            ranking_mode = "controlled"
        elif ranking_mode in global_ranking_aliases:
            use_global_feature_selection_ranking = True
            ranking_mode = "feature_selection_global"
        else:
            raise ValueError(f"Modo de ranking no soportado: {feature_ranking_mode}")

        requested_protocol = str(experimental_protocol or "").strip().lower()
        use_frozen_tuning_ablation = requested_protocol in {
            FROZEN_TUNING_ABLATION_PROTOCOL_FAMILY,
            "cross_frozen_tuning_ablation",
            "frozen_tuning_cross_ablation",
            "ablacion_cruzada_tuning_congelado",
        }
        if requested_protocol and not use_frozen_tuning_ablation:
            if requested_protocol not in {
                "controlled_comparison",
                "comparacion_controlada",
                "modelos_por_k",
            }:
                raise ValueError(
                    f"Protocolo experimental no soportado: {experimental_protocol}"
                )
        if use_frozen_tuning_ablation:
            use_global_feature_selection_ranking = False
            ranking_mode = "controlled"

        protocol_family = (
            FROZEN_TUNING_ABLATION_PROTOCOL_FAMILY
            if use_frozen_tuning_ablation
            else (
                "modelos_por_k"
                if use_global_feature_selection_ranking
                else "controlled_comparison"
            )
        )

        if use_frozen_tuning_ablation:
            k_grid_global = []
            common_k_grid = _k_grid_values(
                k_min=int(k_min),
                k_max=int(k_max),
                k_step=int(k_step),
                feature_count=len(base_cols),
            )
            k_grid_by_set = {
                feature_set_name: list(common_k_grid)
                for feature_set_name in FROZEN_TUNING_ABLATION_FEATURE_SETS
            }
        elif use_global_feature_selection_ranking:
            k_grid_global = _k_grid_values(
                k_min=int(k_min),
                k_max=int(k_max),
                k_step=int(k_step),
                feature_count=len(all_cols),
            )
            k_grid_by_set = {
                feature_set_name: list(k_grid_global)
                for feature_set_name in CONTROLLED_COMPARISON_FEATURE_SETS
            }
        else:
            k_grid_global = []
            k_grid_by_set = {
                "Base": _k_grid_values(
                    k_min=int(k_min),
                    k_max=int(k_max),
                    k_step=int(k_step),
                    feature_count=len(base_cols),
                ),
                "Cluster": _k_grid_values(
                    k_min=int(k_min),
                    k_max=int(k_max),
                    k_step=int(k_step),
                    feature_count=len(cluster_cols),
                ),
                "Base + Cluster": _k_grid_values(
                    k_min=int(k_min),
                    k_max=int(k_max),
                    k_step=int(k_step),
                    feature_count=len(all_cols),
                ),
            }
        if any(not values for values in k_grid_by_set.values()):
            raise ValueError("La grilla de K no es valida para al menos un conjunto.")

        protocol = {
            "protocol_family": protocol_family,
            "split_mode": "Temporal",
            "metric": objective_metric,
            "objective_metric": objective_metric,
            "objective_label": objective_label,
            "objective_direction": objective_direction,
            "optuna_objective_metric": objective_metric,
            "optuna_objective_label": objective_label,
            "threshold_protocols": list(resolved_threshold_protocols),
            "threshold_objective": threshold_objective,
            "threshold_objective_label": threshold_objective_label,
            "calibration_method": calibration_method,
            "far_target": float(far_target),
            "alerts_per_day": float(alerts_per_day),
            "fn_cost": float(fn_cost),
            "fp_cost": float(fp_cost),
            "robust_folds": int(robust_folds),
            "test_only_final": True,
            "models": list(resolved_models),
            "feature_sets": (
                list(FROZEN_TUNING_ABLATION_FEATURE_SETS)
                if use_frozen_tuning_ablation
                else list(CONTROLLED_COMPARISON_FEATURE_SETS)
            ),
            "balance_modes": list(CONTROLLED_COMPARISON_BALANCE_MODES),
            "test_size": float(test_size),
            "val_size": float(val_size),
            "k_min": int(k_min),
            "k_max": int(k_max),
            "k_step": int(k_step),
            "k_grid_by_set": {key: list(values) for key, values in k_grid_by_set.items()},
            "k_grid_global": list(k_grid_global),
            "feature_ranking_mode": ranking_mode,
            "ranking_protocol": (
                "feature_selection_tab"
                if use_global_feature_selection_ranking
                else "controlled_train_only_per_feature_set"
            ),
            "feature_selection_params": {
                "n_estimators": int(feature_selection_n_estimators),
                "max_depth": (
                    None
                    if feature_selection_max_depth in {None, 0, "0"}
                    else int(feature_selection_max_depth)
                ),
                "random_state": int(self.random_state),
                "class_weight": "balanced",
                "criterion": "gini",
                "n_jobs": int(feature_selection_n_jobs),
            },
            "n_trials": int(n_trials),
            "timeout": int(timeout),
            "optuna_n_jobs": int(optuna_n_jobs),
            "parallel_jobs": int(parallel_jobs),
            "xgb_parallel_jobs": int(xgb_parallel_jobs),
            "search_space_config": search_space_config,
            "segment_info": dict(segment_info or {}),
            "event_path": str(event_path or ""),
            "features_path": str(features_path or ""),
            "dataset_date_start": (
                None if dataset_date_start is None else str(pd.Timestamp(dataset_date_start))
            ),
            "dataset_date_end": (
                None if dataset_date_end is None else str(pd.Timestamp(dataset_date_end))
            ),
        }
        if use_frozen_tuning_ablation:
            protocol["ablation_config"] = dict(FROZEN_TUNING_ABLATION_CONFIG)
        context = build_controlled_comparison_context(
            event_path=event_path,
            features_path=features_path,
            segment_info=segment_info,
            protocol=protocol,
        )

        preview = preview_controlled_comparison_checkpoint(
            context,
            checkpoint_root=checkpoint_root,
        )
        effective_run_id = None
        manifest: Optional[Dict[str, object]] = None
        auto_resumed = False
        loaded_from_checkpoint = False

        if checkpoint_run_id_override:
            effective_run_id = str(checkpoint_run_id_override)
        elif bool(auto_resume) and not bool(start_fresh) and preview.get("compatible"):
            effective_run_id = str(preview.get("run_id") or "")

        if effective_run_id:
            paths = _controlled_comparison_paths(
                _controlled_comparison_run_dir(
                    effective_run_id, checkpoint_root=checkpoint_root
                )
            )
            manifest = _load_manifest(paths["manifest"])
            compatible, reason = _manifest_is_compatible(manifest, context)
            if not compatible:
                raise ValueError(reason)
            checkpoint_status = str((manifest or {}).get("status") or "")
            if checkpoint_status == "completed" and not bool(start_fresh):
                grid_results_df = _read_checkpoint_frame(paths["grid_results"])
                best_summary_df = _read_checkpoint_frame(paths["best_summary"])
                curves_df = _read_checkpoint_frame(paths["curves"])
                ablation_deltas_df = _read_checkpoint_frame(paths["ablation_deltas"])
                loaded_from_checkpoint = True
                return self._assemble_controlled_payload(
                    run_id=effective_run_id,
                    protocol=dict((manifest or {}).get("protocol") or protocol),
                    manifest=dict(manifest or {}),
                    paths=paths,
                    grid_results_df=grid_results_df,
                    best_summary_df=best_summary_df,
                    curves_df=curves_df,
                    auto_resumed=False,
                    loaded_from_checkpoint=True,
                    ablation_deltas_df=ablation_deltas_df,
                )
            auto_resumed = True
        else:
            if use_frozen_tuning_ablation:
                run_prefix = "controlled_frozen_ablation"
            elif use_global_feature_selection_ranking:
                run_prefix = "controlled_modelos_k"
            else:
                run_prefix = "controlled"
            effective_run_id = (
                f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{str(context['computed_run_id'])[:8]}"
            )
            paths = _controlled_comparison_paths(
                _controlled_comparison_run_dir(
                    effective_run_id, checkpoint_root=checkpoint_root
                )
            )

        if manifest is None:
            step_sequence = [
                "dataset_build",
                "split_freeze",
            ]
            if use_global_feature_selection_ranking:
                step_sequence.append("ranking_global_feature_selection")
            elif use_frozen_tuning_ablation:
                step_sequence.extend(
                    [
                        "ranking_base",
                        "ranking_base_cluster",
                    ]
                )
            else:
                step_sequence.extend(
                    [
                        "ranking_base",
                        "ranking_cluster",
                        "ranking_base_cluster",
                    ]
                )
            if use_frozen_tuning_ablation:
                common_k_values = list(k_grid_by_set["Base"])
                for model_name, source_feature_set, balance_mode, threshold_protocol, k_value in itertools.product(
                    resolved_models,
                    FROZEN_TUNING_ABLATION_FEATURE_SETS,
                    CONTROLLED_COMPARISON_BALANCE_MODES,
                    resolved_threshold_protocols,
                    common_k_values,
                ):
                    source_step_id = _ablation_combo_id(
                        phase="source_tuning",
                        model_name=model_name,
                        params_source_feature_set=source_feature_set,
                        target_feature_set=source_feature_set,
                        balance_mode=balance_mode,
                        k=int(k_value),
                        threshold_protocol=threshold_protocol,
                    )
                    step_sequence.append(source_step_id)
                    for target_feature_set in FROZEN_TUNING_ABLATION_FEATURE_SETS:
                        if target_feature_set == source_feature_set:
                            continue
                        step_sequence.append(
                            _ablation_combo_id(
                                phase="cross_eval",
                                model_name=model_name,
                                params_source_feature_set=source_feature_set,
                                target_feature_set=target_feature_set,
                                balance_mode=balance_mode,
                                k=int(k_value),
                                threshold_protocol=threshold_protocol,
                            )
                        )
            else:
                for feature_set_name, k_values in k_grid_by_set.items():
                    for model_name, balance_mode, threshold_protocol, k_value in itertools.product(
                        resolved_models,
                        CONTROLLED_COMPARISON_BALANCE_MODES,
                        resolved_threshold_protocols,
                        k_values,
                    ):
                        step_sequence.append(
                            _combo_id(
                                model_name,
                                feature_set_name,
                                balance_mode,
                                int(k_value),
                                threshold_protocol,
                            )
                        )
            manifest = {
                "protocol_version": CONTROLLED_COMPARISON_PROTOCOL_VERSION,
                "run_id": effective_run_id,
                "computed_run_id": context["computed_run_id"],
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "status": "running",
                "result_status": "running",
                "last_error": None,
                "protocol": protocol,
                "input_fingerprints": {
                    "event": context["event_fingerprint"],
                    "features": context["features_fingerprint"],
                    "segment_info": context["segment_info"],
                },
                "step_sequence": step_sequence,
                "steps_index": {},
            }
            _refresh_manifest_progress(manifest)
            _atomic_write_json(paths["manifest"], manifest)
        else:
            manifest["status"] = "running"
            manifest["result_status"] = "running"
            manifest["last_error"] = None
            manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _atomic_write_json(paths["manifest"], manifest)

        _atomic_write_json(paths["protocol"], dict(protocol))
        if progress_callback:
            progress_callback(2, "Preparando dataset para comparación controlada.")
        _mark_step(
            paths,
            manifest,
            step_id="dataset_build",
            status="completed",
            message="Dataset etiquetado listo.",
            metadata={
                "rows": int(len(base_df)),
                "base_features": int(len(base_cols)),
                "cluster_features": int(len(cluster_cols)),
                "all_features": int(len(all_cols)),
            },
        )

        if progress_callback:
            progress_callback(5, "Congelando split temporal train/val/test.")
        if (
            str((manifest.get("steps_index") or {}).get("split_freeze", {}).get("status") or "")
            == "completed"
        ):
            train_df, val_df, test_df = self._load_controlled_splits(paths)
        else:
            train_val_df, test_df = temporal_train_test_split(
                base_df, test_size=float(test_size)
            )
            train_df, val_df = temporal_train_test_split(
                train_val_df, test_size=float(val_size)
            )
            split_artifacts = self._persist_controlled_splits(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                paths=paths,
            )
            _mark_step(
                paths,
                manifest,
                step_id="split_freeze",
                status="completed",
                message="Split temporal congelado.",
                artifact_paths=split_artifacts,
                metadata={
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "test_rows": int(len(test_df)),
                },
            )

        rankings: Dict[str, pd.DataFrame] = {}
        if use_global_feature_selection_ranking:
            step_id = "ranking_global_feature_selection"
            ranking_path = paths["ranking_global"]
            if progress_callback:
                progress_callback(
                    10,
                    "Calculando ranking global estilo Feature selection.",
                )
            completed = (
                str(
                    (manifest.get("steps_index") or {})
                    .get(step_id, {})
                    .get("status")
                    or ""
                )
                == "completed"
            )
            ranking_df = (
                _read_checkpoint_frame(ranking_path) if completed else pd.DataFrame()
            )
            if ranking_df.empty:
                fs_max_depth = (
                    None
                    if feature_selection_max_depth in {None, 0, "0"}
                    else int(feature_selection_max_depth)
                )
                ranking_df = self.calculate_feature_importance(
                    base_df,
                    all_cols,
                    target_col="target",
                    n_estimators=int(feature_selection_n_estimators),
                    max_depth=fs_max_depth,
                    n_jobs=int(feature_selection_n_jobs),
                )
                _write_checkpoint_frame(ranking_df, ranking_path)
                _mark_step(
                    paths,
                    manifest,
                    step_id=step_id,
                    status="completed",
                    message="Ranking global estilo Feature selection listo.",
                    artifact_paths={"csv": str(ranking_path)},
                    metadata={
                        "variables": int(len(ranking_df)),
                        "n_estimators": int(feature_selection_n_estimators),
                        "max_depth": fs_max_depth,
                        "random_state": int(self.random_state),
                        "class_weight": "balanced",
                        "criterion": "gini",
                        "n_jobs": int(feature_selection_n_jobs),
                    },
                )
            rankings["__global__"] = ranking_df
        else:
            if use_frozen_tuning_ablation:
                ranking_specs = [
                    ("Base", "ranking_base", paths["ranking_base"]),
                    (
                        "Base + Cluster",
                        "ranking_base_cluster",
                        paths["ranking_base_cluster"],
                    ),
                ]
            else:
                ranking_specs = [
                    ("Base", "ranking_base", paths["ranking_base"]),
                    ("Cluster", "ranking_cluster", paths["ranking_cluster"]),
                    (
                        "Base + Cluster",
                        "ranking_base_cluster",
                        paths["ranking_base_cluster"],
                    ),
                ]
            for feature_set_name, step_id, ranking_path in ranking_specs:
                if progress_callback:
                    progress_callback(10, f"Calculando ranking {feature_set_name}.")
                completed = str((manifest.get("steps_index") or {}).get(step_id, {}).get("status") or "") == "completed"
                ranking_df = _read_checkpoint_frame(ranking_path) if completed else pd.DataFrame()
                if ranking_df.empty:
                    cols = feature_sets[feature_set_name]
                    ranking_df = self.calculate_feature_importance(
                        train_df,
                        cols,
                        target_col="target",
                        n_estimators=max(200, min(1000, len(cols) * 8)),
                        n_jobs=int(parallel_jobs),
                    )
                    _write_checkpoint_frame(ranking_df, ranking_path)
                    _mark_step(
                        paths,
                        manifest,
                        step_id=step_id,
                        status="completed",
                        message=f"Ranking {feature_set_name} listo.",
                        artifact_paths={"csv": str(ranking_path)},
                        metadata={"variables": int(len(ranking_df))},
                    )
                rankings[feature_set_name] = ranking_df

        existing_grid_df = _read_checkpoint_frame(paths["grid_results"])
        existing_combo_ids = set(
            existing_grid_df["combo_id"].astype(str).tolist()
        ) if not existing_grid_df.empty and "combo_id" in existing_grid_df.columns else set()
        grid_records = (
            existing_grid_df.to_dict(orient="records")
            if not existing_grid_df.empty
            else []
        )

        if use_frozen_tuning_ablation:
            common_k_values = list(k_grid_by_set["Base"])

            def _ablation_selected_features(
                feature_set_name: str,
                k_value: int,
            ) -> List[str]:
                return (
                    rankings[feature_set_name]["variable"]
                    .astype(str)
                    .head(int(k_value))
                    .tolist()
                )

            def _source_key_from_values(
                model_name: object,
                source_feature_set: object,
                balance_mode: object,
                threshold_protocol: object,
                k_value: object,
            ) -> Tuple[str, str, str, str, int]:
                return (
                    str(model_name),
                    str(source_feature_set),
                    str(balance_mode),
                    str(threshold_protocol),
                    int(float(k_value)),
                )

            source_params_by_key: Dict[Tuple[str, str, str, str, int], Dict[str, object]] = {}
            for existing_record in grid_records:
                if str(existing_record.get("status") or "") != "completed":
                    continue
                if str(existing_record.get("ablation_phase") or "") != "source_tuning":
                    continue
                source_key = _source_key_from_values(
                    existing_record.get("model_name"),
                    existing_record.get("params_source_feature_set"),
                    existing_record.get("balance_mode"),
                    existing_record.get("threshold_protocol"),
                    existing_record.get("k"),
                )
                source_params_by_key[source_key] = {
                    "source_combo_id": str(existing_record.get("combo_id") or ""),
                    "best_params": _json_dict(existing_record.get("best_params")),
                    "smote_params": _json_dict(existing_record.get("smote_params")),
                }

            def _ablation_base_payload(
                *,
                phase: str,
                combo_id: str,
                source_combo_id: str,
                model_name: str,
                params_source_feature_set: str,
                target_feature_set: str,
                balance_mode: str,
                threshold_protocol: str,
                k_value: int,
                selected_features: List[str],
                frozen_tuning: bool,
            ) -> Dict[str, object]:
                counts = _selected_feature_family_counts(
                    selected_features,
                    cluster_cols,
                )
                payload = {
                    "experiment": "Controlled comparison",
                    "protocol_family": protocol_family,
                    "run_id": effective_run_id,
                    "computed_run_id": context["computed_run_id"],
                    "combo_id": combo_id,
                    "source_combo_id": source_combo_id,
                    "ablation_phase": phase,
                    "params_source_feature_set": params_source_feature_set,
                    "target_feature_set": target_feature_set,
                    "frozen_tuning": bool(frozen_tuning),
                    "threshold_freeze_policy": "recalibrate_per_target",
                    "model_name": model_name,
                    "feature_set": target_feature_set,
                    "balance_mode": balance_mode,
                    "threshold_protocol": threshold_protocol,
                    "threshold_protocol_label": THRESHOLD_PROTOCOL_LABELS.get(
                        threshold_protocol, threshold_protocol
                    ),
                    "threshold_objective": threshold_objective,
                    "threshold_objective_label": threshold_objective_label,
                    "calibration_method": calibration_method,
                    "objective_metric": objective_metric,
                    "objective_label": objective_label,
                    "objective_direction": objective_direction,
                    "optuna_objective_metric": objective_metric,
                    "optuna_objective_label": objective_label,
                    "k": int(k_value),
                    "k_global": None,
                    "effective_k": int(len(selected_features)),
                    "feature_ranking_mode": ranking_mode,
                    "ranking_protocol": "controlled_train_only_per_feature_set",
                    "selected_features": json.dumps(
                        selected_features,
                        ensure_ascii=True,
                    ),
                    "selected_features_global": json.dumps([], ensure_ascii=True),
                    "selected_feature_count": int(len(selected_features)),
                    "status": "pending",
                    "error": None,
                    "event_path": str(event_path or ""),
                    "features_path": str(features_path or ""),
                    "optuna_n_jobs": int(optuna_n_jobs),
                    "parallel_jobs": int(parallel_jobs),
                    "xgb_parallel_jobs": int(xgb_parallel_jobs),
                }
                payload.update(counts)
                return payload

            source_specs = list(
                itertools.product(
                    resolved_models,
                    FROZEN_TUNING_ABLATION_FEATURE_SETS,
                    CONTROLLED_COMPARISON_BALANCE_MODES,
                    resolved_threshold_protocols,
                    common_k_values,
                )
            )
            cross_specs = [
                (
                    model_name,
                    source_feature_set,
                    target_feature_set,
                    balance_mode,
                    threshold_protocol,
                    k_value,
                )
                for model_name, source_feature_set, balance_mode, threshold_protocol, k_value in source_specs
                for target_feature_set in FROZEN_TUNING_ABLATION_FEATURE_SETS
                if target_feature_set != source_feature_set
            ]
            total_ablation_combos = max(1, len(source_specs) + len(cross_specs))
            ablation_combo_index = 0

            for (
                model_name,
                source_feature_set,
                balance_mode,
                threshold_protocol,
                k_value,
            ) in source_specs:
                ablation_combo_index += 1
                combo_step_id = _ablation_combo_id(
                    phase="source_tuning",
                    model_name=model_name,
                    params_source_feature_set=source_feature_set,
                    target_feature_set=source_feature_set,
                    balance_mode=balance_mode,
                    k=int(k_value),
                    threshold_protocol=threshold_protocol,
                )
                source_key = _source_key_from_values(
                    model_name,
                    source_feature_set,
                    balance_mode,
                    threshold_protocol,
                    k_value,
                )
                if combo_step_id in existing_combo_ids:
                    continue
                if progress_callback:
                    progress_callback(
                        min(
                            95,
                            15
                            + int(
                                round(
                                    (ablation_combo_index / total_ablation_combos) * 75
                                )
                            ),
                        ),
                        (
                            f"Tuneando fuente {model_name} | {source_feature_set} | "
                            f"{balance_mode} | {threshold_protocol} | K={int(k_value)}"
                        ),
                    )
                selected_features = _ablation_selected_features(
                    source_feature_set,
                    int(k_value),
                )
                payload = _ablation_base_payload(
                    phase="source_tuning",
                    combo_id=combo_step_id,
                    source_combo_id=combo_step_id,
                    model_name=model_name,
                    params_source_feature_set=source_feature_set,
                    target_feature_set=source_feature_set,
                    balance_mode=balance_mode,
                    threshold_protocol=threshold_protocol,
                    k_value=int(k_value),
                    selected_features=selected_features,
                    frozen_tuning=False,
                )
                if not selected_features:
                    payload["status"] = "skipped_no_features"
                    payload["error"] = "No hay variables seleccionadas para la fuente."
                    _mark_step(
                        paths,
                        manifest,
                        step_id=combo_step_id,
                        status="completed",
                        message="Fuente omitida sin variables efectivas.",
                        metadata={
                            "model_name": model_name,
                            "params_source_feature_set": source_feature_set,
                            "target_feature_set": source_feature_set,
                            "balance_mode": balance_mode,
                            "threshold_protocol": threshold_protocol,
                            "k": int(k_value),
                        },
                    )
                    grid_records.append(payload)
                    grid_results_df = pd.DataFrame(grid_records)
                    _write_checkpoint_frame(grid_results_df, paths["grid_results"])
                    if result_callback:
                        result_callback(dict(payload))
                    continue
                try:
                    result = self._optimize_controlled_combo(
                        model_name=model_name,
                        feature_set=source_feature_set,
                        balance_mode=balance_mode,
                        objective_metric=objective_metric,
                        threshold_protocol=threshold_protocol,
                        threshold_objective=threshold_objective,
                        calibration_method=calibration_method,
                        far_target=float(far_target),
                        alerts_per_day=float(alerts_per_day),
                        fn_cost=float(fn_cost),
                        fp_cost=float(fp_cost),
                        robust_folds=int(robust_folds),
                        selected_features=selected_features,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        n_trials=int(n_trials),
                        timeout=int(timeout),
                        optuna_n_jobs=int(optuna_n_jobs),
                        search_space_config=search_space_config,
                        parallel_jobs=int(parallel_jobs),
                        xgb_parallel_jobs=int(xgb_parallel_jobs),
                    )
                    payload.update(
                        _controlled_payload_updates_from_result(
                            result,
                            threshold_protocol=threshold_protocol,
                            threshold_objective=threshold_objective,
                            threshold_objective_label=threshold_objective_label,
                            calibration_method=calibration_method,
                            k_global=None,
                            effective_k=int(len(selected_features)),
                            balance_mode=balance_mode,
                            optuna_n_jobs=int(optuna_n_jobs),
                            parallel_jobs=int(parallel_jobs),
                            xgb_parallel_jobs=int(xgb_parallel_jobs),
                        )
                    )
                    source_params_by_key[source_key] = {
                        "source_combo_id": combo_step_id,
                        "best_params": dict(result.get("best_params") or {}),
                        "smote_params": dict(result.get("smote_params") or {}),
                    }
                    trials_path = paths["trials_dir"] / f"{combo_step_id}.csv"
                    if isinstance(result.get("trials_df"), pd.DataFrame) and not result["trials_df"].empty:
                        _write_checkpoint_frame(result["trials_df"], trials_path)
                    _mark_step(
                        paths,
                        manifest,
                        step_id=combo_step_id,
                        status="completed",
                        message="Fuente tuneada para ablación.",
                        artifact_paths={"trials_csv": str(trials_path)},
                        metadata={
                            "model_name": model_name,
                            "params_source_feature_set": source_feature_set,
                            "target_feature_set": source_feature_set,
                            "balance_mode": balance_mode,
                            "threshold_protocol": threshold_protocol,
                            "k": int(k_value),
                            "objective_metric": objective_metric,
                            "val_objective_score": payload.get("val_objective_score"),
                        },
                    )
                except Exception as exc:
                    payload["status"] = "failed"
                    payload["error"] = str(exc)
                    _mark_step(
                        paths,
                        manifest,
                        step_id=combo_step_id,
                        status="failed",
                        message=f"Fuente de ablación falló: {exc}",
                        metadata={
                            "model_name": model_name,
                            "params_source_feature_set": source_feature_set,
                            "target_feature_set": source_feature_set,
                            "balance_mode": balance_mode,
                            "threshold_protocol": threshold_protocol,
                            "k": int(k_value),
                        },
                    )
                grid_records.append(payload)
                grid_results_df = pd.DataFrame(grid_records)
                _write_checkpoint_frame(grid_results_df, paths["grid_results"])
                if result_callback:
                    result_callback(dict(payload))

            for (
                model_name,
                source_feature_set,
                target_feature_set,
                balance_mode,
                threshold_protocol,
                k_value,
            ) in cross_specs:
                ablation_combo_index += 1
                combo_step_id = _ablation_combo_id(
                    phase="cross_eval",
                    model_name=model_name,
                    params_source_feature_set=source_feature_set,
                    target_feature_set=target_feature_set,
                    balance_mode=balance_mode,
                    k=int(k_value),
                    threshold_protocol=threshold_protocol,
                )
                source_key = _source_key_from_values(
                    model_name,
                    source_feature_set,
                    balance_mode,
                    threshold_protocol,
                    k_value,
                )
                source_payload = source_params_by_key.get(source_key) or {}
                source_combo_id = str(source_payload.get("source_combo_id") or "")
                if combo_step_id in existing_combo_ids:
                    continue
                if progress_callback:
                    progress_callback(
                        min(
                            95,
                            15
                            + int(
                                round(
                                    (ablation_combo_index / total_ablation_combos) * 75
                                )
                            ),
                        ),
                        (
                            f"Evaluando tuning congelado {model_name} | "
                            f"{source_feature_set} -> {target_feature_set} | "
                            f"{balance_mode} | {threshold_protocol} | K={int(k_value)}"
                        ),
                    )
                selected_features = _ablation_selected_features(
                    target_feature_set,
                    int(k_value),
                )
                payload = _ablation_base_payload(
                    phase="cross_eval",
                    combo_id=combo_step_id,
                    source_combo_id=source_combo_id,
                    model_name=model_name,
                    params_source_feature_set=source_feature_set,
                    target_feature_set=target_feature_set,
                    balance_mode=balance_mode,
                    threshold_protocol=threshold_protocol,
                    k_value=int(k_value),
                    selected_features=selected_features,
                    frozen_tuning=True,
                )
                if not selected_features:
                    payload["status"] = "skipped_no_features"
                    payload["error"] = "No hay variables seleccionadas para el target."
                elif not source_payload:
                    payload["status"] = "skipped_missing_source_params"
                    payload["error"] = (
                        "No hay hiperparámetros fuente completados para esta celda."
                    )
                if payload["status"] != "pending":
                    _mark_step(
                        paths,
                        manifest,
                        step_id=combo_step_id,
                        status="completed",
                        message="Evaluación cruzada omitida.",
                        metadata={
                            "model_name": model_name,
                            "params_source_feature_set": source_feature_set,
                            "target_feature_set": target_feature_set,
                            "balance_mode": balance_mode,
                            "threshold_protocol": threshold_protocol,
                            "k": int(k_value),
                            "status": payload["status"],
                        },
                    )
                    grid_records.append(payload)
                    grid_results_df = pd.DataFrame(grid_records)
                    _write_checkpoint_frame(grid_results_df, paths["grid_results"])
                    if result_callback:
                        result_callback(dict(payload))
                    continue
                try:
                    result = self._evaluate_controlled_combo_with_frozen_params(
                        model_name=model_name,
                        feature_set=target_feature_set,
                        balance_mode=balance_mode,
                        objective_metric=objective_metric,
                        threshold_protocol=threshold_protocol,
                        threshold_objective=threshold_objective,
                        calibration_method=calibration_method,
                        far_target=float(far_target),
                        alerts_per_day=float(alerts_per_day),
                        fn_cost=float(fn_cost),
                        fp_cost=float(fp_cost),
                        robust_folds=int(robust_folds),
                        selected_features=selected_features,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        frozen_model_params=dict(source_payload.get("best_params") or {}),
                        frozen_smote_params=dict(source_payload.get("smote_params") or {}),
                        optuna_n_jobs=int(optuna_n_jobs),
                        parallel_jobs=int(parallel_jobs),
                        xgb_parallel_jobs=int(xgb_parallel_jobs),
                    )
                    payload.update(
                        _controlled_payload_updates_from_result(
                            result,
                            threshold_protocol=threshold_protocol,
                            threshold_objective=threshold_objective,
                            threshold_objective_label=threshold_objective_label,
                            calibration_method=calibration_method,
                            k_global=None,
                            effective_k=int(len(selected_features)),
                            balance_mode=balance_mode,
                            optuna_n_jobs=int(optuna_n_jobs),
                            parallel_jobs=int(parallel_jobs),
                            xgb_parallel_jobs=int(xgb_parallel_jobs),
                        )
                    )
                    _mark_step(
                        paths,
                        manifest,
                        step_id=combo_step_id,
                        status="completed",
                        message="Evaluación cruzada con tuning congelado lista.",
                        metadata={
                            "model_name": model_name,
                            "params_source_feature_set": source_feature_set,
                            "target_feature_set": target_feature_set,
                            "source_combo_id": source_combo_id,
                            "balance_mode": balance_mode,
                            "threshold_protocol": threshold_protocol,
                            "k": int(k_value),
                            "objective_metric": objective_metric,
                            "val_objective_score": payload.get("val_objective_score"),
                        },
                    )
                except Exception as exc:
                    payload["status"] = "failed"
                    payload["error"] = str(exc)
                    _mark_step(
                        paths,
                        manifest,
                        step_id=combo_step_id,
                        status="failed",
                        message=f"Evaluación cruzada falló: {exc}",
                        metadata={
                            "model_name": model_name,
                            "params_source_feature_set": source_feature_set,
                            "target_feature_set": target_feature_set,
                            "source_combo_id": source_combo_id,
                            "balance_mode": balance_mode,
                            "threshold_protocol": threshold_protocol,
                            "k": int(k_value),
                        },
                    )
                grid_records.append(payload)
                grid_results_df = pd.DataFrame(grid_records)
                _write_checkpoint_frame(grid_results_df, paths["grid_results"])
                if result_callback:
                    result_callback(dict(payload))

        combo_specs: List[Tuple[str, str, str, str, int]] = []
        if not use_frozen_tuning_ablation:
            for feature_set_name, k_values in k_grid_by_set.items():
                combo_specs.extend(
                    list(
                        itertools.product(
                            resolved_models,
                            [feature_set_name],
                            CONTROLLED_COMPARISON_BALANCE_MODES,
                            resolved_threshold_protocols,
                            k_values,
                        )
                    )
                )
        total_combos = max(1, len(combo_specs))

        for combo_index, (
            model_name,
            feature_set_name,
            balance_mode,
            threshold_protocol,
            k_value,
        ) in enumerate(
            combo_specs,
            start=1,
        ):
            combo_step_id = _combo_id(
                model_name,
                feature_set_name,
                balance_mode,
                int(k_value),
                threshold_protocol,
            )
            if combo_step_id in existing_combo_ids:
                continue
            if progress_callback:
                progress_callback(
                    min(95, 15 + int(round((combo_index / total_combos) * 75))),
                    (
                        f"Evaluando {model_name} | {feature_set_name} | "
                        f"{balance_mode} | {threshold_protocol} | K={int(k_value)}"
                    ),
                )
            selected_features_global: List[str] = []
            if use_global_feature_selection_ranking:
                selected_features_global = (
                    rankings["__global__"]["variable"]
                    .astype(str)
                    .head(int(k_value))
                    .tolist()
                )
                allowed_features = set(feature_sets[feature_set_name])
                selected_features = [
                    feature
                    for feature in selected_features_global
                    if feature in allowed_features
                ]
            else:
                selected_features = (
                    rankings[feature_set_name]["variable"].astype(str).head(int(k_value)).tolist()
                )
            payload: Dict[str, object] = {
                "experiment": "Controlled comparison",
                "protocol_family": (
                    "modelos_por_k"
                    if use_global_feature_selection_ranking
                    else "controlled_comparison"
                ),
                "run_id": effective_run_id,
                "computed_run_id": context["computed_run_id"],
                "combo_id": combo_step_id,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "balance_mode": balance_mode,
                "threshold_protocol": threshold_protocol,
                "threshold_protocol_label": THRESHOLD_PROTOCOL_LABELS.get(
                    threshold_protocol, threshold_protocol
                ),
                "threshold_objective": threshold_objective,
                "threshold_objective_label": threshold_objective_label,
                "calibration_method": calibration_method,
                "objective_metric": objective_metric,
                "objective_label": objective_label,
                "objective_direction": objective_direction,
                "optuna_objective_metric": objective_metric,
                "optuna_objective_label": objective_label,
                "k": int(k_value),
                "k_global": (
                    int(k_value) if use_global_feature_selection_ranking else None
                ),
                "effective_k": int(len(selected_features)),
                "feature_ranking_mode": ranking_mode,
                "ranking_protocol": (
                    "feature_selection_tab"
                    if use_global_feature_selection_ranking
                    else "controlled_train_only_per_feature_set"
                ),
                "selected_features": json.dumps(selected_features, ensure_ascii=True),
                "selected_features_global": json.dumps(
                    selected_features_global,
                    ensure_ascii=True,
                ),
                "selected_feature_count": int(len(selected_features)),
                "status": "pending",
                "error": None,
                "event_path": str(event_path or ""),
                "features_path": str(features_path or ""),
                "optuna_n_jobs": int(optuna_n_jobs),
                "parallel_jobs": int(parallel_jobs),
                "xgb_parallel_jobs": int(xgb_parallel_jobs),
            }
            if not selected_features:
                payload["status"] = "skipped_no_features"
                payload["error"] = (
                    "K global no incluye variables para este conjunto."
                    if use_global_feature_selection_ranking
                    else "No hay variables seleccionadas para esta combinación."
                )
                _mark_step(
                    paths,
                    manifest,
                    step_id=combo_step_id,
                    status="completed",
                    message="Combinación omitida sin variables efectivas.",
                    metadata={
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "balance_mode": balance_mode,
                        "threshold_protocol": threshold_protocol,
                        "k": int(k_value),
                        "k_global": (
                            int(k_value)
                            if use_global_feature_selection_ranking
                            else None
                        ),
                        "effective_k": 0,
                    },
                )
                grid_records.append(payload)
                grid_results_df = pd.DataFrame(grid_records)
                _write_checkpoint_frame(grid_results_df, paths["grid_results"])
                if result_callback:
                    result_callback(dict(payload))
                continue
            try:
                result = self._optimize_controlled_combo(
                    model_name=model_name,
                    feature_set=feature_set_name,
                    balance_mode=balance_mode,
                    objective_metric=objective_metric,
                    threshold_protocol=threshold_protocol,
                    threshold_objective=threshold_objective,
                    calibration_method=calibration_method,
                    far_target=float(far_target),
                    alerts_per_day=float(alerts_per_day),
                    fn_cost=float(fn_cost),
                    fp_cost=float(fp_cost),
                    robust_folds=int(robust_folds),
                    selected_features=selected_features,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    n_trials=int(n_trials),
                    timeout=int(timeout),
                    optuna_n_jobs=int(optuna_n_jobs),
                    search_space_config=search_space_config,
                    parallel_jobs=int(parallel_jobs),
                    xgb_parallel_jobs=int(xgb_parallel_jobs),
                )
                payload.update(
                    _controlled_payload_updates_from_result(
                        result,
                        threshold_protocol=threshold_protocol,
                        threshold_objective=threshold_objective,
                        threshold_objective_label=threshold_objective_label,
                        calibration_method=calibration_method,
                        k_global=(
                            int(k_value)
                            if use_global_feature_selection_ranking
                            else None
                        ),
                        effective_k=int(len(selected_features)),
                        balance_mode=balance_mode,
                        optuna_n_jobs=int(optuna_n_jobs),
                        parallel_jobs=int(parallel_jobs),
                        xgb_parallel_jobs=int(xgb_parallel_jobs),
                    )
                )
                trials_path = paths["trials_dir"] / f"{combo_step_id}.csv"
                if isinstance(result.get("trials_df"), pd.DataFrame) and not result["trials_df"].empty:
                    _write_checkpoint_frame(result["trials_df"], trials_path)
                _mark_step(
                    paths,
                    manifest,
                    step_id=combo_step_id,
                    status="completed",
                    message="Combinación evaluada.",
                    artifact_paths={"trials_csv": str(trials_path)},
                    metadata={
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "balance_mode": balance_mode,
                        "threshold_protocol": threshold_protocol,
                        "k": int(k_value),
                        "objective_metric": objective_metric,
                        "val_objective_score": payload.get("val_objective_score"),
                    },
                )
            except Exception as exc:
                payload["status"] = "failed"
                payload["error"] = str(exc)
                _mark_step(
                    paths,
                    manifest,
                    step_id=combo_step_id,
                    status="failed",
                    message=f"Combinación falló: {exc}",
                    metadata={
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "balance_mode": balance_mode,
                        "threshold_protocol": threshold_protocol,
                        "k": int(k_value),
                    },
                )
            grid_records.append(payload)
            grid_results_df = pd.DataFrame(grid_records)
            _write_checkpoint_frame(grid_results_df, paths["grid_results"])
            if result_callback:
                result_callback(dict(payload))

        grid_results_df = _read_checkpoint_frame(paths["grid_results"])
        completed_grid_df = grid_results_df.copy()
        if not completed_grid_df.empty:
            completed_grid_df = completed_grid_df[
                completed_grid_df["status"].astype(str) == "completed"
            ].copy()
            for metric_col in [
                "decision_threshold",
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
            ]:
                if metric_col in completed_grid_df.columns:
                    completed_grid_df[metric_col] = pd.to_numeric(
                        completed_grid_df[metric_col], errors="coerce"
                    )

        best_summary_rows: List[Dict[str, object]] = []
        if not completed_grid_df.empty:
            objective_sort_ascending = objective_direction == "minimize"
            if use_frozen_tuning_ablation:
                grouping_cols = [
                    "model_name",
                    "params_source_feature_set",
                    "target_feature_set",
                    "balance_mode",
                    "threshold_protocol",
                ]
                sort_cols = grouping_cols + ["val_objective_score", "k"]
                sort_ascending = [True] * len(grouping_cols) + [
                    objective_sort_ascending,
                    True,
                ]
            else:
                grouping_cols = [
                    "model_name",
                    "feature_set",
                    "balance_mode",
                    "threshold_protocol",
                ]
                sort_cols = grouping_cols + ["val_objective_score", "k"]
                sort_ascending = [True] * len(grouping_cols) + [
                    objective_sort_ascending,
                    True,
                ]
            sort_df = completed_grid_df.sort_values(
                sort_cols,
                ascending=sort_ascending,
            )
            grouped = sort_df.groupby(
                grouping_cols,
                dropna=False,
            )
            for _, group in grouped:
                best_row = group.iloc[0]
                best_summary_rows.append(
                    {
                        "experiment": "Controlled comparison",
                        "protocol_family": best_row.get(
                            "protocol_family", "controlled_comparison"
                        ),
                        "run_id": effective_run_id,
                        "model_name": best_row.get("model_name"),
                        "feature_set": best_row.get("feature_set"),
                        "ablation_phase": best_row.get("ablation_phase"),
                        "params_source_feature_set": best_row.get(
                            "params_source_feature_set"
                        ),
                        "target_feature_set": best_row.get("target_feature_set"),
                        "source_combo_id": best_row.get("source_combo_id"),
                        "frozen_tuning": best_row.get("frozen_tuning"),
                        "threshold_freeze_policy": best_row.get(
                            "threshold_freeze_policy"
                        ),
                        "balance_mode": best_row.get("balance_mode"),
                        "threshold_protocol": best_row.get("threshold_protocol"),
                        "threshold_protocol_label": best_row.get(
                            "threshold_protocol_label"
                        ),
                        "threshold_objective": best_row.get("threshold_objective"),
                        "threshold_objective_label": best_row.get(
                            "threshold_objective_label"
                        ),
                        "calibration_method": best_row.get("calibration_method"),
                        "objective_metric": best_row.get("objective_metric"),
                        "objective_label": best_row.get("objective_label"),
                        "objective_direction": best_row.get("objective_direction"),
                        "optuna_objective_metric": best_row.get(
                            "optuna_objective_metric",
                            best_row.get("objective_metric"),
                        ),
                        "optuna_objective_label": best_row.get(
                            "optuna_objective_label",
                            best_row.get("objective_label"),
                        ),
                        "decision_threshold": best_row.get("decision_threshold"),
                        "val_objective_score": best_row.get("val_objective_score"),
                        "test_objective_score": best_row.get("test_objective_score"),
                        "best_test_accuracy": best_row.get("test_accuracy"),
                        "best_test_recall": best_row.get("test_recall"),
                        "best_test_sensitivity": best_row.get("test_sensitivity"),
                        "best_test_roc_auc": best_row.get("test_roc_auc"),
                        "best_test_pr_auc": best_row.get("test_pr_auc"),
                        "best_test_brier_score": best_row.get("test_brier_score"),
                        "val_roc_auc": best_row.get("val_roc_auc"),
                        "val_brier_score": best_row.get("val_brier_score"),
                        "best_test_f1": best_row.get("test_f1"),
                        "best_test_f1_global": best_row.get("test_f1_global"),
                        "best_test_balanced_f1": best_row.get(
                            "test_balanced_f1",
                            best_row.get("test_f1_global"),
                        ),
                        "best_test_f1_class_0": best_row.get("test_f1_class_0"),
                        "best_test_f1_class_1": best_row.get(
                            "test_f1_class_1",
                            best_row.get("test_f1"),
                        ),
                        "best_test_false_negatives": best_row.get(
                            "test_false_negatives"
                        ),
                        "best_test_false_positives": best_row.get(
                            "test_false_positives"
                        ),
                        "best_test_true_negatives": best_row.get(
                            "test_true_negatives"
                        ),
                        "best_test_true_positives": best_row.get(
                            "test_true_positives"
                        ),
                        "best_test_confusion_matrix": best_row.get(
                            "test_confusion_matrix"
                        ),
                        "best_test_alerts_per_day": best_row.get(
                            "test_alerts_per_day"
                        ),
                        "best_test_false_alarms_per_day": best_row.get(
                            "test_false_alarms_per_day"
                        ),
                        "best_test_event_recall_approx": best_row.get(
                            "test_event_recall_approx"
                        ),
                        "best_test_operational_cost": best_row.get(
                            "test_operational_cost"
                        ),
                        "best_test_cost_per_day": best_row.get("test_cost_per_day"),
                        "alerts_per_day_budget": best_row.get(
                            "alerts_per_day_budget"
                        ),
                        "fn_cost": best_row.get("fn_cost"),
                        "fp_cost": best_row.get("fp_cost"),
                        "val_f1": best_row.get("val_f1"),
                        "best_test_mcc": best_row.get("test_mcc"),
                        "val_mcc": best_row.get("val_mcc"),
                        "k_optimo": int(best_row.get("k") or 0),
                        "k_global": best_row.get("k_global"),
                        "effective_k": best_row.get(
                            "effective_k",
                            best_row.get("selected_feature_count"),
                        ),
                        "selected_base_feature_count": best_row.get(
                            "selected_base_feature_count"
                        ),
                        "selected_cluster_feature_count": best_row.get(
                            "selected_cluster_feature_count"
                        ),
                        "feature_ranking_mode": best_row.get(
                            "feature_ranking_mode"
                        ),
                        "ranking_protocol": best_row.get("ranking_protocol"),
                        "smote_optimo": bool(best_row.get("balance_mode") == "smote"),
                        "selected_features_global": best_row.get(
                            "selected_features_global"
                        ),
                        "selected_features": best_row.get("selected_features"),
                        "best_params": best_row.get("best_params"),
                        "effective_model_params": best_row.get(
                            "effective_model_params"
                        ),
                        "effective_threshold_n_jobs": best_row.get(
                            "effective_threshold_n_jobs"
                        ),
                        "smote_params": best_row.get("smote_params"),
                        "status": best_row.get("status"),
                        "error": best_row.get("error"),
                    }
                )
        best_summary_df = pd.DataFrame(best_summary_rows)

        curves_df = completed_grid_df[
            [
                col
                for col in [
                    "experiment",
                    "run_id",
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
                    "objective_metric",
                    "objective_label",
                    "k",
                    "k_global",
                    "effective_k",
                    "selected_base_feature_count",
                    "selected_cluster_feature_count",
                    "protocol_family",
                    "feature_ranking_mode",
                    "ranking_protocol",
                    "val_objective_score",
                    "val_roc_auc",
                    "val_brier_score",
                    "val_f1",
                    "val_pr_auc",
                    "val_f1_global",
                    "val_balanced_f1",
                    "val_mcc",
                    "val_alerts_per_day",
                    "val_false_alarms_per_day",
                    "val_event_recall_approx",
                    "val_operational_cost",
                    "val_cost_per_day",
                    "selected_feature_count",
                    "status",
                ]
                if col in completed_grid_df.columns
            ]
        ].copy() if not completed_grid_df.empty else pd.DataFrame()
        ablation_deltas_df = (
            _build_frozen_tuning_ablation_deltas(
                completed_grid_df,
                run_id=effective_run_id,
            )
            if use_frozen_tuning_ablation
            else pd.DataFrame()
        )

        _write_checkpoint_frame(best_summary_df, paths["best_summary"])
        _write_checkpoint_frame(curves_df, paths["curves"])
        if use_frozen_tuning_ablation:
            _write_checkpoint_frame(ablation_deltas_df, paths["ablation_deltas"])

        manifest["status"] = "completed"
        manifest["result_status"] = "completed"
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["artifacts"] = {
            "protocol": str(paths["protocol"]),
            "grid_results": str(paths["grid_results"]),
            "best_summary": str(paths["best_summary"]),
            "curves": str(paths["curves"]),
        }
        if use_frozen_tuning_ablation:
            manifest["artifacts"]["ablation_deltas"] = str(paths["ablation_deltas"])
        _refresh_manifest_progress(manifest)
        _atomic_write_json(paths["manifest"], manifest)
        _persist_live_event(
            paths,
            manifest,
            step_id="completed",
            status="completed",
            message="Comparación controlada completada.",
            extra={"artifact_paths": manifest.get("artifacts") or {}},
        )
        if progress_callback:
            progress_callback(100, "Comparación controlada completada.")

        return self._assemble_controlled_payload(
            run_id=effective_run_id,
            protocol=protocol,
            manifest=manifest,
            paths=paths,
            grid_results_df=grid_results_df,
            best_summary_df=best_summary_df,
            curves_df=curves_df,
            auto_resumed=auto_resumed,
            loaded_from_checkpoint=loaded_from_checkpoint,
            ablation_deltas_df=ablation_deltas_df,
        )
