#!/usr/bin/env python3
"""
Experiment runner and Streamlit tab for Neural drift staged studies.

The module intentionally avoids importing ``src.Neural_drift_app`` or
``src.drift_detection_app`` at import time. Those modules are loaded lazily
inside the execution helpers to prevent circular imports.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import bootstrap as scipy_bootstrap
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, confusion_matrix

try:
    import optuna  # type: ignore
    from optuna.pruners import MedianPruner  # type: ignore
    from optuna.samplers import TPESampler  # type: ignore
except Exception:
    optuna = None
    MedianPruner = None
    TPESampler = None


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
DEFAULT_FEATURE_EXPORT_PATH = RESULTS_DIR / "drift_flow_features_20260320_165122.duckdb"
NEURAL_DRIFT_EXPERIMENTS_DIR = RESULTS_DIR / "neural_drift_experiments"
RUN_TYPE = "neural_drift_experiment_sweep"

BASE_START = pd.Timestamp("2018-01-01")
BASE_END = pd.Timestamp("2018-12-31")
STREAM_START = pd.Timestamp("2019-01-01")
DEV_START = pd.Timestamp("2019-01-01")
DEV_END = pd.Timestamp("2022-12-31 23:59:59")
HOLDOUT_START = pd.Timestamp("2023-01-01")
HOLDOUT_END = pd.Timestamp("2024-09-30 23:59:59")

EXPERIMENT_SEEDS: Tuple[int, ...] = (42, 7, 123)
PAIRWISE_ALPHA = 0.05 / 3.0
PRACTICAL_DELTA_THRESHOLD = 0.01
BALANCE_MODE_OPTIONS: Tuple[str, ...] = ("none", "smote")

STUDY_CUMULATIVE = "cumulative"
STUDY_ADWIN = "adwin"
STUDY_NEURAL = "neural"
STUDY_ALL = "all"
AVAILABLE_STUDIES: Tuple[str, ...] = (STUDY_ALL, STUDY_CUMULATIVE, STUDY_ADWIN, STUDY_NEURAL)
AVAILABLE_PHASES: Tuple[str, ...] = ("all", "1", "2", "3", "4")
PHASE_LABELS: Dict[int, str] = {
    1: "Fase 1 · Detector",
    2: "Fase 2 · Arquitectura",
    3: "Fase 3 · Optimizador XGBoost",
    4: "Fase 4 · Joint",
}
STUDY_PHASE_BUDGETS: Dict[str, Dict[int, int]] = {
    STUDY_ADWIN: {1: 40, 3: 20, 4: 20},
    STUDY_NEURAL: {1: 40, 2: 70, 3: 30, 4: 40},
}
PHASE_TOP_K: Dict[int, int] = {
    1: 1,
    2: 3,
    3: 3,
    4: 3,
}
SCORE_WEIGHTS = {
    "action_cost": 0.05,
    "stability_penalty": 0.10,
}

SESSION_DEFAULTS: Dict[str, Any] = {
    "neural_drift_experiments_balance_mode": "none",
    "neural_drift_experiments_selected_study": STUDY_ALL,
    "neural_drift_experiments_selected_phase": "all",
    "neural_drift_experiments_active_run_id": None,
    "neural_drift_experiments_active_manifest_path": None,
    "neural_drift_experiments_history_selected_run_id": None,
    "neural_drift_experiments_loaded_payload": None,
}


def init_state() -> None:
    for key, default_value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default_value)


def _lazy_neural_drift_app():
    return importlib.import_module("src.Neural_drift_app")


def _lazy_drift_detection_app():
    return importlib.import_module("src.drift_detection_app")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_to_json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_to_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_to_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_json_file(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_json_safe(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
    tmp_path.replace(path)


def _atomic_write_df_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _append_jsonl_record(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json_safe(payload), ensure_ascii=True))
        handle.write("\n")


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _slugify_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "na"
    chars: List[str] = []
    previous_sep = False
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
            previous_sep = False
        elif not previous_sep:
            chars.append("_")
            previous_sep = True
    return "".join(chars).strip("_") or "na"


def _build_run_id(dataset_context: Dict[str, Any], balance_mode: str) -> str:
    created_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw = json.dumps(
        {
            "created_token": created_token,
            "source": str(dataset_context.get("source") or ""),
            "rows_total": int(dataset_context.get("rows_total") or 0),
            "balance_mode": str(balance_mode),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"run_{created_token}_{digest[:8]}"


def _run_dir(run_id: str) -> Path:
    return NEURAL_DRIFT_EXPERIMENTS_DIR / str(run_id)


def _run_paths(run_id: str) -> Dict[str, Path]:
    run_dir = _run_dir(run_id)
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "live_status": run_dir / "live_status.json",
        "live_events": run_dir / "live_events.jsonl",
        "artifacts_dir": run_dir / "artifacts",
        "optuna_dir": run_dir / "optuna",
    }


def _ensure_run_dirs(paths: Dict[str, Path]) -> None:
    for key in ("run_dir", "artifacts_dir", "optuna_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _build_live_status(
    manifest: Dict[str, Any],
    *,
    label: str,
    detail: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    progress = dict(manifest.get("progress") or {})
    return {
        "timestamp": _now_iso(),
        "run_id": str(manifest.get("run_id") or ""),
        "status": str(manifest.get("status") or "unknown"),
        "result_status": str(manifest.get("result_status") or "unknown"),
        "completed_units": int(progress.get("completed_units", 0)),
        "total_units": int(progress.get("total_units", 0)),
        "progress_ratio": float(progress.get("progress_ratio", 0.0)),
        "label": str(label),
        "detail": str(detail),
        "context": _to_json_safe(context or {}),
    }


def _persist_live_status(paths: Dict[str, Path], manifest: Dict[str, Any], *, label: str, detail: str = "", context: Optional[Dict[str, Any]] = None) -> None:
    payload = _build_live_status(manifest, label=label, detail=detail, context=context)
    _atomic_write_json(paths["live_status"], payload)


def _emit_progress_callback(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    *,
    run_id: str,
    completed_units: float,
    total_units: float,
    label: str,
    detail: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> None:
    if progress_callback is None:
        return
    safe_total_units = max(float(total_units), 1.0)
    safe_completed_units = min(max(float(completed_units), 0.0), safe_total_units)
    progress_callback(
        {
            "run_id": str(run_id),
            "timestamp": _now_iso(),
            "completed_units": float(safe_completed_units),
            "total_units": float(safe_total_units),
            "progress_ratio": float(safe_completed_units / safe_total_units),
            "label": str(label),
            "detail": str(detail),
            "context": _to_json_safe(context or {}),
        }
    )


def _log_live_event(paths: Dict[str, Path], *, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    _append_jsonl_record(
        paths["live_events"],
        {
            "timestamp": _now_iso(),
            "event": str(event),
            "payload": _to_json_safe(payload or {}),
        },
    )


def _make_study_status_template(study_name: str) -> Dict[str, Any]:
    return {
        "study": study_name,
        "phases": {
            f"phase_{phase}": {
                "phase": int(phase),
                "status": "pending",
                "storage_path": "",
                "study_name": "",
                "n_trials_budget": int(budget),
                "completed_trials": 0,
                "best_value": None,
                "best_trial_number": None,
                "error": None,
            }
            for phase, budget in STUDY_PHASE_BUDGETS[study_name].items()
        },
    }


def _initial_manifest(
    *,
    run_id: str,
    dataset_context: Dict[str, Any],
    balance_mode: str,
    selected_study: str,
    selected_phase: str,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    created_at = _now_iso()
    return {
        "schema_version": 1,
        "run_id": str(run_id),
        "run_type": RUN_TYPE,
        "status": "running",
        "result_status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "dataset_context": _to_json_safe(dataset_context),
        "balance_mode": str(balance_mode),
        "selected_study": str(selected_study),
        "selected_phase": str(selected_phase),
        "base_config": _to_json_safe(base_config),
        "baseline": {
            "status": "pending",
            "artifact_paths": {},
            "seed_metrics": [],
        },
        "studies": {
            STUDY_ADWIN: _make_study_status_template(STUDY_ADWIN),
            STUDY_NEURAL: _make_study_status_template(STUDY_NEURAL),
        },
        "artifacts": {},
        "winner": {},
        "last_error": None,
        "progress": {},
    }


def _update_manifest_progress(manifest: Dict[str, Any]) -> None:
    total_units = 1  # baseline
    completed_units = 1 if str((manifest.get("baseline") or {}).get("status") or "") == "completed" else 0
    for study_name, study_payload in (manifest.get("studies") or {}).items():
        for phase_key, phase_payload in dict(study_payload.get("phases") or {}).items():
            _ = phase_key
            total_units += 1
            if str(phase_payload.get("status") or "") == "completed":
                completed_units += 1
    manifest["progress"] = {
        "completed_units": int(completed_units),
        "total_units": int(total_units),
        "progress_ratio": float(completed_units / max(total_units, 1)),
    }


def _persist_manifest(paths: Dict[str, Path], manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = _now_iso()
    _update_manifest_progress(manifest)
    _atomic_write_json(paths["manifest"], manifest)


def _read_artifact_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for col in ("timestamp", "month", "start", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _list_persisted_runs(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    base_dir = Path(root) if root is not None else NEURAL_DRIFT_EXPERIMENTS_DIR
    if not base_dir.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for manifest_path in sorted(base_dir.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        manifest = dict(_load_json_file(manifest_path, default={}) or {})
        if not manifest:
            continue
        progress = dict(manifest.get("progress") or {})
        entries.append(
            {
                "run_id": str(manifest.get("run_id") or manifest_path.parent.name),
                "manifest_path": str(manifest_path),
                "status": str(manifest.get("status") or "unknown"),
                "result_status": str(manifest.get("result_status") or "unknown"),
                "updated_at": str(manifest.get("updated_at") or manifest.get("created_at") or ""),
                "source": str((manifest.get("dataset_context") or {}).get("source") or ""),
                "rows_total": int((manifest.get("dataset_context") or {}).get("rows_total") or 0),
                "balance_mode": str(manifest.get("balance_mode") or ""),
                "completed_units": int(progress.get("completed_units", 0)),
                "total_units": int(progress.get("total_units", 0)),
                "label": (
                    f"{manifest.get('run_id') or manifest_path.parent.name} | "
                    f"{manifest.get('status') or 'unknown'} | "
                    f"{manifest.get('updated_at') or manifest.get('created_at') or '-'} | "
                    f"{manifest.get('balance_mode') or 'na'}"
                ),
            }
        )
    return entries


def _load_persisted_run(manifest_path: Path) -> Dict[str, Any]:
    manifest = dict(_load_json_file(manifest_path, default={}) or {})
    artifacts = dict(manifest.get("artifacts") or {})
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "leaderboard_dev": _read_artifact_csv(Path(str(artifacts.get("leaderboard_dev") or ""))),
        "leaderboard_holdout": _read_artifact_csv(Path(str(artifacts.get("leaderboard_holdout") or ""))),
        "monthly_metrics": _read_artifact_csv(Path(str(artifacts.get("monthly_metrics") or ""))),
        "pairwise_stats": _read_artifact_csv(Path(str(artifacts.get("pairwise_stats") or ""))),
        "param_importances": _read_artifact_csv(Path(str(artifacts.get("param_importances") or ""))),
        "pareto": _read_artifact_csv(Path(str(artifacts.get("pareto") or ""))),
        "winner_config": dict(_load_json_file(Path(str(artifacts.get("winner_config") or "")), default={}) or {}),
    }


def _ensure_optuna_available() -> None:
    if optuna is None or TPESampler is None or MedianPruner is None:
        raise ImportError(
            "Neural drift experiments require `optuna` installed in the active environment."
        )


def _bundle_df(dataset_bundle: Dict[str, Any]) -> pd.DataFrame:
    df = dataset_bundle.get("df")
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if df is None:
        return pd.DataFrame()
    return pd.DataFrame(df).copy()


def _dataframe_or_empty(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    return pd.DataFrame(value).copy()


def _configured_balance_modes(current_config: Dict[str, Any]) -> List[str]:
    configured = list(current_config.get("balance_modes") or [])
    return [str(item) for item in configured if str(item) in BALANCE_MODE_OPTIONS]


def _balance_mode_selection_status(current_config: Dict[str, Any]) -> Dict[str, Any]:
    configured = _configured_balance_modes(current_config)
    has_exactly_one_active = len(configured) == 1
    resolved_balance_mode = configured[0] if configured else BALANCE_MODE_OPTIONS[0]
    return {
        "configured_balance_modes": configured,
        "has_exactly_one_active": bool(has_exactly_one_active),
        "resolved_balance_mode": str(resolved_balance_mode),
    }


def _safe_pr_auc(y_true: Sequence[Any], scores: Sequence[Any]) -> float:
    y = pd.to_numeric(pd.Series(list(y_true)), errors="coerce").dropna().astype(int).to_numpy()
    s = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna().astype(float).to_numpy()
    if len(y) != len(s) or len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def _binary_classification_metrics(
    y_true: Sequence[Any],
    scores: Sequence[Any],
    predictions: Optional[Sequence[Any]] = None,
    *,
    threshold: float = 0.5,
    beta: float = 2.0,
) -> Dict[str, float]:
    y = pd.to_numeric(pd.Series(list(y_true)), errors="coerce").fillna(0).astype(int).to_numpy()
    s = pd.to_numeric(pd.Series(list(scores)), errors="coerce").fillna(0.0).astype(float).to_numpy()
    if predictions is None:
        preds = (s >= float(threshold)).astype(int)
    else:
        preds = pd.to_numeric(pd.Series(list(predictions)), errors="coerce").fillna(0).astype(int).to_numpy()
    if len(y) == 0:
        return {
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f_beta": float("nan"),
        }
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    beta_sq = float(beta) ** 2
    f_beta = (
        float((1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall))
        if (beta_sq * precision + recall) > 0
        else 0.0
    )
    return {
        "pr_auc": _safe_pr_auc(y, s),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f_beta": f_beta,
    }


def _month_start(timestamp_series: pd.Series) -> pd.Series:
    return pd.to_datetime(timestamp_series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _monthly_metrics_from_records(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df is None or records_df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "pr_auc",
                "precision",
                "recall",
                "specificity",
                "f_beta",
                "n_rows",
                "n_actions",
            ]
        )
    work = records_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    work["month"] = _month_start(work["timestamp"])
    rows: List[Dict[str, Any]] = []
    for month, group in work.groupby("month", dropna=False, sort=True):
        metrics = _binary_classification_metrics(
            group["y_true"].to_numpy(),
            group["score"].to_numpy(),
            predictions=group["prediction"].to_numpy() if "prediction" in group.columns else None,
            threshold=float(group["decision_threshold"].dropna().iloc[-1]) if "decision_threshold" in group.columns and group["decision_threshold"].notna().any() else 0.5,
            beta=2.0,
        )
        rows.append(
            {
                "month": pd.Timestamp(month),
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f_beta": metrics["f_beta"],
                "n_rows": int(len(group)),
                "n_actions": int(group["action_taken"].astype(str).ne("none").sum()) if "action_taken" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _split_records_by_window(records_df: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=list(records_df.columns) if isinstance(records_df, pd.DataFrame) else [])
    work = records_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    return work.loc[work["timestamp"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")].reset_index(drop=True)


def _build_experiment_score(monthly_df: pd.DataFrame) -> Dict[str, float]:
    if monthly_df is None or monthly_df.empty:
        return {
            "monthly_pr_auc_median": float("nan"),
            "monthly_pr_auc_std": float("nan"),
            "n_actions": 0.0,
            "action_cost": 0.0,
            "stability_penalty": 0.0,
            "score": float("nan"),
        }
    pr_auc_series = pd.to_numeric(monthly_df["pr_auc"], errors="coerce")
    monthly_pr_auc_median = float(pr_auc_series.median()) if pr_auc_series.notna().any() else float("nan")
    monthly_pr_auc_std = float(pr_auc_series.std(ddof=0)) if pr_auc_series.notna().sum() > 1 else 0.0
    n_actions = float(pd.to_numeric(monthly_df["n_actions"], errors="coerce").fillna(0).sum())
    n_months = max(1.0, float(len(monthly_df)))
    action_cost = float(min(n_actions / n_months, 1.0))
    stability_penalty = float(max(monthly_pr_auc_std, 0.0))
    score = (
        float(monthly_pr_auc_median)
        - float(SCORE_WEIGHTS["action_cost"]) * action_cost
        - float(SCORE_WEIGHTS["stability_penalty"]) * stability_penalty
        if np.isfinite(monthly_pr_auc_median)
        else float("nan")
    )
    return {
        "monthly_pr_auc_median": float(monthly_pr_auc_median),
        "monthly_pr_auc_std": float(monthly_pr_auc_std),
        "n_actions": float(n_actions),
        "action_cost": float(action_cost),
        "stability_penalty": float(stability_penalty),
        "score": float(score),
    }


def _aggregate_records(records_df: pd.DataFrame, *, split_name: str) -> Dict[str, Any]:
    monthly_df = _monthly_metrics_from_records(records_df)
    monthly_df["split"] = str(split_name)
    summary = _build_experiment_score(monthly_df)
    global_metrics = _binary_classification_metrics(
        records_df["y_true"].to_numpy() if not records_df.empty else [],
        records_df["score"].to_numpy() if not records_df.empty else [],
        predictions=records_df["prediction"].to_numpy() if not records_df.empty and "prediction" in records_df.columns else None,
        threshold=float(records_df["decision_threshold"].dropna().iloc[-1]) if not records_df.empty and "decision_threshold" in records_df.columns and records_df["decision_threshold"].notna().any() else 0.5,
        beta=2.0,
    )
    return {
        **summary,
        "monthly": monthly_df,
        "f_beta": float(global_metrics["f_beta"]),
        "sensitivity": float(global_metrics["recall"]),
        "specificity": float(global_metrics["specificity"]),
    }


def _seed_aggregate(rows: Sequence[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    if not rows:
        return {
            "split": split_name,
            "score": float("nan"),
            "monthly_pr_auc_median": float("nan"),
            "monthly_pr_auc_std": float("nan"),
            "n_actions": 0.0,
            "action_cost": 0.0,
            "stability_penalty": 0.0,
            "f_beta": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }
    frame = pd.DataFrame(rows)
    out: Dict[str, Any] = {"split": split_name}
    for col in [
        "score",
        "monthly_pr_auc_median",
        "monthly_pr_auc_std",
        "n_actions",
        "action_cost",
        "stability_penalty",
        "f_beta",
        "sensitivity",
        "specificity",
    ]:
        series = pd.to_numeric(frame[col], errors="coerce")
        out[col] = float(series.median()) if series.notna().any() else float("nan")
    return out


def _aggregate_seed_runs(seed_runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dev_rows = [dict(item["dev"]) for item in seed_runs]
    holdout_rows = [dict(item["holdout"]) for item in seed_runs]
    return {
        "dev": _seed_aggregate(dev_rows, "dev"),
        "holdout": _seed_aggregate(holdout_rows, "holdout"),
        "seed_metrics": [
            {
                "seed": int(item["seed"]),
                "dev": _to_json_safe(item["dev"]),
                "holdout": _to_json_safe(item["holdout"]),
            }
            for item in seed_runs
        ],
    }


def _default_drift_xgb_params() -> Dict[str, Any]:
    return {
        "max_depth": 6,
        "eta": 0.1,
        "gamma": 1.0,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "subsample": 0.7,
        "nrounds": 100,
    }


def _normalize_window_params(params: Dict[str, Any], *, min_key: str, max_key: str) -> Dict[str, Any]:
    normalized = dict(params)
    lower = int(normalized[min_key])
    upper = int(normalized[max_key])
    if lower > upper:
        lower, upper = upper, lower
    normalized[min_key] = lower
    normalized[max_key] = upper
    return normalized


def _normalize_neural_candidate_config(base_config: Dict[str, Any], params: Dict[str, Any], *, balance_mode: str, rows_total: int) -> Dict[str, Any]:
    nd = _lazy_neural_drift_app()
    merged = {
        **copy.deepcopy(base_config),
        **dict(params),
    }
    merged = _normalize_window_params(merged, min_key="xgb_fine_tune_window_min", max_key="xgb_fine_tune_window_max")
    merged = _normalize_window_params(merged, min_key="xgb_fine_tune_rounds_min", max_key="xgb_fine_tune_rounds_max")
    merged.update(
        {
            "split_mode": "fixed_dates",
            "base_start": BASE_START.date().isoformat(),
            "base_end": BASE_END.date().isoformat(),
            "stream_start": STREAM_START.date().isoformat(),
            "dataset_percent": 100,
            "models": [nd.MODEL_XGBOOST],
            "strategies": [nd.STRATEGY_FINE_TUNING],
            "balance_modes": [str(balance_mode)],
            "xgb_parallel_neural_enabled": True,
            "max_stream_rows": max(int(rows_total), int(merged.get("max_stream_rows", 0) or 0)),
        }
    )
    return merged


def _normalize_adwin_candidate_config(base_params: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        **_default_drift_xgb_params(),
        **dict(base_params or {}),
        **dict(params or {}),
    }
    return {
        "adwin_delta": float(merged.get("adwin_delta", 0.002)),
        "min_window": int(merged.get("min_window", 45_000)),
        "min_retrain_size": int(merged.get("min_retrain_size", 512)),
        "max_depth": int(merged.get("max_depth", 6)),
        "eta": float(merged.get("eta", 0.1)),
        "gamma": float(merged.get("gamma", 1.0)),
        "colsample_bytree": float(merged.get("colsample_bytree", 0.8)),
        "min_child_weight": float(merged.get("min_child_weight", 5.0)),
        "subsample": float(merged.get("subsample", 0.7)),
        "nrounds": int(merged.get("nrounds", 100)),
    }


def _stream_records_frame(records: Sequence[Dict[str, Any]], *, study_name: str, seed: int) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "record_index",
                "strategy",
                "model",
                "balance_mode",
                "run_seed",
                "run_order",
                "prediction_year",
                "y_true",
                "raw_score",
                "score",
                "prediction",
                "decision_threshold",
                "raw_threshold",
                "action_taken",
                "study",
                "seed",
            ]
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["study"] = str(study_name)
    df["seed"] = int(seed)
    return df.sort_values("timestamp").reset_index(drop=True)


def _evaluate_records(records_df: pd.DataFrame) -> Dict[str, Any]:
    dev_records = _split_records_by_window(records_df, start=DEV_START, end=DEV_END)
    holdout_records = _split_records_by_window(records_df, start=HOLDOUT_START, end=HOLDOUT_END)
    return {
        "dev": _aggregate_records(dev_records, split_name="dev"),
        "holdout": _aggregate_records(holdout_records, split_name="holdout"),
        "records": records_df,
    }


def _evaluate_neural_candidate(
    dataset_bundle: Dict[str, Any],
    *,
    base_config: Dict[str, Any],
    candidate_params: Dict[str, Any],
    balance_mode: str,
    seed: int,
) -> Dict[str, Any]:
    nd = _lazy_neural_drift_app()
    normalized_config = _normalize_neural_candidate_config(
        base_config,
        candidate_params,
        balance_mode=balance_mode,
        rows_total=len(_bundle_df(dataset_bundle)),
    )
    normalized_config["random_state"] = int(seed)
    results = nd.run_backtest_pipeline(dataset_bundle, config=normalized_config)
    records = _dataframe_or_empty(results.get("stream_metrics"))
    if records.empty:
        raise ValueError("Neural drift candidate did not produce stream records.")
    records = records.loc[:, [col for col in [
        "timestamp",
        "y_true",
        "score",
        "prediction",
        "decision_threshold",
        "action_taken",
        "model",
        "strategy",
        "balance_mode",
    ] if col in records.columns]].copy()
    records["raw_score"] = records["score"]
    records["record_index"] = np.arange(len(records), dtype=int)
    records["run_seed"] = int(seed)
    records["run_order"] = 1
    records["prediction_year"] = pd.to_datetime(records["timestamp"], errors="coerce").dt.year
    records_df = _stream_records_frame(records.to_dict(orient="records"), study_name=STUDY_NEURAL, seed=seed)
    payload = _evaluate_records(records_df)
    return {
        "seed": int(seed),
        "config": normalized_config,
        "dev": payload["dev"],
        "holdout": payload["holdout"],
        "records": payload["records"],
    }


def _evaluate_cumulative_baseline(
    dataset_bundle: Dict[str, Any],
    *,
    balance_mode: str,
    seed: int,
    fixed_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    drift_app = _lazy_drift_detection_app()
    records: List[Dict[str, Any]] = []

    def recorder(payload: Dict[str, Any]) -> None:
        records.append(dict(payload))

    df = _bundle_df(dataset_bundle)
    fixed_model_params = {**_default_drift_xgb_params(), **dict(fixed_params or {})}
    drift_app.run_yearly_strategy(
        df,
        strategy="cumulative",
        feature_cols=list(dataset_bundle.get("feature_cols") or []),
        target_col="target",
        time_col="interval_start",
        model_names=["XGBoost"],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=int(seed),
        fast_mode=True,
        grid_limit=1,
        balance_mode=str(balance_mode),
        fixed_params=fixed_model_params,
        stream_record_callback=recorder,
    )
    records_df = _stream_records_frame(records, study_name="cumulative", seed=seed)
    if records_df.empty:
        raise ValueError("Cumulative baseline did not produce stream records.")
    payload = _evaluate_records(records_df)
    return {
        "seed": int(seed),
        "config": fixed_model_params,
        "dev": payload["dev"],
        "holdout": payload["holdout"],
        "records": payload["records"],
    }


def _evaluate_adwin_candidate(
    dataset_bundle: Dict[str, Any],
    *,
    candidate_params: Dict[str, Any],
    balance_mode: str,
    seed: int,
) -> Dict[str, Any]:
    drift_app = _lazy_drift_detection_app()
    records: List[Dict[str, Any]] = []

    def recorder(payload: Dict[str, Any]) -> None:
        records.append(dict(payload))

    config = _normalize_adwin_candidate_config({}, candidate_params)
    drift_app.run_adaptive_strategy(
        _bundle_df(dataset_bundle),
        feature_cols=list(dataset_bundle.get("feature_cols") or []),
        target_col="target",
        time_col="interval_start",
        model_names=["XGBoost"],
        base_year=2018,
        random_state=int(seed),
        validation_size=0.2,
        folds=3,
        fast_mode=True,
        grid_limit=1,
        adwin_delta=float(config["adwin_delta"]),
        min_window=int(config["min_window"]),
        min_retrain_size=int(config["min_retrain_size"]),
        balance_mode=str(balance_mode),
        fixed_params={
            "max_depth": int(config["max_depth"]),
            "eta": float(config["eta"]),
            "gamma": float(config["gamma"]),
            "colsample_bytree": float(config["colsample_bytree"]),
            "min_child_weight": float(config["min_child_weight"]),
            "subsample": float(config["subsample"]),
            "nrounds": int(config["nrounds"]),
        },
        stream_record_callback=recorder,
    )
    records_df = _stream_records_frame(records, study_name=STUDY_ADWIN, seed=seed)
    if records_df.empty:
        raise ValueError("Adaptive ADWIN candidate did not produce stream records.")
    payload = _evaluate_records(records_df)
    return {
        "seed": int(seed),
        "config": config,
        "dev": payload["dev"],
        "holdout": payload["holdout"],
        "records": payload["records"],
    }


def _trial_candidates_from_study(study: Any, *, top_k: int) -> List[Dict[str, Any]]:
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    if optuna is None:
        return []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
            continue
        candidate_config = dict(trial.user_attrs.get("candidate_config") or {})
        if not candidate_config:
            continue
        candidates.append((float(trial.value), candidate_config))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [dict(config) for _, config in candidates[: max(1, int(top_k))]]


def _study_storage_path(paths: Dict[str, Path], study_name: str, phase: int) -> Path:
    return paths["optuna_dir"] / f"{study_name}_phase_{int(phase)}.sqlite"


def _study_display_name(study_name: str, phase: int, run_id: str) -> str:
    return f"{run_id}_{study_name}_phase_{int(phase)}"


def _phase_search_space(study_name: str, phase: int) -> Dict[str, Sequence[Any]]:
    if study_name == STUDY_ADWIN:
        if phase == 1:
            return {
                "adwin_delta": [0.0005, 0.0010, 0.0020, 0.0050],
                "min_window": [5_000, 10_000, 20_000, 45_000],
                "min_retrain_size": [128, 256, 512, 1024],
            }
        if phase == 3:
            return {
                "max_depth": [3, 6, 9],
                "eta": [0.03, 0.10, 0.20],
                "gamma": [0.0, 1.0, 5.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1.0, 5.0, 10.0],
                "subsample": [0.6, 0.8, 1.0],
                "nrounds": [60, 100, 140],
            }
        if phase == 4:
            return {
                "adwin_delta": [0.0005, 0.0010, 0.0020, 0.0050],
                "min_window": [5_000, 10_000, 20_000, 45_000],
                "min_retrain_size": [128, 256, 512, 1024],
                "max_depth": [3, 6, 9],
                "eta": [0.03, 0.10, 0.20],
                "gamma": [0.0, 1.0, 5.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1.0, 5.0, 10.0],
                "subsample": [0.6, 0.8, 1.0],
                "nrounds": [60, 100, 140],
            }
        return {}
    if phase == 1:
        return {
            "detector_adwin_delta": [0.0005, 0.0010, 0.0020, 0.0050],
            "recent_window_size": [48, 72, 96, 144],
            "severity_threshold": [0.35, 0.50, 0.65],
            "lookback_steps": [8, 12, 16],
            "rolling_metric_window": [24, 48, 72],
        }
    if phase == 2:
        return {
            "mlp_hidden_dim": [64, 96, 128],
            "mlp_embedding_dim": [16, 24, 32],
            "mlp_dropout": [0.05, 0.10, 0.20],
            "attention_top_k": [6, 8, 10],
            "drift_monitor_bottleneck_dim": [4, 6, 8],
            "drift_monitor_reconstruction_weight": [0.50, 0.65, 0.80],
            "drift_monitor_architecture": ["Autoencoder clasico", "Attention temporal"],
            "mlp_learning_rate": [5e-4, 1e-3, 2e-3],
            "mlp_epochs": [10, 20, 30],
            "mlp_batch_size": [32, 64, 96],
        }
    if phase == 3:
        return {
            "xgb_fine_tune_selection_metric": ["f_beta_recall", "pr_auc", "brier"],
            "xgb_fine_tune_window_min": [24, 32, 48],
            "xgb_fine_tune_window_max": [96, 160, 224],
            "xgb_fine_tune_rounds_min": [2, 4, 6],
            "xgb_fine_tune_rounds_max": [16, 24, 32],
            "xgb_fine_tune_eta_multiplier_max": [1.25, 1.75, 2.50],
            "xgb_fine_tune_recent_weight_max": [2.0, 4.0, 6.0],
        }
    if phase == 4:
        return {
            "detector_adwin_delta": [0.0005, 0.0010, 0.0020, 0.0050],
            "recent_window_size": [48, 72, 96, 144],
            "severity_threshold": [0.35, 0.50, 0.65],
            "xgb_fine_tune_selection_metric": ["f_beta_recall", "pr_auc", "brier"],
            "xgb_fine_tune_window_min": [24, 32, 48],
            "xgb_fine_tune_window_max": [96, 160, 224],
            "xgb_fine_tune_rounds_min": [2, 4, 6],
            "xgb_fine_tune_rounds_max": [16, 24, 32],
            "xgb_fine_tune_eta_multiplier_max": [1.25, 1.75, 2.50],
            "xgb_fine_tune_recent_weight_max": [2.0, 4.0, 6.0],
        }
    return {}


def _suggest_trial_params(trial: Any, search_space: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    return {
        str(param_name): trial.suggest_categorical(str(param_name), list(values))
        for param_name, values in search_space.items()
    }


def _run_optuna_phase(
    *,
    manifest: Dict[str, Any],
    paths: Dict[str, Path],
    dataset_bundle: Dict[str, Any],
    base_config: Dict[str, Any],
    balance_mode: str,
    study_name: str,
    phase: int,
    candidate_pool: Sequence[Dict[str, Any]],
    progress_state: Dict[str, float],
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Any:
    _ensure_optuna_available()
    storage_path = _study_storage_path(paths, study_name, phase)
    storage_url = f"sqlite:///{storage_path}"
    budget = int(STUDY_PHASE_BUDGETS[study_name][phase])
    study_name_full = _study_display_name(study_name, phase, str(manifest["run_id"]))
    sampler = TPESampler(seed=42 + int(phase))
    pruner = MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name_full,
        storage=storage_url,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    completed_trials = len([trial for trial in study.trials if trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL}])
    phase_manifest = manifest["studies"][study_name]["phases"][f"phase_{phase}"]
    phase_manifest.update(
        {
            "status": "running",
            "storage_path": str(storage_path),
            "study_name": study_name_full,
            "completed_trials": int(completed_trials),
            "n_trials_budget": int(budget),
            "error": None,
        }
    )
    _persist_manifest(paths, manifest)
    _persist_live_status(
        paths,
        manifest,
        label="Ejecutando estudio",
        detail=f"{study_name} | {PHASE_LABELS.get(phase, phase)}",
        context={"study": study_name, "phase": int(phase)},
    )
    _log_live_event(
        paths,
        event="phase_start",
        payload={
            "study": study_name,
            "phase": int(phase),
            "budget": int(budget),
            "completed_trials": int(completed_trials),
            "label": "Ejecutando estudio",
            "detail": f"{study_name} | {PHASE_LABELS.get(phase, phase)}",
            "context": {"study": study_name, "phase": int(phase)},
            "completed_units": float(progress_state["completed_units"]),
            "total_units": float(progress_state["total_units"]),
            "progress_ratio": float(progress_state["completed_units"] / max(progress_state["total_units"], 1.0)),
        },
    )
    _emit_progress_callback(
        progress_callback,
        run_id=str(manifest["run_id"]),
        completed_units=float(progress_state["completed_units"]),
        total_units=float(progress_state["total_units"]),
        label="Ejecutando estudio",
        detail=f"{study_name} | {PHASE_LABELS.get(phase, phase)}",
        context={"study": study_name, "phase": int(phase), "completed_trials": int(completed_trials), "budget": int(budget)},
    )

    search_space = _phase_search_space(study_name, phase)
    active_pool = list(candidate_pool or [{}])
    phase_progress = {"counted_trials": int(completed_trials)}

    def objective(trial: Any) -> float:
        base_idx = 0
        if len(active_pool) > 1:
            base_idx = int(trial.suggest_categorical("base_candidate_index", list(range(len(active_pool)))))
        phase_params = _suggest_trial_params(trial, search_space)
        candidate_config = {
            **dict(active_pool[base_idx]),
            **dict(phase_params),
        }
        seed_runs: List[Dict[str, Any]] = []
        for step_idx, seed in enumerate(EXPERIMENT_SEEDS):
            if study_name == STUDY_NEURAL:
                seed_run = _evaluate_neural_candidate(
                    dataset_bundle,
                    base_config=base_config,
                    candidate_params=candidate_config,
                    balance_mode=balance_mode,
                    seed=int(seed),
                )
            else:
                seed_run = _evaluate_adwin_candidate(
                    dataset_bundle,
                    candidate_params=candidate_config,
                    balance_mode=balance_mode,
                    seed=int(seed),
                )
            seed_runs.append(seed_run)
            trial.report(
                float(_aggregate_seed_runs(seed_runs)["dev"]["score"]),
                step=step_idx,
            )
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned after seed {int(seed)}")

        aggregated = _aggregate_seed_runs(seed_runs)
        trial.set_user_attr("candidate_config", _to_json_safe(candidate_config))
        trial.set_user_attr("phase", int(phase))
        trial.set_user_attr("study", str(study_name))
        trial.set_user_attr("dev_summary", _to_json_safe(aggregated["dev"]))
        trial.set_user_attr("holdout_summary", _to_json_safe(aggregated["holdout"]))
        trial.set_user_attr("seed_metrics", _to_json_safe(aggregated["seed_metrics"]))
        return float(aggregated["dev"]["score"])

    remaining_trials = max(0, int(budget) - int(completed_trials))
    if remaining_trials > 0:
        def _after_trial(_study: Any, _trial: Any) -> None:
            del _study, _trial
            current_completed = len(
                [
                    trial
                    for trial in study.trials
                    if trial.state in {
                        optuna.trial.TrialState.COMPLETE,
                        optuna.trial.TrialState.PRUNED,
                        optuna.trial.TrialState.FAIL,
                    }
                ]
            )
            delta = max(0, int(current_completed) - int(phase_progress["counted_trials"]))
            if delta > 0:
                progress_state["completed_units"] += float(delta)
                phase_progress["counted_trials"] = int(current_completed)
            phase_manifest["completed_trials"] = int(current_completed)
            best_trials_local = [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
            ]
            if best_trials_local:
                best_trial_local = sorted(best_trials_local, key=lambda item: float(item.value), reverse=True)[0]
                phase_manifest["best_value"] = float(best_trial_local.value)
                phase_manifest["best_trial_number"] = int(best_trial_local.number)
            _persist_manifest(paths, manifest)
            _persist_live_status(
                paths,
                manifest,
                label="Ejecutando estudio",
                detail=f"{study_name} | {PHASE_LABELS.get(phase, phase)} | trial {current_completed}/{budget}",
                context={"study": study_name, "phase": int(phase), "completed_trials": int(current_completed), "budget": int(budget)},
            )
            _emit_progress_callback(
                progress_callback,
                run_id=str(manifest["run_id"]),
                completed_units=float(progress_state["completed_units"]),
                total_units=float(progress_state["total_units"]),
                label="Ejecutando estudio",
                detail=f"{study_name} | {PHASE_LABELS.get(phase, phase)} | trial {current_completed}/{budget}",
                context={"study": study_name, "phase": int(phase), "completed_trials": int(current_completed), "budget": int(budget)},
            )
            _log_live_event(
                paths,
                event="trial_complete",
                payload={
                    "study": study_name,
                    "phase": int(phase),
                    "completed_trials": int(current_completed),
                    "budget": int(budget),
                    "best_value": phase_manifest.get("best_value"),
                    "best_trial_number": phase_manifest.get("best_trial_number"),
                    "label": "Ejecutando estudio",
                    "detail": f"{study_name} | {PHASE_LABELS.get(phase, phase)} | trial {current_completed}/{budget}",
                    "context": {"study": study_name, "phase": int(phase)},
                    "completed_units": float(progress_state["completed_units"]),
                    "total_units": float(progress_state["total_units"]),
                    "progress_ratio": float(progress_state["completed_units"] / max(progress_state["total_units"], 1.0)),
                },
            )

        study.optimize(
            objective,
            n_trials=int(remaining_trials),
            show_progress_bar=False,
            callbacks=[_after_trial],
        )

    phase_manifest["completed_trials"] = int(
        len(
            [
                trial
                for trial in study.trials
                if trial.state in {
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                    optuna.trial.TrialState.FAIL,
                }
            ]
        )
    )
    best_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    if best_trials:
        best_trial = sorted(best_trials, key=lambda item: float(item.value), reverse=True)[0]
        phase_manifest["best_value"] = float(best_trial.value)
        phase_manifest["best_trial_number"] = int(best_trial.number)
    phase_manifest["status"] = "completed"
    _persist_manifest(paths, manifest)
    _persist_live_status(
        paths,
        manifest,
        label="Fase completada",
        detail=f"{study_name} | {PHASE_LABELS.get(phase, phase)}",
        context={"study": study_name, "phase": int(phase), "completed_trials": int(phase_manifest["completed_trials"]), "budget": int(budget)},
    )
    _emit_progress_callback(
        progress_callback,
        run_id=str(manifest["run_id"]),
        completed_units=float(progress_state["completed_units"]),
        total_units=float(progress_state["total_units"]),
        label="Fase completada",
        detail=f"{study_name} | {PHASE_LABELS.get(phase, phase)}",
        context={"study": study_name, "phase": int(phase), "completed_trials": int(phase_manifest["completed_trials"]), "budget": int(budget)},
    )
    _log_live_event(
        paths,
        event="phase_complete",
        payload={
            "study": study_name,
            "phase": int(phase),
            "completed_trials": int(phase_manifest["completed_trials"]),
            "budget": int(budget),
            "best_value": phase_manifest.get("best_value"),
            "best_trial_number": phase_manifest.get("best_trial_number"),
            "label": "Fase completada",
            "detail": f"{study_name} | {PHASE_LABELS.get(phase, phase)}",
            "context": {"study": study_name, "phase": int(phase)},
            "completed_units": float(progress_state["completed_units"]),
            "total_units": float(progress_state["total_units"]),
            "progress_ratio": float(progress_state["completed_units"] / max(progress_state["total_units"], 1.0)),
        },
    )
    return study


def _completed_study(study_name: str, phase: int, storage_path: Path, run_id: str) -> Any:
    _ensure_optuna_available()
    storage_url = f"sqlite:///{storage_path}"
    return optuna.load_study(
        study_name=_study_display_name(study_name, phase, run_id),
        storage=storage_url,
    )


def _trial_summary_rows(study_name: str, phase: int, study: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if optuna is None:
        return rows
    for trial in study.trials:
        dev_summary = dict(trial.user_attrs.get("dev_summary") or {})
        holdout_summary = dict(trial.user_attrs.get("holdout_summary") or {})
        candidate_config = dict(trial.user_attrs.get("candidate_config") or {})
        rows.append(
            {
                "study": str(study_name),
                "phase": int(phase),
                "trial_number": int(trial.number),
                "state": str(trial.state.name).lower(),
                "objective_value": float(trial.value) if trial.value is not None else float("nan"),
                "dev_score": pd.to_numeric(dev_summary.get("score"), errors="coerce"),
                "dev_monthly_pr_auc_median": pd.to_numeric(dev_summary.get("monthly_pr_auc_median"), errors="coerce"),
                "dev_monthly_pr_auc_std": pd.to_numeric(dev_summary.get("monthly_pr_auc_std"), errors="coerce"),
                "dev_n_actions": pd.to_numeric(dev_summary.get("n_actions"), errors="coerce"),
                "dev_f_beta": pd.to_numeric(dev_summary.get("f_beta"), errors="coerce"),
                "dev_sensitivity": pd.to_numeric(dev_summary.get("sensitivity"), errors="coerce"),
                "dev_specificity": pd.to_numeric(dev_summary.get("specificity"), errors="coerce"),
                "holdout_score": pd.to_numeric(holdout_summary.get("score"), errors="coerce"),
                "holdout_monthly_pr_auc_median": pd.to_numeric(holdout_summary.get("monthly_pr_auc_median"), errors="coerce"),
                "holdout_monthly_pr_auc_std": pd.to_numeric(holdout_summary.get("monthly_pr_auc_std"), errors="coerce"),
                "holdout_n_actions": pd.to_numeric(holdout_summary.get("n_actions"), errors="coerce"),
                "candidate_config_json": json.dumps(_to_json_safe(candidate_config), sort_keys=True, ensure_ascii=True),
            }
        )
    return rows


def _study_importance_rows(study_name: str, phase: int, study: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if optuna is None:
        return rows
    try:
        evaluator_cls = getattr(optuna.importance, "FanovaImportanceEvaluator", None)
        evaluator = evaluator_cls() if evaluator_cls is not None else None
        if evaluator is None:
            importances = optuna.importance.get_param_importances(study)
        else:
            importances = optuna.importance.get_param_importances(study, evaluator=evaluator)
    except Exception:
        return rows
    for param_name, importance in dict(importances).items():
        rows.append(
            {
                "study": str(study_name),
                "phase": int(phase),
                "parameter": str(param_name),
                "importance": float(importance),
            }
        )
    return rows


def _pairwise_series(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    left = left_df.loc[:, ["month", "pr_auc"]].rename(columns={"pr_auc": "left_pr_auc"})
    right = right_df.loc[:, ["month", "pr_auc"]].rename(columns={"pr_auc": "right_pr_auc"})
    merged = left.merge(right, on="month", how="inner")
    merged["delta"] = merged["left_pr_auc"] - merged["right_pr_auc"]
    return merged.dropna(subset=["delta"]).reset_index(drop=True)


def _bootstrap_delta_ci(delta_values: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(list(delta_values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "delta_median": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    try:
        bootstrap_result = scipy_bootstrap(
            (values,),
            np.median,
            confidence_level=0.95,
            n_resamples=10_000,
            method="BCa",
            random_state=42,
            vectorized=False,
        )
        return {
            "delta_median": float(np.median(values)),
            "ci_low": float(bootstrap_result.confidence_interval.low),
            "ci_high": float(bootstrap_result.confidence_interval.high),
        }
    except Exception:
        return {
            "delta_median": float(np.median(values)),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }


def _pairwise_stat_row(left_label: str, right_label: str, left_df: pd.DataFrame, right_df: pd.DataFrame) -> Dict[str, Any]:
    paired = _pairwise_series(left_df, right_df)
    p_value = float("nan")
    if len(paired) >= 2:
        try:
            p_value = float(wilcoxon(paired["delta"], alternative="greater").pvalue)
        except Exception:
            p_value = float("nan")
    ci_payload = _bootstrap_delta_ci(paired["delta"].to_numpy())
    delta_median = float(ci_payload["delta_median"])
    ci_low = float(ci_payload["ci_low"])
    ci_high = float(ci_payload["ci_high"])
    winner_gate = bool(
        np.isfinite(p_value)
        and p_value < PAIRWISE_ALPHA
        and np.isfinite(ci_low)
        and ci_low > 0.0
        and np.isfinite(delta_median)
        and delta_median > PRACTICAL_DELTA_THRESHOLD
    )
    return {
        "split": "holdout",
        "left": str(left_label),
        "right": str(right_label),
        "n_months": int(len(paired)),
        "wilcoxon_p": p_value,
        "bonferroni_alpha": float(PAIRWISE_ALPHA),
        "delta_median": delta_median,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "practical_delta_threshold": float(PRACTICAL_DELTA_THRESHOLD),
        "passes_gate": bool(winner_gate),
    }


def _final_monthly_series(label: str, seed_runs: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for item in seed_runs:
        seed = int(item["seed"])
        records_df = pd.DataFrame(item["records"]).copy()
        dev_monthly = _aggregate_records(
            _split_records_by_window(records_df, start=DEV_START, end=DEV_END),
            split_name="dev",
        )["monthly"]
        holdout_monthly = _aggregate_records(
            _split_records_by_window(records_df, start=HOLDOUT_START, end=HOLDOUT_END),
            split_name="holdout",
        )["monthly"]
        monthly_df = pd.concat([dev_monthly, holdout_monthly], ignore_index=True)
        monthly_df["label"] = str(label)
        monthly_df["seed"] = int(seed)
        rows.append(monthly_df)
    if not rows:
        return pd.DataFrame()
    per_seed = pd.concat(rows, ignore_index=True)
    aggregated = (
        per_seed.groupby(["label", "split", "month"], dropna=False, as_index=False)[
            ["pr_auc", "precision", "recall", "specificity", "f_beta", "n_rows", "n_actions"]
        ]
        .median(numeric_only=True)
        .sort_values(["label", "split", "month"])
        .reset_index(drop=True)
    )
    return aggregated


def _validate_experiment_dataset(dataset_bundle: Dict[str, Any]) -> Dict[str, Any]:
    df = _bundle_df(dataset_bundle)
    if df.empty:
        raise ValueError("No active dataset is available for Neural drift experiments.")
    if "interval_start" not in df.columns:
        raise ValueError("The active dataset requires `interval_start` for the fixed temporal protocol.")
    df["interval_start"] = pd.to_datetime(df["interval_start"], errors="coerce")
    df = df.dropna(subset=["interval_start"]).sort_values("interval_start").reset_index(drop=True)
    if df.empty:
        raise ValueError("The active dataset does not contain valid timestamps.")
    min_ts = pd.Timestamp(df["interval_start"].min())
    max_ts = pd.Timestamp(df["interval_start"].max())
    required_start_date = BASE_START.normalize()
    required_end_date = HOLDOUT_END.normalize()
    full_coverage = bool(
        min_ts.normalize() <= required_start_date
        and max_ts.normalize() >= required_end_date
    )
    return {
        "source": str(dataset_bundle.get("source") or ""),
        "rows_total": int(len(df)),
        "feature_count": int(len(dataset_bundle.get("feature_cols") or [])),
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "required_start_date": required_start_date,
        "required_end_date": required_end_date,
        "full_coverage": full_coverage,
        "feature_export_path": str(dataset_bundle.get("feature_export_path") or ""),
    }


def _selected_phases_for(study_name: str, selected_phase: str) -> List[int]:
    if study_name not in STUDY_PHASE_BUDGETS:
        return []
    available = sorted(STUDY_PHASE_BUDGETS[study_name].keys())
    if str(selected_phase) == "all":
        return available
    phase_value = int(selected_phase)
    return [phase_value] if phase_value in available else []


def _selected_studies_for(selected_study: str) -> List[str]:
    if str(selected_study) == STUDY_ALL:
        return [STUDY_ADWIN, STUDY_NEURAL]
    if str(selected_study) == STUDY_CUMULATIVE:
        return []
    return [str(selected_study)]


def _planned_total_units(selected_study: str, selected_phase: str) -> int:
    total_units = len(EXPERIMENT_SEEDS) + 1
    for study_name in _selected_studies_for(selected_study):
        for phase in _selected_phases_for(study_name, selected_phase):
            total_units += int(STUDY_PHASE_BUDGETS[study_name][phase])
    return int(max(total_units, 1))


def _resumed_completed_units(manifest: Dict[str, Any], selected_study: str, selected_phase: str) -> int:
    completed_units = len(EXPERIMENT_SEEDS) if str((manifest.get("baseline") or {}).get("status") or "") == "completed" else 0
    for study_name in _selected_studies_for(selected_study):
        for phase in _selected_phases_for(study_name, selected_phase):
            phase_payload = dict((manifest.get("studies") or {}).get(study_name, {}).get("phases", {}).get(f"phase_{phase}") or {})
            budget = int(STUDY_PHASE_BUDGETS[study_name][phase])
            completed_trials = int(phase_payload.get("completed_trials", 0) or 0)
            completed_units += min(max(completed_trials, 0), budget)
    if str(manifest.get("result_status") or "") == "success":
        completed_units += 1
    return int(completed_units)


def _candidate_pool_from_previous_phase(paths: Dict[str, Path], manifest: Dict[str, Any], study_name: str, phase: int) -> List[Dict[str, Any]]:
    previous_phases = [value for value in sorted(STUDY_PHASE_BUDGETS[study_name].keys()) if value < int(phase)]
    if not previous_phases:
        return [{}]
    previous_phase = previous_phases[-1]
    phase_manifest = dict(manifest["studies"][study_name]["phases"].get(f"phase_{previous_phase}") or {})
    raw_storage_path = str(phase_manifest.get("storage_path") or "").strip()
    if not raw_storage_path:
        return [{}]
    storage_path = Path(raw_storage_path)
    if not storage_path.exists() or not storage_path.is_file():
        return [{}]
    study = _completed_study(study_name, previous_phase, storage_path, str(manifest["run_id"]))
    candidates = _trial_candidates_from_study(study, top_k=int(PHASE_TOP_K.get(previous_phase, 1)))
    return candidates or [{}]


def _finalize_run(
    *,
    manifest: Dict[str, Any],
    paths: Dict[str, Path],
    dataset_bundle: Dict[str, Any],
    base_config: Dict[str, Any],
    balance_mode: str,
) -> Dict[str, Any]:
    baseline_seed_runs: List[Dict[str, Any]] = []
    default_fixed_params = _default_drift_xgb_params()
    for seed in EXPERIMENT_SEEDS:
        baseline_seed_runs.append(
            _evaluate_cumulative_baseline(
                dataset_bundle,
                balance_mode=balance_mode,
                seed=int(seed),
                fixed_params=default_fixed_params,
            )
        )

    winners: Dict[str, Dict[str, Any]] = {}
    leaderboard_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    for study_name in (STUDY_ADWIN, STUDY_NEURAL):
        for phase in sorted(STUDY_PHASE_BUDGETS[study_name].keys()):
            phase_manifest = dict(manifest["studies"][study_name]["phases"].get(f"phase_{phase}") or {})
            raw_storage_path = str(phase_manifest.get("storage_path") or "").strip()
            if not raw_storage_path:
                continue
            storage_path = Path(raw_storage_path)
            if not storage_path.exists() or not storage_path.is_file():
                continue
            study = _completed_study(study_name, phase, storage_path, str(manifest["run_id"]))
            leaderboard_rows.extend(_trial_summary_rows(study_name, phase, study))
            importance_rows.extend(_study_importance_rows(study_name, phase, study))
            phase_candidates = _trial_candidates_from_study(study, top_k=1)
            if phase_candidates:
                winners[study_name] = {
                    "phase": int(phase),
                    "candidate_config": dict(phase_candidates[0]),
                }

    baseline_aggregate = _aggregate_seed_runs(baseline_seed_runs)
    monthly_frames = [
        _final_monthly_series("cumulative", baseline_seed_runs),
    ]
    leaderboard_dev_rows: List[Dict[str, Any]] = [
        {
            "label": "cumulative",
            "study": "cumulative",
            "phase": 0,
            "dev_score": float(baseline_aggregate["dev"]["score"]),
            "dev_monthly_pr_auc_median": float(baseline_aggregate["dev"]["monthly_pr_auc_median"]),
            "dev_monthly_pr_auc_std": float(baseline_aggregate["dev"]["monthly_pr_auc_std"]),
            "dev_n_actions": float(baseline_aggregate["dev"]["n_actions"]),
            "dev_f_beta": float(baseline_aggregate["dev"]["f_beta"]),
            "dev_sensitivity": float(baseline_aggregate["dev"]["sensitivity"]),
            "dev_specificity": float(baseline_aggregate["dev"]["specificity"]),
        }
    ]
    leaderboard_holdout_rows: List[Dict[str, Any]] = [
        {
            "label": "cumulative",
            "study": "cumulative",
            "phase": 0,
            "holdout_score": float(baseline_aggregate["holdout"]["score"]),
            "holdout_monthly_pr_auc_median": float(baseline_aggregate["holdout"]["monthly_pr_auc_median"]),
            "holdout_monthly_pr_auc_std": float(baseline_aggregate["holdout"]["monthly_pr_auc_std"]),
            "holdout_n_actions": float(baseline_aggregate["holdout"]["n_actions"]),
            "holdout_f_beta": float(baseline_aggregate["holdout"]["f_beta"]),
            "holdout_sensitivity": float(baseline_aggregate["holdout"]["sensitivity"]),
            "holdout_specificity": float(baseline_aggregate["holdout"]["specificity"]),
        }
    ]

    winner_seed_runs: Dict[str, List[Dict[str, Any]]] = {}
    for study_name, winner_payload in winners.items():
        candidate_config = dict(winner_payload["candidate_config"])
        phase = int(winner_payload["phase"])
        seed_runs: List[Dict[str, Any]] = []
        for seed in EXPERIMENT_SEEDS:
            if study_name == STUDY_NEURAL:
                seed_runs.append(
                    _evaluate_neural_candidate(
                        dataset_bundle,
                        base_config=base_config,
                        candidate_params=candidate_config,
                        balance_mode=balance_mode,
                        seed=int(seed),
                    )
                )
            else:
                seed_runs.append(
                    _evaluate_adwin_candidate(
                        dataset_bundle,
                        candidate_params=candidate_config,
                        balance_mode=balance_mode,
                        seed=int(seed),
                    )
                )
        winner_seed_runs[study_name] = seed_runs
        aggregate = _aggregate_seed_runs(seed_runs)
        monthly_frames.append(_final_monthly_series(study_name, seed_runs))
        leaderboard_dev_rows.append(
            {
                "label": study_name,
                "study": study_name,
                "phase": int(phase),
                "dev_score": float(aggregate["dev"]["score"]),
                "dev_monthly_pr_auc_median": float(aggregate["dev"]["monthly_pr_auc_median"]),
                "dev_monthly_pr_auc_std": float(aggregate["dev"]["monthly_pr_auc_std"]),
                "dev_n_actions": float(aggregate["dev"]["n_actions"]),
                "dev_f_beta": float(aggregate["dev"]["f_beta"]),
                "dev_sensitivity": float(aggregate["dev"]["sensitivity"]),
                "dev_specificity": float(aggregate["dev"]["specificity"]),
                "candidate_config_json": json.dumps(_to_json_safe(candidate_config), sort_keys=True, ensure_ascii=True),
            }
        )
        leaderboard_holdout_rows.append(
            {
                "label": study_name,
                "study": study_name,
                "phase": int(phase),
                "holdout_score": float(aggregate["holdout"]["score"]),
                "holdout_monthly_pr_auc_median": float(aggregate["holdout"]["monthly_pr_auc_median"]),
                "holdout_monthly_pr_auc_std": float(aggregate["holdout"]["monthly_pr_auc_std"]),
                "holdout_n_actions": float(aggregate["holdout"]["n_actions"]),
                "holdout_f_beta": float(aggregate["holdout"]["f_beta"]),
                "holdout_sensitivity": float(aggregate["holdout"]["sensitivity"]),
                "holdout_specificity": float(aggregate["holdout"]["specificity"]),
                "candidate_config_json": json.dumps(_to_json_safe(candidate_config), sort_keys=True, ensure_ascii=True),
            }
        )

    monthly_metrics = pd.concat([frame for frame in monthly_frames if isinstance(frame, pd.DataFrame) and not frame.empty], ignore_index=True) if monthly_frames else pd.DataFrame()
    leaderboard_dev = pd.DataFrame(leaderboard_dev_rows).sort_values("dev_score", ascending=False, na_position="last").reset_index(drop=True)
    leaderboard_holdout = pd.DataFrame(leaderboard_holdout_rows).sort_values("holdout_score", ascending=False, na_position="last").reset_index(drop=True)
    pairwise_rows: List[Dict[str, Any]] = []
    cumulative_holdout = monthly_metrics.loc[(monthly_metrics["label"].astype(str) == "cumulative") & (monthly_metrics["split"].astype(str) == "holdout")].copy()
    adwin_holdout = monthly_metrics.loc[(monthly_metrics["label"].astype(str) == STUDY_ADWIN) & (monthly_metrics["split"].astype(str) == "holdout")].copy()
    neural_holdout = monthly_metrics.loc[(monthly_metrics["label"].astype(str) == STUDY_NEURAL) & (monthly_metrics["split"].astype(str) == "holdout")].copy()
    if not adwin_holdout.empty and not cumulative_holdout.empty:
        pairwise_rows.append(_pairwise_stat_row(STUDY_ADWIN, "cumulative", adwin_holdout, cumulative_holdout))
    if not neural_holdout.empty and not cumulative_holdout.empty:
        pairwise_rows.append(_pairwise_stat_row(STUDY_NEURAL, "cumulative", neural_holdout, cumulative_holdout))
    if not neural_holdout.empty and not adwin_holdout.empty:
        pairwise_rows.append(_pairwise_stat_row(STUDY_NEURAL, STUDY_ADWIN, neural_holdout, adwin_holdout))
    pairwise_stats = pd.DataFrame(
        pairwise_rows,
        columns=[
            "split",
            "left",
            "right",
            "n_months",
            "wilcoxon_p",
            "bonferroni_alpha",
            "delta_median",
            "ci_low",
            "ci_high",
            "practical_delta_threshold",
            "passes_gate",
        ],
    )
    param_importances = pd.DataFrame(importance_rows).sort_values(["study", "phase", "importance"], ascending=[True, True, False], na_position="last").reset_index(drop=True) if importance_rows else pd.DataFrame()
    pareto = leaderboard_dev.loc[leaderboard_dev["study"].astype(str).isin([STUDY_ADWIN, STUDY_NEURAL, "cumulative"])].copy()
    if not pareto.empty:
        pareto["dev_n_actions"] = pd.to_numeric(pareto["dev_n_actions"], errors="coerce").fillna(0.0)
        pareto["dev_monthly_pr_auc_median"] = pd.to_numeric(pareto["dev_monthly_pr_auc_median"], errors="coerce")
        pareto["pareto_optimal"] = True
        for idx, row in pareto.iterrows():
            dominated = pareto.loc[
                (pareto.index != idx)
                & (pd.to_numeric(pareto["dev_monthly_pr_auc_median"], errors="coerce") >= float(row["dev_monthly_pr_auc_median"]))
                & (pd.to_numeric(pareto["dev_n_actions"], errors="coerce") <= float(row["dev_n_actions"]))
                & (
                    (pd.to_numeric(pareto["dev_monthly_pr_auc_median"], errors="coerce") > float(row["dev_monthly_pr_auc_median"]))
                    | (pd.to_numeric(pareto["dev_n_actions"], errors="coerce") < float(row["dev_n_actions"]))
                )
            ]
            pareto.loc[idx, "pareto_optimal"] = dominated.empty

    winner_config_payload: Dict[str, Any] = {}
    neural_vs_cumulative = (
        pairwise_stats.loc[
            pairwise_stats["left"].astype(str).eq(STUDY_NEURAL)
            & pairwise_stats["right"].astype(str).eq("cumulative")
        ]
        if not pairwise_stats.empty
        else pd.DataFrame()
    )
    if STUDY_NEURAL in winners:
        gate_row = neural_vs_cumulative.iloc[0].to_dict() if not neural_vs_cumulative.empty else {}
        winner_config_payload = {
            "study": STUDY_NEURAL,
            "phase": int(winners[STUDY_NEURAL]["phase"]),
            "config": _to_json_safe(
                _normalize_neural_candidate_config(
                    base_config,
                    winners[STUDY_NEURAL]["candidate_config"],
                    balance_mode=balance_mode,
                    rows_total=len(_bundle_df(dataset_bundle)),
                )
            ),
            "eligible_for_promotion": bool(gate_row.get("passes_gate", False)),
            "holdout_gate": _to_json_safe(gate_row),
            "created_at": _now_iso(),
        }

    artifacts = {
        "leaderboard_dev": paths["artifacts_dir"] / "leaderboard_dev.csv",
        "leaderboard_holdout": paths["artifacts_dir"] / "leaderboard_holdout.csv",
        "monthly_metrics": paths["artifacts_dir"] / "monthly_metrics.csv",
        "pairwise_stats": paths["artifacts_dir"] / "pairwise_stats.csv",
        "param_importances": paths["artifacts_dir"] / "param_importances.csv",
        "pareto": paths["artifacts_dir"] / "pareto.csv",
        "winner_config": paths["artifacts_dir"] / "winner_config.json",
    }
    _atomic_write_df_csv(artifacts["leaderboard_dev"], leaderboard_dev)
    _atomic_write_df_csv(artifacts["leaderboard_holdout"], leaderboard_holdout)
    _atomic_write_df_csv(artifacts["monthly_metrics"], monthly_metrics)
    _atomic_write_df_csv(artifacts["pairwise_stats"], pairwise_stats)
    _atomic_write_df_csv(artifacts["param_importances"], param_importances)
    _atomic_write_df_csv(artifacts["pareto"], pareto)
    _atomic_write_json(artifacts["winner_config"], winner_config_payload)
    manifest["artifacts"] = {key: str(path) for key, path in artifacts.items()}
    manifest["winner"] = _to_json_safe(winner_config_payload)
    return {
        "leaderboard_dev": leaderboard_dev,
        "leaderboard_holdout": leaderboard_holdout,
        "monthly_metrics": monthly_metrics,
        "pairwise_stats": pairwise_stats,
        "param_importances": param_importances,
        "pareto": pareto,
        "winner_config": winner_config_payload,
    }


def run_experiment_plan(
    dataset_bundle: Dict[str, Any],
    *,
    base_config: Dict[str, Any],
    balance_mode: str,
    selected_study: str = STUDY_ALL,
    selected_phase: str = "all",
    resume_run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    dataset_context = _validate_experiment_dataset(dataset_bundle)
    if not bool(dataset_context["full_coverage"]):
        raise ValueError(
            "The active dataset does not cover the required range 2018-01-01 .. 2024-09-30."
        )
    if str(balance_mode) not in BALANCE_MODE_OPTIONS:
        raise ValueError(f"Unsupported balance mode: {balance_mode}")

    if resume_run_id:
        paths = _run_paths(str(resume_run_id))
        _ensure_run_dirs(paths)
        manifest = dict(_load_json_file(paths["manifest"], default={}) or {})
        if not manifest:
            raise FileNotFoundError(f"Experiment run `{resume_run_id}` was not found.")
        run_id = str(manifest.get("run_id") or resume_run_id)
    else:
        run_id = _build_run_id(dataset_context, balance_mode)
        paths = _run_paths(run_id)
        _ensure_run_dirs(paths)
        manifest = _initial_manifest(
            run_id=run_id,
            dataset_context=dataset_context,
            balance_mode=balance_mode,
            selected_study=selected_study,
            selected_phase=selected_phase,
            base_config=base_config,
        )
        _persist_manifest(paths, manifest)

    progress_state = {
        "completed_units": float(_resumed_completed_units(manifest, str(selected_study), str(selected_phase))),
        "total_units": float(_planned_total_units(str(selected_study), str(selected_phase))),
    }

    manifest["status"] = "running"
    manifest["result_status"] = "running"
    manifest["last_error"] = None
    _persist_manifest(paths, manifest)
    _persist_live_status(paths, manifest, label="Iniciando experimento", detail=f"balance_mode={balance_mode}")
    _emit_progress_callback(
        progress_callback,
        run_id=str(run_id),
        completed_units=float(progress_state["completed_units"]),
        total_units=float(progress_state["total_units"]),
        label="Iniciando experimento",
        detail=f"balance_mode={balance_mode}",
        context={"selected_study": str(selected_study), "selected_phase": str(selected_phase)},
    )
    _log_live_event(paths, event="run_start", payload={"run_id": run_id, "balance_mode": balance_mode})

    try:
        if str((manifest.get("baseline") or {}).get("status") or "") != "completed":
            baseline_seed_metrics = []
            for seed_idx, seed in enumerate(EXPERIMENT_SEEDS, start=1):
                baseline_seed_run = _evaluate_cumulative_baseline(
                    dataset_bundle,
                    balance_mode=balance_mode,
                    seed=int(seed),
                    fixed_params=_default_drift_xgb_params(),
                )
                aggregate = {
                    "seed": int(seed),
                    "dev": _to_json_safe(baseline_seed_run["dev"]),
                    "holdout": _to_json_safe(baseline_seed_run["holdout"]),
                }
                baseline_seed_metrics.append(aggregate)
                progress_state["completed_units"] += 1.0
                _persist_live_status(
                    paths,
                    manifest,
                    label="Ejecutando baseline cumulative",
                    detail=f"seed {seed_idx}/{len(EXPERIMENT_SEEDS)}",
                    context={"seed": int(seed), "seed_index": int(seed_idx), "total_seeds": int(len(EXPERIMENT_SEEDS))},
                )
                _emit_progress_callback(
                    progress_callback,
                    run_id=str(run_id),
                    completed_units=float(progress_state["completed_units"]),
                    total_units=float(progress_state["total_units"]),
                    label="Ejecutando baseline cumulative",
                    detail=f"seed {seed_idx}/{len(EXPERIMENT_SEEDS)}",
                    context={"seed": int(seed), "seed_index": int(seed_idx), "total_seeds": int(len(EXPERIMENT_SEEDS))},
                )
                _log_live_event(
                    paths,
                    event="baseline_seed_complete",
                    payload={
                        "seed": int(seed),
                        "seed_index": int(seed_idx),
                        "total_seeds": int(len(EXPERIMENT_SEEDS)),
                        "label": "Ejecutando baseline cumulative",
                        "detail": f"seed {seed_idx}/{len(EXPERIMENT_SEEDS)}",
                        "context": {"seed": int(seed), "seed_index": int(seed_idx), "total_seeds": int(len(EXPERIMENT_SEEDS))},
                        "completed_units": float(progress_state["completed_units"]),
                        "total_units": float(progress_state["total_units"]),
                        "progress_ratio": float(progress_state["completed_units"] / max(progress_state["total_units"], 1.0)),
                    },
                )
            manifest["baseline"]["status"] = "completed"
            manifest["baseline"]["seed_metrics"] = _to_json_safe(baseline_seed_metrics)
            _persist_manifest(paths, manifest)
            _log_live_event(
                paths,
                event="baseline_complete",
                payload={
                    "seeds": list(EXPERIMENT_SEEDS),
                    "label": "Baseline completado",
                    "detail": "cumulative",
                    "completed_units": float(progress_state["completed_units"]),
                    "total_units": float(progress_state["total_units"]),
                    "progress_ratio": float(progress_state["completed_units"] / max(progress_state["total_units"], 1.0)),
                },
            )

        requested_studies = _selected_studies_for(str(selected_study))
        for study_name in requested_studies:
            for phase in _selected_phases_for(study_name, selected_phase):
                phase_manifest = manifest["studies"][study_name]["phases"][f"phase_{phase}"]
                if str(phase_manifest.get("status") or "") == "completed":
                    continue
                candidate_pool = _candidate_pool_from_previous_phase(paths, manifest, study_name, phase)
                _run_optuna_phase(
                    manifest=manifest,
                    paths=paths,
                    dataset_bundle=dataset_bundle,
                    base_config=base_config,
                    balance_mode=balance_mode,
                    study_name=study_name,
                    phase=int(phase),
                    candidate_pool=candidate_pool,
                    progress_state=progress_state,
                    progress_callback=progress_callback,
                )

        _persist_live_status(paths, manifest, label="Finalizando artefactos", detail=run_id)
        _emit_progress_callback(
            progress_callback,
            run_id=str(run_id),
            completed_units=float(progress_state["completed_units"]),
            total_units=float(progress_state["total_units"]),
            label="Finalizando artefactos",
            detail=run_id,
        )
        _log_live_event(
            paths,
            event="finalizing",
            payload={
                "label": "Finalizando artefactos",
                "detail": str(run_id),
                "completed_units": float(progress_state["completed_units"]),
                "total_units": float(progress_state["total_units"]),
                "progress_ratio": float(progress_state["completed_units"] / max(progress_state["total_units"], 1.0)),
            },
        )
        outputs = _finalize_run(
            manifest=manifest,
            paths=paths,
            dataset_bundle=dataset_bundle,
            base_config=base_config,
            balance_mode=balance_mode,
        )
        progress_state["completed_units"] = float(progress_state["total_units"])
        manifest["status"] = "completed"
        manifest["result_status"] = "success"
        _persist_manifest(paths, manifest)
        _persist_live_status(paths, manifest, label="Experimento completado", detail=run_id)
        _emit_progress_callback(
            progress_callback,
            run_id=str(run_id),
            completed_units=float(progress_state["completed_units"]),
            total_units=float(progress_state["total_units"]),
            label="Experimento completado",
            detail=run_id,
        )
        _log_live_event(
            paths,
            event="run_complete",
            payload={
                "run_id": run_id,
                "label": "Experimento completado",
                "detail": str(run_id),
                "completed_units": float(progress_state["completed_units"]),
                "total_units": float(progress_state["total_units"]),
                "progress_ratio": 1.0,
            },
        )
        return {
            "run_id": run_id,
            "manifest": manifest,
            "manifest_path": str(paths["manifest"]),
            **outputs,
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["result_status"] = "failed"
        manifest["last_error"] = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _persist_manifest(paths, manifest)
        _persist_live_status(paths, manifest, label="Experimento fallido", detail=str(exc))
        _emit_progress_callback(
            progress_callback,
            run_id=str(run_id),
            completed_units=float(progress_state["completed_units"]),
            total_units=float(progress_state["total_units"]),
            label="Experimento fallido",
            detail=str(exc),
        )
        _log_live_event(
            paths,
            event="run_failed",
            payload={
                "error": str(exc),
                "label": "Experimento fallido",
                "detail": str(exc),
                "completed_units": float(progress_state["completed_units"]),
                "total_units": float(progress_state["total_units"]),
                "progress_ratio": float(progress_state["completed_units"] / max(progress_state["total_units"], 1.0)),
            },
        )
        raise


def _resolve_balance_mode_from_config(current_config: Dict[str, Any]) -> str:
    return str(_balance_mode_selection_status(current_config)["resolved_balance_mode"])


def _default_experiment_base_config(current_config: Dict[str, Any]) -> Dict[str, Any]:
    nd = _lazy_neural_drift_app()
    return {
        **dict(nd.DEFAULT_CONFIG),
        **dict(current_config or {}),
    }


def _current_setup_state(dataset_bundle: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_context = _validate_experiment_dataset(dataset_bundle)
    balance_mode_status = _balance_mode_selection_status(current_config)
    balance_mode = str(
        st.session_state.get(
            "neural_drift_experiments_balance_mode",
            balance_mode_status["resolved_balance_mode"],
        )
    )
    if balance_mode not in BALANCE_MODE_OPTIONS:
        balance_mode = str(balance_mode_status["resolved_balance_mode"])
    selected_study = str(
        st.session_state.get(
            "neural_drift_experiments_selected_study",
            STUDY_ALL,
        )
    )
    if selected_study not in AVAILABLE_STUDIES:
        selected_study = STUDY_ALL
    selected_phase = str(
        st.session_state.get(
            "neural_drift_experiments_selected_phase",
            "all",
        )
    )
    if selected_phase not in AVAILABLE_PHASES:
        selected_phase = "all"
    return {
        "dataset_context": dataset_context,
        "balance_mode_status": balance_mode_status,
        "balance_mode": balance_mode,
        "selected_study": selected_study,
        "selected_phase": selected_phase,
    }


def _render_setup_subtab(dataset_bundle: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
    setup_state = _current_setup_state(dataset_bundle, current_config)
    dataset_context = setup_state["dataset_context"]
    balance_mode_status = setup_state["balance_mode_status"]
    st.markdown("**Protocolo fijo**")
    st.caption(
        "Base 2018, dev stream 2019-2022 y holdout 2023-2024. "
        "El tab ignora `dataset_percent` y exige cobertura temporal completa."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", int(dataset_context["rows_total"]))
    metric_cols[1].metric("Features", int(dataset_context["feature_count"]))
    metric_cols[2].metric("Inicio", str(dataset_context["min_timestamp"].date()))
    metric_cols[3].metric("Fin", str(dataset_context["max_timestamp"].date()))
    if not bool(dataset_context["full_coverage"]):
        st.error("La fuente activa no cubre 2018-01-01 .. 2024-09-30.")
    configured_balance_modes = balance_mode_status["configured_balance_modes"]
    if balance_mode_status["has_exactly_one_active"]:
        st.caption(f"Balance mode activo detectado: `{configured_balance_modes[0]}`.")
    else:
        configured_label = ", ".join(configured_balance_modes) if configured_balance_modes else "ninguno"
        st.warning(
            "La configuracion actual no tiene exactamente un `balance_mode` activo "
            f"(actual: {configured_label}). El experimento correrá con un único modo seleccionado abajo."
        )
    balance_mode = st.selectbox(
        "Balance mode del experimento",
        BALANCE_MODE_OPTIONS,
        key="neural_drift_experiments_balance_mode",
        index=BALANCE_MODE_OPTIONS.index(setup_state["balance_mode"]) if setup_state["balance_mode"] in BALANCE_MODE_OPTIONS else 0,
        help="El sweep exige exactamente un modo de balanceo activo.",
    )
    selected_study = st.selectbox(
        "Study",
        AVAILABLE_STUDIES,
        key="neural_drift_experiments_selected_study",
        format_func=lambda value: {
            STUDY_ALL: "Todos",
            STUDY_CUMULATIVE: "cumulative",
            STUDY_ADWIN: "adaptive_adwin",
            STUDY_NEURAL: "neural_drift",
        }.get(str(value), str(value)),
    )
    selected_phase = st.selectbox(
        "Phase",
        AVAILABLE_PHASES,
        key="neural_drift_experiments_selected_phase",
        format_func=lambda value: "Todas" if str(value) == "all" else PHASE_LABELS.get(int(value), str(value)),
    )
    st.markdown("**Score de ranking**")
    st.code(
        "score = median_seed(monthly_pr_auc_median) - 0.05 * action_cost - 0.10 * stability_penalty",
        language="text",
    )
    st.caption(
        "Las métricas estadísticas finales usan meses calendario y comparan contra `cumulative` "
        "con Wilcoxon unilateral + bootstrap BCa."
    )
    return {
        "dataset_context": dataset_context,
        "balance_mode": str(balance_mode),
        "selected_study": str(selected_study),
        "selected_phase": str(selected_phase),
        "balance_mode_status": balance_mode_status,
    }


def _render_execution_subtab(dataset_bundle: Dict[str, Any], current_config: Dict[str, Any]) -> None:
    setup = _current_setup_state(dataset_bundle, current_config)
    dataset_context = setup["dataset_context"]
    st.markdown("**Configuración activa**")
    st.caption(
        "Los parámetros de estudio, fase y balance mode se definen en `Setup`. "
        "Aquí solo se ejecuta la configuración actualmente seleccionada."
    )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Balance", str(setup["balance_mode"]))
    summary_cols[1].metric("Study", str(setup["selected_study"]))
    summary_cols[2].metric("Phase", "all" if str(setup["selected_phase"]) == "all" else PHASE_LABELS.get(int(setup["selected_phase"]), str(setup["selected_phase"])))
    summary_cols[3].metric("Cobertura", "ok" if bool(dataset_context["full_coverage"]) else "incompleta")
    if not bool(dataset_context["full_coverage"]):
        st.error("La fuente activa no cubre 2018-01-01 .. 2024-09-30.")
        return
    selected_study = str(setup["selected_study"])
    requires_optuna = selected_study in {STUDY_ALL, STUDY_ADWIN, STUDY_NEURAL}
    if requires_optuna and optuna is None:
        st.warning("`optuna` no está disponible en el entorno activo. El runner no puede ejecutarse.")
        return
    runs = _list_persisted_runs()
    resume_candidates = [None] + [str(entry["run_id"]) for entry in runs]
    resume_run_id = st.selectbox(
        "Resume run",
        resume_candidates,
        format_func=lambda value: "Nueva corrida" if value is None else str(value),
    )
    active_manifest_path = st.session_state.get("neural_drift_experiments_active_manifest_path")
    if active_manifest_path:
        live_status_path = Path(str(active_manifest_path)).with_name("live_status.json")
        live_events_path = Path(str(active_manifest_path)).with_name("live_events.jsonl")
        live_status = dict(_load_json_file(live_status_path, default={}) or {})
        live_events = _read_jsonl_records(live_events_path)[-20:]
        if live_status:
            st.markdown("**Estado en vivo**")
            st.json(live_status)
        if live_events:
            st.markdown("**Eventos recientes**")
            st.dataframe(pd.DataFrame(live_events), width="stretch")
    run_clicked = st.button("Run experiment plan", key="neural_drift_experiments_run")
    if not run_clicked:
        return
    progress_status = st.empty()
    progress_bar = st.progress(0.0)
    progress_detail = st.empty()

    def _ui_progress_callback(payload: Dict[str, Any]) -> None:
        ratio = float(payload.get("progress_ratio", 0.0) or 0.0)
        completed_units = float(payload.get("completed_units", 0.0) or 0.0)
        total_units = float(payload.get("total_units", 1.0) or 1.0)
        label = str(payload.get("label") or "Ejecutando plan de experimentación...")
        detail = str(payload.get("detail") or "")
        progress_bar.progress(max(0.0, min(1.0, ratio)))
        progress_status.markdown(f"**{label}**")
        progress_detail.caption(f"{detail} · {completed_units:.0f}/{total_units:.0f}")

    _ui_progress_callback(
        {
            "progress_ratio": 0.0,
            "completed_units": 0.0,
            "total_units": float(_planned_total_units(selected_study, str(setup["selected_phase"]))),
            "label": "Ejecutando plan de experimentación...",
            "detail": "Preparando runner",
        }
    )
    try:
        results = run_experiment_plan(
            dataset_bundle,
            base_config=_default_experiment_base_config(current_config),
            balance_mode=str(setup["balance_mode"]),
            selected_study=selected_study,
            selected_phase=str(setup["selected_phase"]),
            resume_run_id=None if resume_run_id is None else str(resume_run_id),
            progress_callback=_ui_progress_callback,
        )
    except Exception as exc:
        progress_status.markdown("**Ejecución fallida**")
        progress_detail.caption(str(exc))
        raise
    st.session_state["neural_drift_experiments_active_run_id"] = str(results["run_id"])
    st.session_state["neural_drift_experiments_active_manifest_path"] = str(results["manifest_path"])
    st.session_state["neural_drift_experiments_loaded_payload"] = _load_persisted_run(Path(str(results["manifest_path"])))
    progress_bar.progress(1.0)
    progress_status.markdown("**Experimento completado**")
    progress_detail.caption(str(results["run_id"]))
    st.success(f"Experimento completado: {results['run_id']}")


def _render_results_subtab(apply_winner_config_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    payload = st.session_state.get("neural_drift_experiments_loaded_payload")
    if not isinstance(payload, dict):
        st.info("No hay una corrida de experimentos cargada.")
        return
    leaderboard_dev = payload.get("leaderboard_dev")
    leaderboard_holdout = payload.get("leaderboard_holdout")
    monthly_metrics = payload.get("monthly_metrics")
    pairwise_stats = payload.get("pairwise_stats")
    param_importances = payload.get("param_importances")
    pareto = payload.get("pareto")
    winner_config = dict(payload.get("winner_config") or {})

    if isinstance(leaderboard_dev, pd.DataFrame) and not leaderboard_dev.empty:
        st.markdown("**Leaderboard dev**")
        st.dataframe(leaderboard_dev, width="stretch")
    if isinstance(leaderboard_holdout, pd.DataFrame) and not leaderboard_holdout.empty:
        st.markdown("**Leaderboard holdout**")
        st.dataframe(leaderboard_holdout, width="stretch")
    if isinstance(monthly_metrics, pd.DataFrame) and not monthly_metrics.empty:
        st.markdown("**PR-AUC mensual**")
        chart_df = monthly_metrics.pivot_table(
            index="month",
            columns="label",
            values="pr_auc",
            aggfunc="last",
        ).sort_index()
        st.line_chart(chart_df, width="stretch")
        st.dataframe(monthly_metrics, width="stretch")
    if isinstance(pairwise_stats, pd.DataFrame) and not pairwise_stats.empty:
        st.markdown("**Comparaciones estadísticas**")
        st.dataframe(pairwise_stats, width="stretch")
    if isinstance(param_importances, pd.DataFrame) and not param_importances.empty:
        st.markdown("**Importancias fANOVA**")
        st.dataframe(param_importances, width="stretch")
    if isinstance(pareto, pd.DataFrame) and not pareto.empty:
        st.markdown("**Pareto PR-AUC vs acciones**")
        st.dataframe(pareto, width="stretch")

    if winner_config:
        st.markdown("**Winner config**")
        st.code(json.dumps(_to_json_safe(winner_config), indent=2, sort_keys=True, ensure_ascii=True), language="json")
        if apply_winner_config_callback is not None and bool(winner_config.get("eligible_for_promotion", False)):
            if st.button("Promover ganador neural a la configuración actual", key="neural_drift_promote_winner"):
                apply_winner_config_callback(dict(winner_config.get("config") or {}))
                st.success("Configuración promotida al estado actual de Neural drift.")


def _render_history_subtab(apply_winner_config_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    runs = _list_persisted_runs()
    if not runs:
        st.info("Todavía no hay corridas persistidas de experimentos.")
        return
    run_ids = [str(entry["run_id"]) for entry in runs]
    if st.session_state.get("neural_drift_experiments_history_selected_run_id") not in run_ids:
        st.session_state["neural_drift_experiments_history_selected_run_id"] = run_ids[0]
    selected_run_id = st.selectbox(
        "Corrida",
        run_ids,
        key="neural_drift_experiments_history_selected_run_id",
    )
    selected_entry = next(entry for entry in runs if str(entry["run_id"]) == str(selected_run_id))
    st.dataframe(pd.DataFrame(runs), width="stretch")
    manifest_path = Path(str(selected_entry["manifest_path"]))
    manifest = dict(_load_json_file(manifest_path, default={}) or {})
    st.caption(f"Manifest: {manifest_path}")
    if st.button("Cargar corrida seleccionada", key="neural_drift_experiments_load_history"):
        payload = _load_persisted_run(manifest_path)
        st.session_state["neural_drift_experiments_loaded_payload"] = payload
        st.session_state["neural_drift_experiments_active_run_id"] = str(selected_run_id)
        st.session_state["neural_drift_experiments_active_manifest_path"] = str(manifest_path)
        st.success(f"Corrida cargada: {selected_run_id}")
    winner_config = dict((manifest.get("winner") or {}))
    if apply_winner_config_callback is not None and bool(winner_config.get("eligible_for_promotion", False)):
        if st.button("Promover winner de la corrida seleccionada", key="neural_drift_experiments_promote_history_winner"):
            apply_winner_config_callback(dict(winner_config.get("config") or {}))
            st.success("Winner aplicado a la configuración actual.")


def render_experiments_tab(
    dataset_bundle: Dict[str, Any],
    current_config: Dict[str, Any],
    *,
    apply_winner_config_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    init_state()
    setup_tab, execution_tab, results_tab, history_tab = st.tabs(
        ["Setup", "Ejecución", "Resultados", "Historial"]
    )
    with setup_tab:
        _render_setup_subtab(dataset_bundle, current_config)
    with execution_tab:
        _render_execution_subtab(dataset_bundle, current_config)
    with results_tab:
        _render_results_subtab(apply_winner_config_callback=apply_winner_config_callback)
    with history_tab:
        _render_history_subtab(apply_winner_config_callback=apply_winner_config_callback)


__all__ = [
    "AVAILABLE_PHASES",
    "AVAILABLE_STUDIES",
    "BALANCE_MODE_OPTIONS",
    "DEFAULT_FEATURE_EXPORT_PATH",
    "EXPERIMENT_SEEDS",
    "NEURAL_DRIFT_EXPERIMENTS_DIR",
    "PAIRWISE_ALPHA",
    "PRACTICAL_DELTA_THRESHOLD",
    "RUN_TYPE",
    "STUDY_CUMULATIVE",
    "_balance_mode_selection_status",
    "_bootstrap_delta_ci",
    "_build_experiment_score",
    "_final_monthly_series",
    "_list_persisted_runs",
    "_load_persisted_run",
    "_monthly_metrics_from_records",
    "_pairwise_stat_row",
    "_validate_experiment_dataset",
    "render_experiments_tab",
    "run_experiment_plan",
]
