#!/usr/bin/env python3
"""
Streamlit app to monitor experiment results in real time.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src.drift_bias_variance import (
    BIAS_VARIANCE_NOISE_COLUMNS,
    build_bias_variance_noise_lookup,
    drift_row_group_key,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
DRIFT_RUNS_DIR = RESULTS_DIR / "drift_recalibration_runs"
NEURAL_DRIFT_EXPERIMENTS_DIR = RESULTS_DIR / "neural_drift_experiments"
NLP_PAPER_RUNS_DIR = RESULTS_DIR / "nlp_in_severity" / "paper_replication"
NLP_LANGUAGE_MODELING_LIVE_DIR = RESULTS_DIR / "nlp_in_severity" / "language_modeling_live"
PAPER_MODEL_CODES = ("M1", "M2", "M3")


def _list_live_db_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        RESULTS_DIR.glob("experiment_live_*.sqlite"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_json_file(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _load_pickle_file(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return pd.read_pickle(path)
    except Exception:
        return default


def _read_jsonl_records(path: Path) -> list[Dict[str, object]]:
    if not path.exists():
        return []
    rows: list[Dict[str, object]] = []
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
                    rows.append(payload)
    except Exception:
        return []
    return rows


def _safe_int(value: object, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return int(default)
    return int(numeric)


def _maybe_int(value: object) -> Optional[int]:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return int(numeric)


def _maybe_float(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _sequence_to_numeric(values: object) -> list[float]:
    if values is None or isinstance(values, (str, bytes)):
        return []
    try:
        raw_values = list(values)
    except TypeError:
        return []
    out: list[float] = []
    for value in raw_values:
        numeric = pd.to_numeric(value, errors="coerce")
        out.append(float("nan") if pd.isna(numeric) else float(numeric))
    return out


def _streamlit_arrow_safe_df(data: object) -> object:
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return data
    work = data.copy()
    for col in work.columns:
        if pd.api.types.is_object_dtype(work[col]):
            work[col] = work[col].astype("string")
    return work


def _ensure_balance_strategy_column(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    work = df.copy()
    if "balance_strategy" in work.columns:
        if "balance_mode" in work.columns:
            work["balance_strategy"] = work["balance_strategy"].where(
                work["balance_strategy"].notna(), work["balance_mode"]
            )
        return work
    if "balance_mode" in work.columns:
        work["balance_strategy"] = work["balance_mode"]
        return work
    if "smote_optimo" in work.columns:
        flag = work["smote_optimo"].astype(str).str.lower()
        work["balance_strategy"] = flag.map(
            {
                "true": "smote",
                "1": "smote",
                "yes": "smote",
                "false": "none",
                "0": "none",
                "no": "none",
            }
        ).fillna("not_specified")
    return work


def _average_precision_from_scores(y_true: list[int], scores: list[float]) -> float:
    if not y_true or len(y_true) != len(scores):
        return float("nan")
    total_pos = int(sum(int(v) for v in y_true))
    if total_pos <= 0 or total_pos >= len(y_true):
        return float("nan")
    ranked = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    ap = 0.0
    idx = 0
    while idx < len(ranked):
        current_score = ranked[idx][0]
        pos_in_group = 0
        neg_in_group = 0
        while idx < len(ranked) and ranked[idx][0] == current_score:
            if int(ranked[idx][1]) == 1:
                pos_in_group += 1
            else:
                neg_in_group += 1
            idx += 1
        tp += pos_in_group
        fp += neg_in_group
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall_prev = (tp - pos_in_group) / float(total_pos)
        recall = tp / float(total_pos)
        ap += (recall - recall_prev) * precision
    return float(ap)


def _brier_score_from_probabilities(y_true: object, scores: object) -> float:
    y = _sequence_to_numeric(y_true)
    p = _sequence_to_numeric(scores)
    if not y or not p or len(y) != len(p):
        return float("nan")
    work = pd.DataFrame({"y": y, "p": p}).dropna(subset=["y", "p"])
    if work.empty:
        return float("nan")
    work["y"] = work["y"].clip(0.0, 1.0).round()
    work["p"] = work["p"].clip(0.0, 1.0)
    return float(((work["p"] - work["y"]) ** 2).mean())


def _probability_scores_for_brier(roc_item: Dict[str, object], *, expected_length: int) -> list[float]:
    calibrated_scores = _sequence_to_numeric(roc_item.get("calibrated_scores"))
    if len(calibrated_scores) == expected_length and pd.Series(calibrated_scores).notna().any():
        return calibrated_scores
    return _sequence_to_numeric(roc_item.get("scores"))


def _derive_f1_from_rates(row: Dict[str, object], y_true: list[int]) -> float:
    sensitivity = _maybe_float(row.get("sensitivity"))
    specificity = _maybe_float(row.get("specificity"))
    if not math.isfinite(sensitivity) or not math.isfinite(specificity):
        return float("nan")
    positives = int(sum(int(v) for v in y_true))
    negatives = int(len(y_true) - positives)
    tp = int(round(sensitivity * positives))
    tn = int(round(specificity * negatives))
    fp = max(0, negatives - tn)
    fn = max(0, positives - tp)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _is_missing_numeric(value: object) -> bool:
    numeric = pd.to_numeric(value, errors="coerce")
    return bool(pd.isna(numeric))


def _row_metric_key(row: Dict[str, object]) -> tuple[object, ...]:
    return (
        str(row.get("strategy") or ""),
        str(row.get("model") or ""),
        str(row.get("balance_mode") or "not_applicable"),
        _maybe_int(row.get("run_seed")),
        _maybe_int(row.get("run_order")),
    )


def _roc_metric_key(item: Dict[str, object]) -> tuple[object, ...]:
    return (
        str(item.get("strategy") or ""),
        str(item.get("model") or ""),
        str(item.get("balance_mode") or "not_applicable"),
        _maybe_int(item.get("run_seed")),
        _maybe_int(item.get("run_order")),
    )


def _apply_derived_metrics(
    row: Dict[str, object],
    roc_item: Optional[Dict[str, object]],
    *,
    decomposition_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    if not isinstance(roc_item, dict):
        if isinstance(decomposition_metrics, dict):
            for col in BIAS_VARIANCE_NOISE_COLUMNS:
                if _is_missing_numeric(row.get(col)):
                    row[col] = decomposition_metrics.get(col)
        return row
    y_true_numeric = _sequence_to_numeric(roc_item.get("y_true"))
    y_true = [int(v) for v in y_true_numeric if not pd.isna(v)]
    scores = _sequence_to_numeric(roc_item.get("scores"))
    if y_true and scores and len(y_true) == len(scores) and _is_missing_numeric(row.get("pr_auc")):
        row["pr_auc"] = _average_precision_from_scores(y_true, scores)
    if y_true and _is_missing_numeric(row.get("brier_score")):
        probability_scores = _probability_scores_for_brier(roc_item, expected_length=len(y_true))
        row["brier_score"] = _brier_score_from_probabilities(y_true, probability_scores)
    if y_true and _is_missing_numeric(row.get("f1")):
        row["f1"] = _derive_f1_from_rates(row, y_true)
    if isinstance(decomposition_metrics, dict):
        for col in BIAS_VARIANCE_NOISE_COLUMNS:
            if _is_missing_numeric(row.get(col)):
                row[col] = decomposition_metrics.get(col)
    return row


def _enrich_payload_result_rows(
    rows: list[Dict[str, object]],
    roc_payload: list[Dict[str, object]],
    *,
    yearly: bool,
) -> list[Dict[str, object]]:
    enriched = [dict(row) for row in rows if isinstance(row, dict)]
    roc_items = [dict(item) for item in roc_payload if isinstance(item, dict)]
    if not enriched or not roc_items:
        return enriched
    decomposition_lookup = build_bias_variance_noise_lookup(roc_items)

    if yearly:
        exact_lookup: dict[tuple[object, ...], Dict[str, object]] = {}
        grouped_lookup: dict[tuple[object, ...], list[Dict[str, object]]] = {}
        for item in roc_items:
            common_key = _roc_metric_key(item)
            grouped_lookup.setdefault(common_key, []).append(item)
            exact_lookup[common_key + (str(item.get("segment") or ""),)] = item
        for row in enriched:
            common_key = _row_metric_key(row)
            match: Optional[Dict[str, object]] = None
            prediction_year = _maybe_int(row.get("prediction_year"))
            if prediction_year is not None:
                match = exact_lookup.get(common_key + (str(prediction_year),))
            if match is None and len(grouped_lookup.get(common_key, [])) == 1:
                match = grouped_lookup[common_key][0]
            _apply_derived_metrics(
                row,
                match,
                decomposition_metrics=decomposition_lookup.get(
                    drift_row_group_key(row, yearly=True)
                ),
            )
        return enriched

    grouped_rocs: dict[tuple[object, ...], list[Dict[str, object]]] = {}
    grouped_rows: dict[tuple[object, ...], list[int]] = {}
    for item in roc_items:
        grouped_rocs.setdefault(_roc_metric_key(item), []).append(item)
    for idx, row in enumerate(enriched):
        grouped_rows.setdefault(_row_metric_key(row), []).append(idx)
    for key, row_indices in grouped_rows.items():
        candidates = grouped_rocs.get(key, [])
        for ordinal, row_idx in enumerate(row_indices):
            match = candidates[ordinal] if ordinal < len(candidates) else None
            _apply_derived_metrics(
                enriched[row_idx],
                match,
                decomposition_metrics=decomposition_lookup.get(
                    drift_row_group_key(enriched[row_idx], yearly=False)
                ),
            )
    return enriched


def _list_drift_manifest_files() -> list[Path]:
    if not DRIFT_RUNS_DIR.exists():
        return []
    return sorted(
        DRIFT_RUNS_DIR.glob("*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _list_paper_replication_manifest_files() -> list[Path]:
    if not NLP_PAPER_RUNS_DIR.exists():
        return []
    return sorted(
        NLP_PAPER_RUNS_DIR.glob("*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _list_neural_drift_experiment_manifest_files() -> list[Path]:
    if not NEURAL_DRIFT_EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        NEURAL_DRIFT_EXPERIMENTS_DIR.glob("*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _list_language_modeling_manifest_files() -> list[Path]:
    if not NLP_LANGUAGE_MODELING_LIVE_DIR.exists():
        return []
    return sorted(
        NLP_LANGUAGE_MODELING_LIVE_DIR.glob("*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _build_live_sources() -> list[Dict[str, object]]:
    entries: list[Dict[str, object]] = []
    for path in _list_drift_manifest_files():
        manifest = _load_json_file(path, default={})
        run_id = str((manifest or {}).get("run_id") or path.parent.name)
        status = str((manifest or {}).get("status") or "unknown")
        updated_at = str((manifest or {}).get("updated_at") or (manifest or {}).get("started_at") or "-")
        entries.append(
            {
                "type": "drift_recalibration",
                "path": path,
                "sort_key": float(path.stat().st_mtime),
                "label": f"Drift recalibration | {run_id} | {status} | {updated_at}",
            }
        )
    for path in _list_paper_replication_manifest_files():
        manifest = _load_json_file(path, default={})
        run_id = str((manifest or {}).get("run_id") or path.parent.name)
        status = str((manifest or {}).get("status") or "unknown")
        updated_at = str((manifest or {}).get("updated_at") or (manifest or {}).get("created_at") or "-")
        entries.append(
            {
                "type": "paper_replication",
                "path": path,
                "sort_key": float(path.stat().st_mtime),
                "label": f"Paper replication | {run_id} | {status} | {updated_at}",
            }
        )
    for path in _list_neural_drift_experiment_manifest_files():
        manifest = _load_json_file(path, default={})
        run_id = str((manifest or {}).get("run_id") or path.parent.name)
        status = str((manifest or {}).get("status") or "unknown")
        updated_at = str((manifest or {}).get("updated_at") or (manifest or {}).get("created_at") or "-")
        entries.append(
            {
                "type": "neural_drift_experiment",
                "path": path,
                "sort_key": float(path.stat().st_mtime),
                "label": f"Neural drift experiments | {run_id} | {status} | {updated_at}",
            }
        )
    for path in _list_language_modeling_manifest_files():
        manifest = _load_json_file(path, default={})
        run_id = str((manifest or {}).get("run_id") or path.parent.name)
        run_type = str((manifest or {}).get("run_type") or "language_modeling")
        status = str((manifest or {}).get("status") or "unknown")
        updated_at = str((manifest or {}).get("updated_at") or (manifest or {}).get("created_at") or "-")
        entries.append(
            {
                "type": "language_modeling",
                "path": path,
                "sort_key": float(path.stat().st_mtime),
                "label": f"Language modeling | {run_type} | {run_id} | {status} | {updated_at}",
            }
        )
    for path in _list_live_db_files():
        entries.append(
            {
                "type": "sqlite",
                "path": path,
                "sort_key": float(path.stat().st_mtime),
                "label": f"SQLite | {path.name}",
            }
        )
    entries.sort(key=lambda item: float(item.get("sort_key", 0.0)), reverse=True)
    return entries


def _drift_strategy_priority(name: object) -> int:
    priority = {
        "static": 0,
        "period_aligned": 1,
        "cumulative": 2,
        "adaptive_adwin": 3,
        "adaptive_arf": 4,
        "adaptive_kswin": 5,
    }
    return int(priority.get(str(name), 99))


def _drift_model_priority(name: object) -> int:
    priority = {
        "NNet": 0,
        "AdaBoost": 1,
        "Random Forest": 2,
        "XGBoost": 3,
    }
    return int(priority.get(str(name), 99))


def _drift_block_sort_key(row: Dict[str, object]) -> tuple[int, int, int, int, int, str, str]:
    balance = str(row.get("balance_mode") or "not_applicable")
    balance_priority = {"none": 0, "not_applicable": 0, "smote": 1}
    return (
        _safe_int(row.get("run_order")),
        _safe_int(row.get("run_seed")),
        _drift_strategy_priority(row.get("strategy")),
        int(balance_priority.get(balance, 9)),
        _drift_model_priority(row.get("model")),
        str(row.get("model") or ""),
        str(row.get("detector_variant") or ""),
    )


def _json_cell(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    except Exception:
        return str(value)


def _parse_jsonish_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _jsonish_to_text(value: object) -> str:
    parsed = _parse_jsonish_cell(value)
    if isinstance(parsed, list):
        return ", ".join(str(item) for item in parsed)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    if parsed is None:
        return ""
    return str(parsed)


def _coerce_confusion_matrix_cell(value: object) -> Optional[list[list[int]]]:
    parsed = _parse_jsonish_cell(value)
    if isinstance(parsed, list) and len(parsed) == 4 and not isinstance(
        parsed[0], (list, tuple)
    ):
        try:
            tn, fp, fn, tp = [int(item) for item in parsed]
            return [[tn, fp], [fn, tp]]
        except Exception:
            return None
    if (
        isinstance(parsed, list)
        and len(parsed) == 2
        and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in parsed)
    ):
        try:
            return [
                [int(parsed[0][0]), int(parsed[0][1])],
                [int(parsed[1][0]), int(parsed[1][1])],
            ]
        except Exception:
            return None
    return None


def _confusion_matrix_text(value: object) -> str:
    matrix = _coerce_confusion_matrix_cell(value)
    if matrix is None:
        return _jsonish_to_text(value)
    return json.dumps(matrix, ensure_ascii=False)


def _first_present_metric(row: Dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _controlled_live_best_payload(
    row: Dict[str, object],
    *,
    objective_label: str,
) -> Dict[str, object]:
    return {
        "model_name": row.get("model_name"),
        "feature_set": row.get("feature_set"),
        "balance_mode": row.get("balance_mode"),
        "k": row.get("k", row.get("k_optimo")),
        "objective": row.get("objective_label") or objective_label,
        "val_objective_score": row.get("val_objective_score"),
        "test_objective_score": row.get("test_objective_score"),
        "test_accuracy": _first_present_metric(row, "test_accuracy", "best_test_accuracy"),
        "test_recall": _first_present_metric(row, "test_recall", "best_test_recall"),
        "test_sensitivity": _first_present_metric(
            row,
            "test_sensitivity",
            "best_test_sensitivity",
        ),
        "test_f1_global": _first_present_metric(
            row,
            "test_f1_global",
            "best_test_f1_global",
        ),
        "test_f1_class_0": _first_present_metric(
            row,
            "test_f1_class_0",
            "best_test_f1_class_0",
        ),
        "test_f1_class_1": _first_present_metric(
            row,
            "test_f1_class_1",
            "best_test_f1_class_1",
        ),
        "test_false_negatives": _first_present_metric(
            row,
            "test_false_negatives",
            "best_test_false_negatives",
        ),
        "test_false_positives": _first_present_metric(
            row,
            "test_false_positives",
            "best_test_false_positives",
        ),
        "test_roc_auc": _first_present_metric(row, "test_roc_auc", "best_test_roc_auc"),
        "test_pr_auc": _first_present_metric(row, "test_pr_auc", "best_test_pr_auc"),
        "test_mcc": _first_present_metric(row, "test_mcc", "best_test_mcc"),
        "decision_threshold": row.get("decision_threshold"),
    }


def _artifact_context_dict(artifact: Dict[str, object]) -> Dict[str, object]:
    context = artifact.get("strategy_context")
    if isinstance(context, dict):
        return dict(context)
    if isinstance(context, str) and context.strip():
        try:
            parsed = json.loads(context)
        except Exception:
            return {"raw": context}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {}


def _flatten_mapping_columns(payload: object, prefix: str) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    rows: Dict[str, object] = {}
    for key, value in sorted(payload.items(), key=lambda item: str(item[0])):
        col = f"{prefix}_{str(key)}"
        if isinstance(value, dict):
            nested = _flatten_mapping_columns(value, col)
            if nested:
                rows.update(nested)
            else:
                rows[col] = "{}"
            continue
        if isinstance(value, (list, tuple, set)):
            rows[col] = _json_cell(list(value))
            continue
        rows[col] = value
    return rows


def _build_drift_tuning_trials_frame(artifacts: list[Dict[str, object]]) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        study_id = str(artifact.get("study_id") or "")
        model = str(artifact.get("model_name") or artifact.get("model") or "")
        balance_mode = str(artifact.get("balance_mode") or "not_applicable")
        stage = str(artifact.get("stage") or "tuning")
        for trial in artifact.get("trials") or []:
            if not isinstance(trial, dict):
                continue
            rows.append(
                {
                    "study_id": study_id,
                    "model": model,
                    "balance_mode": balance_mode,
                    "stage": stage,
                    "trial_number": _safe_int(trial.get("trial_number"), default=len(rows)),
                    "cv_auc": pd.to_numeric(trial.get("cv_auc"), errors="coerce"),
                    "state": str(trial.get("state") or ""),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["study_id", "model", "balance_mode", "stage", "trial_number", "cv_auc", "state"]
        )
    return pd.DataFrame(rows).sort_values(
        ["model", "balance_mode", "stage", "trial_number"]
    ).reset_index(drop=True)


def _build_drift_tuning_params_frame(artifacts: list[Dict[str, object]]) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        context = _artifact_context_dict(artifact)
        best_params = dict(artifact.get("best_params") or {})
        row: Dict[str, object] = {
            "study_id": str(artifact.get("study_id") or ""),
            "tuning_key": str(artifact.get("tuning_key") or ""),
            "model": str(artifact.get("model_name") or artifact.get("model") or ""),
            "balance_mode": str(artifact.get("balance_mode") or "not_applicable"),
            "stage": str(artifact.get("stage") or "tuning"),
            "best_cv_auc": pd.to_numeric(artifact.get("best_value"), errors="coerce"),
            "n_trials": _safe_int(artifact.get("n_trials"), default=0),
            "requested_trials": _safe_int(artifact.get("requested_trials"), default=0),
            "search_space_size": _safe_int(artifact.get("search_space_size"), default=0),
            "invalid_trial_count": _safe_int(artifact.get("invalid_trial_count"), default=0),
            "has_valid_trial": bool(artifact.get("has_valid_trial", False)),
            "n_train": _safe_int(artifact.get("n_train"), default=0),
            "positive_rows": _safe_int(artifact.get("positive_rows"), default=0),
            "positive_rate": pd.to_numeric(artifact.get("positive_rate"), errors="coerce"),
            "train_signature": str(artifact.get("train_signature") or ""),
            "best_params_json": _json_cell(best_params),
        }
        row.update(_flatten_mapping_columns(context, "ctx"))
        row.update(_flatten_mapping_columns(best_params, "param"))
        rows.append(row)

    base_columns = [
        "study_id",
        "tuning_key",
        "model",
        "balance_mode",
        "stage",
        "best_cv_auc",
        "best_cv_auc_none",
        "cv_auc_delta_vs_none",
        "n_trials",
        "requested_trials",
        "search_space_size",
        "invalid_trial_count",
        "has_valid_trial",
        "n_train",
        "positive_rows",
        "positive_rate",
        "train_signature",
        "best_params_json",
    ]
    if not rows:
        return pd.DataFrame(columns=base_columns)

    out = pd.DataFrame(rows)
    compare_cols = [
        "model",
        "stage",
        "train_signature",
        "ctx_strategy",
        "ctx_window_kind",
        "ctx_training_year",
        "ctx_prediction_year",
        "ctx_training_years",
        "n_train",
        "positive_rows",
    ]
    for col in compare_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out["_compare_key"] = out[compare_cols].astype("string").fillna("").agg("|".join, axis=1)
    none_ref = (
        out.loc[out["balance_mode"].astype(str).isin(["none", "not_applicable"])]
        .groupby("_compare_key", dropna=False)["best_cv_auc"]
        .max()
    )
    out["best_cv_auc_none"] = out["_compare_key"].map(none_ref)
    out["cv_auc_delta_vs_none"] = out["best_cv_auc"] - out["best_cv_auc_none"]
    balance_priority = {"none": 0, "not_applicable": 0, "smote": 1}
    out["_balance_priority"] = out["balance_mode"].map(balance_priority).fillna(9)

    context_cols = sorted(
        col for col in out.columns if col.startswith("ctx_")
    )
    param_cols = sorted(
        col for col in out.columns if col.startswith("param_")
    )
    ordered_cols = base_columns[:5] + context_cols + base_columns[5:] + param_cols
    for col in ordered_cols:
        if col not in out.columns:
            out[col] = pd.NA
    return (
        out[ordered_cols + ["_balance_priority"]]
        .sort_values(
            ["model", "stage", *([col for col in context_cols if col in out.columns]), "_balance_priority", "best_cv_auc"],
            ascending=[True] * (3 + len(context_cols)) + [False],
            na_position="last",
        )
        .drop(columns=["_balance_priority"])
        .reset_index(drop=True)
    )


def _build_drift_execution_memory_trace(execution_log: pd.DataFrame) -> pd.DataFrame:
    if execution_log is None or execution_log.empty:
        return pd.DataFrame(columns=["order", "metric", "value", "phase", "status"])
    work = execution_log.copy()
    for col in ["order", "rss_before_mb", "rss_after_mb", "training_time_sec"]:
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = pd.to_numeric(work[col], errors="coerce")
    rows: list[Dict[str, object]] = []
    for metric in ["rss_before_mb", "rss_after_mb", "training_time_sec"]:
        metric_rows = work.loc[work[metric].notna(), ["order", "phase", "status", metric]].copy()
        if metric_rows.empty:
            continue
        metric_rows = metric_rows.rename(columns={metric: "value"})
        metric_rows["metric"] = metric
        rows.extend(metric_rows.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=["order", "metric", "value", "phase", "status"])
    return pd.DataFrame(rows).sort_values(["metric", "order"]).reset_index(drop=True)


def _build_drift_average_roc_curves(
    roc_payload: list[Dict[str, object]],
    *,
    n_points: int = 101,
) -> pd.DataFrame:
    if not roc_payload:
        return pd.DataFrame(columns=["strategy", "model", "balance_mode", "fpr", "tpr", "label"])

    fpr_grid = pd.Series([idx / float(max(1, n_points - 1)) for idx in range(int(n_points))], dtype=float)
    rows: list[Dict[str, object]] = []
    keys = sorted(
        {
            (
                str(item.get("strategy") or ""),
                str(item.get("model") or ""),
                str(item.get("balance_mode") or "not_applicable"),
            )
            for item in roc_payload
            if isinstance(item, dict)
        }
    )
    for strategy, model, balance_mode in keys:
        curves: list[pd.Series] = []
        for item in roc_payload:
            if not isinstance(item, dict):
                continue
            if (
                str(item.get("strategy") or "") != strategy
                or str(item.get("model") or "") != model
                or str(item.get("balance_mode") or "not_applicable") != balance_mode
            ):
                continue
            y_true = pd.Series(item.get("y_true") or [], dtype="float64").dropna().astype(int)
            scores = pd.Series(item.get("scores") or [], dtype="float64").dropna().astype(float)
            if y_true.empty or scores.empty or len(y_true) != len(scores) or y_true.nunique() < 2:
                continue
            roc_df = pd.DataFrame({"y_true": y_true.to_numpy(), "scores": scores.to_numpy()}).sort_values(
                "scores",
                ascending=False,
                kind="stable",
            )
            positives = float((roc_df["y_true"] == 1).sum())
            negatives = float((roc_df["y_true"] == 0).sum())
            if positives <= 0 or negatives <= 0:
                continue
            roc_df["tp"] = (roc_df["y_true"] == 1).cumsum()
            roc_df["fp"] = (roc_df["y_true"] == 0).cumsum()
            roc_df["tpr"] = roc_df["tp"] / positives
            roc_df["fpr"] = roc_df["fp"] / negatives
            curve_df = pd.concat(
                [
                    pd.DataFrame({"fpr": [0.0], "tpr": [0.0]}),
                    roc_df[["fpr", "tpr"]],
                    pd.DataFrame({"fpr": [1.0], "tpr": [1.0]}),
                ],
                ignore_index=True,
            ).drop_duplicates(subset=["fpr"], keep="last")
            interpolated = (
                curve_df.set_index("fpr")["tpr"]
                .reindex(sorted(set(curve_df["fpr"]).union(set(fpr_grid.tolist()))))
                .interpolate(method="index")
                .reindex(fpr_grid.tolist())
            )
            if interpolated.empty:
                continue
            curves.append(interpolated.reset_index(drop=True))
        if not curves:
            continue
        mean_tpr = pd.concat(curves, axis=1).mean(axis=1)
        label = f"{strategy} | {model} | {balance_mode}"
        for fpr_value, tpr_value in zip(fpr_grid.tolist(), mean_tpr.tolist()):
            rows.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "balance_mode": balance_mode,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "label": label,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["strategy", "model", "balance_mode", "fpr", "tpr", "label"])
    return pd.DataFrame(rows)


def _build_drift_partial_summary(yearly_df: pd.DataFrame, adaptive_df: pd.DataFrame) -> pd.DataFrame:
    calibration_cols = [
        "sensitivity_before_calibration",
        "specificity_before_calibration",
        "sensitivity_after_calibration",
        "specificity_after_calibration",
    ]
    ordered_cols = [
        "strategy",
        "model",
        "detector_variant",
        "balance_mode",
        "auc",
        "pr_auc",
        "brier_score",
        "bias2",
        "variance",
        "noise",
        "f1",
        "sensitivity",
        "specificity",
        "sensitivity_before_calibration",
        "specificity_before_calibration",
        "sensitivity_after_calibration",
        "specificity_after_calibration",
        "error_rate",
        "training_time_sec",
        "n_segments",
        "n_repetitions",
    ]
    frames: list[pd.DataFrame] = []

    for df in [yearly_df, adaptive_df]:
        if df is None or df.empty:
            continue
        work = df.copy()
        if "model" not in work.columns:
            continue
        if "balance_mode" not in work.columns:
            work["balance_mode"] = "not_applicable"
        if "run_seed" not in work.columns:
            work["run_seed"] = pd.NA
        if "detector_variant" not in work.columns:
            work["detector_variant"] = "-"
        work["detector_variant"] = (
            work["detector_variant"].astype(str).replace({"": "-", "nan": "-"})
        )
        if "base_model" in work.columns:
            base_model = work["base_model"].astype(str)
            actual_model = work["model"].astype(str)
            work["summary_model"] = actual_model.where(
                base_model.str.strip().eq("") | base_model.eq("nan"),
                base_model,
            )
        else:
            work["summary_model"] = work["model"].astype(str)
        for col in calibration_cols:
            if col not in work.columns:
                work[col] = pd.NA
        for col in BIAS_VARIANCE_NOISE_COLUMNS:
            if col not in work.columns:
                work[col] = pd.NA
        for col in [
            "auc",
            "pr_auc",
            "brier_score",
            "bias2",
            "variance",
            "noise",
            "f1",
            "sensitivity",
            "specificity",
            "sensitivity_before_calibration",
            "specificity_before_calibration",
            "sensitivity_after_calibration",
            "specificity_after_calibration",
            "error_rate",
            "training_time_sec",
            "run_seed",
        ]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        grouped = (
            work.groupby(["strategy", "summary_model", "detector_variant", "balance_mode"], dropna=False)
            .agg(
                auc=("auc", "mean"),
                pr_auc=("pr_auc", "mean"),
                brier_score=("brier_score", "mean"),
                bias2=("bias2", "mean"),
                variance=("variance", "mean"),
                noise=("noise", "mean"),
                f1=("f1", "mean"),
                sensitivity=("sensitivity", "mean"),
                specificity=("specificity", "mean"),
                sensitivity_before_calibration=("sensitivity_before_calibration", "mean"),
                specificity_before_calibration=("specificity_before_calibration", "mean"),
                sensitivity_after_calibration=("sensitivity_after_calibration", "mean"),
                specificity_after_calibration=("specificity_after_calibration", "mean"),
                error_rate=("error_rate", "mean"),
                training_time_sec=("training_time_sec", "mean"),
                n_segments=("model", "size"),
                n_repetitions=(
                    "run_seed",
                    lambda s: int(s.dropna().astype(int).nunique()) if s.notna().any() else 1,
                ),
            )
            .reset_index()
        )
        grouped = grouped.rename(columns={"summary_model": "model"})
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=ordered_cols)

    out = pd.concat(frames, ignore_index=True)
    for col in ordered_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out["_strategy_priority"] = out["strategy"].map(_drift_strategy_priority).fillna(99)
    out["_model_priority"] = out["model"].map(_drift_model_priority).fillna(99)
    return out[ordered_cols + ["_strategy_priority", "_model_priority"]].sort_values(
        ["_strategy_priority", "_model_priority", "model", "detector_variant", "balance_mode"]
    ).drop(columns=["_strategy_priority", "_model_priority"]).reset_index(drop=True)


def _display_cell(value: object, *, pending: bool = False) -> object:
    if pending:
        return "Pendiente"
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if value.strip() else "-"
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return "-"
        return value.isoformat(sep=" ", timespec="seconds")
    if pd.isna(value):
        return "-"
    if isinstance(value, bool):
        return value
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return value
    if not math.isfinite(numeric_value):
        return str(numeric_value)
    if abs(numeric_value - round(numeric_value)) < 1e-9:
        return int(round(numeric_value))
    return round(numeric_value, 4)
    return value


def _yearly_training_label(strategy: str, *, base_year: Optional[int], prediction_year: int) -> str:
    if base_year is None:
        return "-"
    if strategy == "static":
        return f"{int(base_year)}"
    if strategy == "period_aligned":
        return f"{int(prediction_year) - 1}"
    if strategy == "cumulative":
        return f"[<= {int(prediction_year) - 1}]"
    return "-"


def _expected_yearly_table(
    manifest: Dict[str, object],
    yearly_df: pd.DataFrame,
    *,
    strategy: str,
) -> pd.DataFrame:
    run_manifest = dict(manifest.get("run_manifest") or {})
    selected_strategies = [str(item) for item in run_manifest.get("strategies") or []]
    if strategy not in selected_strategies and (
        yearly_df is None
        or yearly_df.empty
        or not yearly_df["strategy"].astype(str).eq(strategy).any()
    ):
        return pd.DataFrame()

    base_year_raw = run_manifest.get("base_year")
    base_year = None if pd.isna(pd.to_numeric(base_year_raw, errors="coerce")) else _safe_int(base_year_raw)
    prediction_years = [
        _safe_int(year)
        for year in (run_manifest.get("prediction_years") or [])
        if not pd.isna(pd.to_numeric(year, errors="coerce"))
    ]
    models = [str(item) for item in run_manifest.get("models") or []]
    balance_modes = [str(item) for item in run_manifest.get("balance_modes") or []]
    repetition_seeds = [_safe_int(item) for item in (run_manifest.get("repetition_seeds") or [])]

    if not prediction_years and isinstance(yearly_df, pd.DataFrame) and not yearly_df.empty and "prediction_year" in yearly_df.columns:
        prediction_years = sorted(
            yearly_df["prediction_year"].dropna().astype(int).unique().tolist()
        )
    if not models and isinstance(yearly_df, pd.DataFrame) and not yearly_df.empty:
        models = sorted(yearly_df["model"].dropna().astype(str).unique().tolist())
    if not balance_modes and isinstance(yearly_df, pd.DataFrame) and not yearly_df.empty:
        balance_modes = sorted(yearly_df["balance_mode"].dropna().astype(str).unique().tolist())
    if not repetition_seeds and isinstance(yearly_df, pd.DataFrame) and not yearly_df.empty and "run_seed" in yearly_df.columns:
        repetition_seeds = sorted(yearly_df["run_seed"].dropna().astype(int).unique().tolist())

    display_cols = [
        "status",
        "iteration",
        "training_year",
        "prediction_year",
        "model",
        "balance_mode",
        "auc",
        "pr_auc",
        "brier_score",
        "bias2",
        "variance",
        "noise",
        "f1",
        "sensitivity",
        "specificity",
        "sensitivity_before_calibration",
        "specificity_before_calibration",
        "sensitivity_after_calibration",
        "specificity_after_calibration",
        "error_rate",
        "training_time_sec",
        "threshold",
        "n_train",
        "n_test",
        "run_seed",
        "run_order",
    ]
    if not models or not prediction_years or not repetition_seeds:
        return pd.DataFrame(columns=display_cols)

    work = pd.DataFrame() if yearly_df is None else yearly_df.copy()
    if not work.empty:
        for col in ["prediction_year", "run_seed", "run_order"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

    rows: list[Dict[str, object]] = []
    for run_order, seed in enumerate(repetition_seeds, start=1):
        for prediction_year in prediction_years:
            for model in models:
                for balance_mode in balance_modes:
                    match = pd.DataFrame()
                    if not work.empty:
                        mask = (
                            work["strategy"].astype(str).eq(strategy)
                            & work["model"].astype(str).eq(model)
                            & work["balance_mode"].astype(str).eq(balance_mode)
                            & pd.to_numeric(work["prediction_year"], errors="coerce").eq(int(prediction_year))
                            & pd.to_numeric(work["run_seed"], errors="coerce").eq(int(seed))
                            & pd.to_numeric(work["run_order"], errors="coerce").eq(int(run_order))
                        )
                        match = work.loc[mask].head(1)
                    if not match.empty:
                        source_row = match.iloc[0].to_dict()
                        row = {
                            "status": "Completado",
                            "iteration": _display_cell(source_row.get("iteration")),
                            "training_year": _display_cell(source_row.get("training_year")),
                            "prediction_year": _display_cell(source_row.get("prediction_year")),
                            "model": _display_cell(source_row.get("model")),
                            "balance_mode": _display_cell(source_row.get("balance_mode")),
                            "auc": _display_cell(source_row.get("auc")),
                            "pr_auc": _display_cell(source_row.get("pr_auc")),
                            "brier_score": _display_cell(source_row.get("brier_score")),
                            "bias2": _display_cell(source_row.get("bias2")),
                            "variance": _display_cell(source_row.get("variance")),
                            "noise": _display_cell(source_row.get("noise")),
                            "f1": _display_cell(source_row.get("f1")),
                            "sensitivity": _display_cell(source_row.get("sensitivity")),
                            "specificity": _display_cell(source_row.get("specificity")),
                            "sensitivity_before_calibration": _display_cell(source_row.get("sensitivity_before_calibration")),
                            "specificity_before_calibration": _display_cell(source_row.get("specificity_before_calibration")),
                            "sensitivity_after_calibration": _display_cell(source_row.get("sensitivity_after_calibration")),
                            "specificity_after_calibration": _display_cell(source_row.get("specificity_after_calibration")),
                            "error_rate": _display_cell(source_row.get("error_rate")),
                            "training_time_sec": _display_cell(source_row.get("training_time_sec")),
                            "threshold": _display_cell(source_row.get("threshold")),
                            "n_train": _display_cell(source_row.get("n_train")),
                            "n_test": _display_cell(source_row.get("n_test")),
                            "run_seed": _display_cell(source_row.get("run_seed")),
                            "run_order": _display_cell(source_row.get("run_order")),
                        }
                    else:
                        row = {
                            "status": "Pendiente",
                            "iteration": _display_cell(
                                None if base_year is None else int(prediction_year) - int(base_year)
                            ),
                            "training_year": _yearly_training_label(
                                strategy,
                                base_year=base_year,
                                prediction_year=int(prediction_year),
                            ),
                            "prediction_year": int(prediction_year),
                            "model": model,
                            "balance_mode": balance_mode,
                            "auc": "Pendiente",
                            "pr_auc": "Pendiente",
                            "brier_score": "Pendiente",
                            "bias2": "Pendiente",
                            "variance": "Pendiente",
                            "noise": "Pendiente",
                            "f1": "Pendiente",
                            "sensitivity": "Pendiente",
                            "specificity": "Pendiente",
                            "sensitivity_before_calibration": "Pendiente",
                            "specificity_before_calibration": "Pendiente",
                            "sensitivity_after_calibration": "Pendiente",
                            "specificity_after_calibration": "Pendiente",
                            "error_rate": "Pendiente",
                            "training_time_sec": "Pendiente",
                            "threshold": "Pendiente",
                            "n_train": "Pendiente",
                            "n_test": "Pendiente",
                            "run_seed": int(seed),
                            "run_order": int(run_order),
                        }
                    rows.append(row)

    return pd.DataFrame(rows, columns=display_cols)


def _expected_adaptive_table(
    manifest: Dict[str, object],
    block_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
) -> pd.DataFrame:
    run_manifest = dict(manifest.get("run_manifest") or {})
    selected_strategies = [str(item) for item in run_manifest.get("strategies") or []]
    adaptive_strategies = [
        strategy
        for strategy in ["adaptive_adwin", "adaptive_arf", "adaptive_kswin"]
        if strategy in selected_strategies
    ]

    display_cols = [
        "status",
        "strategy",
        "drift",
        "drift_date",
        "prediction_year",
        "balance_mode",
        "segment_rows",
        "n_internal_drifts",
        "n_internal_warnings",
        "vote_count",
        "vote_threshold",
        "monitor_feature_count",
        "retrain_rows",
        "retrain_positive_rows",
        "base_model",
        "detector_variant",
        "detected_features",
        "monitored_features",
        "W",
        "W0",
        "W1",
        "remaining_periods",
        "model",
        "auc",
        "pr_auc",
        "brier_score",
        "bias2",
        "variance",
        "noise",
        "f1",
        "sensitivity",
        "specificity",
        "sensitivity_before_calibration",
        "specificity_before_calibration",
        "sensitivity_after_calibration",
        "specificity_after_calibration",
        "error_rate",
        "training_time_sec",
        "threshold",
        "run_seed",
        "run_order",
        "error_message",
    ]

    rows: list[Dict[str, object]] = []
    adaptive_work = pd.DataFrame() if adaptive_df is None else adaptive_df.copy()
    if not adaptive_work.empty:
        available_cols = [col for col in display_cols if col in adaptive_work.columns]
        actual_df = adaptive_work[available_cols].copy()
        actual_df.insert(0, "status", "Completado")
        for col in display_cols:
            if col not in actual_df.columns:
                actual_df[col] = "-"
        actual_df = actual_df[display_cols]
        for record in actual_df.to_dict(orient="records"):
            rows.append({key: _display_cell(value) for key, value in record.items()})

    if isinstance(block_df, pd.DataFrame) and not block_df.empty:
        adaptive_row_counts = pd.to_numeric(
            block_df["adaptive_rows"] if "adaptive_rows" in block_df.columns else pd.Series(0, index=block_df.index),
            errors="coerce",
        ).fillna(0)
        completed_empty_blocks = block_df.loc[
            block_df["strategy"].astype(str).isin(adaptive_strategies)
            & block_df["status"].astype(str).str.lower().eq("completed")
            & adaptive_row_counts.eq(0)
        ].copy()
        for block in completed_empty_blocks.to_dict(orient="records"):
            rows.append(
                {
                    "status": "Completado sin filas",
                    "strategy": str(block.get("strategy") or "-"),
                    "drift": "Sin segmentos persistidos",
                    "drift_date": "-",
                    "prediction_year": "-",
                    "balance_mode": str(block.get("balance_mode") or "not_applicable"),
                    "segment_rows": 0,
                    "n_internal_drifts": "-",
                    "n_internal_warnings": "-",
                    "vote_count": "-",
                    "vote_threshold": "-",
                    "monitor_feature_count": "-",
                    "retrain_rows": "-",
                    "retrain_positive_rows": "-",
                    "base_model": str(block.get("model") or "-"),
                    "detector_variant": _display_cell(block.get("detector_variant")),
                    "detected_features": "-",
                    "monitored_features": "-",
                    "W": "-",
                    "W0": "-",
                    "W1": "-",
                    "remaining_periods": "-",
                    "model": str(block.get("model") or "-"),
                    "auc": "-",
                    "pr_auc": "-",
                    "brier_score": "-",
                    "bias2": "-",
                    "variance": "-",
                    "noise": "-",
                    "f1": "-",
                    "sensitivity": "-",
                    "specificity": "-",
                    "sensitivity_before_calibration": "-",
                    "specificity_before_calibration": "-",
                    "sensitivity_after_calibration": "-",
                    "specificity_after_calibration": "-",
                    "error_rate": "-",
                    "training_time_sec": "-",
                    "threshold": "-",
                    "run_seed": _display_cell(block.get("run_seed")),
                    "run_order": _display_cell(block.get("run_order")),
                    "error_message": _display_cell(block.get("error_message")),
                }
            )
        failed_blocks = block_df.loc[
            block_df["strategy"].astype(str).isin(adaptive_strategies)
            & block_df["status"].astype(str).str.lower().isin({"failed", "error", "skipped_failed"})
        ].copy()
        for block in failed_blocks.to_dict(orient="records"):
            rows.append(
                {
                    "status": "Error",
                    "strategy": str(block.get("strategy") or "-"),
                    "drift": "Error en bloque",
                    "drift_date": "-",
                    "prediction_year": "-",
                    "balance_mode": str(block.get("balance_mode") or "not_applicable"),
                    "segment_rows": 0,
                    "n_internal_drifts": "-",
                    "n_internal_warnings": "-",
                    "vote_count": "-",
                    "vote_threshold": "-",
                    "monitor_feature_count": "-",
                    "retrain_rows": "-",
                    "retrain_positive_rows": "-",
                    "base_model": str(block.get("model") or "-"),
                    "detector_variant": _display_cell(block.get("detector_variant")),
                    "detected_features": "-",
                    "monitored_features": "-",
                    "W": "-",
                    "W0": "-",
                    "W1": "-",
                    "remaining_periods": "-",
                    "model": str(block.get("model") or "-"),
                    "auc": "-",
                    "pr_auc": "-",
                    "brier_score": "-",
                    "bias2": "-",
                    "variance": "-",
                    "noise": "-",
                    "f1": "-",
                    "sensitivity": "-",
                    "specificity": "-",
                    "sensitivity_before_calibration": "-",
                    "specificity_before_calibration": "-",
                    "sensitivity_after_calibration": "-",
                    "specificity_after_calibration": "-",
                    "error_rate": "-",
                    "training_time_sec": "-",
                    "threshold": "-",
                    "run_seed": _display_cell(block.get("run_seed")),
                    "run_order": _display_cell(block.get("run_order")),
                    "error_message": _display_cell(block.get("error_message")),
                }
            )
        pending_blocks = block_df.loc[
            block_df["strategy"].astype(str).isin(adaptive_strategies)
            & block_df["status"].astype(str).str.lower().eq("pending")
        ].copy()
        for block in pending_blocks.to_dict(orient="records"):
            rows.append(
                {
                    "status": "Pendiente",
                    "strategy": str(block.get("strategy") or "-"),
                    "drift": "Pendiente",
                    "drift_date": "Pendiente",
                    "prediction_year": "Pendiente",
                    "balance_mode": str(block.get("balance_mode") or "not_applicable"),
                    "segment_rows": "Pendiente",
                    "n_internal_drifts": "Pendiente",
                    "n_internal_warnings": "Pendiente",
                    "vote_count": "Pendiente",
                    "vote_threshold": "Pendiente",
                    "monitor_feature_count": "Pendiente",
                    "retrain_rows": "Pendiente",
                    "retrain_positive_rows": "Pendiente",
                    "base_model": "Pendiente",
                    "detector_variant": _display_cell(block.get("detector_variant")),
                    "detected_features": "Pendiente",
                    "monitored_features": "Pendiente",
                    "W": "Pendiente",
                    "W0": "Pendiente",
                    "W1": "Pendiente",
                    "remaining_periods": "Pendiente",
                    "model": str(block.get("model") or "-"),
                    "auc": "Pendiente",
                    "pr_auc": "Pendiente",
                    "brier_score": "Pendiente",
                    "bias2": "Pendiente",
                    "variance": "Pendiente",
                    "noise": "Pendiente",
                    "f1": "Pendiente",
                    "sensitivity": "Pendiente",
                    "specificity": "Pendiente",
                    "sensitivity_before_calibration": "Pendiente",
                    "specificity_before_calibration": "Pendiente",
                    "sensitivity_after_calibration": "Pendiente",
                    "specificity_after_calibration": "Pendiente",
                    "error_rate": "Pendiente",
                    "training_time_sec": "Pendiente",
                    "threshold": "Pendiente",
                    "run_seed": _display_cell(block.get("run_seed")),
                    "run_order": _display_cell(block.get("run_order")),
                    "error_message": "-",
                }
            )

    if not rows:
        return pd.DataFrame(columns=display_cols)

    out = pd.DataFrame(rows, columns=display_cols)
    sort_cols = [col for col in ["status", "strategy", "model", "balance_mode", "run_seed", "run_order"] if col in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def _build_drift_live_result_tables(
    manifest: Dict[str, object],
    block_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    adaptive_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    return {
        "A.6": _expected_yearly_table(manifest, yearly_df, strategy="static"),
        "A.7": _expected_yearly_table(manifest, yearly_df, strategy="period_aligned"),
        "A.8": _expected_yearly_table(manifest, yearly_df, strategy="cumulative"),
        "A.9": _expected_adaptive_table(manifest, block_df, adaptive_df),
    }


def _read_drift_run(manifest_path: Path) -> Dict[str, object]:
    manifest = _load_json_file(manifest_path, default={}) or {}
    run_dir = manifest_path.parent
    blocks_dir = run_dir / "blocks"
    tuning_dir = run_dir / "tuning"
    live_status_path = run_dir / "live_status.json"
    live_events_path = run_dir / "live_events.jsonl"

    payloads_by_block: Dict[str, Dict[str, object]] = {}
    block_payloads: list[Dict[str, object]] = []
    if blocks_dir.exists():
        for block_path in sorted(blocks_dir.glob("*.json")):
            payload = _load_json_file(block_path, default=None)
            if not isinstance(payload, dict):
                continue
            block_id = str(payload.get("block_id") or block_path.stem)
            payloads_by_block[block_id] = payload
            block_payloads.append(payload)
    block_payloads.sort(
        key=lambda payload: _drift_block_sort_key(
            {
                "run_order": payload.get("run_order"),
                "run_seed": payload.get("run_seed"),
                **dict(payload.get("block") or {}),
            }
        )
    )

    block_rows: list[Dict[str, object]] = []
    block_index = dict(manifest.get("block_index") or {})
    all_block_ids = sorted(
        set(str(key) for key in block_index.keys()).union(payloads_by_block.keys()),
        key=lambda block_id: _drift_block_sort_key(
            {
                "run_order": (block_index.get(block_id) or {}).get("run_order") or (payloads_by_block.get(block_id) or {}).get("run_order"),
                "run_seed": (block_index.get(block_id) or {}).get("run_seed") or (payloads_by_block.get(block_id) or {}).get("run_seed"),
                **dict((block_index.get(block_id) or {}).copy()),
                **dict((payloads_by_block.get(block_id) or {}).get("block") or {}),
            }
        ),
    )
    for block_id in all_block_ids:
        info = dict(block_index.get(block_id) or {})
        payload = dict(payloads_by_block.get(block_id) or {})
        block_config = dict(payload.get("block") or {})
        payload_execution_log = [
            entry
            for entry in (payload.get("execution_log") or [])
            if isinstance(entry, dict)
        ]
        payload_errors = [
            entry
            for entry in payload_execution_log
            if str(entry.get("status", "")).lower() == "error"
        ]
        inferred_status = str(info.get("status") or ("completed" if payload else "pending"))
        if (
            payload
            and inferred_status == "completed"
            and payload_errors
            and not (payload.get("yearly_rows") or [])
            and not (payload.get("adaptive_rows") or [])
        ):
            inferred_status = "error"
        latest_error = dict(payload_errors[-1]) if payload_errors else {}
        row = {
            "block_id": block_id,
            "status": inferred_status,
            "strategy": str(info.get("strategy") or block_config.get("strategy") or ""),
            "model": str(info.get("model") or block_config.get("model") or ""),
            "detector_variant": str(info.get("detector_variant") or block_config.get("detector_variant") or ""),
            "balance_mode": str(info.get("balance_mode") or block_config.get("balance_mode") or "not_applicable"),
            "run_seed": _safe_int(info.get("run_seed") or payload.get("run_seed")),
            "run_order": _safe_int(info.get("run_order") or payload.get("run_order")),
            "saved_at": payload.get("saved_at"),
            "yearly_rows": int(len(payload.get("yearly_rows") or [])),
            "adaptive_rows": int(len(payload.get("adaptive_rows") or [])),
            "execution_log_rows": int(len(payload.get("execution_log") or [])),
            "error_message": str(latest_error.get("error") or latest_error.get("message") or info.get("error") or ""),
        }
        detector_variant = str(row["detector_variant"] or "")
        balance_mode = str(row["balance_mode"] or "not_applicable")
        parts = [str(row["strategy"]), str(row["model"])]
        if detector_variant:
            parts.append(detector_variant)
        if balance_mode and balance_mode != "not_applicable":
            parts.append(balance_mode)
        row["block_label"] = " | ".join(part for part in parts if part)
        row["seed_label"] = f"seed {int(row['run_seed'])}"
        block_rows.append(row)
    block_df = pd.DataFrame(block_rows)
    if not block_df.empty:
        block_df = block_df.sort_values(
            ["run_order", "run_seed", "strategy", "balance_mode", "model", "detector_variant"]
        ).reset_index(drop=True)

    yearly_frames: list[pd.DataFrame] = []
    adaptive_frames: list[pd.DataFrame] = []
    roc_payload_rows: list[Dict[str, object]] = []
    execution_log_rows = list(manifest.get("global_execution_log") or [])
    for payload in block_payloads:
        roc_payload = [
            item
            for item in (payload.get("roc_payload") or [])
            if isinstance(item, dict)
        ]
        yearly_rows = _enrich_payload_result_rows(
            [row for row in (payload.get("yearly_rows") or []) if isinstance(row, dict)],
            roc_payload,
            yearly=True,
        )
        adaptive_rows = _enrich_payload_result_rows(
            [row for row in (payload.get("adaptive_rows") or []) if isinstance(row, dict)],
            roc_payload,
            yearly=False,
        )
        roc_payload_rows.extend(
            roc_payload
        )
        if yearly_rows:
            yearly_frames.append(pd.DataFrame(yearly_rows))
        if adaptive_rows:
            adaptive_frames.append(pd.DataFrame(adaptive_rows))
        execution_log_rows.extend(list(payload.get("execution_log") or []))
    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    adaptive_df = pd.concat(adaptive_frames, ignore_index=True) if adaptive_frames else pd.DataFrame()
    for idx, row in enumerate(execution_log_rows, start=1):
        row["order"] = idx
    execution_log_df = pd.DataFrame(execution_log_rows)

    tuning_artifacts: list[Dict[str, object]] = []
    if tuning_dir.exists():
        for tuning_path in sorted(tuning_dir.glob("*.json")):
            artifact = _load_json_file(tuning_path, default=None)
            if isinstance(artifact, dict):
                tuning_artifacts.append(artifact)
    tuning_trials_df = _build_drift_tuning_trials_frame(tuning_artifacts)
    summary_df = _build_drift_partial_summary(yearly_df, adaptive_df)
    memory_trace_df = _build_drift_execution_memory_trace(execution_log_df)
    average_roc_df = _build_drift_average_roc_curves(roc_payload_rows)
    tuning_params_df = _build_drift_tuning_params_frame(tuning_artifacts)

    live_status = _load_json_file(live_status_path, default={}) or {}
    live_event_rows = _read_jsonl_records(live_events_path)
    if not live_event_rows and isinstance(live_status, dict) and live_status:
        live_event_rows = [live_status]
    if not live_event_rows and isinstance(manifest, dict) and manifest:
        progress = dict(manifest.get("progress") or {})
        live_event_rows = [
            {
                "timestamp": manifest.get("updated_at") or manifest.get("started_at"),
                "completed_units": progress.get("completed_units", 0.0),
                "total_units": progress.get("total_units", 0),
                "progress_ratio": (
                    float(progress.get("completed_units", 0.0)) / float(max(1, int(progress.get("total_units", 0) or 1)))
                ),
                "label": "Checkpoint state",
                "detail": "",
                "context": {},
            }
        ]
    live_events_df = pd.DataFrame(live_event_rows)
    if not live_events_df.empty:
        if "progress_ratio" not in live_events_df.columns:
            live_events_df["progress_ratio"] = (
                pd.to_numeric(live_events_df.get("completed_units"), errors="coerce")
                / pd.to_numeric(live_events_df.get("total_units"), errors="coerce").clip(lower=1)
            )
        live_events_df["progress_pct"] = 100.0 * pd.to_numeric(live_events_df["progress_ratio"], errors="coerce")
        live_events_df["event_index"] = range(1, len(live_events_df) + 1)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_dir": run_dir,
        "block_df": block_df,
        "yearly_df": yearly_df,
        "adaptive_df": adaptive_df,
        "summary_df": summary_df,
        "execution_log_df": execution_log_df,
        "memory_trace_df": memory_trace_df,
        "roc_payload": roc_payload_rows,
        "average_roc_df": average_roc_df,
        "tuning_trials_df": tuning_trials_df,
        "tuning_params_df": tuning_params_df,
        "tuning_artifacts": tuning_artifacts,
        "live_status": live_status,
        "live_events_df": live_events_df,
        "live_status_path": live_status_path,
        "live_events_path": live_events_path,
    }


def _load_csv_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _neural_drift_experiment_artifact_path(
    run_dir: Path,
    manifest: Dict[str, object],
    artifact_name: str,
    *,
    suffix: str,
) -> Path:
    artifacts = dict(manifest.get("artifacts") or {})
    candidate = str(artifacts.get(artifact_name) or "").strip()
    if candidate:
        return Path(candidate)
    return run_dir / "artifacts" / f"{artifact_name}.{suffix}"


def _normalize_neural_drift_experiment_live_event(
    payload: Dict[str, object],
    *,
    manifest: Dict[str, object],
) -> Dict[str, object]:
    nested = dict(payload.get("payload") or {}) if isinstance(payload.get("payload"), dict) else {}
    manifest_progress = dict(manifest.get("progress") or {})
    context = payload.get("context")
    if not isinstance(context, dict):
        context = nested.get("context")
    if not isinstance(context, dict):
        context = {}

    completed_units = pd.to_numeric(
        payload.get("completed_units", nested.get("completed_units", manifest_progress.get("completed_units", 0.0))),
        errors="coerce",
    )
    total_units = pd.to_numeric(
        payload.get("total_units", nested.get("total_units", manifest_progress.get("total_units", 0.0))),
        errors="coerce",
    )
    progress_ratio = pd.to_numeric(
        payload.get("progress_ratio", nested.get("progress_ratio")),
        errors="coerce",
    )
    if pd.isna(completed_units):
        completed_units = 0.0
    if pd.isna(total_units) or float(total_units) <= 0.0:
        total_units = max(1.0, float(manifest_progress.get("total_units", 1.0) or 1.0))
    if pd.isna(progress_ratio):
        progress_ratio = float(completed_units) / float(max(1.0, float(total_units)))

    return {
        "timestamp": str(payload.get("timestamp") or nested.get("timestamp") or manifest.get("updated_at") or manifest.get("created_at") or ""),
        "event_type": str(payload.get("event") or nested.get("event") or ""),
        "status": str(payload.get("status") or nested.get("status") or manifest.get("status") or ""),
        "result_status": str(payload.get("result_status") or nested.get("result_status") or manifest.get("result_status") or ""),
        "label": str(payload.get("label") or nested.get("label") or payload.get("event") or "checkpoint"),
        "detail": str(payload.get("detail") or nested.get("detail") or ""),
        "completed_units": float(completed_units),
        "total_units": float(total_units),
        "progress_ratio": max(0.0, min(float(progress_ratio), 1.0)),
        "study": str(context.get("study") or nested.get("study") or ""),
        "phase": str(context.get("phase") or nested.get("phase") or ""),
        "context": context,
    }


def _read_neural_drift_experiment_run(manifest_path: Path) -> Dict[str, object]:
    manifest = _load_json_file(manifest_path, default={}) or {}
    run_dir = manifest_path.parent
    live_status_path = run_dir / "live_status.json"
    live_events_path = run_dir / "live_events.jsonl"
    live_status = _load_json_file(live_status_path, default={}) or {}
    live_event_rows = _read_jsonl_records(live_events_path)
    if not live_event_rows and isinstance(live_status, dict) and live_status:
        live_event_rows = [live_status]
    if not live_event_rows and manifest:
        progress = dict(manifest.get("progress") or {})
        live_event_rows = [
            {
                "timestamp": manifest.get("updated_at") or manifest.get("created_at"),
                "completed_units": progress.get("completed_units", 0.0),
                "total_units": progress.get("total_units", 1.0),
                "progress_ratio": float(progress.get("progress_ratio", 0.0) or 0.0),
                "label": "Checkpoint state",
                "detail": "",
                "context": {},
            }
        ]
    normalized_events = [
        _normalize_neural_drift_experiment_live_event(row, manifest=manifest)
        for row in live_event_rows
        if isinstance(row, dict)
    ]
    live_events_df = pd.DataFrame(normalized_events)
    if not live_events_df.empty:
        live_events_df["progress_pct"] = 100.0 * pd.to_numeric(live_events_df["progress_ratio"], errors="coerce")
        live_events_df["event_index"] = range(1, len(live_events_df) + 1)

    leaderboard_dev = _load_csv_file(
        _neural_drift_experiment_artifact_path(run_dir, manifest, "leaderboard_dev", suffix="csv")
    )
    leaderboard_holdout = _load_csv_file(
        _neural_drift_experiment_artifact_path(run_dir, manifest, "leaderboard_holdout", suffix="csv")
    )
    monthly_metrics = _load_csv_file(
        _neural_drift_experiment_artifact_path(run_dir, manifest, "monthly_metrics", suffix="csv")
    )
    pairwise_stats = _load_csv_file(
        _neural_drift_experiment_artifact_path(run_dir, manifest, "pairwise_stats", suffix="csv")
    )
    param_importances = _load_csv_file(
        _neural_drift_experiment_artifact_path(run_dir, manifest, "param_importances", suffix="csv")
    )
    pareto = _load_csv_file(
        _neural_drift_experiment_artifact_path(run_dir, manifest, "pareto", suffix="csv")
    )
    winner_config = _load_json_file(
        _neural_drift_experiment_artifact_path(run_dir, manifest, "winner_config", suffix="json"),
        default={},
    ) or {}

    for df in [leaderboard_dev, leaderboard_holdout, monthly_metrics, pairwise_stats, param_importances, pareto]:
        if isinstance(df, pd.DataFrame):
            for col in ["month", "timestamp"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

    phase_rows: list[Dict[str, object]] = []
    for study_name, study_payload in dict(manifest.get("studies") or {}).items():
        phases = dict((study_payload or {}).get("phases") or {})
        for phase_key, phase_payload in phases.items():
            del phase_key
            budget = _safe_int(phase_payload.get("n_trials_budget"), default=0)
            completed_trials = _safe_int(phase_payload.get("completed_trials"), default=0)
            phase_rows.append(
                {
                    "study": str(study_name),
                    "phase": _safe_int(phase_payload.get("phase"), default=0),
                    "status": str(phase_payload.get("status") or "pending"),
                    "completed_trials": completed_trials,
                    "n_trials_budget": budget,
                    "progress_ratio": float(completed_trials) / float(max(1, budget)) if budget > 0 else float("nan"),
                    "best_value": pd.to_numeric(phase_payload.get("best_value"), errors="coerce"),
                    "best_trial_number": pd.to_numeric(phase_payload.get("best_trial_number"), errors="coerce"),
                    "error": str(phase_payload.get("error") or ""),
                    "storage_path": str(phase_payload.get("storage_path") or ""),
                }
            )
    phase_status_df = pd.DataFrame(phase_rows)
    if not phase_status_df.empty:
        phase_status_df["progress_pct"] = 100.0 * pd.to_numeric(phase_status_df["progress_ratio"], errors="coerce")
        phase_status_df = phase_status_df.sort_values(["study", "phase"]).reset_index(drop=True)

    baseline_seed_rows: list[Dict[str, object]] = []
    baseline = dict(manifest.get("baseline") or {})
    for item in baseline.get("seed_metrics") or []:
        if not isinstance(item, dict):
            continue
        dev = dict(item.get("dev") or {})
        holdout = dict(item.get("holdout") or {})
        baseline_seed_rows.append(
            {
                "seed": _safe_int(item.get("seed"), default=0),
                "dev_score": pd.to_numeric(dev.get("score"), errors="coerce"),
                "dev_monthly_pr_auc_median": pd.to_numeric(dev.get("monthly_pr_auc_median"), errors="coerce"),
                "dev_monthly_pr_auc_std": pd.to_numeric(dev.get("monthly_pr_auc_std"), errors="coerce"),
                "holdout_score": pd.to_numeric(holdout.get("score"), errors="coerce"),
                "holdout_monthly_pr_auc_median": pd.to_numeric(holdout.get("monthly_pr_auc_median"), errors="coerce"),
                "holdout_monthly_pr_auc_std": pd.to_numeric(holdout.get("monthly_pr_auc_std"), errors="coerce"),
            }
        )
    baseline_seed_df = pd.DataFrame(baseline_seed_rows)
    if not baseline_seed_df.empty:
        baseline_seed_df = baseline_seed_df.sort_values("seed").reset_index(drop=True)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_dir": run_dir,
        "live_status": live_status,
        "live_events_df": live_events_df,
        "live_status_path": live_status_path,
        "live_events_path": live_events_path,
        "phase_status_df": phase_status_df,
        "baseline_seed_df": baseline_seed_df,
        "leaderboard_dev": leaderboard_dev,
        "leaderboard_holdout": leaderboard_holdout,
        "monthly_metrics": monthly_metrics,
        "pairwise_stats": pairwise_stats,
        "param_importances": param_importances,
        "pareto": pareto,
        "winner_config": winner_config,
    }


def _paper_route_dir(run_dir: Path, route_name: str) -> Path:
    return run_dir / str(route_name)


def _paper_stage_step_summary(step_df: pd.DataFrame, prefix: str) -> Dict[str, object]:
    if not isinstance(step_df, pd.DataFrame) or step_df.empty or "step_id" not in step_df.columns:
        return {
            "status": "pending",
            "current_step_id": "",
            "status_message": "",
            "completed_steps": 0,
            "total_steps": 0,
        }
    prefix_text = str(prefix)
    stage_mask = step_df["step_id"].astype(str).str.startswith(prefix_text)
    scoped = step_df.loc[stage_mask].copy()
    if scoped.empty:
        return {
            "status": "pending",
            "current_step_id": "",
            "status_message": "",
            "completed_steps": 0,
            "total_steps": 0,
        }
    statuses = scoped["status"].astype(str).str.lower()
    if statuses.eq("failed").any():
        status = "failed"
    elif statuses.eq("blocked").any():
        status = "blocked"
    elif statuses.eq("running").any():
        status = "running"
    elif statuses.eq("completed").all():
        status = "completed"
    elif statuses.eq("completed").any():
        status = "partial"
    else:
        status = "pending"
    scoped = scoped.sort_values(["order", "started_at", "completed_at"], kind="stable").reset_index(drop=True)
    current_row = scoped.iloc[-1].to_dict() if not scoped.empty else {}
    return {
        "status": status,
        "current_step_id": str(current_row.get("step_id") or ""),
        "status_message": str(current_row.get("last_message") or current_row.get("error") or ""),
        "completed_steps": int(statuses.isin(["completed", "blocked"]).sum()),
        "total_steps": int(len(scoped)),
    }


def _paper_normalize_live_event(
    row: Dict[str, object],
    *,
    manifest_progress: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row = dict(row or {})
    progress = dict(row.get("progress") or {})
    base_progress = dict(manifest_progress or {})
    completed_units = pd.to_numeric(
        row.get("completed_units", progress.get("completed_units", base_progress.get("completed_units", 0.0))),
        errors="coerce",
    )
    total_units = pd.to_numeric(
        row.get("total_units", progress.get("total_units", base_progress.get("total_units", 0.0))),
        errors="coerce",
    )
    if pd.isna(completed_units):
        completed_units = 0.0
    if pd.isna(total_units):
        total_units = 0.0
    progress_ratio = pd.to_numeric(row.get("progress_ratio"), errors="coerce")
    if pd.isna(progress_ratio):
        progress_ratio = float(completed_units) / float(max(1.0, float(total_units or 0.0)))
    step_id = str(row.get("step_id") or progress.get("current_step_id") or "")
    stage = str(row.get("stage") or progress.get("current_stage") or "")
    if not stage and step_id:
        stage = step_id.split(".", 1)[0]
    label = str(row.get("label") or step_id or "Checkpoint state")
    detail = str(row.get("detail") or row.get("message") or "")
    timestamp = str(row.get("timestamp") or row.get("updated_at") or row.get("created_at") or "")
    return {
        "timestamp": timestamp,
        "completed_units": float(completed_units),
        "total_units": float(total_units),
        "progress_ratio": max(0.0, min(float(progress_ratio), 1.0)),
        "progress_pct": 100.0 * max(0.0, min(float(progress_ratio), 1.0)),
        "label": label,
        "detail": detail,
        "stage": stage,
        "step_id": step_id,
        "step_status": str(row.get("step_status") or row.get("status") or ""),
        "result_status": str(row.get("result_status") or ""),
        "status": str(row.get("status") or ""),
        "event_type": str(row.get("event_type") or ""),
    }


def _paper_build_step_df(manifest: Dict[str, object]) -> pd.DataFrame:
    steps_index = dict(manifest.get("steps_index") or {})
    step_sequence = list(manifest.get("step_sequence") or steps_index.keys())
    rows: list[Dict[str, object]] = []
    seen: set[str] = set()
    for step_id in [*step_sequence, *sorted(steps_index.keys())]:
        step_key = str(step_id)
        if step_key in seen:
            continue
        seen.add(step_key)
        entry = dict(steps_index.get(step_key) or {})
        artifact_paths = entry.get("artifact_paths") or {}
        rows.append(
            {
                "step_id": step_key,
                "stage": str(entry.get("stage") or ""),
                "description": str(entry.get("description") or ""),
                "status": str(entry.get("status") or "pending"),
                "order": _safe_int(entry.get("order"), default=len(rows) + 1),
                "started_at": entry.get("started_at"),
                "completed_at": entry.get("completed_at"),
                "last_message": str(entry.get("last_message") or ""),
                "error": str(entry.get("error") or ""),
                "artifact_count": len(artifact_paths) if isinstance(artifact_paths, dict) else 0,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "step_id",
                "stage",
                "description",
                "status",
                "order",
                "started_at",
                "completed_at",
                "last_message",
                "error",
                "artifact_count",
            ]
        )
    return pd.DataFrame(rows).sort_values("order", kind="stable").reset_index(drop=True)


def _paper_load_compare_payload(run_dir: Path) -> Dict[str, object]:
    compare_dir = run_dir / "compare"
    payload = _load_pickle_file(compare_dir / "payload.pkl", default=None)
    if isinstance(payload, dict):
        return payload
    summary = _load_json_file(compare_dir / "summary.json", default={}) or {}
    return {
        "status": summary.get("status"),
        "reason": summary.get("reason"),
        "passed": summary.get("passed"),
        "max_numeric_diff": summary.get("max_numeric_diff"),
        "tolerance": summary.get("tolerance"),
        "diff_df": _load_csv_file(compare_dir / "diff.csv"),
    }


def _paper_load_export_payload(run_dir: Path) -> Dict[str, object]:
    export_dir = run_dir / "export"
    payload = _load_pickle_file(export_dir / "final_payload.pkl", default=None)
    if isinstance(payload, dict):
        return payload
    summary = _load_json_file(export_dir / "payload.json", default={}) or {}
    return {
        "candidate_paths": summary.get("candidate_paths") or {},
        "promoted_paths": summary.get("promoted_paths") or {},
        "latex_promoted": bool(summary.get("latex_promoted")),
        "result_status": summary.get("result_status"),
    }


def _paper_collect_route_models(route_dir: Path, route_name: str) -> Tuple[list[Dict[str, object]], pd.DataFrame, pd.DataFrame]:
    model_payloads: list[Dict[str, object]] = []
    model_rows: list[Dict[str, object]] = []
    k_frames: list[pd.DataFrame] = []
    models_root = route_dir / "models"
    model_dirs: list[Path] = []
    for model_code in PAPER_MODEL_CODES:
        model_dir = models_root / model_code
        if model_dir.exists():
            model_dirs.append(model_dir)
    if models_root.exists():
        for model_dir in sorted(models_root.iterdir()):
            if model_dir.is_dir() and model_dir not in model_dirs:
                model_dirs.append(model_dir)

    for model_dir in model_dirs:
        model_code = model_dir.name
        summary = _load_json_file(model_dir / "final_summary.json", default={}) or {}
        metrics = _load_json_file(model_dir / "metrics.json", default={}) or {}
        k_search_df = _load_pickle_file(model_dir / "k_search.pkl", default=pd.DataFrame())
        if not isinstance(k_search_df, pd.DataFrame):
            k_search_df = pd.DataFrame()
        if k_search_df.empty:
            k_result_rows = _read_jsonl_records(model_dir / "k_results.jsonl")
            if not k_result_rows:
                k_result_rows = []
                k_results_dir = model_dir / "k_results"
                if k_results_dir.exists():
                    for k_path in sorted(k_results_dir.glob("k_*.json")):
                        payload = _load_json_file(k_path, default=None)
                        if isinstance(payload, dict):
                            k_result_rows.append(payload)
            if k_result_rows:
                k_search_df = pd.DataFrame(k_result_rows)
        if isinstance(k_search_df, pd.DataFrame) and not k_search_df.empty:
            work_k_df = k_search_df.copy()
            if "route_name" in work_k_df.columns:
                work_k_df["route_name"] = str(route_name)
            else:
                work_k_df.insert(0, "route_name", str(route_name))
            resolved_model_code = str(summary.get("model_code") or model_code)
            resolved_model_title = str(summary.get("model_title") or model_code)
            if "model_code" in work_k_df.columns:
                work_k_df["model_code"] = resolved_model_code
            else:
                work_k_df.insert(1, "model_code", resolved_model_code)
            if "model_title" in work_k_df.columns:
                work_k_df["model_title"] = resolved_model_title
            else:
                insert_loc = 2 if "model_code" in work_k_df.columns else len(work_k_df.columns)
                work_k_df.insert(insert_loc, "model_title", resolved_model_title)
            k_frames.append(work_k_df)

        final_available = bool(summary or metrics)
        if not final_available and k_search_df.empty:
            continue
        class_metrics = metrics.get("class_metrics") or {}
        model_payload = {
            "route_name": str(route_name),
            "model_code": str(summary.get("model_code") or model_code),
            "model_title": str(summary.get("model_title") or model_code),
            "selected_k": _maybe_int(summary.get("selected_k")),
            "feature_group": str(summary.get("feature_group") or ""),
            "candidate_feature_count": _maybe_int(summary.get("candidate_feature_count")),
            "best_cv_score": pd.to_numeric(summary.get("best_cv_score"), errors="coerce"),
            "accuracy": pd.to_numeric(metrics.get("accuracy"), errors="coerce"),
            "precision": pd.to_numeric(metrics.get("precision"), errors="coerce"),
            "recall": pd.to_numeric(metrics.get("recall"), errors="coerce"),
            "f1_score": pd.to_numeric(metrics.get("f1_score"), errors="coerce"),
            "roc_auc": pd.to_numeric(metrics.get("roc_auc"), errors="coerce"),
            "false_negatives_positive_class": _maybe_int(metrics.get("false_negatives_positive_class")),
            "no_marc_f1": pd.to_numeric((class_metrics.get("0") or {}).get("f1_score"), errors="coerce"),
            "marc_f1": pd.to_numeric((class_metrics.get("1") or {}).get("f1_score"), errors="coerce"),
            "optimization_backend": str((summary.get("optimization") or {}).get("backend") or ""),
            "requested_optimization_backend": str((summary.get("optimization") or {}).get("requested_backend") or ""),
            "k_results_count": int(len(k_search_df)) if isinstance(k_search_df, pd.DataFrame) else 0,
            "final_available": bool(final_available),
            "status": "completed" if final_available else "partial",
        }
        model_payloads.append(
            {
                **model_payload,
                "summary": summary,
                "metrics": metrics,
                "k_search_df": k_search_df if isinstance(k_search_df, pd.DataFrame) else pd.DataFrame(),
            }
        )
        model_rows.append(model_payload)

    model_df = pd.DataFrame(model_rows)
    if not model_df.empty:
        model_df = model_df.sort_values(["route_name", "model_code"], kind="stable").reset_index(drop=True)
    k_progress_df = pd.concat(k_frames, ignore_index=True) if k_frames else pd.DataFrame()
    if not k_progress_df.empty and "k" in k_progress_df.columns:
        k_progress_df["k"] = pd.to_numeric(k_progress_df["k"], errors="coerce")
        k_progress_df = k_progress_df.sort_values(["route_name", "model_code", "k"], kind="stable").reset_index(drop=True)
    return model_payloads, model_df, k_progress_df


def _paper_route_status_rows(
    *,
    manifest: Dict[str, object],
    step_df: pd.DataFrame,
    frozen_payload: Dict[str, object],
    raw_payload: Dict[str, object],
    raw_build_payload: Dict[str, object],
    compare_payload: Dict[str, object],
    export_payload: Dict[str, object],
    partial_models_df: pd.DataFrame,
    k_progress_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for route_name, route_payload in (("frozen", frozen_payload), ("raw", raw_payload)):
        validation = route_payload.get("dataset_validation") or {}
        step_summary = _paper_stage_step_summary(step_df, f"{route_name}.")
        route_model_df = (
            partial_models_df.loc[partial_models_df["route_name"].astype(str) == route_name].copy()
            if isinstance(partial_models_df, pd.DataFrame) and not partial_models_df.empty and "route_name" in partial_models_df.columns
            else pd.DataFrame()
        )
        route_k_df = (
            k_progress_df.loc[k_progress_df["route_name"].astype(str) == route_name].copy()
            if isinstance(k_progress_df, pd.DataFrame) and not k_progress_df.empty and "route_name" in k_progress_df.columns
            else pd.DataFrame()
        )
        status = str(route_payload.get("status") or step_summary.get("status") or "pending")
        if status == "completed":
            status = "ok"
        if not route_payload and not route_model_df.empty and status == "pending":
            status = "partial"
        rows.append(
            {
                "stage": route_name,
                "status": status,
                "status_message": str(route_payload.get("status_message") or step_summary.get("status_message") or ""),
                "current_step_id": str(step_summary.get("current_step_id") or ""),
                "rows": validation.get("rows"),
                "flow_features": validation.get("flow_features"),
                "embedding_features": validation.get("embedding_features"),
                "total_features": validation.get("total_features"),
                "completed_models": int(
                    route_model_df["final_available"].fillna(False).astype(bool).sum()
                ) if not route_model_df.empty and "final_available" in route_model_df.columns else 0,
                "expected_models": len(PAPER_MODEL_CODES),
                "k_results": int(len(route_k_df)),
            }
        )

    raw_build_step = _paper_stage_step_summary(step_df, "raw.build.")
    raw_build_embedding_meta = dict(raw_build_payload.get("embedding_meta") or {})
    selected_embedding_cols = raw_build_payload.get("selected_embedding_cols") or []
    raw_build_dataset = raw_build_payload.get("dataset_df")
    rows.append(
        {
            "stage": "raw_build",
            "status": str(raw_build_step.get("status") or "pending"),
            "status_message": str(raw_build_step.get("status_message") or ""),
            "current_step_id": str(raw_build_step.get("current_step_id") or ""),
            "rows": int(len(raw_build_dataset)) if isinstance(raw_build_dataset, pd.DataFrame) else None,
            "flow_features": None,
            "embedding_features": int(raw_build_embedding_meta.get("selected_embedding_count") or len(selected_embedding_cols or [])),
            "total_features": None,
            "completed_models": None,
            "expected_models": None,
            "k_results": None,
        }
    )

    compare_step = _paper_stage_step_summary(step_df, "compare.")
    compare_diff_df = compare_payload.get("diff_df")
    rows.append(
        {
            "stage": "compare",
            "status": str(compare_payload.get("status") or compare_step.get("status") or "pending"),
            "status_message": str(compare_payload.get("reason") or compare_step.get("status_message") or ""),
            "current_step_id": str(compare_step.get("current_step_id") or ""),
            "rows": None,
            "flow_features": None,
            "embedding_features": None,
            "total_features": None,
            "completed_models": int(bool(compare_payload)),
            "expected_models": None,
            "k_results": int(len(compare_diff_df)) if isinstance(compare_diff_df, pd.DataFrame) else 0,
        }
    )

    export_step = _paper_stage_step_summary(step_df, "export.")
    rows.append(
        {
            "stage": "export",
            "status": str(export_payload.get("result_status") or export_step.get("status") or "pending"),
            "status_message": (
                "Assets promovidos a LaTeX."
                if bool(export_payload.get("latex_promoted"))
                else str(export_step.get("status_message") or "")
            ),
            "current_step_id": str(export_step.get("current_step_id") or ""),
            "rows": None,
            "flow_features": None,
            "embedding_features": None,
            "total_features": None,
            "completed_models": int(len(export_payload.get("promoted_paths") or {})),
            "expected_models": int(len(export_payload.get("candidate_paths") or {})),
            "k_results": None,
        }
    )
    return pd.DataFrame(rows)


def _paper_payload_summary(payload: object) -> object:
    if isinstance(payload, pd.DataFrame):
        return {
            "type": "DataFrame",
            "rows": int(len(payload)),
            "columns": list(payload.columns),
        }
    if isinstance(payload, dict):
        out: Dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, pd.DataFrame):
                out[str(key)] = {
                    "type": "DataFrame",
                    "rows": int(len(value)),
                    "columns": list(value.columns),
                }
            elif isinstance(value, dict):
                out[str(key)] = _paper_payload_summary(value)
            elif isinstance(value, (list, tuple)):
                if value and isinstance(value[0], dict):
                    out[str(key)] = {"type": "list[dict]", "length": len(value)}
                elif len(value) > 10:
                    out[str(key)] = {"type": "list", "length": len(value)}
                else:
                    out[str(key)] = list(value)
            else:
                out[str(key)] = value
        return out
    return payload


def _read_paper_replication_run(manifest_path: Path) -> Dict[str, object]:
    manifest = _load_json_file(manifest_path, default={}) or {}
    run_dir = manifest_path.parent
    live_status_path = run_dir / "live_status.json"
    live_events_path = run_dir / "live_events.jsonl"
    frozen_dir = _paper_route_dir(run_dir, "frozen")
    raw_dir = _paper_route_dir(run_dir, "raw")
    raw_build_dir = run_dir / "raw_build"

    live_status = _load_json_file(live_status_path, default={}) or {}
    manifest_progress = dict(manifest.get("progress") or {})
    live_event_rows = _read_jsonl_records(live_events_path)
    if not live_event_rows and isinstance(live_status, dict) and live_status:
        live_event_rows = [live_status]
    if not live_event_rows and manifest:
        live_event_rows = [
            {
                "updated_at": manifest.get("updated_at") or manifest.get("created_at"),
                "progress": manifest_progress,
                "step_id": manifest_progress.get("current_step_id"),
                "status": manifest.get("status"),
                "result_status": manifest.get("result_status"),
            }
        ]
    normalized_events = [
        _paper_normalize_live_event(row, manifest_progress=manifest_progress)
        for row in live_event_rows
        if isinstance(row, dict)
    ]
    live_events_df = pd.DataFrame(normalized_events)
    if not live_events_df.empty:
        live_events_df["event_index"] = range(1, len(live_events_df) + 1)

    step_df = _paper_build_step_df(manifest)
    frozen_payload = _load_pickle_file(frozen_dir / "route_payload.pkl", default={}) or {}
    raw_payload = _load_pickle_file(raw_dir / "route_payload.pkl", default={}) or {}
    if not frozen_payload:
        frozen_payload = {
            "route_name": "frozen",
            "dataset_validation": _load_json_file(frozen_dir / "dataset_validation.json", default={}) or {},
        }
    if not raw_payload:
        raw_payload = {
            "route_name": "raw",
            "dataset_validation": _load_json_file(raw_dir / "dataset_validation.json", default={}) or {},
        }
    raw_build_payload = _load_pickle_file(raw_build_dir / "payload.pkl", default={}) or {}
    compare_payload = _paper_load_compare_payload(run_dir)
    export_payload = _paper_load_export_payload(run_dir)

    frozen_models, frozen_model_df, frozen_k_df = _paper_collect_route_models(frozen_dir, "frozen")
    raw_models, raw_model_df, raw_k_df = _paper_collect_route_models(raw_dir, "raw")
    partial_models_df = pd.concat(
        [df for df in [frozen_model_df, raw_model_df] if isinstance(df, pd.DataFrame) and not df.empty],
        ignore_index=True,
    ) if any(isinstance(df, pd.DataFrame) and not df.empty for df in [frozen_model_df, raw_model_df]) else pd.DataFrame()
    k_progress_df = pd.concat(
        [df for df in [frozen_k_df, raw_k_df] if isinstance(df, pd.DataFrame) and not df.empty],
        ignore_index=True,
    ) if any(isinstance(df, pd.DataFrame) and not df.empty for df in [frozen_k_df, raw_k_df]) else pd.DataFrame()

    if isinstance(compare_payload.get("diff_df"), pd.DataFrame):
        compare_diff_df = compare_payload.get("diff_df")
    else:
        compare_diff_df = _load_csv_file(run_dir / "compare" / "diff.csv")
        compare_payload["diff_df"] = compare_diff_df

    compare_summary_df = pd.DataFrame(
        [
            {
                "status": str(compare_payload.get("status") or ""),
                "passed": bool(compare_payload.get("passed")),
                "reason": str(compare_payload.get("reason") or ""),
                "max_numeric_diff": compare_payload.get("max_numeric_diff"),
                "tolerance": compare_payload.get("tolerance"),
                "diff_rows": int(len(compare_diff_df)) if isinstance(compare_diff_df, pd.DataFrame) else 0,
            }
        ]
    )
    export_summary_df = pd.DataFrame(
        [
            {
                "result_status": str(export_payload.get("result_status") or ""),
                "latex_promoted": bool(export_payload.get("latex_promoted")),
                "candidate_count": int(len(export_payload.get("candidate_paths") or {})),
                "promoted_count": int(len(export_payload.get("promoted_paths") or {})),
            }
        ]
    )
    route_status_df = _paper_route_status_rows(
        manifest=manifest,
        step_df=step_df,
        frozen_payload=frozen_payload,
        raw_payload=raw_payload,
        raw_build_payload=raw_build_payload,
        compare_payload=compare_payload,
        export_payload=export_payload,
        partial_models_df=partial_models_df,
        k_progress_df=k_progress_df,
    )
    current_context = {
        "status": str(live_status.get("status") or manifest.get("status") or ""),
        "result_status": str(live_status.get("result_status") or manifest.get("result_status") or ""),
        "current_stage": str(manifest_progress.get("current_stage") or ""),
        "current_step_id": str(live_status.get("step_id") or manifest_progress.get("current_step_id") or ""),
        "message": str(live_status.get("message") or ""),
        "updated_at": str(live_status.get("updated_at") or manifest.get("updated_at") or ""),
    }
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_dir": run_dir,
        "live_status": live_status,
        "live_events_df": live_events_df,
        "live_status_path": live_status_path,
        "live_events_path": live_events_path,
        "step_df": step_df,
        "route_status_df": route_status_df,
        "partial_models_df": partial_models_df,
        "k_progress_df": k_progress_df,
        "compare_summary_df": compare_summary_df,
        "compare_diff_df": compare_diff_df,
        "export_summary_df": export_summary_df,
        "current_context": current_context,
        "frozen_payload": frozen_payload,
        "raw_payload": raw_payload,
        "raw_build_payload": raw_build_payload,
        "compare_payload": compare_payload,
        "export_payload": export_payload,
        "model_payloads": frozen_models + raw_models,
    }


def _language_modeling_artifact_path(
    run_dir: Path,
    manifest: Dict[str, object],
    key: str,
) -> Optional[Path]:
    artifacts = dict(manifest.get("artifacts") or {})
    raw_path = artifacts.get(key)
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate


def _load_optional_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    return _load_csv_file(path)


def _load_optional_json(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    payload = _load_json_file(path, default={}) or {}
    return payload if isinstance(payload, dict) else {}


def _language_modeling_normalize_live_event(
    row: Dict[str, object],
    *,
    manifest: Dict[str, object],
) -> Dict[str, object]:
    payload = dict(row or {})
    progress_ratio = pd.to_numeric(
        payload.get("progress_ratio", manifest.get("progress_ratio", 0.0)),
        errors="coerce",
    )
    if pd.isna(progress_ratio):
        progress_ratio = 0.0
    progress_ratio = max(0.0, min(float(progress_ratio), 1.0))
    normalized: Dict[str, object] = {
        "timestamp": str(payload.get("timestamp") or payload.get("updated_at") or manifest.get("updated_at") or ""),
        "stage": str(payload.get("stage") or ""),
        "label": str(payload.get("label") or payload.get("message") or payload.get("event_type") or "Checkpoint state"),
        "detail": str(payload.get("detail") or ""),
        "status": str(payload.get("status") or manifest.get("status") or ""),
        "result_status": str(payload.get("result_status") or manifest.get("result_status") or ""),
        "event_type": str(payload.get("event_type") or ""),
        "progress_ratio": progress_ratio,
        "progress_pct": 100.0 * progress_ratio,
        "message": str(payload.get("message") or ""),
        "run_type": str(payload.get("run_type") or manifest.get("run_type") or ""),
        "title": str(payload.get("title") or manifest.get("title") or ""),
        "phase": str(payload.get("phase") or ""),
        "model_name": str(payload.get("model_name") or ""),
    }
    for key in [
        "trial_index",
        "config_rank",
        "trainer_seed",
        "objective",
        "epoch",
        "global_step",
        "loss",
        "eval_loss",
        "eval_accuracy",
        "eval_f1",
        "eval_balanced_f1",
        "learning_rate",
    ]:
        numeric = pd.to_numeric(payload.get(key), errors="coerce")
        normalized[key] = None if pd.isna(numeric) else float(numeric)
    return normalized


def _language_modeling_search_curve(
    trials_df: pd.DataFrame,
    *,
    greater_is_better: bool,
) -> pd.DataFrame:
    if not isinstance(trials_df, pd.DataFrame) or trials_df.empty:
        return pd.DataFrame(columns=["trial_index", "objective", "best_so_far"])
    if "trial_index" not in trials_df.columns or "objective" not in trials_df.columns:
        return pd.DataFrame(columns=["trial_index", "objective", "best_so_far"])
    work = trials_df.copy()
    work["trial_index"] = pd.to_numeric(work["trial_index"], errors="coerce")
    work["objective"] = pd.to_numeric(work["objective"], errors="coerce")
    if "status" in work.columns:
        work = work.loc[work["status"].astype(str) == "ok"].copy()
    work = work.dropna(subset=["trial_index", "objective"]).sort_values("trial_index")
    if work.empty:
        return pd.DataFrame(columns=["trial_index", "objective", "best_so_far"])
    work["best_so_far"] = (
        work["objective"].cummax()
        if greater_is_better
        else work["objective"].cummin()
    )
    return work[["trial_index", "objective", "best_so_far"]].reset_index(drop=True)


def _read_language_modeling_run(manifest_path: Path) -> Dict[str, object]:
    manifest = _load_json_file(manifest_path, default={}) or {}
    run_dir = manifest_path.parent
    live_status_path = run_dir / "live_status.json"
    live_events_path = run_dir / "live_events.jsonl"
    live_status = _load_json_file(live_status_path, default={}) or {}
    live_event_rows = _read_jsonl_records(live_events_path)
    if not live_event_rows and isinstance(live_status, dict) and live_status:
        live_event_rows = [live_status]
    normalized_events = [
        _language_modeling_normalize_live_event(row, manifest=manifest)
        for row in live_event_rows
        if isinstance(row, dict)
    ]
    live_events_df = pd.DataFrame(normalized_events)
    if not live_events_df.empty:
        live_events_df["event_index"] = range(1, len(live_events_df) + 1)

    search_trials_df = _load_optional_csv(
        _language_modeling_artifact_path(run_dir, manifest, "search_trials_csv")
    )
    confirmation_trials_df = _load_optional_csv(
        _language_modeling_artifact_path(run_dir, manifest, "confirmation_trials_csv")
    )
    confirmation_summary_df = _load_optional_csv(
        _language_modeling_artifact_path(run_dir, manifest, "confirmation_summary_csv")
    )
    history_df = _load_optional_csv(
        _language_modeling_artifact_path(run_dir, manifest, "history_csv")
    )
    best_history_df = _load_optional_csv(
        _language_modeling_artifact_path(run_dir, manifest, "best_history_csv")
    )
    search_summary = _load_optional_json(
        _language_modeling_artifact_path(run_dir, manifest, "search_summary_json")
    )
    best_result = _load_optional_json(
        _language_modeling_artifact_path(run_dir, manifest, "best_result_json")
    )
    finetune_result = _load_optional_json(
        _language_modeling_artifact_path(run_dir, manifest, "finetune_result_json")
    )
    failure_summary = _load_optional_json(
        _language_modeling_artifact_path(run_dir, manifest, "search_failure_summary_json")
    )

    current_context = {
        "status": str(live_status.get("status") or manifest.get("status") or ""),
        "result_status": str(live_status.get("result_status") or manifest.get("result_status") or ""),
        "stage": str(live_status.get("stage") or ""),
        "message": str(live_status.get("message") or manifest.get("last_message") or ""),
        "updated_at": str(live_status.get("updated_at") or manifest.get("updated_at") or ""),
    }
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "run_dir": run_dir,
        "live_status": live_status,
        "live_events_df": live_events_df,
        "live_status_path": live_status_path,
        "live_events_path": live_events_path,
        "search_trials_df": search_trials_df,
        "confirmation_trials_df": confirmation_trials_df,
        "confirmation_summary_df": confirmation_summary_df,
        "history_df": history_df,
        "best_history_df": best_history_df,
        "search_summary": search_summary,
        "best_result": best_result,
        "finetune_result": finetune_result,
        "failure_summary": failure_summary,
        "current_context": current_context,
    }


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def _read_live_db(
    path: Path,
) -> Tuple[Dict[str, object], pd.DataFrame, Optional[Dict[str, object]]]:
    meta: Dict[str, object] = {}
    rows_df = pd.DataFrame()
    best_row: Optional[Dict[str, object]] = None

    con = sqlite3.connect(path, timeout=1)
    try:
        if _table_exists(con, "meta"):
            rows = con.execute("SELECT key, value FROM meta").fetchall()
            for key, value in rows:
                try:
                    meta[key] = json.loads(value)
                except Exception:
                    meta[key] = value

        if _table_exists(con, "results"):
            rows = con.execute(
                "SELECT id, created_at, payload_json FROM results ORDER BY id"
            ).fetchall()
            payloads = []
            for row_id, created_at, payload_json in rows:
                try:
                    payload = json.loads(payload_json)
                except Exception:
                    payload = {"raw": payload_json}
                payload["_row_id"] = row_id
                payload["_created_at"] = created_at
                payloads.append(payload)
            if payloads:
                rows_df = pd.DataFrame(payloads)

        if _table_exists(con, "best"):
            row = con.execute(
                "SELECT payload_json FROM best ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                try:
                    best_row = json.loads(row[0])
                except Exception:
                    best_row = {"raw": row[0]}
    finally:
        con.close()
    return meta, rows_df, best_row


def _render_find_samples_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: Find samples sizes")
    metric_options = {
        "best_f1": "F1",
        "accuracy": "Accuracy",
        "recall": "Recall",
        "precision": "Precision",
        "roc_auc": "ROC-AUC",
        "fnr": "FNR (menor es mejor)",
    }
    available_metrics = {k: v for k, v in metric_options.items() if k in df.columns}
    if not available_metrics:
        st.info("No hay metricas disponibles para graficar.")
        st.dataframe(df, width="stretch")
        return

    metric_labels = list(available_metrics.values())
    selected_metric_label = st.selectbox(
        "Metrica a graficar",
        metric_labels,
        key="live_find_samples_metric",
    )
    metric_key = next(
        k for k, v in available_metrics.items() if v == selected_metric_label
    )

    plot_df = df.copy()
    if "error" in plot_df.columns:
        plot_df = plot_df[
            plot_df["error"].isna() | (plot_df["error"] == "")
        ]

    if best_row is None and not plot_df.empty and metric_key in plot_df.columns:
        if metric_key == "fnr":
            best_row = plot_df.loc[plot_df[metric_key].idxmin()].to_dict()
        else:
            best_row = plot_df.loc[plot_df[metric_key].idxmax()].to_dict()

    if best_row:
        st.markdown("**Resultado optimo**")
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
        ]
        metrics_payload = {
            key: best_row.get(key) for key in metrics_cols if key in best_row
        }
        if metrics_payload:
            st.json(metrics_payload)
        model_path = best_row.get("model_path")
        if model_path and isinstance(model_path, str):
            st.caption(f"Modelo: {model_path}")

    tab_viz, tab_data = st.tabs(["Grafico", "Datos"])
    with tab_viz:
        if "candidate_rank" in plot_df.columns and metric_key in plot_df.columns:
            try:
                import altair as alt
                chart = (
                    alt.Chart(plot_df)
                    .mark_line(point=True)
                    .encode(
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
                    )
                    .interactive()
                )
                st.altair_chart(chart, width="stretch")
            except ImportError:
                st.warning("Altair no instalado.")
        else:
            st.info("No hay columnas suficientes para graficar.")

        if {"window_days", "accidents_per_day"}.issubset(plot_df.columns):
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
                    .interactive()
                )
                st.altair_chart(scatter, width="stretch")
            except ImportError:
                pass

    with tab_data:
        st.dataframe(df, width="stretch")


def _render_features_sampler_view(df: pd.DataFrame) -> None:
    if "best_f1" in df.columns and "type" in df.columns:
        st.caption("Mejor F1 por estrategia:")
        best_by_type = df.loc[df.groupby("type")["best_f1"].idxmax()]
        if not best_by_type.empty:
            cols = st.columns(len(best_by_type))
            for idx, row in enumerate(best_by_type.itertuples(), start=0):
                with cols[idx]:
                    delta_label = ""
                    if "k" in best_by_type.columns:
                        delta_label = f"k={row.k}"
                    st.metric(
                        label=row.type,
                        value=f"{row.best_f1:.4f}",
                        delta=delta_label,
                    )

    tab_viz, tab_data = st.tabs(["Grafico", "Datos"])
    with tab_viz:
        if "k" in df.columns:
            metric_options = {
                "best_f1": "Best F1 Score",
                "accuracy": "Accuracy",
                "recall": "Recall (Sens)",
                "precision": "Precision",
                "roc_auc": "ROC-AUC",
                "fnr": "FNR",
            }
            available_metrics = {
                k: v for k, v in metric_options.items() if k in df.columns
            }
            if not available_metrics:
                available_metrics = (
                    {"best_f1": "Best F1 Score"} if "best_f1" in df.columns else {}
                )
            selected_metric_key = "best_f1"
            if available_metrics:
                col_sel, _ = st.columns([0.3, 0.7])
                with col_sel:
                    selected_metric_label = st.selectbox(
                        "Metrica a graficar",
                        options=list(available_metrics.values()),
                        index=0,
                        key="live_features_metric",
                    )
                    selected_metric_key = next(
                        k
                        for k, v in available_metrics.items()
                        if v == selected_metric_label
                    )
            if selected_metric_key in df.columns and "type" in df.columns:
                try:
                    import altair as alt
                    y_min = df[selected_metric_key].min()
                    y_max = df[selected_metric_key].max()
                    padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
                    chart = (
                        alt.Chart(df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "k", axis=alt.Axis(title="Top K Features")
                            ),
                            y=alt.Y(
                                selected_metric_key,
                                scale=alt.Scale(
                                    domain=[
                                        max(0, y_min - padding),
                                        min(1, y_max + padding),
                                    ]
                                ),
                                axis=alt.Axis(
                                    title=available_metrics[selected_metric_key]
                                ),
                            ),
                            color="type",
                            tooltip=["k", selected_metric_key, "type", "n_features"],
                        )
                        .interactive()
                    )
                    st.altair_chart(chart, width="stretch")
                except ImportError:
                    st.warning("Altair no instalado.")
            else:
                st.info("Columnas insuficientes para graficar.")
        else:
            st.info("No hay columna 'k' para graficar.")

    with tab_data:
        st.dataframe(df, width="stretch")


def _render_controlled_comparison_live_view(
    meta: Dict[str, object],
    df: pd.DataFrame,
    best_row: Optional[Dict[str, object]],
) -> None:
    st.caption("Experimento detectado: Comparación controlada con Clusters")
    plot_df = _ensure_balance_strategy_column(df.copy())
    if plot_df.empty:
        st.warning("No hay resultados en la base de datos.")
        return

    objective_metric = str(meta.get("objective_metric") or "").strip().lower()
    if not objective_metric and "objective_metric" in plot_df.columns:
        metric_values = plot_df["objective_metric"].dropna().astype(str)
        if not metric_values.empty:
            objective_metric = str(metric_values.iloc[0]).strip().lower()
    if not objective_metric:
        objective_metric = "roc_auc"

    objective_label = str(meta.get("objective_label") or "").strip()
    if not objective_label and "objective_label" in plot_df.columns:
        label_values = plot_df["objective_label"].dropna().astype(str)
        if not label_values.empty:
            objective_label = str(label_values.iloc[0]).strip()
    if not objective_label:
        objective_label = objective_metric.upper()

    metric_col = (
        "val_objective_score"
        if "val_objective_score" in plot_df.columns
        else {
            "roc_auc": "val_roc_auc",
            "f1": "val_f1",
            "mcc": "val_mcc",
        }.get(objective_metric, "val_roc_auc")
    )

    numeric_cols = [
        "k",
        "val_objective_score",
        "test_objective_score",
        "test_accuracy",
        "test_recall",
        "test_sensitivity",
        "val_roc_auc",
        "test_roc_auc",
        "test_pr_auc",
        "val_f1",
        "test_f1",
        "test_f1_global",
        "test_f1_class_0",
        "test_f1_class_1",
        "val_mcc",
        "test_mcc",
        "test_false_negatives",
        "test_false_positives",
        "test_true_negatives",
        "test_true_positives",
        "decision_threshold",
        "train_rows",
        "val_rows",
        "test_rows",
        "optuna_trials_completed",
    ]
    for col in numeric_cols:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    completed_df = plot_df.copy()
    if "status" in completed_df.columns:
        completed_df = completed_df[
            completed_df["status"].astype(str).str.lower() == "completed"
        ].copy()

    status_counts = (
        plot_df["status"].astype(str).value_counts().to_dict()
        if "status" in plot_df.columns
        else {}
    )
    completed_count = int(status_counts.get("completed", len(completed_df)))
    failed_count = int(status_counts.get("failed", 0))
    pending_count = int(
        sum(
            count
            for status, count in status_counts.items()
            if status not in {"completed", "failed"}
        )
    )

    segment_info = meta.get("segment_info")
    if isinstance(segment_info, dict) and segment_info:
        st.caption(
            f"Tramo: {segment_info.get('segment_label') or segment_info}"
        )
    st.caption(
        f"Objetivo: {objective_label} | "
        f"Eventos: {meta.get('dataset_name') or '-'} | "
        f"Features: {meta.get('features_name') or '-'}"
    )

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5, kpi_6 = st.columns(6)
    kpi_1.metric("Estado", str(meta.get("run_mode") or "live"))
    kpi_2.metric("Completadas", f"{completed_count}")
    kpi_3.metric("Fallidas", f"{failed_count}")
    kpi_4.metric("Pendientes", f"{pending_count}")
    kpi_5.metric("Objetivo", objective_label)
    kpi_6.metric("Optuna jobs", str(meta.get("optuna_n_jobs") or "-"))

    if completed_df.empty:
        st.info("Aún no hay combinaciones completadas.")
        st.dataframe(_streamlit_arrow_safe_df(plot_df), width="stretch")
        return

    if best_row is None and metric_col in completed_df.columns:
        candidate_df = completed_df.dropna(subset=[metric_col]).copy()
        if not candidate_df.empty:
            best_row = candidate_df.sort_values(
                [metric_col, "k"],
                ascending=[False, True],
            ).iloc[0].to_dict()

    if best_row:
        st.markdown("**Mejor combinación observada hasta ahora**")
        best_payload = _controlled_live_best_payload(
            dict(best_row),
            objective_label=objective_label,
        )
        st.json(best_payload)
        best_matrix = _coerce_confusion_matrix_cell(
            _first_present_metric(
                dict(best_row),
                "test_confusion_matrix",
                "best_test_confusion_matrix",
            )
        )
        if best_matrix is not None:
            st.caption("Matriz de confusión de test")
            st.dataframe(
                pd.DataFrame(
                    best_matrix,
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                ),
                width="stretch",
            )

    summary_df = completed_df.copy()
    summary_cols = [
        "model_name",
        "feature_set",
        metric_col,
        "test_objective_score",
        "test_accuracy",
        "test_recall",
        "test_sensitivity",
        "test_roc_auc",
        "test_pr_auc",
        "test_f1_global",
        "test_f1_class_0",
        "test_f1_class_1",
        "test_mcc",
        "test_false_negatives",
        "test_false_positives",
        "test_confusion_matrix",
        "k",
        "balance_mode",
        "decision_threshold",
        "selected_features",
        "best_params",
        "smote_params",
    ]
    summary_df = summary_df[[col for col in summary_cols if col in summary_df.columns]]
    if {"model_name", "feature_set", metric_col}.issubset(completed_df.columns):
        best_idx = (
            completed_df.dropna(subset=[metric_col])
            .sort_values(["model_name", "feature_set", metric_col, "k"], ascending=[True, True, False, True])
            .groupby(["model_name", "feature_set"], dropna=False)
            .head(1)
            .index
        )
        summary_df = completed_df.loc[best_idx, [col for col in summary_cols if col in completed_df.columns]].copy()
    if "selected_features" in summary_df.columns:
        summary_df["selected_features"] = summary_df["selected_features"].apply(_jsonish_to_text)
    if "best_params" in summary_df.columns:
        summary_df["best_params"] = summary_df["best_params"].apply(_jsonish_to_text)
    if "smote_params" in summary_df.columns:
        summary_df["smote_params"] = summary_df["smote_params"].apply(_jsonish_to_text)
    if "test_confusion_matrix" in summary_df.columns:
        summary_df["test_confusion_matrix"] = summary_df["test_confusion_matrix"].apply(
            _confusion_matrix_text
        )

    tab_summary, tab_curves, tab_data = st.tabs(["Resumen", "Curvas", "Datos"])
    with tab_summary:
        st.markdown("**Mejor resultado por modelo y conjunto**")
        st.dataframe(_streamlit_arrow_safe_df(summary_df), width="stretch")
        if "test_confusion_matrix" in completed_df.columns:
            with st.expander("Matrices de confusión de test", expanded=False):
                matrix_df = completed_df.copy()
                if {"model_name", "feature_set", metric_col}.issubset(matrix_df.columns):
                    best_idx = (
                        matrix_df.dropna(subset=[metric_col])
                        .sort_values(
                            ["model_name", "feature_set", metric_col, "k"],
                            ascending=[True, True, False, True],
                        )
                        .groupby(["model_name", "feature_set"], dropna=False)
                        .head(1)
                        .index
                    )
                    matrix_df = matrix_df.loc[best_idx].copy()
                for _, row in matrix_df.iterrows():
                    matrix = _coerce_confusion_matrix_cell(row.get("test_confusion_matrix"))
                    if matrix is None:
                        continue
                    st.markdown(
                        f"**{row.get('model_name', '-')} | {row.get('feature_set', '-')} | "
                        f"K={row.get('k', '-')}**"
                    )
                    st.dataframe(
                        pd.DataFrame(
                            matrix,
                            index=["Actual 0", "Actual 1"],
                            columns=["Pred 0", "Pred 1"],
                        ),
                        width="stretch",
                    )
        if failed_count > 0 and "status" in plot_df.columns:
            failed_df = plot_df[
                plot_df["status"].astype(str).str.lower() == "failed"
            ].copy()
            if not failed_df.empty:
                visible_cols = [
                    col
                    for col in [
                        "model_name",
                        "feature_set",
                        "balance_mode",
                        "k",
                        "error",
                    ]
                    if col in failed_df.columns
                ]
                st.markdown("**Combinaciones fallidas**")
                st.dataframe(_streamlit_arrow_safe_df(failed_df[visible_cols]), width="stretch")

    with tab_curves:
        balance_options = (
            sorted(completed_df["balance_mode"].dropna().astype(str).unique().tolist())
            if "balance_mode" in completed_df.columns
            else []
        )
        if not balance_options:
            balance_options = ["none"]
            completed_df["balance_mode"] = "none"
        selected_balance = st.selectbox(
            "Balanceo",
            options=balance_options,
            key="live_controlled_balance_mode",
        )
        curve_df = completed_df[
            completed_df["balance_mode"].astype(str) == str(selected_balance)
        ].copy()
        if curve_df.empty or metric_col not in curve_df.columns:
            st.info("No hay datos suficientes para graficar.")
        else:
            try:
                import altair as alt
                chart = (
                    alt.Chart(curve_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("k:Q", axis=alt.Axis(title="K")),
                        y=alt.Y(
                            f"{metric_col}:Q",
                            axis=alt.Axis(title=f"Validación {objective_label}"),
                        ),
                        color=alt.Color("feature_set:N", title="Conjunto"),
                        row=alt.Row(
                            "model_name:N",
                            title=None,
                            header=alt.Header(labelAngle=0),
                        ),
                        tooltip=[
                            alt.Tooltip("model_name:N", title="Modelo"),
                            alt.Tooltip("feature_set:N", title="Conjunto"),
                            alt.Tooltip("balance_mode:N", title="Balanceo"),
                            alt.Tooltip("k:Q", title="K"),
                            alt.Tooltip(f"{metric_col}:Q", title=f"Val {objective_label}", format=".4f"),
                            alt.Tooltip("test_accuracy:Q", title="Test Accuracy", format=".4f"),
                            alt.Tooltip("test_recall:Q", title="Test Recall", format=".4f"),
                            alt.Tooltip("test_sensitivity:Q", title="Test Sensitivity", format=".4f"),
                            alt.Tooltip("test_roc_auc:Q", title="Test ROC-AUC", format=".4f"),
                            alt.Tooltip("test_pr_auc:Q", title="Test PR-AUC", format=".4f"),
                            alt.Tooltip("test_f1_global:Q", title="Test F1 Global", format=".4f"),
                            alt.Tooltip("test_f1_class_0:Q", title="Test F1 Clase 0", format=".4f"),
                            alt.Tooltip("test_f1_class_1:Q", title="Test F1 Clase 1", format=".4f"),
                            alt.Tooltip("test_mcc:Q", title="Test MCC", format=".4f"),
                            alt.Tooltip("test_false_negatives:Q", title="FN Test"),
                            alt.Tooltip("test_false_positives:Q", title="FP Test"),
                        ],
                    )
                    .properties(height=150)
                    .interactive()
                )
                st.altair_chart(chart, width="stretch")
            except ImportError:
                pivot_df = curve_df.pivot_table(
                    index="k",
                    columns=["model_name", "feature_set"],
                    values=metric_col,
                    aggfunc="max",
                ).sort_index()
                st.line_chart(pivot_df, width="stretch")

    with tab_data:
        st.dataframe(_streamlit_arrow_safe_df(plot_df), width="stretch")


def _render_gnn_optuna_objectives_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: GNN Optuna Objectives")
    plot_df = _ensure_balance_strategy_column(df.copy())
    metric_cols = [
        "test_f1",
        "test_precision",
        "test_recall",
        "test_accuracy",
        "test_far",
        "test_auprc",
        "test_mcc",
    ]
    for col in metric_cols:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    if "gnn_variant" in plot_df.columns:
        variant_options = sorted(
            plot_df["gnn_variant"].dropna().astype(str).unique().tolist()
        )
        if variant_options:
            selected_variants = st.multiselect(
                "Variantes GNN",
                options=variant_options,
                default=variant_options,
                key="live_gnn_optuna_variants",
            )
            if selected_variants:
                plot_df = plot_df[
                    plot_df["gnn_variant"].astype(str).isin(selected_variants)
                ].copy()

    if "balance_strategy" in plot_df.columns:
        balance_options = sorted(
            plot_df["balance_strategy"].dropna().astype(str).unique().tolist()
        )
        if balance_options:
            selected_balance = st.multiselect(
                "Balanceo",
                options=balance_options,
                default=balance_options,
                key="live_gnn_optuna_balance_filter",
            )
            if selected_balance:
                plot_df = plot_df[
                    plot_df["balance_strategy"].astype(str).isin(selected_balance)
                ].copy()

    if plot_df.empty:
        st.info("No hay datos para los filtros seleccionados.")
        st.dataframe(df, width="stretch")
        return

    if best_row is None and "test_f1" in plot_df.columns:
        valid = plot_df[
            pd.to_numeric(plot_df["test_f1"], errors="coerce").notna()
        ]
        if not valid.empty:
            best_row = valid.loc[valid["test_f1"].idxmax()].to_dict()

    if best_row:
        balance_value = best_row.get("balance_strategy")
        if balance_value in (None, ""):
            balance_value = best_row.get("balance_mode")
        st.markdown("**Resultado optimo (test_f1)**")
        metrics_payload = {
            "gnn_variant": best_row.get("gnn_variant"),
            "balance_strategy": balance_value,
            "objective": best_row.get("objective_label"),
            "test_f1": best_row.get("test_f1"),
            "test_precision": best_row.get("test_precision"),
            "test_recall": best_row.get("test_recall"),
            "test_accuracy": best_row.get("test_accuracy"),
            "test_far": best_row.get("test_far"),
            "test_auprc": best_row.get("test_auprc"),
            "test_mcc": best_row.get("test_mcc"),
        }
        st.json(metrics_payload)
        model_path = best_row.get("model_path")
        if model_path:
            st.caption(f"Modelo: {model_path}")

    tab_viz, tab_data = st.tabs(["Grafico", "Datos"])
    with tab_viz:
        if "gnn_variant" in plot_df.columns:
            summary_metric = "test_auprc" if "test_auprc" in plot_df.columns else "test_f1"
            if summary_metric in plot_df.columns:
                tmp = plot_df.copy()
                tmp[summary_metric] = pd.to_numeric(tmp[summary_metric], errors="coerce")
                tmp = tmp[tmp[summary_metric].notna()]
                if not tmp.empty:
                    idx = tmp.groupby("gnn_variant")[summary_metric].idxmax()
                    summary = tmp.loc[idx].copy()
                    keep_cols = [
                        "gnn_variant",
                        "balance_strategy",
                        "objective_label",
                        "test_f1",
                        "test_auprc",
                        "test_auc",
                        "test_mcc",
                    ]
                    keep_cols = [c for c in keep_cols if c in summary.columns]
                    st.markdown("**Mejor corrida por variante**")
                    st.dataframe(
                        summary[keep_cols].sort_values(summary_metric, ascending=False),
                        width="stretch",
                    )

        if {"objective_label", "test_f1"}.issubset(plot_df.columns):
            plot_df = plot_df[plot_df["test_f1"].notna()].copy()
            if plot_df.empty:
                st.info("No hay datos numericos para graficar.")
                st.dataframe(df, width="stretch")
            else:
                if float(plot_df["test_f1"].max() or 0.0) == 0.0:
                    st.info("Todos los valores de test_f1 son 0.0 en esta corrida.")
                try:
                    import altair as alt
                    tooltip_fields = [
                        alt.Tooltip("objective_label:N", title="Objetivo"),
                    ]
                    if "balance_strategy" in plot_df.columns:
                        tooltip_fields.append(
                            alt.Tooltip("balance_strategy:N", title="Balanceo")
                        )
                    if "gnn_variant" in plot_df.columns:
                        tooltip_fields.append(
                            alt.Tooltip("gnn_variant:N", title="Variante GNN")
                        )
                    tooltip_fields += [
                        alt.Tooltip("test_f1:Q", title="Test F1", format=".4f"),
                        alt.Tooltip("test_precision:Q", title="Test Precision", format=".4f"),
                        alt.Tooltip("test_recall:Q", title="Test Recall", format=".4f"),
                        alt.Tooltip("test_far:Q", title="Test FAR", format=".4f"),
                    ]
                    color_enc = (
                        alt.Color("balance_strategy:N", title="Balanceo")
                        if "balance_strategy" in plot_df.columns
                        else alt.value("#1f77b4")
                    )
                    enc_kwargs = dict(
                        x=alt.X(
                            "objective_label:N",
                            axis=alt.Axis(title="Objetivo"),
                        ),
                        y=alt.Y(
                            "test_f1:Q",
                            axis=alt.Axis(title="Test F1"),
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        color=color_enc,
                        tooltip=tooltip_fields,
                    )
                    if "gnn_variant" in plot_df.columns:
                        n_variants = int(plot_df["gnn_variant"].astype(str).nunique())
                        if n_variants > 1:
                            enc_kwargs["column"] = alt.Column(
                                "gnn_variant:N",
                                title="Variante GNN",
                                header=alt.Header(labelAngle=0),
                            )
                    base = alt.Chart(plot_df).encode(**enc_kwargs)
                    bars = base.mark_bar(opacity=0.8)
                    points = base.mark_circle(size=60)
                    labels = base.mark_text(
                        dy=-8, size=10, color="#444"
                    ).encode(text=alt.Text("test_f1:Q", format=".3f"))
                    chart = (bars + points + labels).interactive()
                    st.altair_chart(chart, width="stretch")
                except ImportError:
                    st.warning("Altair no instalado.")
        else:
            st.info("No hay columnas suficientes para graficar.")

    with tab_data:
        st.dataframe(df, width="stretch")


def _render_gnn_recursive_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: Opt.Recursiva (Optuna vs Ray)")
    plot_df = _ensure_balance_strategy_column(df.copy())

    include_errors = st.checkbox(
        "Incluir registros con error",
        value=False,
        key="live_gnn_recursive_include_errors",
    )
    if not include_errors and "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"] == "ok"]

    if plot_df.empty:
        st.info("No hay datos suficientes para graficar.")
        st.dataframe(df, width="stretch")
        return

    objective_options = (
        sorted(plot_df["objective_label"].dropna().astype(str).unique())
        if "objective_label" in plot_df.columns
        else []
    )
    balance_options = (
        sorted(plot_df["balance_strategy"].dropna().astype(str).unique())
        if "balance_strategy" in plot_df.columns
        else []
    )
    optimizer_options = (
        sorted(plot_df["optimizer"].dropna().astype(str).unique())
        if "optimizer" in plot_df.columns
        else []
    )
    variant_options = (
        sorted(plot_df["gnn_variant"].dropna().astype(str).unique())
        if "gnn_variant" in plot_df.columns
        else []
    )

    col_filters_1, col_filters_2, col_filters_3 = st.columns(3)
    with col_filters_1:
        selected_objective = st.selectbox(
            "Objetivo",
            options=objective_options or ["(sin objetivo)"],
            key="live_gnn_recursive_objective",
        )
    with col_filters_2:
        selected_balance = st.selectbox(
            "Balanceo",
            options=balance_options or ["(sin balanceo)"],
            key="live_gnn_recursive_balance",
        )
    with col_filters_3:
        selected_optimizer = st.multiselect(
            "Optimizadores",
            options=optimizer_options or ["Optuna", "Ray"],
            default=optimizer_options or ["Optuna", "Ray"],
            key="live_gnn_recursive_optimizer",
        )
    if variant_options:
        selected_variants = st.multiselect(
            "Variantes GNN",
            options=variant_options,
            default=variant_options,
            key="live_gnn_recursive_variants",
        )
    else:
        selected_variants = []

    if "objective_label" in plot_df.columns and objective_options:
        plot_df = plot_df[
            plot_df["objective_label"].astype(str) == str(selected_objective)
        ]
    if "balance_strategy" in plot_df.columns and balance_options:
        plot_df = plot_df[
            plot_df["balance_strategy"].astype(str) == str(selected_balance)
        ]
    if "optimizer" in plot_df.columns and selected_optimizer:
        plot_df = plot_df[
            plot_df["optimizer"].astype(str).isin(selected_optimizer)
        ]
    if "gnn_variant" in plot_df.columns and selected_variants:
        plot_df = plot_df[
            plot_df["gnn_variant"].astype(str).isin(selected_variants)
        ]

    if "iteration" in plot_df.columns:
        plot_df["iteration"] = pd.to_numeric(
            plot_df["iteration"], errors="coerce"
        )
    else:
        plot_df["iteration"] = pd.to_numeric(
            plot_df.get("_row_id", pd.Series(range(len(plot_df)))),
            errors="coerce",
        )

    metric_candidates = [
        "test_f1",
        "test_precision",
        "test_recall",
        "test_accuracy",
        "test_far",
        "test_auprc",
        "test_mcc",
    ]
    available_metrics = [
        m for m in metric_candidates if m in plot_df.columns
    ]
    if not available_metrics:
        st.info("No hay metricas disponibles para graficar.")
        st.dataframe(df, width="stretch")
        return

    selected_metric = st.selectbox(
        "Metrica principal",
        available_metrics,
        index=0,
        key="live_gnn_recursive_metric",
    )

    for col in available_metrics:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    if best_row is None and selected_metric in plot_df.columns:
        valid = plot_df[plot_df[selected_metric].notna()]
        if not valid.empty:
            best_row = valid.loc[valid[selected_metric].idxmax()].to_dict()

    if best_row:
        st.markdown("**Resultado optimo (metrica seleccionada)**")
        metrics_payload = {
            "gnn_variant": best_row.get("gnn_variant"),
            "objective": best_row.get("objective_label"),
            "optimizer": best_row.get("optimizer"),
            "iteration": best_row.get("iteration"),
            "test_f1": best_row.get("test_f1"),
            "test_precision": best_row.get("test_precision"),
            "test_recall": best_row.get("test_recall"),
            "test_accuracy": best_row.get("test_accuracy"),
            "test_far": best_row.get("test_far"),
            "test_auprc": best_row.get("test_auprc"),
            "test_mcc": best_row.get("test_mcc"),
        }
        st.json(metrics_payload)
        model_path = best_row.get("model_path")
        if model_path:
            st.caption(f"Modelo: {model_path}")

    tab_viz, tab_data = st.tabs(["Grafico", "Datos"])
    with tab_viz:
        try:
            import altair as alt

            base = plot_df.dropna(subset=["iteration"]).copy()
            if base.empty:
                st.info("No hay datos suficientes para graficar.")
            else:
                line_tooltip = [
                    alt.Tooltip("iteration:Q", title="Iteracion"),
                    alt.Tooltip("optimizer:N", title="Optimizador"),
                ]
                if "gnn_variant" in base.columns:
                    line_tooltip.append(
                        alt.Tooltip("gnn_variant:N", title="Variante GNN")
                    )
                if "balance_strategy" in base.columns:
                    line_tooltip.append(
                        alt.Tooltip("balance_strategy:N", title="Balanceo")
                    )
                if "objective_label" in base.columns:
                    line_tooltip.append(
                        alt.Tooltip("objective_label:N", title="Objetivo")
                    )
                line_tooltip.append(
                    alt.Tooltip(f"{selected_metric}:Q", title=selected_metric, format=".4f")
                )
                chart = (
                    alt.Chart(base)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "iteration:Q",
                            axis=alt.Axis(title="Iteracion"),
                        ),
                        y=alt.Y(
                            f"{selected_metric}:Q",
                            axis=alt.Axis(title=selected_metric),
                        ),
                        color=alt.Color(
                            "optimizer:N",
                            title="Optimizador",
                        ),
                        tooltip=line_tooltip,
                    )
                    .interactive()
                )
                st.altair_chart(chart, width="stretch")

            long_rows = []
            for metric in available_metrics:
                metric_cols_base = ["iteration", "optimizer", metric]
                if "gnn_variant" in plot_df.columns:
                    metric_cols_base.append("gnn_variant")
                metric_df = plot_df[metric_cols_base].copy()
                metric_df["metric"] = metric
                metric_df = metric_df.rename(columns={metric: "value"})
                long_rows.append(metric_df)
            long_df = pd.concat(long_rows, ignore_index=True)
            long_df = long_df.dropna(subset=["value", "iteration"])

            if not long_df.empty:
                multi_tooltip = ["iteration", "optimizer", "metric", "value"]
                if "gnn_variant" in long_df.columns:
                    multi_tooltip.insert(2, "gnn_variant")
                multi = (
                    alt.Chart(long_df)
                    .mark_line(point=True, opacity=0.8)
                    .encode(
                        x=alt.X(
                            "iteration:Q",
                            axis=alt.Axis(title="Iteracion"),
                        ),
                        y=alt.Y(
                            "value:Q",
                            axis=alt.Axis(title="Valor"),
                        ),
                        color=alt.Color("optimizer:N", title="Optimizador"),
                        column=alt.Column(
                            "metric:N",
                            title="Metricas",
                            header=alt.Header(labelAngle=0),
                        ),
                        tooltip=multi_tooltip,
                    )
                    .properties(height=220)
                    .interactive()
                )
                st.altair_chart(multi, width="stretch")

            if "alert_level" in plot_df.columns:
                alert_map = {"none": 0, "yellow": 1, "red": 2}
                alert_cols = ["iteration", "optimizer", "alert_level"]
                if "gnn_variant" in plot_df.columns:
                    alert_cols.append("gnn_variant")
                alert_df = plot_df[alert_cols].copy()
                alert_df["alert_value"] = (
                    alert_df["alert_level"].astype(str).str.lower().map(alert_map)
                )
                alert_df = alert_df.dropna(subset=["alert_value", "iteration"])
                if not alert_df.empty:
                    alert_tooltip = ["iteration", "optimizer", "alert_level"]
                    if "gnn_variant" in alert_df.columns:
                        alert_tooltip.insert(2, "gnn_variant")
                    alert_chart = (
                        alt.Chart(alert_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(
                                "iteration:Q",
                                axis=alt.Axis(title="Iteracion"),
                            ),
                            y=alt.Y(
                                "alert_value:Q",
                                axis=alt.Axis(
                                    title="Alerta (0=none, 1=yellow, 2=red)"
                                ),
                            ),
                            color=alt.Color("optimizer:N", title="Optimizador"),
                            tooltip=alert_tooltip,
                        )
                        .interactive()
                    )
                    st.altair_chart(alert_chart, width="stretch")
        except ImportError:
            st.warning("Altair no instalado.")

    with tab_data:
        st.dataframe(plot_df, width="stretch")


def _render_gnn_sampler_memory_budget_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: GNN Sampler Memory Budget")
    plot_df = df.copy()

    numeric_cols = [
        "batch_size",
        "memory_peak_bytes",
        "memory_peak_gb",
        "memory_peak_fraction_total",
        "memory_peak_fraction_budget",
        "budget_bytes",
        "budget_gb",
        "probe_batches",
        "adaptive_jump",
    ]
    for col in numeric_cols:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    if "memory_peak_gb" not in plot_df.columns and "memory_peak_bytes" in plot_df.columns:
        plot_df["memory_peak_gb"] = plot_df["memory_peak_bytes"] / (1024 ** 3)

    role_options = (
        sorted(plot_df["role"].dropna().astype(str).unique().tolist())
        if "role" in plot_df.columns
        else []
    )
    if role_options:
        default_roles = (
            ["best_per_config"] if "best_per_config" in role_options else role_options
        )
        selected_roles = st.multiselect(
            "Roles",
            options=role_options,
            default=default_roles,
            key="live_gnn_mem_roles",
        )
        if selected_roles:
            plot_df = plot_df[plot_df["role"].astype(str).isin(selected_roles)].copy()

    include_errors = st.checkbox(
        "Incluir errores/OOM",
        value=False,
        key="live_gnn_mem_include_errors",
    )
    if "status" in plot_df.columns and not include_errors:
        plot_df = plot_df[plot_df["status"] == "ok"].copy()

    if "train_sampler_mode" in plot_df.columns:
        sampler_options = sorted(
            plot_df["train_sampler_mode"].dropna().astype(str).unique().tolist()
        )
        if sampler_options:
            selected_samplers = st.multiselect(
                "Sampler mode",
                options=sampler_options,
                default=sampler_options,
                key="live_gnn_mem_sampler_filter",
            )
            if selected_samplers:
                plot_df = plot_df[
                    plot_df["train_sampler_mode"].astype(str).isin(selected_samplers)
                ].copy()

    if "status" not in plot_df.columns:
        plot_df["status"] = "unknown"
    plot_df["status_norm"] = plot_df["status"].astype(str).str.lower()
    if "memory_peak_fraction_budget" in plot_df.columns:
        plot_df["usage_pct"] = 100.0 * plot_df["memory_peak_fraction_budget"]
    else:
        plot_df["usage_pct"] = pd.NA

    def _budget_bucket(row: pd.Series) -> str:
        status = str(row.get("status_norm", "unknown"))
        if status != "ok":
            if "oom" in status:
                return "OOM"
            if "error" in status:
                return "Error"
            return "No-OK"
        val = pd.to_numeric(row.get("usage_pct"), errors="coerce")
        if pd.isna(val):
            return "Sin medición"
        if val < 60:
            return "Muy bajo (<60%)"
        if val < 85:
            return "Bajo (60-85%)"
        if val <= 100:
            return "Objetivo (85-100%)"
        if val <= 110:
            return "Sobre presupuesto (100-110%)"
        return "Muy sobrepresupuesto (>110%)"

    plot_df["budget_state"] = plot_df.apply(_budget_bucket, axis=1)

    if plot_df.empty:
        st.info("No hay datos para los filtros seleccionados.")
        st.dataframe(df, width="stretch")
        return

    if best_row is None:
        valid = plot_df.copy()
        if {"status", "memory_peak_fraction_budget"}.issubset(valid.columns):
            valid = valid[
                (valid["status"] == "ok")
                & valid["memory_peak_fraction_budget"].notna()
            ]
            under = valid[valid["memory_peak_fraction_budget"] <= 1.0]
            if not under.empty:
                best_row = under.loc[
                    under["memory_peak_fraction_budget"].idxmax()
                ].to_dict()
            elif not valid.empty:
                over = valid[valid["memory_peak_fraction_budget"] > 1.0]
                if not over.empty:
                    over = over.assign(
                        _delta=(over["memory_peak_fraction_budget"] - 1.0).abs()
                    )
                    best_row = over.loc[over["_delta"].idxmin()].to_dict()

    ok_df = plot_df[
        (plot_df["status_norm"] == "ok")
        & pd.to_numeric(plot_df["usage_pct"], errors="coerce").notna()
    ].copy()
    under_df = ok_df[ok_df["usage_pct"] <= 100.0].copy()
    over_df = ok_df[ok_df["usage_pct"] > 100.0].copy()

    total_eval = int(len(plot_df))
    total_ok = int(len(ok_df))
    total_under = int(len(under_df))
    under_ratio = (100.0 * total_under / total_ok) if total_ok > 0 else 0.0
    best_under_pct = (
        float(under_df["usage_pct"].max())
        if not under_df.empty
        else None
    )
    min_over_pct = (
        float(over_df["usage_pct"].min())
        if not over_df.empty
        else None
    )

    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
    kpi_a.metric("Evaluaciones", f"{total_eval}")
    kpi_b.metric("Corridas OK", f"{total_ok}")
    kpi_c.metric(
        "Bajo presupuesto",
        f"{total_under}",
        f"{under_ratio:.1f}% de OK",
    )
    if best_under_pct is not None:
        kpi_d.metric(
            "Mejor uso bajo presupuesto",
            f"{best_under_pct:.1f}%",
            f"gap {100.0 - best_under_pct:.1f}%",
        )
    elif min_over_pct is not None:
        kpi_d.metric(
            "Exceso mínimo",
            f"{min_over_pct:.1f}%",
            f"+{min_over_pct - 100.0:.1f}%",
        )
    else:
        kpi_d.metric("Mejor uso bajo presupuesto", "N/A")

    st.caption(
        "Lectura rápida: `85-100%` es zona objetivo, `>100%` excede presupuesto, "
        "`<85%` está subutilizando memoria."
    )

    if best_row:
        st.markdown("**Configuración recomendada**")
        payload = {
            "config_name": best_row.get("config_name"),
            "train_sampler_mode": best_row.get("train_sampler_mode"),
            "num_neighbors": best_row.get("num_neighbors"),
            "batch_size": best_row.get("batch_size"),
            "memory_peak_gb": best_row.get("memory_peak_gb"),
            "memory_peak_fraction_budget": best_row.get("memory_peak_fraction_budget"),
            "budget_gb": best_row.get("budget_gb"),
            "status": best_row.get("status"),
        }
        st.json(payload)

    tab_viz, tab_data = st.tabs(["Grafico", "Datos"])
    with tab_viz:
        if {"batch_size", "usage_pct"}.issubset(plot_df.columns):
            chart_df = plot_df.dropna(
                subset=["batch_size", "usage_pct"]
            ).copy()
            if chart_df.empty:
                st.info("No hay datos numéricos suficientes para graficar.")
            else:
                try:
                    import altair as alt

                    max_usage = float(chart_df["usage_pct"].max())
                    y_max = max(120.0, max_usage + 5.0)
                    color_domain = [
                        "Objetivo (85-100%)",
                        "Bajo (60-85%)",
                        "Muy bajo (<60%)",
                        "Sobre presupuesto (100-110%)",
                        "Muy sobrepresupuesto (>110%)",
                        "OOM",
                        "Error",
                        "No-OK",
                        "Sin medición",
                    ]
                    color_range = [
                        "#2ca02c",
                        "#1f77b4",
                        "#9ecae1",
                        "#ff7f0e",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                        "#7f7f7f",
                        "#c7c7c7",
                    ]
                    color_enc = alt.Color(
                        "budget_state:N",
                        title="Estado",
                        scale=alt.Scale(domain=color_domain, range=color_range),
                    )
                    tooltip = [
                        "config_name",
                        "train_sampler_mode",
                        "batch_size",
                        "usage_pct",
                        "memory_peak_gb",
                        "status",
                        "adaptive_jump",
                        "role",
                    ]
                    base = (
                        alt.Chart()
                        .mark_circle(size=95, opacity=0.9)
                        .encode(
                            x=alt.X("batch_size:Q", axis=alt.Axis(title="Batch size")),
                            y=alt.Y(
                                "usage_pct:Q",
                                axis=alt.Axis(title="Uso del presupuesto (%)"),
                                scale=alt.Scale(domain=[0, y_max]),
                            ),
                            color=color_enc,
                            shape=alt.Shape("status_norm:N", title="Status"),
                            tooltip=tooltip,
                        )
                    )
                    line_85 = (
                        alt.Chart()
                        .mark_rule(color="#1f77b4", strokeDash=[6, 4], opacity=0.8)
                        .encode(y=alt.datum(85.0))
                    )
                    line_95 = (
                        alt.Chart()
                        .mark_rule(color="#2ca02c", strokeDash=[6, 4], opacity=0.8)
                        .encode(y=alt.datum(95.0))
                    )
                    line_100 = (
                        alt.Chart()
                        .mark_rule(color="#d62728", strokeDash=[6, 4], opacity=0.9)
                        .encode(y=alt.datum(100.0))
                    )
                    layered = alt.layer(
                        base,
                        line_85,
                        line_95,
                        line_100,
                        data=chart_df,
                    ).interactive()

                    if (
                        "train_sampler_mode" in chart_df.columns
                        and chart_df["train_sampler_mode"].nunique() > 1
                    ):
                        composed = layered.facet(
                            column=alt.Column(
                                "train_sampler_mode:N",
                                title="Sampler mode",
                                header=alt.Header(labelAngle=0),
                            )
                        )
                        st.altair_chart(composed, width="stretch")
                    else:
                        st.altair_chart(layered, width="stretch")

                    st.markdown("**Top configuraciones bajo presupuesto**")
                    rank_df = chart_df[
                        (chart_df["status_norm"] == "ok")
                        & (chart_df["usage_pct"] <= 100.0)
                    ].copy()
                    if rank_df.empty:
                        st.info("No hay configuraciones bajo presupuesto en los datos actuales.")
                    else:
                        rank_df["rank_label"] = (
                            rank_df["train_sampler_mode"].astype(str)
                            + " | bs="
                            + rank_df["batch_size"].astype(int).astype(str)
                        )
                        rank_df = rank_df.sort_values("usage_pct", ascending=False).head(15)
                        bar = (
                            alt.Chart(rank_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("usage_pct:Q", axis=alt.Axis(title="Uso del presupuesto (%)")),
                                y=alt.Y(
                                    "rank_label:N",
                                    sort="-x",
                                    axis=alt.Axis(title="Configuración"),
                                ),
                                color=alt.Color(
                                    "train_sampler_mode:N",
                                    title="Sampler",
                                ),
                                tooltip=[
                                    "config_name",
                                    "train_sampler_mode",
                                    "batch_size",
                                    "usage_pct",
                                    "memory_peak_gb",
                                ],
                            )
                        )
                        ref_95 = (
                            alt.Chart(pd.DataFrame({"x": [95.0]}))
                            .mark_rule(color="#2ca02c", strokeDash=[6, 4], opacity=0.9)
                            .encode(x="x:Q")
                        )
                        ref_100 = (
                            alt.Chart(pd.DataFrame({"x": [100.0]}))
                            .mark_rule(color="#d62728", strokeDash=[6, 4], opacity=0.9)
                            .encode(x="x:Q")
                        )
                        st.altair_chart((bar + ref_95 + ref_100).interactive(), width="stretch")
                except ImportError:
                    st.warning("Altair no instalado.")
        else:
            st.info("No hay columnas suficientes para graficar.")

    with tab_data:
        view_df = plot_df.copy()
        preferred_cols = [
            "config_name",
            "train_sampler_mode",
            "role",
            "status",
            "probe_mode",
            "sampler_impl",
            "batch_size",
            "usage_pct",
            "memory_peak_gb",
            "budget_gb",
            "budget_state",
            "adaptive_jump",
            "probe_batches",
            "error",
        ]
        cols = [c for c in preferred_cols if c in view_df.columns]
        if cols:
            view_df = view_df[cols]
        if "usage_pct" in view_df.columns:
            view_df = view_df.sort_values("usage_pct", ascending=False, na_position="last")
        st.dataframe(view_df, width="stretch")


def _render_best_highway_section_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: Best highway section")

    plot_df = df.copy()
    if "error" in plot_df.columns:
        plot_df = plot_df[
            plot_df["error"].isna() | (plot_df["error"] == "")
        ]
    if plot_df.empty:
        st.info("No hay resultados validos para graficar.")
        st.dataframe(df, width="stretch")
        return

    dataset_types = []
    if "type" in plot_df.columns:
        dataset_types = sorted(
            [t for t in plot_df["type"].dropna().unique().tolist() if t]
        )
    selected_type = "Todos"
    if dataset_types:
        selected_type = st.selectbox(
            "Dataset",
            ["Todos"] + dataset_types,
            key="live_best_section_type",
        )
    if selected_type != "Todos" and "type" in plot_df.columns:
        plot_df = plot_df[plot_df["type"] == selected_type]
        if plot_df.empty:
            st.info("No hay resultados para el dataset seleccionado.")
            return

    metric_cols = [
        col
        for col in ("accuracy", "recall", "roc_auc")
        if col in plot_df.columns
    ]
    if not metric_cols:
        st.info("No hay metricas disponibles para graficar.")
        st.dataframe(df, width="stretch")
        return

    def _segment_label(row: pd.Series) -> str:
        last = row.get("segment_portico_last")
        nxt = row.get("segment_portico_next")
        eje = row.get("segment_eje")
        calzada = row.get("segment_calzada")
        last = "?" if pd.isna(last) else str(last)
        nxt = "?" if pd.isna(nxt) else str(nxt)
        label = f"{last}->{nxt}"
        if pd.notna(eje) or pd.notna(calzada):
            eje_val = "-" if pd.isna(eje) else str(eje)
            calzada_val = "-" if pd.isna(calzada) else str(calzada)
            label = f"{eje_val}/{calzada_val} {label}"
        return label

    plot_df = plot_df.copy()
    plot_df["segment_label"] = plot_df.apply(_segment_label, axis=1)
    if "segment_index" in plot_df.columns:
        plot_df["segment_index"] = pd.to_numeric(
            plot_df["segment_index"], errors="coerce"
        )
    else:
        plot_df["segment_index"] = range(1, len(plot_df) + 1)

    long_df = plot_df.melt(
        id_vars=["segment_label", "segment_index"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"])
    if long_df.empty:
        st.info("No hay datos validos para graficar.")
        st.dataframe(df, width="stretch")
        return

    metric_labels = {
        "accuracy": "Accuracy",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
    }
    long_df["metric_label"] = long_df["metric"].map(metric_labels)
    order = (
        plot_df.sort_values("segment_index")["segment_label"]
        .drop_duplicates()
        .tolist()
    )

    try:
        import plotly.express as px
    except ImportError:
        pivot = (
            long_df.pivot_table(
                index="segment_label", columns="metric_label", values="value"
            )
            .reindex(order)
        )
        st.line_chart(pivot)
    else:
        fig = px.line(
            long_df,
            x="segment_label",
            y="value",
            color="metric_label",
            markers=True,
            category_orders={"segment_label": order},
        )
        fig.update_layout(
            xaxis_title="Tramo",
            yaxis_title="Metrica",
            legend_title_text="Metrica",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Datos")
    st.dataframe(_streamlit_arrow_safe_df(plot_df), width="stretch")


def _render_drift_block_matrix(block_df: pd.DataFrame) -> None:
    if block_df is None or block_df.empty:
        st.info("No hay bloques registrados en el checkpoint.")
        return
    chart_df = block_df.copy()
    chart_df["status"] = chart_df["status"].astype(str).str.lower()
    status_short = {
        "pending": "P",
        "running": "R",
        "completed": "OK",
        "failed": "ERR",
    }
    chart_df["status_short"] = chart_df["status"].map(status_short).fillna(
        chart_df["status"].str[:3].str.upper()
    )
    block_order = chart_df["block_label"].drop_duplicates().tolist()
    seed_order = (
        chart_df.sort_values(["run_order", "run_seed"])["seed_label"]
        .drop_duplicates()
        .tolist()
    )

    try:
        import altair as alt
    except ImportError:
        table_df = chart_df[
            ["seed_label", "block_label", "status", "strategy", "model", "balance_mode"]
        ].copy()
        st.dataframe(_streamlit_arrow_safe_df(table_df), width="stretch")
        return

    color_domain = ["pending", "running", "completed", "failed"]
    color_range = ["#c7c7c7", "#1f77b4", "#2ca02c", "#d62728"]
    tooltip = [
        "seed_label",
        "block_label",
        "status",
        "strategy",
        "model",
        "detector_variant",
        "balance_mode",
        "saved_at",
    ]
    base = (
        alt.Chart(chart_df)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X(
                "block_label:N",
                title="Bloques",
                sort=block_order,
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y("seed_label:N", title="Seeds", sort=seed_order),
            color=alt.Color(
                "status:N",
                title="Estado",
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=tooltip,
        )
        .properties(height=max(140, 40 * len(seed_order)))
    )
    text = (
        alt.Chart(chart_df)
        .mark_text(color="white", fontSize=10)
        .encode(
            x=alt.X("block_label:N", sort=block_order),
            y=alt.Y("seed_label:N", sort=seed_order),
            text="status_short:N",
        )
    )
    st.altair_chart((base + text).interactive(), width="stretch")


def _render_drift_roc_curves(
    average_roc_df: pd.DataFrame,
    *,
    key_prefix: str = "live_drift_roc",
) -> None:
    if average_roc_df is None or average_roc_df.empty:
        st.info("No hay curvas ROC-AUC disponibles todavia.")
        return

    plot_df = average_roc_df.copy()
    strategies = plot_df["strategy"].astype(str).drop_duplicates().tolist()
    selected_strategies = st.multiselect(
        "Estrategias ROC",
        options=strategies,
        default=strategies,
        key=f"{key_prefix}_strategies",
    )
    if selected_strategies:
        plot_df = plot_df.loc[
            plot_df["strategy"].astype(str).isin([str(item) for item in selected_strategies])
        ].copy()

    if plot_df.empty:
        st.info("No hay curvas ROC-AUC para las estrategias seleccionadas.")
        return

    st.caption("Curvas promedio agregadas desde los bloques completados.")

    try:
        import plotly.express as px
    except ImportError:
        roc_pivot = plot_df.pivot_table(
            index="fpr",
            columns="label",
            values="tpr",
            aggfunc="last",
        ).sort_index()
        st.line_chart(roc_pivot, width="stretch")
    else:
        fig = px.line(
            plot_df,
            x="fpr",
            y="tpr",
            color="label",
            facet_col="strategy",
            facet_col_wrap=2,
            line_group="label",
            hover_data=["strategy", "model", "balance_mode"],
        )
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend_title_text="Serie",
        )
        fig.update_xaxes(range=[0.0, 1.0])
        fig.update_yaxes(range=[0.0, 1.0])
        st.plotly_chart(fig, width="stretch")

    st.dataframe(_streamlit_arrow_safe_df(plot_df), width="stretch")


def _render_drift_recalibration_view(data: Dict[str, object]) -> None:
    manifest = dict(data.get("manifest") or {})
    live_status = dict(data.get("live_status") or {})
    live_events_df = data.get("live_events_df")
    block_df = data.get("block_df")
    summary_df = data.get("summary_df")
    tuning_trials_df = data.get("tuning_trials_df")
    tuning_params_df = data.get("tuning_params_df")
    memory_trace_df = data.get("memory_trace_df")
    average_roc_df = data.get("average_roc_df")
    yearly_df = data.get("yearly_df")
    adaptive_df = data.get("adaptive_df")
    execution_log_df = data.get("execution_log_df")
    live_result_tables = _build_drift_live_result_tables(
        manifest,
        block_df if isinstance(block_df, pd.DataFrame) else pd.DataFrame(),
        yearly_df if isinstance(yearly_df, pd.DataFrame) else pd.DataFrame(),
        adaptive_df if isinstance(adaptive_df, pd.DataFrame) else pd.DataFrame(),
    )

    st.caption("Experimento detectado: Drift recalibration")
    st.caption(f"Checkpoint: {data.get('manifest_path')}")
    st.caption(f"Run dir: {data.get('run_dir')}")

    progress = dict(manifest.get("progress") or {})
    total_units = _safe_int(
        live_status.get("total_units", progress.get("total_units", 0)),
        default=1,
    )
    completed_units = pd.to_numeric(
        live_status.get("completed_units", progress.get("completed_units", 0.0)),
        errors="coerce",
    )
    if pd.isna(completed_units):
        completed_units = 0.0
    progress_ratio = pd.to_numeric(
        live_status.get("progress_ratio"), errors="coerce"
    )
    if pd.isna(progress_ratio):
        progress_ratio = float(completed_units) / float(max(1, total_units))
    progress_ratio = max(0.0, min(float(progress_ratio), 1.0))
    status = str(
        live_status.get("status")
        or manifest.get("status")
        or "unknown"
    )
    updated_at = str(
        live_status.get("timestamp")
        or manifest.get("updated_at")
        or manifest.get("started_at")
        or "-"
    )
    context = dict(live_status.get("context") or {})
    current_label = str(live_status.get("label") or "Sin actividad registrada")
    current_detail = str(live_status.get("detail") or "")

    if status == "failed":
        last_error = dict(manifest.get("last_error") or {})
        st.error(
            f"Checkpoint fallido en fase `{last_error.get('phase', 'desconocida')}`: {last_error.get('error', 'sin detalle')}"
        )
    elif status == "completed":
        st.success("Corrida completada. Se muestran resultados finales y trazas de ejecución.")
    else:
        st.info("Corrida en progreso o reanudable. La vista se actualiza con el checkpoint persistido.")

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5, kpi_6 = st.columns(6)
    kpi_1.metric("Estado", status)
    kpi_2.metric("Progreso", f"{100.0 * progress_ratio:.1f}%")
    kpi_3.metric(
        "Tunings",
        f"{_safe_int(progress.get('completed_tuning_tasks'))}/{_safe_int(progress.get('total_tuning_tasks'))}",
    )
    kpi_4.metric(
        "Bloques",
        f"{_safe_int(progress.get('completed_blocks'))}/{_safe_int(progress.get('total_blocks'))}",
    )
    kpi_5.metric("Ultima actualizacion", updated_at)
    kpi_6.metric("SMOTE cache", f"{len(manifest.get('smote_index') or {})}")

    st.progress(progress_ratio)
    st.caption(current_label)
    if current_detail:
        st.caption(current_detail)

    active_col_1, active_col_2, active_col_3, active_col_4 = st.columns(4)
    active_col_1.metric("Fase activa", str(context.get("phase") or "-"))
    active_col_2.metric("Estrategia", str(context.get("strategy") or "-"))
    active_col_3.metric("Modelo", str(context.get("model") or "-"))
    active_col_4.metric("Seed", str(context.get("run_seed") or "-"))

    live_tab, partial_tab, data_tab = st.tabs(
        ["Live calculations", "Partial results", "Raw data"]
    )

    with live_tab:
        st.markdown("**Progress over time**")
        if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty:
            history_df = live_events_df.copy()
            if "event_index" not in history_df.columns:
                history_df["event_index"] = range(1, len(history_df) + 1)
            if "progress_pct" not in history_df.columns:
                history_df["progress_pct"] = 100.0 * pd.to_numeric(
                    history_df.get("progress_ratio"), errors="coerce"
                )
            plot_df = history_df[["event_index", "progress_pct"]].copy()
            plot_df = plot_df.dropna(subset=["progress_pct"])
            if not plot_df.empty:
                st.line_chart(plot_df.set_index("event_index")["progress_pct"], width="stretch")
            st.dataframe(
                _streamlit_arrow_safe_df(
                    history_df[["event_index", "timestamp", "label", "detail", "progress_pct"]]
                ),
                width="stretch",
            )
        else:
            st.info("No hay eventos live persistidos todavía.")

        st.markdown("**Block execution matrix**")
        _render_drift_block_matrix(
            block_df if isinstance(block_df, pd.DataFrame) else pd.DataFrame()
        )

        st.markdown("**Progress breakdown**")
        block_status_df = (
            pd.DataFrame(columns=["status", "count"])
            if not isinstance(block_df, pd.DataFrame) or block_df.empty
            else block_df["status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
        )
        if not block_status_df.empty:
            st.bar_chart(
                block_status_df.set_index("status")["count"],
                width="stretch",
            )
            st.dataframe(_streamlit_arrow_safe_df(block_status_df), width="stretch")
        else:
            st.info("No hay estados de bloques para resumir.")

        st.markdown("**Tuning evolution**")
        if isinstance(tuning_trials_df, pd.DataFrame) and not tuning_trials_df.empty:
            tuning_plot = tuning_trials_df.copy()
            tuning_plot["series_label"] = (
                tuning_plot["model"].astype(str)
                + " | "
                + tuning_plot["balance_mode"].astype(str)
                + " | "
                + tuning_plot["stage"].astype(str)
            )
            tuning_plot["best_so_far"] = tuning_plot.groupby("series_label")["cv_auc"].cummax()
            tuning_pivot = tuning_plot.pivot_table(
                index="trial_number",
                columns="series_label",
                values="best_so_far",
                aggfunc="max",
            ).sort_index()
            st.line_chart(tuning_pivot, width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(tuning_plot), width="stretch")
        else:
            st.info("No hay tuning trials persistidos todavía.")

        st.markdown("**Execution resource trace**")
        if isinstance(memory_trace_df, pd.DataFrame) and not memory_trace_df.empty:
            trace_metric = st.selectbox(
                "Trace metric",
                options=sorted(memory_trace_df["metric"].astype(str).unique().tolist()),
                index=0,
                key="live_drift_trace_metric",
            )
            trace_plot = memory_trace_df.loc[
                memory_trace_df["metric"].astype(str) == str(trace_metric)
            ].copy()
            if not trace_plot.empty:
                trace_plot["series_label"] = (
                    trace_plot["phase"].astype(str) + " | " + trace_plot["status"].astype(str)
                )
                trace_pivot = trace_plot.pivot_table(
                    index="order",
                    columns="series_label",
                    values="value",
                    aggfunc="last",
                ).sort_index()
                st.line_chart(trace_pivot, width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(trace_plot), width="stretch")
        else:
            st.info("No hay trazas de recursos disponibles.")

    with partial_tab:
        st.markdown("**Experiment result tables**")
        has_live_tables = any(
            isinstance(table_df, pd.DataFrame) and not table_df.empty
            for table_df in live_result_tables.values()
        )
        if has_live_tables:
            for table_key in ["A.6", "A.7", "A.8", "A.9"]:
                table_df = live_result_tables.get(table_key)
                st.caption(f"Table {table_key}")
                if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                    st.dataframe(_streamlit_arrow_safe_df(table_df), width="stretch")
                else:
                    st.info(f"Table {table_key} no aplica a la configuracion actual.")
        else:
            st.info("No fue posible reconstruir tablas live para la corrida actual.")
        st.caption("Table H.1")
        if isinstance(tuning_params_df, pd.DataFrame) and not tuning_params_df.empty:
            st.caption(
                "Mejor configuracion por proceso de tuning. `cv_auc_delta_vs_none` compara contra la corrida equivalente sin balanceo cuando existe."
            )
            st.dataframe(_streamlit_arrow_safe_df(tuning_params_df), width="stretch")
        else:
            st.info("No hay artefactos de tuning persistidos todavía.")

        st.markdown("**ROC-AUC curves by strategy**")
        _render_drift_roc_curves(
            average_roc_df if isinstance(average_roc_df, pd.DataFrame) else pd.DataFrame(),
            key_prefix="partial_drift_roc",
        )

        st.markdown("**Strategy summary**")
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            if "auc" in summary_df.columns and summary_df["auc"].notna().any():
                chart_df = summary_df.copy()
                detector_variant = (
                    chart_df["detector_variant"].astype(str)
                    if "detector_variant" in chart_df.columns
                    else pd.Series("-", index=chart_df.index)
                )
                chart_df["series_label"] = chart_df["strategy"].astype(str) + " | " + chart_df["model"].astype(str)
                chart_df.loc[detector_variant.ne("-"), "series_label"] = (
                    chart_df.loc[detector_variant.ne("-"), "series_label"]
                    + " | "
                    + detector_variant.loc[detector_variant.ne("-")]
                )
                chart_df["series_label"] = (
                    chart_df["series_label"] + " | " + chart_df["balance_mode"].astype(str)
                )
                st.bar_chart(
                    chart_df.set_index("series_label")["auc"],
                    width="stretch",
                )
            st.dataframe(_streamlit_arrow_safe_df(summary_df), width="stretch")
        else:
            st.info("Aún no hay resultados parciales agregables.")

        st.markdown("**Completed blocks**")
        if isinstance(block_df, pd.DataFrame) and not block_df.empty:
            completed_blocks = block_df.loc[
                block_df["status"].astype(str).str.lower() == "completed"
            ].copy()
            if not completed_blocks.empty:
                st.dataframe(_streamlit_arrow_safe_df(completed_blocks), width="stretch")
            else:
                st.info("Todavía no hay bloques completados.")
        else:
            st.info("No hay bloque index persistido.")

        st.markdown("**Accumulated rows**")
        acc_col_1, acc_col_2, acc_col_3 = st.columns(3)
        acc_col_1.metric(
            "Yearly rows",
            str(len(yearly_df)) if isinstance(yearly_df, pd.DataFrame) else "0",
        )
        acc_col_2.metric(
            "Adaptive rows",
            str(len(adaptive_df)) if isinstance(adaptive_df, pd.DataFrame) else "0",
        )
        acc_col_3.metric(
            "Execution log rows",
            str(len(execution_log_df))
            if isinstance(execution_log_df, pd.DataFrame)
            else "0",
        )
        if isinstance(yearly_df, pd.DataFrame) and not yearly_df.empty:
            st.markdown("**Yearly preview**")
            st.dataframe(_streamlit_arrow_safe_df(yearly_df), width="stretch")
        if isinstance(adaptive_df, pd.DataFrame) and not adaptive_df.empty:
            st.markdown("**Adaptive preview**")
            st.dataframe(_streamlit_arrow_safe_df(adaptive_df), width="stretch")

    with data_tab:
        st.markdown("**Manifest**")
        st.json(manifest, expanded=False)
        if live_status:
            st.markdown("**Live status**")
            st.json(live_status, expanded=False)
        if isinstance(block_df, pd.DataFrame) and not block_df.empty:
            st.markdown("**Block index**")
            st.dataframe(_streamlit_arrow_safe_df(block_df), width="stretch")
        if isinstance(tuning_trials_df, pd.DataFrame) and not tuning_trials_df.empty:
            st.markdown("**Tuning trials**")
            st.dataframe(_streamlit_arrow_safe_df(tuning_trials_df), width="stretch")
        if isinstance(execution_log_df, pd.DataFrame) and not execution_log_df.empty:
            st.markdown("**Execution log**")
            st.dataframe(_streamlit_arrow_safe_df(execution_log_df), width="stretch")


def _render_paper_replication_live_view(data: Dict[str, object]) -> None:
    manifest = dict(data.get("manifest") or {})
    live_status = dict(data.get("live_status") or {})
    live_events_df = data.get("live_events_df")
    step_df = data.get("step_df")
    route_status_df = data.get("route_status_df")
    partial_models_df = data.get("partial_models_df")
    k_progress_df = data.get("k_progress_df")
    compare_summary_df = data.get("compare_summary_df")
    compare_diff_df = data.get("compare_diff_df")
    export_summary_df = data.get("export_summary_df")
    current_context = dict(data.get("current_context") or {})

    st.caption("Experimento detectado: Paper replication")
    st.caption(f"Checkpoint: {data.get('manifest_path')}")
    st.caption(f"Run dir: {data.get('run_dir')}")

    progress = dict(manifest.get("progress") or {})
    total_units = pd.to_numeric(progress.get("total_units", 0.0), errors="coerce")
    completed_units = pd.to_numeric(progress.get("completed_units", 0.0), errors="coerce")
    if pd.isna(total_units):
        total_units = 0.0
    if pd.isna(completed_units):
        completed_units = 0.0
    progress_ratio = float(completed_units) / float(max(1.0, float(total_units or 0.0)))
    if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty and "progress_ratio" in live_events_df.columns:
        live_ratio = pd.to_numeric(live_events_df["progress_ratio"], errors="coerce").dropna()
        if not live_ratio.empty:
            progress_ratio = float(live_ratio.iloc[-1])
    progress_ratio = max(0.0, min(progress_ratio, 1.0))
    status = str(manifest.get("status") or current_context.get("status") or "unknown")
    result_status = str(manifest.get("result_status") or current_context.get("result_status") or status)
    updated_at = str(
        current_context.get("updated_at")
        or live_status.get("updated_at")
        or manifest.get("updated_at")
        or manifest.get("created_at")
        or "-"
    )
    current_stage = str(current_context.get("current_stage") or progress.get("current_stage") or "-")
    current_step_id = str(current_context.get("current_step_id") or progress.get("current_step_id") or "-")
    current_message = str(
        current_context.get("message")
        or live_status.get("message")
        or live_status.get("detail")
        or ""
    )

    if status == "failed":
        st.error(f"Corrida fallida: {manifest.get('last_error') or 'sin detalle persistido'}.")
    elif status == "completed" and result_status == "blocked":
        st.warning("Corrida completada con bloqueo metodologico. Se muestran resultados y diff persistidos.")
    elif status == "completed":
        st.success("Corrida completada. Se muestran resultados finales y parciales persistidos.")
    else:
        st.info("Corrida en progreso o reanudable. La vista usa el checkpoint persistido.")

    frozen_status = "-"
    raw_status = "-"
    compare_status = "-"
    latex_status = "No"
    if isinstance(route_status_df, pd.DataFrame) and not route_status_df.empty:
        for stage_name, target in [("frozen", "frozen_status"), ("raw", "raw_status"), ("compare", "compare_status")]:
            match = route_status_df.loc[route_status_df["stage"].astype(str) == stage_name]
            if not match.empty:
                value = str(match.iloc[0]["status"] or "-")
                if target == "frozen_status":
                    frozen_status = value
                elif target == "raw_status":
                    raw_status = value
                else:
                    compare_status = value
    if isinstance(export_summary_df, pd.DataFrame) and not export_summary_df.empty:
        latex_status = "Si" if bool(export_summary_df.iloc[0]["latex_promoted"]) else "No"

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5, kpi_6 = st.columns(6)
    kpi_1.metric("Estado", status)
    kpi_2.metric("Resultado", result_status)
    kpi_3.metric("Progreso", f"{100.0 * progress_ratio:.1f}%")
    kpi_4.metric(
        "Pasos",
        f"{_safe_int(progress.get('completed_steps'))}/{_safe_int(progress.get('total_steps'))}",
    )
    kpi_5.metric("Frozen / Raw", f"{frozen_status} / {raw_status}")
    kpi_6.metric("Compare / LaTeX", f"{compare_status} / {latex_status}")

    st.progress(progress_ratio)
    st.caption(f"Etapa activa: {current_stage}")
    st.caption(f"Step activo: {current_step_id}")
    if current_message:
        st.caption(current_message)
    st.caption(f"Ultima actualizacion: {updated_at}")

    live_tab, partial_tab, data_tab = st.tabs(
        ["Live calculations", "Partial results", "Raw data"]
    )

    with live_tab:
        st.markdown("**Progress over time**")
        if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty:
            history_df = live_events_df.copy()
            if "event_index" not in history_df.columns:
                history_df["event_index"] = range(1, len(history_df) + 1)
            plot_df = history_df[["event_index", "progress_pct"]].dropna(subset=["progress_pct"])
            if not plot_df.empty:
                st.line_chart(plot_df.set_index("event_index")["progress_pct"], width="stretch")
            columns = [
                col
                for col in ["event_index", "timestamp", "stage", "step_id", "step_status", "label", "detail", "progress_pct"]
                if col in history_df.columns
            ]
            st.dataframe(_streamlit_arrow_safe_df(history_df[columns]), width="stretch")
        else:
            st.info("No hay eventos live persistidos todavía.")

        st.markdown("**Step execution matrix**")
        if isinstance(step_df, pd.DataFrame) and not step_df.empty:
            step_status_df = (
                step_df["status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
            )
            if not step_status_df.empty:
                st.bar_chart(step_status_df.set_index("status")["count"], width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(step_df), width="stretch")
        else:
            st.info("No hay steps persistidos todavía.")

    with partial_tab:
        st.markdown("**Route summary**")
        if isinstance(route_status_df, pd.DataFrame) and not route_status_df.empty:
            st.dataframe(_streamlit_arrow_safe_df(route_status_df), width="stretch")
        else:
            st.info("No hay resumen parcial de rutas disponible.")

        st.markdown("**Partial model results**")
        if isinstance(partial_models_df, pd.DataFrame) and not partial_models_df.empty:
            st.dataframe(_streamlit_arrow_safe_df(partial_models_df), width="stretch")
        else:
            st.info("Todavia no hay modelos finales ni parciales persistidos.")

        st.markdown("**k-search evolution (nested CV on training folds)**")
        if isinstance(k_progress_df, pd.DataFrame) and not k_progress_df.empty:
            st.caption("Los resultados finales de la tabla de modelos usan el holdout final temporal.")
            route_options = sorted(k_progress_df["route_name"].astype(str).unique().tolist())
            selected_route = st.selectbox(
                "Route",
                options=route_options,
                index=0,
                key="paper_live_route_filter",
            )
            route_k_df = k_progress_df.loc[
                k_progress_df["route_name"].astype(str) == str(selected_route)
            ].copy()
            model_options = sorted(route_k_df["model_code"].astype(str).unique().tolist())
            selected_models = st.multiselect(
                "Model codes",
                options=model_options,
                default=model_options,
                key="paper_live_model_filter",
            )
            if selected_models:
                route_k_df = route_k_df.loc[
                    route_k_df["model_code"].astype(str).isin([str(item) for item in selected_models])
                ].copy()
            route_k_df["series_label"] = (
                route_k_df["route_name"].astype(str) + " | " + route_k_df["model_code"].astype(str)
            )
            for metric_name in ["accuracy", "f1_score", "false_negatives_pct", "validation_score"]:
                if metric_name not in route_k_df.columns:
                    continue
                plot_df = route_k_df[["k", "series_label", metric_name]].copy()
                plot_df["k"] = pd.to_numeric(plot_df["k"], errors="coerce")
                plot_df[metric_name] = pd.to_numeric(plot_df[metric_name], errors="coerce")
                plot_df = plot_df.dropna(subset=["k", metric_name])
                if plot_df.empty:
                    continue
                st.caption(metric_name)
                pivot_df = plot_df.pivot_table(
                    index="k",
                    columns="series_label",
                    values=metric_name,
                    aggfunc="last",
                ).sort_index()
                st.line_chart(pivot_df, width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(route_k_df), width="stretch")
        else:
            st.info("No hay resultados parciales de k persistidos todavía.")

        st.markdown("**Compare status**")
        if isinstance(compare_summary_df, pd.DataFrame) and not compare_summary_df.empty:
            st.dataframe(_streamlit_arrow_safe_df(compare_summary_df), width="stretch")
            if isinstance(compare_diff_df, pd.DataFrame) and not compare_diff_df.empty:
                st.dataframe(_streamlit_arrow_safe_df(compare_diff_df), width="stretch")
        else:
            st.info("No hay payload de comparacion disponible todavía.")

        st.markdown("**Export status**")
        if isinstance(export_summary_df, pd.DataFrame) and not export_summary_df.empty:
            st.dataframe(_streamlit_arrow_safe_df(export_summary_df), width="stretch")
            export_payload = dict(data.get("export_payload") or {})
            asset_rows = [
                {
                    "asset_name": str(asset_name),
                    "candidate_path": str(candidate_path),
                    "promoted_path": str((export_payload.get("promoted_paths") or {}).get(str(asset_name)) or ""),
                }
                for asset_name, candidate_path in (export_payload.get("candidate_paths") or {}).items()
            ]
            if asset_rows:
                st.dataframe(_streamlit_arrow_safe_df(pd.DataFrame(asset_rows)), width="stretch")
        else:
            st.info("No hay payload de exportacion disponible todavía.")

    with data_tab:
        st.markdown("**Manifest**")
        st.json(manifest, expanded=False)
        if live_status:
            st.markdown("**Live status**")
            st.json(live_status, expanded=False)
        if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty:
            st.markdown("**Live events**")
            st.dataframe(_streamlit_arrow_safe_df(live_events_df), width="stretch")
        st.markdown("**Payload summaries**")
        st.json(
            {
                "frozen": _paper_payload_summary(data.get("frozen_payload") or {}),
                "raw_build": _paper_payload_summary(data.get("raw_build_payload") or {}),
                "raw": _paper_payload_summary(data.get("raw_payload") or {}),
                "compare": _paper_payload_summary(data.get("compare_payload") or {}),
                "export": _paper_payload_summary(data.get("export_payload") or {}),
            },
            expanded=False,
        )


def _render_language_modeling_history(
    history_df: pd.DataFrame,
) -> None:
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        st.info("No hay historial de entrenamiento persistido todavía.")
        return
    work = history_df.copy()
    axis_col = "epoch" if "epoch" in work.columns else "global_step" if "global_step" in work.columns else None
    if axis_col is None:
        work["history_index"] = range(1, len(work) + 1)
        axis_col = "history_index"
    work[axis_col] = pd.to_numeric(work[axis_col], errors="coerce")
    preferred_metrics = [
        "loss",
        "eval_loss",
        "eval_accuracy",
        "eval_f1",
        "eval_balanced_f1",
        "learning_rate",
    ]
    metric_cols = [col for col in preferred_metrics if col in work.columns]
    if not metric_cols:
        metric_cols = [
            col
            for col in work.columns
            if col != axis_col
            and pd.to_numeric(work[col], errors="coerce").notna().any()
        ][:6]
    if metric_cols:
        plot_df = work[[axis_col, *metric_cols]].copy()
        for col in metric_cols:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=[axis_col])
        if not plot_df.empty:
            st.line_chart(plot_df.set_index(axis_col)[metric_cols], width="stretch")
    st.dataframe(_streamlit_arrow_safe_df(work), width="stretch")


def _render_language_modeling_live_view(data: Dict[str, object]) -> None:
    manifest = dict(data.get("manifest") or {})
    live_status = dict(data.get("live_status") or {})
    live_events_df = data.get("live_events_df")
    search_trials_df = data.get("search_trials_df")
    confirmation_trials_df = data.get("confirmation_trials_df")
    confirmation_summary_df = data.get("confirmation_summary_df")
    history_df = data.get("history_df")
    best_history_df = data.get("best_history_df")
    search_summary = dict(data.get("search_summary") or {})
    best_result = dict(data.get("best_result") or {})
    finetune_result = dict(data.get("finetune_result") or {})
    failure_summary = dict(data.get("failure_summary") or {})
    current_context = dict(data.get("current_context") or {})

    run_type = str(manifest.get("run_type") or "language_modeling")
    title = str(manifest.get("title") or manifest.get("run_id") or run_type)
    status = str(current_context.get("status") or manifest.get("status") or "unknown")
    result_status = str(current_context.get("result_status") or manifest.get("result_status") or status)
    progress_ratio = pd.to_numeric(manifest.get("progress_ratio", 0.0), errors="coerce")
    if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty and "progress_ratio" in live_events_df.columns:
        live_ratio = pd.to_numeric(live_events_df["progress_ratio"], errors="coerce").dropna()
        if not live_ratio.empty:
            progress_ratio = live_ratio.iloc[-1]
    if pd.isna(progress_ratio):
        progress_ratio = 0.0
    progress_ratio = max(0.0, min(float(progress_ratio), 1.0))
    current_stage = str(current_context.get("stage") or live_status.get("stage") or "-")
    current_message = str(current_context.get("message") or live_status.get("message") or "")
    updated_at = str(current_context.get("updated_at") or live_status.get("updated_at") or manifest.get("updated_at") or "-")

    st.caption("Experimento detectado: Language modeling")
    st.caption(f"Checkpoint: {data.get('manifest_path')}")
    st.caption(f"Run dir: {data.get('run_dir')}")

    if status == "failed":
        st.error(manifest.get("last_error") or current_message or "Corrida fallida sin detalle persistido.")
    elif status == "completed":
        st.success("Corrida completada. Se muestran trazas y artefactos persistidos.")
    else:
        st.info("Corrida en progreso. La vista usa el tracker live persistido.")

    history_source = (
        history_df
        if isinstance(history_df, pd.DataFrame) and not history_df.empty
        else best_history_df
    )
    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5, kpi_6 = st.columns(6)
    kpi_1.metric("Run type", run_type)
    kpi_2.metric("Estado", status)
    kpi_3.metric("Resultado", result_status)
    kpi_4.metric("Progreso", f"{100.0 * progress_ratio:.1f}%")
    kpi_5.metric(
        "Trials",
        f"{int(len(search_trials_df)):,}" if isinstance(search_trials_df, pd.DataFrame) and not search_trials_df.empty else "0",
    )
    kpi_6.metric(
        "Hist rows",
        f"{int(len(history_source)):,}" if isinstance(history_source, pd.DataFrame) and not history_source.empty else "0",
    )

    st.progress(progress_ratio)
    st.caption(f"Titulo: {title}")
    st.caption(f"Etapa activa: {current_stage}")
    if current_message:
        st.caption(current_message)
    st.caption(f"Ultima actualizacion: {updated_at}")

    live_tab, artifact_tab, data_tab = st.tabs(
        ["Live calculations", "Artifacts", "Raw data"]
    )

    with live_tab:
        st.markdown("**Progress over time**")
        if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty:
            history_live_df = live_events_df.copy()
            if "event_index" not in history_live_df.columns:
                history_live_df["event_index"] = range(1, len(history_live_df) + 1)
            plot_df = history_live_df[["event_index", "progress_pct"]].dropna(subset=["progress_pct"])
            if not plot_df.empty:
                st.line_chart(plot_df.set_index("event_index")["progress_pct"], width="stretch")
            visible_cols = [
                col
                for col in ["event_index", "timestamp", "stage", "event_type", "label", "progress_pct"]
                if col in history_live_df.columns
            ]
            st.dataframe(_streamlit_arrow_safe_df(history_live_df[visible_cols]), width="stretch")
        else:
            st.info("No hay eventos live persistidos todavía.")

        st.markdown("**Robust hyperparameter search**")
        if isinstance(search_trials_df, pd.DataFrame) and not search_trials_df.empty:
            search_curve_df = _language_modeling_search_curve(
                search_trials_df,
                greater_is_better=bool(search_summary.get("greater_is_better", True)),
            )
            if not search_curve_df.empty:
                st.line_chart(search_curve_df.set_index("trial_index")[["objective", "best_so_far"]], width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(search_trials_df), width="stretch")
        else:
            st.info("No hay trials de busqueda persistidos todavía.")

        st.markdown("**Confirmation stage**")
        if isinstance(confirmation_trials_df, pd.DataFrame) and not confirmation_trials_df.empty:
            st.dataframe(_streamlit_arrow_safe_df(confirmation_trials_df), width="stretch")
            if isinstance(confirmation_summary_df, pd.DataFrame) and not confirmation_summary_df.empty:
                st.dataframe(_streamlit_arrow_safe_df(confirmation_summary_df), width="stretch")
        else:
            st.info("No hay corridas de confirmacion persistidas todavía.")

        st.markdown("**Language model fine-tuning**")
        _render_language_modeling_history(
            history_source if isinstance(history_source, pd.DataFrame) else pd.DataFrame()
        )

    with artifact_tab:
        if search_summary:
            st.markdown("**Search summary**")
            st.json(search_summary, expanded=False)
        if best_result:
            st.markdown("**Best search result**")
            st.json(best_result, expanded=False)
        if finetune_result:
            st.markdown("**Fine-tune result**")
            st.json(finetune_result, expanded=False)
        if failure_summary:
            st.markdown("**Failure summary**")
            st.json(failure_summary, expanded=False)
        if not any([search_summary, best_result, finetune_result, failure_summary]):
            st.info("No hay resumenes de artefactos persistidos todavía.")

    with data_tab:
        st.markdown("**Manifest**")
        st.json(manifest, expanded=False)
        if live_status:
            st.markdown("**Live status**")
            st.json(live_status, expanded=False)
        if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty:
            st.markdown("**Live events**")
            st.dataframe(_streamlit_arrow_safe_df(live_events_df), width="stretch")


def _render_neural_drift_experiment_live_view(data: Dict[str, object]) -> None:
    manifest = dict(data.get("manifest") or {})
    live_status = dict(data.get("live_status") or {})
    live_events_df = data.get("live_events_df")
    phase_status_df = data.get("phase_status_df")
    baseline_seed_df = data.get("baseline_seed_df")
    leaderboard_dev = data.get("leaderboard_dev")
    leaderboard_holdout = data.get("leaderboard_holdout")
    monthly_metrics = data.get("monthly_metrics")
    pairwise_stats = data.get("pairwise_stats")
    param_importances = data.get("param_importances")
    pareto = data.get("pareto")
    winner_config = dict(data.get("winner_config") or {})

    progress = dict(manifest.get("progress") or {})
    total_units = pd.to_numeric(
        live_status.get("total_units", progress.get("total_units", 1.0)),
        errors="coerce",
    )
    completed_units = pd.to_numeric(
        live_status.get("completed_units", progress.get("completed_units", 0.0)),
        errors="coerce",
    )
    progress_ratio = pd.to_numeric(
        live_status.get("progress_ratio", progress.get("progress_ratio", 0.0)),
        errors="coerce",
    )
    if pd.isna(total_units) or float(total_units) <= 0.0:
        total_units = 1.0
    if pd.isna(completed_units):
        completed_units = 0.0
    if pd.isna(progress_ratio):
        progress_ratio = float(completed_units) / float(max(1.0, float(total_units)))
    progress_ratio = max(0.0, min(float(progress_ratio), 1.0))

    status = str(live_status.get("status") or manifest.get("status") or "unknown")
    result_status = str(live_status.get("result_status") or manifest.get("result_status") or status)
    updated_at = str(
        live_status.get("timestamp")
        or manifest.get("updated_at")
        or manifest.get("created_at")
        or "-"
    )
    current_label = str(live_status.get("label") or "Sin actividad registrada")
    current_detail = str(live_status.get("detail") or "")
    current_context = dict(live_status.get("context") or {})

    st.caption("Experimento detectado: Neural drift experiments")
    st.caption(f"Checkpoint: {data.get('manifest_path')}")
    st.caption(f"Run dir: {data.get('run_dir')}")

    if status == "failed":
        st.error(manifest.get("last_error") or "Corrida fallida sin detalle persistido.")
    elif status == "completed":
        st.success("Corrida completada. Se muestran resultados y trazas parciales/finales.")
    else:
        st.info("Corrida en progreso. La vista se alimenta del checkpoint persistido.")

    completed_phases = 0
    total_phases = 0
    if isinstance(phase_status_df, pd.DataFrame) and not phase_status_df.empty:
        total_phases = int(len(phase_status_df))
        completed_phases = int(
            phase_status_df["status"].astype(str).str.lower().eq("completed").sum()
        )
    baseline_completed = int(len(baseline_seed_df)) if isinstance(baseline_seed_df, pd.DataFrame) else 0

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5, kpi_6 = st.columns(6)
    kpi_1.metric("Estado", status)
    kpi_2.metric("Resultado", result_status)
    kpi_3.metric("Progreso", f"{100.0 * progress_ratio:.1f}%")
    kpi_4.metric("Baseline seeds", f"{baseline_completed}/3")
    kpi_5.metric("Fases", f"{completed_phases}/{total_phases}")
    kpi_6.metric("Ultima actualizacion", updated_at)

    st.progress(progress_ratio)
    st.caption(current_label)
    if current_detail:
        st.caption(current_detail)

    active_col_1, active_col_2, active_col_3, active_col_4 = st.columns(4)
    active_col_1.metric("Study activo", str(current_context.get("study") or manifest.get("selected_study") or "-"))
    active_col_2.metric("Phase activa", str(current_context.get("phase") or manifest.get("selected_phase") or "-"))
    active_col_3.metric("Balance", str(manifest.get("balance_mode") or "-"))
    active_col_4.metric("Run id", str(manifest.get("run_id") or "-"))

    live_tab, partial_tab, data_tab = st.tabs(
        ["Live calculations", "Partial results", "Raw data"]
    )

    with live_tab:
        st.markdown("**Progress over time**")
        if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty:
            history_df = live_events_df.copy()
            plot_df = history_df[["event_index", "progress_pct"]].dropna(subset=["progress_pct"])
            if not plot_df.empty:
                st.line_chart(plot_df.set_index("event_index")["progress_pct"], width="stretch")
            visible_cols = [
                col
                for col in ["event_index", "timestamp", "event_type", "label", "detail", "study", "phase", "progress_pct"]
                if col in history_df.columns
            ]
            st.dataframe(_streamlit_arrow_safe_df(history_df[visible_cols]), width="stretch")
        else:
            st.info("No hay eventos live persistidos todavía.")

        st.markdown("**Phase execution matrix**")
        if isinstance(phase_status_df, pd.DataFrame) and not phase_status_df.empty:
            status_df = (
                phase_status_df["status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
            )
            if not status_df.empty:
                st.bar_chart(status_df.set_index("status")["count"], width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(phase_status_df), width="stretch")
        else:
            st.info("No hay fases persistidas todavía.")

        st.markdown("**Baseline cumulative**")
        if isinstance(baseline_seed_df, pd.DataFrame) and not baseline_seed_df.empty:
            st.dataframe(_streamlit_arrow_safe_df(baseline_seed_df), width="stretch")
        else:
            st.info("No hay seeds de baseline persistidas todavía.")

    with partial_tab:
        st.markdown("**Leaderboard dev**")
        if isinstance(leaderboard_dev, pd.DataFrame) and not leaderboard_dev.empty:
            st.dataframe(_streamlit_arrow_safe_df(leaderboard_dev), width="stretch")
        else:
            st.info("Todavía no hay leaderboard dev persistido.")

        st.markdown("**Leaderboard holdout**")
        if isinstance(leaderboard_holdout, pd.DataFrame) and not leaderboard_holdout.empty:
            st.dataframe(_streamlit_arrow_safe_df(leaderboard_holdout), width="stretch")
        else:
            st.info("Todavía no hay leaderboard holdout persistido.")

        st.markdown("**PR-AUC mensual**")
        if isinstance(monthly_metrics, pd.DataFrame) and not monthly_metrics.empty:
            plot_df = monthly_metrics.copy()
            if {"month", "label", "pr_auc"} <= set(plot_df.columns):
                split_options = sorted(plot_df["split"].astype(str).dropna().unique().tolist()) if "split" in plot_df.columns else []
                selected_split = split_options[0] if len(split_options) == 1 else None
                if len(split_options) > 1:
                    selected_split = st.selectbox(
                        "Split mensual",
                        options=split_options,
                        index=0,
                        key="neural_drift_experiment_live_split",
                    )
                if selected_split is not None and "split" in plot_df.columns:
                    plot_df = plot_df.loc[plot_df["split"].astype(str) == str(selected_split)].copy()
                pivot_df = plot_df.pivot_table(
                    index="month",
                    columns="label",
                    values="pr_auc",
                    aggfunc="last",
                ).sort_index()
                if not pivot_df.empty:
                    st.line_chart(pivot_df, width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(monthly_metrics), width="stretch")
        else:
            st.info("No hay métricas mensuales persistidas todavía.")

        st.markdown("**Comparaciones estadísticas**")
        if isinstance(pairwise_stats, pd.DataFrame) and not pairwise_stats.empty:
            st.dataframe(_streamlit_arrow_safe_df(pairwise_stats), width="stretch")
        else:
            st.info("No hay comparaciones pareadas persistidas todavía.")

        st.markdown("**Importancias fANOVA**")
        if isinstance(param_importances, pd.DataFrame) and not param_importances.empty:
            st.dataframe(_streamlit_arrow_safe_df(param_importances), width="stretch")
        else:
            st.info("No hay importancias persistidas todavía.")

        st.markdown("**Pareto frontier**")
        if isinstance(pareto, pd.DataFrame) and not pareto.empty:
            st.dataframe(_streamlit_arrow_safe_df(pareto), width="stretch")
        else:
            st.info("No hay Pareto persistido todavía.")

        st.markdown("**Winner config**")
        if winner_config:
            st.json(winner_config, expanded=False)
        else:
            st.info("No hay winner config persistido todavía.")

    with data_tab:
        st.markdown("**Manifest**")
        st.json(manifest, expanded=False)
        if live_status:
            st.markdown("**Live status**")
            st.json(live_status, expanded=False)
        if isinstance(live_events_df, pd.DataFrame) and not live_events_df.empty:
            st.markdown("**Live events**")
            st.dataframe(_streamlit_arrow_safe_df(live_events_df), width="stretch")
        if isinstance(phase_status_df, pd.DataFrame) and not phase_status_df.empty:
            st.markdown("**Phase status**")
            st.dataframe(_streamlit_arrow_safe_df(phase_status_df), width="stretch")


def main(*, set_page_config: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Experiments Live", layout="wide")
    st.title("Experimentos en vivo")

    sources = _build_live_sources()
    if not sources:
        st.info("No hay fuentes live disponibles.")
        return

    selected_idx = st.selectbox(
        "Fuente en vivo",
        options=list(range(len(sources))),
        index=0,
        format_func=lambda idx: str(sources[idx]["label"]),
    )

    auto_refresh = st.sidebar.checkbox(
        "Actualizar automaticamente", value=True
    )
    refresh_seconds = st.sidebar.number_input(
        "Intervalo (segundos)",
        min_value=1,
        value=10,
        step=1,
    )
    if st.sidebar.button("Actualizar ahora"):
        st.rerun()

    source = sources[int(selected_idx)]
    source_type = str(source.get("type") or "")
    path = Path(str(source.get("path")))
    if source_type == "drift_recalibration":
        run_data = _read_drift_run(path)
        _render_drift_recalibration_view(run_data)
    elif source_type == "neural_drift_experiment":
        run_data = _read_neural_drift_experiment_run(path)
        _render_neural_drift_experiment_live_view(run_data)
    elif source_type == "paper_replication":
        run_data = _read_paper_replication_run(path)
        _render_paper_replication_live_view(run_data)
    elif source_type == "language_modeling":
        run_data = _read_language_modeling_run(path)
        _render_language_modeling_live_view(run_data)
    else:
        meta, df, best_row = _read_live_db(path)
        st.caption(f"Archivo: {path}")
        if meta:
            with st.expander("Meta", expanded=False):
                st.json(meta)

        if df.empty:
            st.warning("No hay resultados en la base de datos.")
        else:
            experiment_name = str(meta.get("experiment", "")).lower()
            is_controlled_comparison = (
                "controlled comparison" in experiment_name
                or "comparación controlada" in experiment_name
                or df.get("experiment", pd.Series())
                .astype(str)
                .str.contains(
                    "controlled comparison|comparaci[oó]n controlada",
                    case=False,
                    na=False,
                )
                .any()
                or {
                    "model_name",
                    "feature_set",
                    "balance_mode",
                    "k",
                    "val_objective_score",
                }.issubset(df.columns)
            )
            is_find_samples = (
                "find samples" in experiment_name
                or df.get("experiment", pd.Series())
                .astype(str)
                .str.contains("find samples", case=False, na=False)
                .any()
                or "candidate_rank" in df.columns
            )
            is_best_section = (
                "best highway section" in experiment_name
                or df.get("experiment", pd.Series())
                .astype(str)
                .str.contains("best highway section", case=False, na=False)
                .any()
            )
            is_gnn_recursive = (
                "opt.recursiva" in experiment_name
                or "opt recursiva" in experiment_name
                or df.get("experiment", pd.Series())
                .astype(str)
                .str.contains("opt\\.recursiva|opt recursiva", case=False, na=False)
                .any()
                or {"optimizer", "iteration"}.issubset(df.columns)
            )
            is_gnn_optuna = (
                "gnn optuna" in experiment_name
                or df.get("experiment", pd.Series())
                .astype(str)
                .str.contains("gnn optuna", case=False, na=False)
                .any()
                or {"objective_label", "test_f1"}.issubset(df.columns)
            )
            is_gnn_sampler_memory = (
                "sampler memory budget" in experiment_name
                or df.get("experiment", pd.Series())
                .astype(str)
                .str.contains("sampler memory budget", case=False, na=False)
                .any()
                or {
                    "memory_peak_fraction_budget",
                    "batch_size",
                }.issubset(df.columns)
            )

            if is_controlled_comparison:
                _render_controlled_comparison_live_view(meta, df, best_row)
            elif is_gnn_sampler_memory:
                _render_gnn_sampler_memory_budget_view(df, best_row)
            elif is_best_section:
                _render_best_highway_section_view(df, best_row)
            elif is_gnn_recursive:
                _render_gnn_recursive_view(df, best_row)
            elif is_gnn_optuna:
                _render_gnn_optuna_objectives_view(df, best_row)
            elif is_find_samples:
                _render_find_samples_view(df, best_row)
            else:
                _render_features_sampler_view(df)

    if auto_refresh:
        time.sleep(float(refresh_seconds))
        st.rerun()


if __name__ == "__main__":
    main()
