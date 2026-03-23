#!/usr/bin/env python3
"""
Streamlit app to monitor experiment results in real time.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
DRIFT_RUNS_DIR = RESULTS_DIR / "drift_recalibration_runs"


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


def _list_drift_manifest_files() -> list[Path]:
    if not DRIFT_RUNS_DIR.exists():
        return []
    return sorted(
        DRIFT_RUNS_DIR.glob("*/manifest.json"),
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


def _summarize_drift_rows(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "strategy",
                "model",
                "balance_mode",
                "auc",
                "sensitivity",
                "specificity",
                "error_rate",
                "training_time_sec",
                "n_rows",
                "n_seeds",
            ]
        )
    work = df.copy()
    if "balance_mode" not in work.columns:
        work["balance_mode"] = "not_applicable"
    for col in ["auc", "sensitivity", "specificity", "error_rate", "training_time_sec", "run_seed"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    agg_map: Dict[str, tuple[str, object]] = {"n_rows": ("strategy", "size")}
    for metric in ["auc", "sensitivity", "specificity", "error_rate", "training_time_sec"]:
        if metric in work.columns:
            agg_map[metric] = (metric, "mean")
    if "run_seed" in work.columns:
        agg_map["n_seeds"] = (
            "run_seed",
            lambda s: int(pd.Series(s).dropna().astype(int).nunique()) if pd.Series(s).notna().any() else 0,
        )
    grouped = (
        work.groupby(["strategy", "model", "balance_mode"], dropna=False)
        .agg(**agg_map)
        .reset_index()
    )
    grouped.insert(0, "source", source)
    if "n_seeds" not in grouped.columns:
        grouped["n_seeds"] = 0
    ordered_cols = [
        "source",
        "strategy",
        "model",
        "balance_mode",
        "auc",
        "sensitivity",
        "specificity",
        "error_rate",
        "training_time_sec",
        "n_rows",
        "n_seeds",
    ]
    for col in ordered_cols:
        if col not in grouped.columns:
            grouped[col] = pd.NA
    return grouped[ordered_cols].sort_values(
        ["source", "strategy", "balance_mode", "model"]
    ).reset_index(drop=True)


def _build_drift_partial_summary(yearly_df: pd.DataFrame, adaptive_df: pd.DataFrame) -> pd.DataFrame:
    frames = [
        _summarize_drift_rows(yearly_df, source="yearly"),
        _summarize_drift_rows(adaptive_df, source="adaptive"),
    ]
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=[
                "source",
                "strategy",
                "model",
                "balance_mode",
                "auc",
                "sensitivity",
                "specificity",
                "error_rate",
                "training_time_sec",
                "n_rows",
                "n_seeds",
            ]
        )
    return pd.concat(non_empty, ignore_index=True)


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
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return int(round(value))
        return round(value, 4)
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
        "sensitivity",
        "specificity",
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
                            "sensitivity": _display_cell(source_row.get("sensitivity")),
                            "specificity": _display_cell(source_row.get("specificity")),
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
                            "sensitivity": "Pendiente",
                            "specificity": "Pendiente",
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
        "sensitivity",
        "specificity",
        "error_rate",
        "training_time_sec",
        "threshold",
        "run_seed",
        "run_order",
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
        pending_blocks = block_df.loc[
            block_df["strategy"].astype(str).isin(adaptive_strategies)
            & ~block_df["status"].astype(str).str.lower().eq("completed")
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
                    "sensitivity": "Pendiente",
                    "specificity": "Pendiente",
                    "error_rate": "Pendiente",
                    "training_time_sec": "Pendiente",
                    "threshold": "Pendiente",
                    "run_seed": _display_cell(block.get("run_seed")),
                    "run_order": _display_cell(block.get("run_order")),
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
        row = {
            "block_id": block_id,
            "status": str(info.get("status") or "completed" if payload else "pending"),
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
        yearly_rows = payload.get("yearly_rows") or []
        adaptive_rows = payload.get("adaptive_rows") or []
        roc_payload_rows.extend(
            item
            for item in (payload.get("roc_payload") or [])
            if isinstance(item, dict)
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
        "tuning_artifacts": tuning_artifacts,
        "live_status": live_status,
        "live_events_df": live_events_df,
        "live_status_path": live_status_path,
        "live_events_path": live_events_path,
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


def _render_gnn_optuna_objectives_view(
    df: pd.DataFrame, best_row: Optional[Dict[str, object]]
) -> None:
    st.caption("Experimento detectado: GNN Optuna Objectives")
    plot_df = df.copy()
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
        st.markdown("**Resultado optimo (test_f1)**")
        metrics_payload = {
            "gnn_variant": best_row.get("gnn_variant"),
            "balance_strategy": best_row.get("balance_strategy"),
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
                        "objective_label",
                        "balance_strategy",
                    ]
                    if "gnn_variant" in plot_df.columns:
                        tooltip_fields.append("gnn_variant")
                    tooltip_fields += [
                        "test_f1",
                        "test_precision",
                        "test_recall",
                        "test_far",
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
    plot_df = df.copy()

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
                    "iteration",
                    "optimizer",
                    "balance_strategy",
                    "objective_label",
                    selected_metric,
                ]
                if "gnn_variant" in base.columns:
                    line_tooltip.insert(2, "gnn_variant")
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
    st.dataframe(plot_df, width="stretch")


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
        st.dataframe(table_df, width="stretch")
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

    st.dataframe(plot_df, width="stretch")


def _render_drift_recalibration_view(data: Dict[str, object]) -> None:
    manifest = dict(data.get("manifest") or {})
    live_status = dict(data.get("live_status") or {})
    live_events_df = data.get("live_events_df")
    block_df = data.get("block_df")
    summary_df = data.get("summary_df")
    tuning_trials_df = data.get("tuning_trials_df")
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
                history_df[["event_index", "timestamp", "label", "detail", "progress_pct"]],
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
            st.dataframe(block_status_df, width="stretch")
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
            st.dataframe(tuning_plot, width="stretch")
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
            st.dataframe(trace_plot, width="stretch")
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
                    st.dataframe(table_df, width="stretch")
                else:
                    st.info(f"Table {table_key} no aplica a la configuracion actual.")
        else:
            st.info("No fue posible reconstruir tablas live para la corrida actual.")

        st.markdown("**ROC-AUC curves by strategy**")
        _render_drift_roc_curves(
            average_roc_df if isinstance(average_roc_df, pd.DataFrame) else pd.DataFrame(),
            key_prefix="partial_drift_roc",
        )

        st.markdown("**Partial strategy summary**")
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            if "auc" in summary_df.columns and summary_df["auc"].notna().any():
                chart_df = summary_df.copy()
                chart_df["series_label"] = (
                    chart_df["source"].astype(str)
                    + " | "
                    + chart_df["strategy"].astype(str)
                    + " | "
                    + chart_df["model"].astype(str)
                    + " | "
                    + chart_df["balance_mode"].astype(str)
                )
                st.bar_chart(
                    chart_df.set_index("series_label")["auc"],
                    width="stretch",
                )
            st.dataframe(summary_df, width="stretch")
        else:
            st.info("Aún no hay resultados parciales agregables.")

        st.markdown("**Completed blocks**")
        if isinstance(block_df, pd.DataFrame) and not block_df.empty:
            completed_blocks = block_df.loc[
                block_df["status"].astype(str).str.lower() == "completed"
            ].copy()
            if not completed_blocks.empty:
                st.dataframe(completed_blocks, width="stretch")
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
            st.dataframe(yearly_df, width="stretch")
        if isinstance(adaptive_df, pd.DataFrame) and not adaptive_df.empty:
            st.markdown("**Adaptive preview**")
            st.dataframe(adaptive_df, width="stretch")

    with data_tab:
        st.markdown("**Manifest**")
        st.json(manifest, expanded=False)
        if live_status:
            st.markdown("**Live status**")
            st.json(live_status, expanded=False)
        if isinstance(block_df, pd.DataFrame) and not block_df.empty:
            st.markdown("**Block index**")
            st.dataframe(block_df, width="stretch")
        if isinstance(tuning_trials_df, pd.DataFrame) and not tuning_trials_df.empty:
            st.markdown("**Tuning trials**")
            st.dataframe(tuning_trials_df, width="stretch")
        if isinstance(execution_log_df, pd.DataFrame) and not execution_log_df.empty:
            st.markdown("**Execution log**")
            st.dataframe(execution_log_df, width="stretch")


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

            if is_gnn_sampler_memory:
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
