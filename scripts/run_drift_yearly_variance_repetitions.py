#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.drift_detection_app as drift_app
from src.drift_bias_variance import (
    build_bias_variance_noise_lookup,
    build_brier_score_lookup,
    drift_roc_group_key,
)


DEFAULT_SOURCE_JSON = Path(
    "Resultados/drift_recalibration_runs/run_6d1e6bd611a2b7a1/"
    "drift_recalibration_optuna_20260409_125138.json"
)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
YEARLY_STRATEGIES = ("static", "period_aligned", "cumulative")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _max_abs_additive_residual(df: pd.DataFrame) -> float | None:
    required = {"brier_score", "bias2", "variance", "noise"}
    if df.empty or not required <= set(df.columns):
        return None
    residual = (
        _numeric_series(df, "brier_score")
        - _numeric_series(df, "bias2")
        - _numeric_series(df, "variance")
        - _numeric_series(df, "noise")
    ).dropna()
    if residual.empty:
        return None
    return float(residual.abs().max())


def _nonzero_count(df: pd.DataFrame, column: str, *, tol: float = 1e-12) -> int:
    values = _numeric_series(df, column).dropna()
    if values.empty:
        return 0
    return int((values.abs() > tol).sum())


def _safe_int(value: Any) -> int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return int(numeric)


def _sequence_signature(values: Any) -> dict[str, Any]:
    if values is None or isinstance(values, (str, bytes)):
        raw_values = []
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = []
    normalized = []
    for value in raw_values:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            normalized.append("nan")
        else:
            normalized.append(str(int(round(float(numeric)))))
    digest = hashlib.sha256(",".join(normalized).encode("utf-8")).hexdigest()
    positives = sum(1 for value in normalized if value == "1")
    return {"length": len(normalized), "positives": int(positives), "sha256": digest}


def _feature_context_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    run_manifest = dict(payload.get("run_manifest") or {})
    context = {}
    for candidate in (
        payload.get("feature_selection_context"),
        run_manifest.get("feature_selection_context"),
    ):
        if isinstance(candidate, dict):
            context.update(candidate)

    if not context.get("feature_export_path"):
        linked_path = context.get("linked_duckdb_path")
        if linked_path:
            context["feature_export_path"] = str(linked_path)
    if not context.get("feature_export_name"):
        linked_name = context.get("linked_duckdb_name")
        if linked_name:
            context["feature_export_name"] = str(linked_name)
    return context


def _load_execution_bundle(payload: dict[str, Any], *, feature_export_path: Path | None = None) -> dict[str, Any]:
    run_manifest = dict(payload.get("run_manifest") or {})
    context = _feature_context_from_payload(payload)
    if feature_export_path is not None:
        context["feature_export_path"] = str(feature_export_path)
        context.setdefault("feature_export_name", feature_export_path.name)

    bundle = drift_app._load_checkpoint_feature_bundle(  # noqa: SLF001 - script entrypoint for local app internals.
        run_manifest={**run_manifest, "feature_selection_context": context},
        feature_selection_context=context,
    )
    if bundle is None:
        raise FileNotFoundError(
            "Could not resolve the feature DuckDB referenced by the source run. "
            "Pass --feature-export-path, or restore the file named in "
            "`feature_selection_context.linked_duckdb_path` / `feature_export_path`."
        )
    if not isinstance(bundle.get("clean_df"), pd.DataFrame) or bundle["clean_df"].empty:
        raise ValueError("Resolved feature DuckDB does not contain a non-empty clean_features table.")
    return bundle


def _execution_config_from_source(
    payload: dict[str, Any],
    *,
    clean_df: pd.DataFrame,
    feature_selection_context: dict[str, Any],
    seeds: tuple[int, ...],
    strategies: tuple[str, ...] = YEARLY_STRATEGIES,
) -> dict[str, Any]:
    run_manifest = dict(payload.get("run_manifest") or {})
    feature_cols = [
        str(col)
        for col in (
            feature_selection_context.get("selected_features")
            or run_manifest.get("feature_cols")
            or []
        )
        if str(col) in set(clean_df.columns)
    ]
    if not feature_cols:
        raise ValueError("No source feature columns are available in the resolved clean dataset.")

    current_config = {
        "feature_cols": feature_cols,
        "target_col": str(run_manifest.get("target_col") or "target"),
        "time_col": str(run_manifest.get("time_col") or "interval_start"),
        "model_names": list(run_manifest.get("models") or drift_app.MODEL_NAMES),
        "strategies": list(strategies),
        "validation_size": float(run_manifest.get("validation_size", 0.2)),
        "folds": int(run_manifest.get("folds", 3)),
        "base_year": _safe_int(run_manifest.get("base_year")),
        "random_state": int(run_manifest.get("random_state_fallback", 42)),
        "fast_mode": bool(run_manifest.get("fast_mode", False)),
        "resource_mode": str(run_manifest.get("resource_mode", drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE)),
        "resource_policy_overrides": dict(run_manifest.get("resource_policy_overrides") or {}),
        "grid_limit": int(run_manifest.get("optuna_trials", 30)),
        "adwin_delta": float(run_manifest.get("adwin_delta", 0.002)),
        "min_window": int(run_manifest.get("min_window", 45_000)),
        "min_retrain_size": (
            None
            if run_manifest.get("min_retrain_size") is None
            else int(run_manifest.get("min_retrain_size"))
        ),
        "arf_variants": list(run_manifest.get("arf_variants") or drift_app.ARF_DEFAULT_VARIANTS),
        "kswin_variants": list(run_manifest.get("kswin_variants") or drift_app.KSWIN_DEFAULT_VARIANTS),
        "kswin_top_k_features": int(run_manifest.get("kswin_top_k_features", 5)),
        "kswin_vote_threshold": int(run_manifest.get("kswin_vote_threshold", 3)),
        "kswin_retrain_days": int(run_manifest.get("kswin_retrain_days", 30)),
        "kswin_min_retrain_rows": int(run_manifest.get("kswin_min_retrain_rows", 50)),
        "repetition_seeds": tuple(seeds),
        "balance_modes": list(run_manifest.get("balance_modes") or drift_app.DEFAULT_BATCH_BALANCE_MODES),
        "feature_selection_context": dict(feature_selection_context),
        "custom_grids": dict(run_manifest.get("custom_grids") or {}),
        "continue_on_block_error": bool(run_manifest.get("continue_on_block_error", False)),
        "neural_drift_config": dict(run_manifest.get("neural_drift_config") or {}),
    }

    config = drift_app._execution_config_from_checkpoint_run_manifest(  # noqa: SLF001
        run_manifest,
        available_columns=list(clean_df.columns),
        current_config=current_config,
        feature_selection_context=feature_selection_context,
    )
    config["strategies"] = list(strategies)
    config["repetition_seeds"] = tuple(seeds)
    config["feature_selection_context"] = dict(feature_selection_context)
    return config


def _seed_tuning_cache_from_source(
    *,
    source_json_path: Path,
    checkpoint_root: Path,
    run_id: str,
    resume: bool,
    reuse_source_tuning: bool,
) -> dict[str, Any]:
    target_run_dir = drift_app._recalibration_run_dir(run_id, checkpoint_root=checkpoint_root)  # noqa: SLF001
    paths = drift_app._recalibration_run_paths(target_run_dir)  # noqa: SLF001
    if target_run_dir.exists() and any(target_run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Checkpoint run directory already exists: {target_run_dir}. "
            "Use --resume to continue it or choose a new --output-dir."
        )
    drift_app._ensure_recalibration_run_dirs(paths)  # noqa: SLF001

    copied_tuning = 0
    source_tuning_dir = source_json_path.parent / "tuning"
    if reuse_source_tuning and source_tuning_dir.exists():
        for source_file in sorted(source_tuning_dir.glob("*.json")):
            target_file = paths["tuning_dir"] / source_file.name
            if target_file.exists():
                continue
            shutil.copy2(source_file, target_file)
            copied_tuning += 1

    manifest_path = paths["manifest"]
    if manifest_path.exists() and resume:
        manifest = _load_json(manifest_path)
    else:
        manifest = {
            "run_id": str(run_id),
            "status": "running",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "completed_block_ids": [],
            "skipped_failed_block_ids": [],
            "pending_block_ids": [],
            "failed_block_id": None,
            "last_error": None,
            "nonfatal_block_errors": [],
            "block_index": {},
            "tuning_index": {},
            "smote_index": {},
            "global_execution_log": [],
            "progress": {},
        }
    _write_json(manifest_path, manifest)
    return {
        "checkpoint_run_dir": str(target_run_dir),
        "copied_tuning_artifacts": int(copied_tuning),
        "source_tuning_dir": str(source_tuning_dir),
    }


def validate_yearly_roc_payload(
    roc_payload: list[dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
    strategies: tuple[str, ...] = YEARLY_STRATEGIES,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in roc_payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("strategy")) not in set(strategies):
            continue
        grouped.setdefault(drift_roc_group_key(item), []).append(item)

    brier_lookup = build_brier_score_lookup(list(roc_payload))
    decomposition_lookup = build_bias_variance_noise_lookup(list(roc_payload))
    errors: list[str] = []
    group_reports: list[dict[str, Any]] = []
    expected_seed_set = {int(seed) for seed in expected_seeds}

    for key, items in sorted(grouped.items()):
        by_seed: dict[int, dict[str, Any]] = {}
        duplicate_seeds: set[int] = set()
        for item in items:
            seed = _safe_int(item.get("run_seed"))
            if seed is None:
                continue
            if seed in by_seed:
                duplicate_seeds.add(seed)
            by_seed[seed] = item

        present_seed_set = set(by_seed)
        missing = sorted(expected_seed_set - present_seed_set)
        extra = sorted(present_seed_set - expected_seed_set)
        signatures = {seed: _sequence_signature(item.get("y_true")) for seed, item in by_seed.items()}
        unique_y_signatures = {json.dumps(sig, sort_keys=True) for sig in signatures.values()}
        if missing:
            errors.append(f"{key}: missing seeds {missing}")
        if extra:
            errors.append(f"{key}: unexpected seeds {extra}")
        if duplicate_seeds:
            errors.append(f"{key}: duplicate seeds {sorted(duplicate_seeds)}")
        if len(unique_y_signatures) > 1:
            errors.append(f"{key}: y_true differs across seeds")

        decomposition = decomposition_lookup.get(key, {})
        brier = brier_lookup.get(key)
        bias2 = decomposition.get("bias2")
        variance = decomposition.get("variance")
        noise = decomposition.get("noise")
        additive_residual = None
        if all(value is not None for value in (brier, bias2, variance, noise)):
            additive_residual = float(abs(float(brier) - float(bias2) - float(variance) - float(noise)))

        group_reports.append(
            {
                "strategy": key[0],
                "model": key[1],
                "balance_mode": key[2],
                "segment": key[3],
                "n_items": int(len(items)),
                "seeds": sorted(present_seed_set),
                "y_length": next(iter(signatures.values()), {}).get("length"),
                "positive_count": next(iter(signatures.values()), {}).get("positives"),
                "brier_score": brier,
                "bias2": bias2,
                "variance": variance,
                "noise": noise,
                "additive_residual": additive_residual,
            }
        )

    nonzero_variance_groups = sum(
        1
        for row in group_reports
        if pd.notna(pd.to_numeric(row.get("variance"), errors="coerce"))
        and abs(float(row.get("variance") or 0.0)) > 1e-12
    )
    return {
        "group_count": int(len(group_reports)),
        "expected_seeds": sorted(expected_seed_set),
        "nonzero_variance_groups": int(nonzero_variance_groups),
        "zero_variance_groups": int(len(group_reports) - nonzero_variance_groups),
        "errors": errors,
        "groups": group_reports,
    }


def _write_roc_payload_gzip(roc_payload: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(drift_app._to_json_safe(roc_payload), fh, ensure_ascii=True)  # noqa: SLF001


def write_variance_outputs(
    outputs: dict[str, Any],
    *,
    output_dir: Path,
    seeds: tuple[int, ...],
    source_json_path: Path,
    cache_report: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    yearly_results = outputs.get("yearly_results")
    summary = outputs.get("summary")
    appendix_tables = dict(outputs.get("appendix_tables") or {})
    appendix_mean = dict(outputs.get("appendix_tables_mean") or {})
    roc_payload = [dict(item) for item in (outputs.get("roc_payload") or []) if isinstance(item, dict)]

    if not isinstance(yearly_results, pd.DataFrame):
        yearly_results = pd.DataFrame()
    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame()

    _write_table(yearly_results, output_dir / "yearly_results_5seeds.csv")
    _write_table(summary, output_dir / "summary_5seeds.csv")

    for key in ("A.6", "A.7", "A.8"):
        table = appendix_tables.get(key, pd.DataFrame())
        mean_table = appendix_mean.get(key, pd.DataFrame())
        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame()
        if not isinstance(mean_table, pd.DataFrame):
            mean_table = pd.DataFrame()
        _write_table(table, output_dir / f"{key.replace('.', '')}_by_seed_5seeds.csv")
        _write_table(mean_table, output_dir / f"{key.replace('.', '')}_5seeds.csv")

    roc_path = output_dir / "roc_payload_5seeds.json.gz"
    _write_roc_payload_gzip(roc_payload, roc_path)

    roc_validation = validate_yearly_roc_payload(roc_payload, expected_seeds=seeds)
    table_reports = {}
    for key in ("A.6", "A.7", "A.8"):
        table = appendix_mean.get(key, pd.DataFrame())
        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame()
        table_reports[key] = {
            "rows": int(len(table)),
            "nonzero_variance_rows": _nonzero_count(table, "variance"),
            "max_abs_additive_residual": _max_abs_additive_residual(table),
            "n_repetitions_values": sorted(
                {
                    int(value)
                    for value in _numeric_series(table, "n_repetitions").dropna().tolist()
                }
            ),
            "seed_lists": sorted(
                {
                    str(value)
                    for value in table.get("seed_list", pd.Series(dtype=str)).dropna().astype(str).unique()
                }
            ),
        }

    validation_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_json": str(source_json_path),
        "output_dir": str(output_dir),
        "seeds": list(seeds),
        "strategies": list(YEARLY_STRATEGIES),
        "run_output_json_path": str(outputs.get("optuna_json_path") or ""),
        "checkpoint_run_dir": str(outputs.get("checkpoint_run_dir") or ""),
        "roc_payload_path": str(roc_path),
        "roc_payload_rows": int(len(roc_payload)),
        "cache": cache_report,
        "roc_alignment": roc_validation,
        "tables": table_reports,
    }
    if roc_validation["errors"]:
        validation_report["status"] = "failed"
    elif int(roc_validation.get("nonzero_variance_groups", 0)) <= 0:
        validation_report["status"] = "warning_zero_variance"
    else:
        validation_report["status"] = "ok"
    _write_json(output_dir / "validation_report.json", validation_report)
    if roc_validation["errors"]:
        raise RuntimeError(
            "ROC payload validation failed: " + "; ".join(roc_validation["errors"][:5])
        )
    return validation_report


def run_variance_repetitions(
    *,
    source_json_path: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    feature_export_path: Path | None = None,
    resume: bool = False,
    reuse_source_tuning: bool = True,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required to estimate non-zero variance.")
    payload = _load_json(source_json_path)
    run_manifest = dict(payload.get("run_manifest") or {})
    run_id = str(run_manifest.get("run_id") or source_json_path.parent.name)
    checkpoint_root = output_dir / "checkpoints"

    bundle = _load_execution_bundle(payload, feature_export_path=feature_export_path)
    clean_df = bundle["clean_df"]
    feature_selection_context = dict(bundle.get("feature_selection_context") or {})
    config = _execution_config_from_source(
        payload,
        clean_df=clean_df,
        feature_selection_context=feature_selection_context,
        seeds=seeds,
        strategies=YEARLY_STRATEGIES,
    )
    cache_report = _seed_tuning_cache_from_source(
        source_json_path=source_json_path,
        checkpoint_root=checkpoint_root,
        run_id=run_id,
        resume=resume,
        reuse_source_tuning=reuse_source_tuning,
    )

    def progress_callback(payload: dict[str, Any]) -> None:
        label = str(payload.get("label", "")).strip()
        detail = str(payload.get("detail", "")).strip()
        completed = payload.get("completed_units")
        total = payload.get("total_units")
        print(f"[drift-variance] {completed}/{total} {label} {detail}".strip(), flush=True)

    outputs = drift_app.run_recalibration_experiments(
        clean_df,
        feature_cols=config["feature_cols"],
        target_col=config["target_col"],
        time_col=config["time_col"],
        model_names=config["model_names"],
        strategies=list(YEARLY_STRATEGIES),
        validation_size=config["validation_size"],
        folds=config["folds"],
        base_year=config["base_year"],
        random_state=config["random_state"],
        fast_mode=config["fast_mode"],
        resource_mode=config["resource_mode"],
        resource_policy_overrides=config["resource_policy_overrides"],
        grid_limit=config["grid_limit"],
        adwin_delta=config["adwin_delta"],
        min_window=config["min_window"],
        min_retrain_size=config["min_retrain_size"],
        arf_variants=config["arf_variants"],
        kswin_variants=config["kswin_variants"],
        kswin_top_k_features=config["kswin_top_k_features"],
        kswin_vote_threshold=config["kswin_vote_threshold"],
        kswin_retrain_days=config["kswin_retrain_days"],
        kswin_min_retrain_rows=config["kswin_min_retrain_rows"],
        repetition_seeds=seeds,
        balance_modes=config["balance_modes"],
        feature_selection_context=config["feature_selection_context"],
        custom_grids=config["custom_grids"],
        checkpoint_root=checkpoint_root,
        checkpoint_run_id_override=run_id,
        auto_resume=True,
        persist_progress=True,
        reuse_tuning_cache=True,
        reuse_smote_cache=True,
        continue_on_block_error=config["continue_on_block_error"],
        progress_callback=progress_callback,
    )
    return write_variance_outputs(
        outputs,
        output_dir=output_dir,
        seeds=seeds,
        source_json_path=source_json_path,
        cache_report=cache_report,
    )


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    seeds = drift_app.parse_repetition_seeds(raw)
    return tuple(int(seed) for seed in seeds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Drift A.6-A.8 with repeated seeds and export bias-variance tables."
    )
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--feature-export-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-reuse-source-tuning", action="store_true")
    args = parser.parse_args()

    source_json_path = args.source_json.resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (ROOT_DIR / "Resultados" / "Drift" / f"variance_repetitions_5seeds_{stamp}").resolve()
    else:
        output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    report = run_variance_repetitions(
        source_json_path=source_json_path,
        output_dir=output_dir,
        seeds=_parse_seed_list(args.seeds),
        feature_export_path=args.feature_export_path.resolve() if args.feature_export_path else None,
        resume=bool(args.resume),
        reuse_source_tuning=not bool(args.no_reuse_source_tuning),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
