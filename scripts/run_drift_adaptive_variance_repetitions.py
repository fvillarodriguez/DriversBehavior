#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.drift_detection_app as drift_app
from scripts.run_drift_yearly_variance_repetitions import (
    _execution_config_from_source,
    _load_execution_bundle,
    _load_json,
    _max_abs_additive_residual,
    _nonzero_count,
    _numeric_series,
    _parse_seed_list,
    _safe_int,
    _sequence_signature,
    _write_json,
    _write_roc_payload_gzip,
    _write_table,
)
from src.drift_bias_variance import (
    build_bias_variance_noise_lookup,
    build_brier_score_lookup,
    drift_roc_group_key,
    enrich_drift_rows_with_bias_variance,
)


DEFAULT_SOURCE_RUN_DIR = Path("Resultados/drift_recalibration_runs/run_6d1e6bd611a2b7a1")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
ADAPTIVE_STRATEGIES = (
    drift_app.ADAPTIVE_ADWIN_STRATEGY,
    drift_app.ADAPTIVE_ARF_STRATEGY,
    drift_app.ADAPTIVE_KSWIN_STRATEGY,
)
ADDITIVE_TOLERANCE = 1e-10
NONZERO_VARIANCE_TOLERANCE = 1e-12


def _load_source_manifest(source_run_dir: Path) -> dict[str, Any]:
    manifest_path = source_run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source run manifest not found: {manifest_path}")
    payload = _load_json(manifest_path)
    if not isinstance(payload.get("run_manifest"), dict):
        raise ValueError(f"Source manifest does not contain run_manifest: {manifest_path}")
    return payload


def _source_payload_from_manifest(source_manifest: dict[str, Any]) -> dict[str, Any]:
    run_manifest = dict(source_manifest.get("run_manifest") or {})
    feature_selection_context = dict(
        source_manifest.get("feature_selection_context")
        or run_manifest.get("feature_selection_context")
        or {}
    )
    return {
        "run_manifest": run_manifest,
        "feature_selection_context": feature_selection_context,
    }


def _adaptive_strategies_from_manifest(run_manifest: dict[str, Any]) -> tuple[str, ...]:
    manifest_strategies = [str(item) for item in (run_manifest.get("strategies") or [])]
    strategies = tuple(strategy for strategy in ADAPTIVE_STRATEGIES if strategy in manifest_strategies)
    if not strategies:
        raise ValueError(
            "The source run does not include A.9 adaptive strategies "
            f"({', '.join(ADAPTIVE_STRATEGIES)})."
        )
    return strategies


def _split_csv_tokens(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).replace(";", ",").split(",") if item.strip()]


def _filter_allowed(values: list[str], allowed: list[str], *, label: str) -> list[str]:
    if not values:
        return list(allowed)
    allowed_set = set(allowed)
    invalid = [value for value in values if value not in allowed_set]
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}. Allowed values: {allowed}")
    return [value for value in allowed if value in set(values)]


def _adaptive_roc_payload(roc_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adaptive_set = set(ADAPTIVE_STRATEGIES)
    return [
        dict(item)
        for item in roc_payload
        if isinstance(item, dict) and str(item.get("strategy")) in adaptive_set
    ]


def _copy_dir_contents(source_dir: Path, target_dir: Path) -> int:
    if not source_dir.exists():
        return 0
    copied = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_dir.iterdir()):
        target_path = target_dir / source_path.name
        if source_path.is_dir():
            if target_path.exists():
                copied += _copy_dir_contents(source_path, target_path)
            else:
                shutil.copytree(source_path, target_path)
                copied += sum(1 for item in target_path.rglob("*") if item.is_file())
            continue
        if target_path.exists():
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def _expected_adaptive_blocks(
    run_manifest: dict[str, Any],
    *,
    strategies: tuple[str, ...],
    model_names: list[str] | None = None,
    balance_modes: list[str] | None = None,
    arf_variants: list[str] | None = None,
    kswin_variants: list[str] | None = None,
) -> list[dict[str, str]]:
    return drift_app._build_experiment_blocks(  # noqa: SLF001 - script runner reuses app checkpoint semantics.
        model_names=list(model_names or run_manifest.get("models") or drift_app.MODEL_NAMES),
        strategies=list(strategies),
        arf_variants=list(arf_variants or run_manifest.get("arf_variants") or drift_app.ARF_DEFAULT_VARIANTS),
        kswin_variants=list(kswin_variants or run_manifest.get("kswin_variants") or drift_app.KSWIN_DEFAULT_VARIANTS),
        balance_modes=list(balance_modes or run_manifest.get("balance_modes") or drift_app.DEFAULT_BATCH_BALANCE_MODES),
    )


def prepare_adaptive_checkpoint_from_source(
    *,
    source_run_dir: Path,
    checkpoint_root: Path,
    run_id: str,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    resume: bool = False,
    model_names: list[str] | None = None,
    strategies: tuple[str, ...] | None = None,
    balance_modes: list[str] | None = None,
    arf_variants: list[str] | None = None,
    kswin_variants: list[str] | None = None,
) -> dict[str, Any]:
    source_run_dir = Path(source_run_dir)
    source_manifest = _load_source_manifest(source_run_dir)
    source_run_manifest = dict(source_manifest.get("run_manifest") or {})
    resolved_strategies = strategies or _adaptive_strategies_from_manifest(source_run_manifest)
    experiment_blocks = _expected_adaptive_blocks(
        source_run_manifest,
        strategies=resolved_strategies,
        model_names=model_names,
        balance_modes=balance_modes,
        arf_variants=arf_variants,
        kswin_variants=kswin_variants,
    )
    if not experiment_blocks:
        raise ValueError("No adaptive experiment blocks were generated from the source manifest.")

    target_run_dir = drift_app._recalibration_run_dir(  # noqa: SLF001
        run_id,
        checkpoint_root=checkpoint_root,
    )
    paths = drift_app._recalibration_run_paths(target_run_dir)  # noqa: SLF001
    if target_run_dir.exists() and any(target_run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Checkpoint run directory already exists: {target_run_dir}. "
            "Use --resume to continue it or choose a new --output-dir."
        )
    drift_app._ensure_recalibration_run_dirs(paths)  # noqa: SLF001

    copied_tuning = _copy_dir_contents(source_run_dir / "tuning", paths["tuning_dir"])
    copied_smote = _copy_dir_contents(source_run_dir / "smote", paths["smote_dir"])

    existing_manifest = _load_json(paths["manifest"]) if paths["manifest"].exists() else {}
    manifest = dict(existing_manifest) if resume and existing_manifest else {}
    completed_ids = {str(item) for item in (manifest.get("completed_block_ids") or [])}
    skipped_ids = {str(item) for item in (manifest.get("skipped_failed_block_ids") or [])}
    block_index = dict(manifest.get("block_index") or {})

    expected_block_ids: list[str] = []
    for run_order, seed in enumerate(seeds, start=1):
        for block in experiment_blocks:
            block_id = drift_app._build_recalibration_block_id(  # noqa: SLF001
                block,
                run_seed=int(seed),
                run_order=int(run_order),
            )
            expected_block_ids.append(block_id)
            previous = dict(block_index.get(block_id) or {})
            block_index[block_id] = {
                **previous,
                "filename": previous.get("filename") or f"{block_id}.json",
                "status": previous.get("status") or "pending",
                "strategy": str(block.get("strategy", "")),
                "model": str(block.get("model", "")),
                "detector_variant": str(block.get("detector_variant", "")),
                "balance_mode": str(block.get("balance_mode", drift_app.BALANCE_MODE_NOT_APPLICABLE)),
                "run_seed": int(seed),
                "run_order": int(run_order),
            }

    expected_id_set = set(expected_block_ids)
    source_seed = int(seeds[0])
    copied_seed_blocks = 0
    copied_source_block_ids: list[str] = []
    ignored_source_block_ids: list[str] = []
    for source_block_path in sorted((source_run_dir / "blocks").glob("*.json")):
        payload = drift_app._load_recalibration_block(source_block_path)  # noqa: SLF001
        if not payload:
            continue
        block = dict(payload.get("block") or {})
        strategy = str(block.get("strategy") or "")
        block_seed = _safe_int(payload.get("run_seed"))
        block_id = str(payload.get("block_id") or source_block_path.stem)
        if strategy not in set(resolved_strategies) or block_seed != source_seed:
            ignored_source_block_ids.append(block_id)
            continue
        if block_id not in expected_id_set:
            ignored_source_block_ids.append(block_id)
            continue
        target_block_path = paths["blocks_dir"] / f"{block_id}.json"
        if not target_block_path.exists():
            shutil.copy2(source_block_path, target_block_path)
            copied_seed_blocks += 1
        completed_ids.add(block_id)
        block_index[block_id] = {
            **dict(block_index.get(block_id) or {}),
            "filename": f"{block_id}.json",
            "status": "completed",
            "strategy": strategy,
            "model": str(block.get("model", "")),
            "detector_variant": str(block.get("detector_variant", "")),
            "balance_mode": str(block.get("balance_mode", drift_app.BALANCE_MODE_NOT_APPLICABLE)),
            "run_seed": int(source_seed),
            "run_order": int(payload.get("run_order") or 1),
        }
        copied_source_block_ids.append(block_id)

    terminal_ids = completed_ids | skipped_ids
    pending_ids = [block_id for block_id in expected_block_ids if block_id not in terminal_ids]
    adapted_run_manifest = {
        **source_run_manifest,
        "models": list(model_names or source_run_manifest.get("models") or drift_app.MODEL_NAMES),
        "strategies": list(resolved_strategies),
        "arf_variants": list(arf_variants or source_run_manifest.get("arf_variants") or drift_app.ARF_DEFAULT_VARIANTS),
        "kswin_variants": list(kswin_variants or source_run_manifest.get("kswin_variants") or drift_app.KSWIN_DEFAULT_VARIANTS),
        "balance_modes": list(balance_modes or source_run_manifest.get("balance_modes") or drift_app.DEFAULT_BATCH_BALANCE_MODES),
        "repetition_seeds": [int(seed) for seed in seeds],
        "run_id": str(run_id),
        "checkpoint_run_dir": str(paths["run_dir"]),
        "checkpoint_manifest_path": str(paths["manifest"]),
        "total_block_units": int(len(expected_block_ids)),
        "total_progress_units": int(len(expected_block_ids)),
    }
    manifest.update(
        {
            "run_id": str(run_id),
            "status": "completed" if not pending_ids else "running",
            "started_at": manifest.get("started_at") or datetime.now().isoformat(timespec="seconds"),
            "run_manifest": drift_app._to_json_safe(adapted_run_manifest),  # noqa: SLF001
            "feature_selection_context": drift_app._to_json_safe(  # noqa: SLF001
                source_manifest.get("feature_selection_context")
                or source_run_manifest.get("feature_selection_context")
                or {}
            ),
            "completed_block_ids": sorted(completed_ids),
            "skipped_failed_block_ids": sorted(skipped_ids),
            "pending_block_ids": pending_ids,
            "failed_block_id": manifest.get("failed_block_id"),
            "last_error": manifest.get("last_error"),
            "nonfatal_block_errors": list(manifest.get("nonfatal_block_errors") or []),
            "block_index": block_index,
            "tuning_index": dict(manifest.get("tuning_index") or {}),
            "smote_index": dict(manifest.get("smote_index") or {}),
            "progress": {
                "completed_units": float(len(terminal_ids)),
                "total_units": int(len(expected_block_ids)),
                "completed_tuning_tasks": 0,
                "total_tuning_tasks": 0,
                "completed_blocks": int(len(completed_ids)),
                "skipped_failed_blocks": int(len(skipped_ids)),
                "total_blocks": int(len(expected_block_ids)),
            },
            "resume": {
                "auto_resumed": bool(resume),
                "checkpoint_status": str(manifest.get("status") or "prepared"),
                "prepared_from_source_run_dir": str(source_run_dir),
            },
            "preflight": dict(manifest.get("preflight") or {}),
            "memory_summary": dict(manifest.get("memory_summary") or {}),
            "global_execution_log": list(manifest.get("global_execution_log") or []),
        }
    )
    drift_app._persist_manifest(paths["manifest"], manifest)  # noqa: SLF001

    return {
        "source_run_dir": str(source_run_dir),
        "checkpoint_run_dir": str(paths["run_dir"]),
        "checkpoint_manifest_path": str(paths["manifest"]),
        "run_id": str(run_id),
        "strategies": list(resolved_strategies),
        "seeds": [int(seed) for seed in seeds],
        "source_seed_reused": int(source_seed),
        "expected_blocks": int(len(expected_block_ids)),
        "copied_seed_blocks": int(copied_seed_blocks),
        "copied_source_block_ids": copied_source_block_ids,
        "ignored_source_block_ids": ignored_source_block_ids,
        "completed_blocks": int(len(completed_ids)),
        "pending_blocks": int(len(pending_ids)),
        "pending_block_ids": pending_ids,
        "copied_tuning_artifacts": int(copied_tuning),
        "copied_smote_artifacts": int(copied_smote),
    }


def validate_adaptive_roc_payload(
    roc_payload: list[dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    adaptive_payload = _adaptive_roc_payload(roc_payload)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in adaptive_payload:
        grouped.setdefault(drift_roc_group_key(item), []).append(item)

    brier_lookup = build_brier_score_lookup(adaptive_payload)
    decomposition_lookup = build_bias_variance_noise_lookup(adaptive_payload)
    expected_seed_set = {int(seed) for seed in expected_seeds}
    errors: list[str] = []
    group_reports: list[dict[str, Any]] = []

    if not grouped:
        errors.append("No adaptive ROC payload items were found.")

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
            if additive_residual > ADDITIVE_TOLERANCE:
                errors.append(f"{key}: additive residual {additive_residual:.12g} exceeds {ADDITIVE_TOLERANCE:g}")

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
        and abs(float(row.get("variance") or 0.0)) > NONZERO_VARIANCE_TOLERANCE
    )
    return {
        "group_count": int(len(group_reports)),
        "expected_seeds": sorted(expected_seed_set),
        "nonzero_variance_groups": int(nonzero_variance_groups),
        "zero_variance_groups": int(len(group_reports) - nonzero_variance_groups),
        "errors": errors,
        "groups": group_reports,
    }


def _validate_a9_mean_table(
    a9_table: pd.DataFrame,
    *,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    expected_seed_list = ",".join(str(int(seed)) for seed in expected_seeds)
    errors: list[str] = []
    if a9_table is None or a9_table.empty:
        return {
            "rows": 0,
            "nonzero_variance_rows": 0,
            "max_abs_additive_residual": None,
            "n_repetitions_values": [],
            "seed_lists": [],
            "errors": ["A.9 mean table is empty."],
        }

    repetitions = sorted(
        {
            int(value)
            for value in _numeric_series(a9_table, "n_repetitions").dropna().tolist()
        }
    )
    seed_lists = sorted(
        {
            str(value)
            for value in a9_table.get("seed_list", pd.Series(dtype=str)).dropna().astype(str).unique()
        }
    )
    if repetitions != [len(expected_seeds)]:
        errors.append(f"A.9 n_repetitions values {repetitions} do not equal {len(expected_seeds)}.")
    if seed_lists != [expected_seed_list]:
        errors.append(f"A.9 seed_list values {seed_lists} do not equal {expected_seed_list!r}.")

    max_residual = _max_abs_additive_residual(a9_table)
    if max_residual is not None and float(max_residual) > ADDITIVE_TOLERANCE:
        errors.append(f"A.9 max additive residual {max_residual:.12g} exceeds {ADDITIVE_TOLERANCE:g}.")

    return {
        "rows": int(len(a9_table)),
        "nonzero_variance_rows": _nonzero_count(a9_table, "variance", tol=NONZERO_VARIANCE_TOLERANCE),
        "max_abs_additive_residual": max_residual,
        "n_repetitions_values": repetitions,
        "seed_lists": seed_lists,
        "errors": errors,
    }


def write_adaptive_variance_outputs(
    outputs: dict[str, Any],
    *,
    output_dir: Path,
    seeds: tuple[int, ...],
    source_run_dir: Path,
    cache_report: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adaptive_results = outputs.get("adaptive_results")
    summary = outputs.get("summary")
    appendix_tables = dict(outputs.get("appendix_tables") or {})
    appendix_mean = dict(outputs.get("appendix_tables_mean") or {})
    roc_payload = _adaptive_roc_payload(
        [dict(item) for item in (outputs.get("roc_payload") or []) if isinstance(item, dict)]
    )

    if not isinstance(adaptive_results, pd.DataFrame):
        adaptive_results = pd.DataFrame()
    if not isinstance(summary, pd.DataFrame):
        summary = pd.DataFrame()

    a9_by_seed = appendix_tables.get("A.9", pd.DataFrame())
    a9_mean = appendix_mean.get("A.9", pd.DataFrame())
    if not isinstance(a9_by_seed, pd.DataFrame):
        a9_by_seed = pd.DataFrame()
    if not isinstance(a9_mean, pd.DataFrame):
        a9_mean = pd.DataFrame()

    _write_table(adaptive_results, output_dir / "adaptive_results_5seeds.csv")
    _write_table(summary, output_dir / "summary_5seeds.csv")
    _write_table(a9_by_seed, output_dir / "A9_by_seed_5seeds.csv")
    _write_table(a9_mean, output_dir / "A9_5seeds.csv")

    roc_path = output_dir / "roc_payload_adaptive_5seeds.json.gz"
    _write_roc_payload_gzip(roc_payload, roc_path)

    roc_validation = validate_adaptive_roc_payload(roc_payload, expected_seeds=seeds)
    a9_report = _validate_a9_mean_table(a9_mean, expected_seeds=seeds)
    validation_errors = list(roc_validation.get("errors") or []) + list(a9_report.get("errors") or [])
    validation_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "seeds": [int(seed) for seed in seeds],
        "strategies": list(ADAPTIVE_STRATEGIES),
        "run_output_json_path": str(outputs.get("optuna_json_path") or ""),
        "checkpoint_run_dir": str(outputs.get("checkpoint_run_dir") or ""),
        "roc_payload_path": str(roc_path),
        "roc_payload_rows": int(len(roc_payload)),
        "cache": cache_report,
        "roc_alignment": roc_validation,
        "tables": {"A.9": a9_report},
        "errors": validation_errors,
    }
    if validation_errors:
        validation_report["status"] = "failed"
    elif int(roc_validation.get("nonzero_variance_groups", 0)) <= 0:
        validation_report["status"] = "warning_zero_variance"
    else:
        validation_report["status"] = "ok"
    _write_json(output_dir / "validation_report.json", validation_report)
    if validation_errors:
        raise RuntimeError(
            "Adaptive A.9 validation failed: " + "; ".join(validation_errors[:5])
        )
    return validation_report


def _read_json_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [dict(item) for item in payload if isinstance(item, dict)]


def merge_adaptive_variance_outputs(
    *,
    input_dirs: list[Path],
    output_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    adaptive_frames: list[pd.DataFrame] = []
    a9_by_seed_frames: list[pd.DataFrame] = []
    a9_frames: list[pd.DataFrame] = []
    roc_payload: list[dict[str, Any]] = []
    source_dirs: list[str] = []

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        source_dirs.append(str(input_dir))
        adaptive_path = input_dir / "adaptive_results_5seeds.csv"
        a9_by_seed_path = input_dir / "A9_by_seed_5seeds.csv"
        a9_path = input_dir / "A9_5seeds.csv"
        roc_path = input_dir / "roc_payload_adaptive_5seeds.json.gz"
        if adaptive_path.exists():
            adaptive_frames.append(pd.read_csv(adaptive_path))
        if a9_by_seed_path.exists():
            a9_by_seed_frames.append(pd.read_csv(a9_by_seed_path))
        if a9_path.exists():
            a9_frames.append(pd.read_csv(a9_path))
        if roc_path.exists():
            roc_payload.extend(_read_json_gzip(roc_path))

    if not input_dirs:
        raise ValueError("At least one --merge-input-dir is required.")
    if not roc_payload:
        raise ValueError("No adaptive ROC payloads were found in merge input directories.")

    adaptive_results = pd.concat(adaptive_frames, ignore_index=True) if adaptive_frames else pd.DataFrame()
    if not adaptive_results.empty:
        adaptive_results = pd.DataFrame(
            enrich_drift_rows_with_bias_variance(
                adaptive_results.to_dict(orient="records"),
                roc_payload,
                yearly=False,
                overwrite_existing=True,
            )
        )
    summary = drift_app.summarize_results(pd.DataFrame(), adaptive_results)
    appendix_tables = drift_app.format_appendix_tables(pd.DataFrame(), adaptive_results)
    appendix_mean = drift_app.format_appendix_tables_mean(pd.DataFrame(), adaptive_results)
    if adaptive_results.empty and a9_by_seed_frames:
        appendix_tables["A.9"] = pd.concat(a9_by_seed_frames, ignore_index=True)
    if adaptive_results.empty and a9_frames:
        appendix_mean["A.9"] = pd.concat(a9_frames, ignore_index=True)
    return write_adaptive_variance_outputs(
        {
            "adaptive_results": adaptive_results,
            "summary": summary,
            "appendix_tables": appendix_tables,
            "appendix_tables_mean": appendix_mean,
            "roc_payload": roc_payload,
            "optuna_json_path": "",
            "checkpoint_run_dir": "",
        },
        output_dir=output_dir,
        seeds=seeds,
        source_run_dir=Path(";".join(source_dirs)),
        cache_report={"merged_input_dirs": source_dirs},
    )


def run_adaptive_variance_repetitions(
    *,
    source_run_dir: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    feature_export_path: Path | None = None,
    resume: bool = False,
    strategy_filter: list[str] | None = None,
    model_filter: list[str] | None = None,
    balance_filter: list[str] | None = None,
    arf_variant_filter: list[str] | None = None,
    kswin_variant_filter: list[str] | None = None,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one seed is required.")
    source_manifest = _load_source_manifest(source_run_dir)
    payload = _source_payload_from_manifest(source_manifest)
    run_manifest = dict(payload.get("run_manifest") or {})
    manifest_strategies = list(_adaptive_strategies_from_manifest(run_manifest))
    strategies = tuple(_filter_allowed(strategy_filter or [], manifest_strategies, label="strategies"))
    model_names = _filter_allowed(
        model_filter or [],
        [str(item) for item in (run_manifest.get("models") or drift_app.MODEL_NAMES)],
        label="models",
    )
    balance_modes = _filter_allowed(
        balance_filter or [],
        [str(item) for item in (run_manifest.get("balance_modes") or drift_app.DEFAULT_BATCH_BALANCE_MODES)],
        label="balance modes",
    )
    arf_variants = _filter_allowed(
        arf_variant_filter or [],
        [str(item) for item in (run_manifest.get("arf_variants") or drift_app.ARF_DEFAULT_VARIANTS)],
        label="ARF variants",
    )
    kswin_variants = _filter_allowed(
        kswin_variant_filter or [],
        [str(item) for item in (run_manifest.get("kswin_variants") or drift_app.KSWIN_DEFAULT_VARIANTS)],
        label="KSWIN variants",
    )
    run_id = str(run_manifest.get("run_id") or Path(source_run_dir).name)
    checkpoint_root = output_dir / "checkpoints"

    try:
        bundle = _load_execution_bundle(payload, feature_export_path=feature_export_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc} For this adaptive A.9 runner, pass --feature-export-path with the restored "
            "DuckDB feature export referenced by the source manifest."
        ) from exc

    clean_df = bundle["clean_df"]
    feature_selection_context = dict(bundle.get("feature_selection_context") or {})
    config = _execution_config_from_source(
        payload,
        clean_df=clean_df,
        feature_selection_context=feature_selection_context,
        seeds=seeds,
        strategies=strategies,
    )
    config["strategies"] = list(strategies)
    config["model_names"] = list(model_names)
    config["balance_modes"] = list(balance_modes)
    config["arf_variants"] = list(arf_variants)
    config["kswin_variants"] = list(kswin_variants)
    config["repetition_seeds"] = tuple(seeds)
    config["feature_selection_context"] = dict(feature_selection_context)

    cache_report = prepare_adaptive_checkpoint_from_source(
        source_run_dir=source_run_dir,
        checkpoint_root=checkpoint_root,
        run_id=run_id,
        seeds=seeds,
        resume=resume,
        model_names=model_names,
        strategies=strategies,
        balance_modes=balance_modes,
        arf_variants=arf_variants,
        kswin_variants=kswin_variants,
    )

    def progress_callback(payload: dict[str, Any]) -> None:
        label = str(payload.get("label", "")).strip()
        detail = str(payload.get("detail", "")).strip()
        completed = payload.get("completed_units")
        total = payload.get("total_units")
        print(f"[drift-adaptive-variance] {completed}/{total} {label} {detail}".strip(), flush=True)

    outputs = drift_app.run_recalibration_experiments(
        clean_df,
        feature_cols=config["feature_cols"],
        target_col=config["target_col"],
        time_col=config["time_col"],
        model_names=config["model_names"],
        strategies=list(strategies),
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
    return write_adaptive_variance_outputs(
        outputs,
        output_dir=output_dir,
        seeds=seeds,
        source_run_dir=source_run_dir,
        cache_report=cache_report,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Drift A.9 adaptive strategies with repeated seeds and export bias-variance tables."
    )
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--feature-export-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strategies", default="", help="Comma-separated adaptive strategies for this shard.")
    parser.add_argument("--models", default="", help="Comma-separated model names for this shard.")
    parser.add_argument("--balance-modes", default="", help="Comma-separated balance modes for this shard.")
    parser.add_argument("--arf-variants", default="", help="Comma-separated ARF variants for this shard.")
    parser.add_argument("--kswin-variants", default="", help="Comma-separated KSWIN variants for this shard.")
    parser.add_argument(
        "--merge-input-dir",
        type=Path,
        action="append",
        default=[],
        help="Merge one or more completed shard output directories instead of running experiments.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "adaptive_variance_repetitions_merged" if args.merge_input_dir else "adaptive_variance_repetitions_5seeds"
        output_dir = (ROOT_DIR / "Resultados" / "Drift" / f"{prefix}_{stamp}").resolve()
    else:
        output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")

    if args.merge_input_dir:
        report = merge_adaptive_variance_outputs(
            input_dirs=[path.resolve() for path in args.merge_input_dir],
            output_dir=output_dir,
            seeds=_parse_seed_list(args.seeds),
        )
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return

    source_run_dir = args.source_run_dir.resolve()
    report = run_adaptive_variance_repetitions(
        source_run_dir=source_run_dir,
        output_dir=output_dir,
        seeds=_parse_seed_list(args.seeds),
        feature_export_path=args.feature_export_path.resolve() if args.feature_export_path else None,
        resume=bool(args.resume),
        strategy_filter=_split_csv_tokens(args.strategies),
        model_filter=_split_csv_tokens(args.models),
        balance_filter=_split_csv_tokens(args.balance_modes),
        arf_variant_filter=_split_csv_tokens(args.arf_variants),
        kswin_variant_filter=_split_csv_tokens(args.kswin_variants),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
