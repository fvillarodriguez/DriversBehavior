#!/usr/bin/env python3
"""
Phase 0 — regenerate dynamic_assignments and dynamic_window_summary.

This wraps `run_dynamic_gmm_clustering` from src/clustering.py with the
configuration declared in config/default.yaml (or override) and persists the
output under the run-specific path expected by the Snakefile.

Behavior:
  - Loads base cluster features from Resultados/cluster_features*.duckdb if
    available, otherwise computes them on the fly.
  - Trains the GMM on frequent drivers as defined by train_params.
  - Iterates the daily sliding window and persists per-window assignments
    plus a fitted model joblib for reproducibility.
  - Uses checkpointing — re-running the script resumes from where it left off
    if `resume_existing=True` and the same config_fingerprint is found.

This is the costly step (~30-50h CPU). For first-time runs we strongly
recommend executing it inside a tmux/screen session.

Run example:
    python scripts/01_run_dynamic_gmm.py \
        --config config/default.yaml \
        --k 5 \
        --output-db results/dynamic_gmm/k5_2018-01-01_2024-09-30_assignments.duckdb \
        --output-model results/dynamic_gmm/k5_2018-01-01_2024-09-30_model.joblib \
        --output-metadata results/dynamic_gmm/k5_2018-01-01_2024-09-30_run.json \
        --parallel-jobs 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# src/ on the import path so we can import the legacy clustering module.
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.trc_paper.lib import load_yaml_config, write_json_atomic  # noqa: E402

# Import lazily to surface clearer errors if dependencies are missing.
from clustering import (  # type: ignore  # noqa: E402
    Clusterization,
    FlowColumns,
    TTC_MAX_BY_PORTICO,
    run_dynamic_gmm_clustering,
)
from utils import load_flujos_range  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--k", type=int)
    p.add_argument("--output-db", required=True, type=Path)
    p.add_argument("--output-model", required=True, type=Path)
    p.add_argument("--output-metadata", required=True, type=Path)
    p.add_argument("--parallel-jobs", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    gmm_cfg = cfg["dynamic_gmm"]
    k = args.k if args.k is not None else gmm_cfg["k"]

    args.output_db.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Dynamic GMM regeneration  K={k}")
    print(f"  range:  {gmm_cfg['date_start']}  →  {gmm_cfg['date_end']}")
    print(f"  window: {gmm_cfg['window_days']}d  step: {gmm_cfg['window_step_days']}d")
    print(f"  output: {args.output_db}")
    print(f"  resume: {gmm_cfg.get('resume_existing', True)}")
    print(f"  parallel_jobs: {args.parallel_jobs}")
    print("=" * 70)

    flow_cols = FlowColumns()
    ttc_max_map = TTC_MAX_BY_PORTICO if gmm_cfg.get("ttc_mode") == "dynamic" else None

    date_start = pd.Timestamp(gmm_cfg["date_start"])
    date_end = pd.Timestamp(gmm_cfg["date_end"])

    print("Loading flow records (DuckDB streaming)…")
    t0 = time.time()
    flujos = load_flujos_range(
        date_start=date_start,
        date_end=date_end,
    )
    print(f"  rows loaded: {len(flujos):,}  ({time.time()-t0:.1f}s)")

    print("Computing per-plate base cluster features…")
    t0 = time.time()
    base_features_df = Clusterization(
        flujos,
        flow_cols=flow_cols,
        ttc_max_map=ttc_max_map,
        ttc_mode=gmm_cfg.get("ttc_mode", "dynamic"),
        fixed_ttc_s=gmm_cfg.get("ttc_fixed_seconds"),
    )
    print(f"  base features: {base_features_df.shape}  ({time.time()-t0:.1f}s)")

    # Use the run-specific DuckDB as both incremental store and final output.
    incremental = args.output_db

    print("Running run_dynamic_gmm_clustering …")
    t0 = time.time()
    result: Dict[str, Any] = run_dynamic_gmm_clustering(
        base_features_df=base_features_df,
        feature_cols=list(gmm_cfg["feature_cols"]),
        flow_cols=flow_cols,
        ttc_max_map=ttc_max_map,
        k=int(k),
        confidence_threshold_proba=float(gmm_cfg["confidence_threshold_proba"]),
        window_days=int(gmm_cfg["window_days"]),
        date_start=date_start,
        date_end=date_end,
        min_window_passes=int(gmm_cfg["min_window_passes"]),
        train_params=dict(gmm_cfg["train_params"]),
        random_state=int(gmm_cfg["random_state"]),
        covariance_type=str(gmm_cfg["covariance_type"]),
        ttc_mode=str(gmm_cfg["ttc_mode"]),
        fixed_ttc_s=gmm_cfg.get("ttc_fixed_seconds"),
        include_membership_probabilities=bool(
            gmm_cfg.get("include_membership_probabilities", True)
        ),
        parallel_jobs=int(args.parallel_jobs),
        checkpoint_enabled=True,
        incremental_db_path=incremental,
        resume_existing=bool(gmm_cfg.get("resume_existing", True)),
        assignment_scope=str(gmm_cfg.get("assignment_scope", "all")),
        load_final_result=False,
        metadata={"source": "papers/dynamic_clusters_trc"},
    )
    duration_s = time.time() - t0
    print(f"  finished in {duration_s/3600.0:.2f}h")

    model_artifact = result.get("model_artifact_path")
    if model_artifact and Path(model_artifact).exists():
        shutil.copy2(model_artifact, args.output_model)
    else:
        print("WARNING: no model artifact returned, --output-model will not be populated")

    metadata = result.get("metadata") or {}
    write_json_atomic(args.output_metadata, {
        "duration_seconds": duration_s,
        "result_summary": {
            "n_windows": metadata.get("n_windows"),
            "k": metadata.get("k"),
            "config_fingerprint": metadata.get("config_fingerprint"),
            "run_id": metadata.get("run_id"),
            "checkpoint_db": str(incremental),
        },
        "metadata": metadata,
    })

    print(f"  assignments DuckDB: {args.output_db}")
    print(f"  model joblib:       {args.output_model}")
    print(f"  metadata JSON:      {args.output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
