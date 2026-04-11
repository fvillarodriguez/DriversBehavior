#!/usr/bin/env python3
"""CLI entrypoint for Neural drift staged experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    from src import neural_drift_experiments as nde

    parser = argparse.ArgumentParser(
        description="Run Neural drift experiments outside Streamlit.",
    )
    parser.add_argument(
        "--study",
        choices=list(nde.AVAILABLE_STUDIES),
        default=nde.STUDY_ALL,
        help="Study to execute.",
    )
    parser.add_argument(
        "--phase",
        choices=list(nde.AVAILABLE_PHASES),
        default="all",
        help="Phase to execute.",
    )
    parser.add_argument(
        "--resume-run-id",
        default=None,
        help="Existing run id to resume.",
    )
    parser.add_argument(
        "--feature-export",
        default=str(nde.DEFAULT_FEATURE_EXPORT_PATH),
        help="DuckDB export with `clean_features` for the experiment protocol.",
    )
    parser.add_argument(
        "--balance-mode",
        choices=list(nde.BALANCE_MODE_OPTIONS),
        default="none",
        help="Single balance mode for the experiment sweep.",
    )
    return parser


def run_cli(
    *,
    study: str,
    phase: str,
    balance_mode: str,
    feature_export: str,
    resume_run_id: Optional[str] = None,
) -> dict:
    from src import Neural_drift_app as neural_drift_app
    from src import neural_drift_experiments as nde

    feature_export_path = Path(feature_export).expanduser().resolve()
    dataset_bundle = neural_drift_app.resolve_dataset_from_context(
        {"feature_export_path": str(feature_export_path)}
    )
    result = nde.run_experiment_plan(
        dataset_bundle,
        base_config=dict(neural_drift_app.DEFAULT_CONFIG),
        balance_mode=str(balance_mode),
        selected_study=str(study),
        selected_phase=str(phase),
        resume_run_id=resume_run_id,
    )
    return {
        "run_id": str(result["run_id"]),
        "manifest_path": str(result["manifest_path"]),
        "artifacts_dir": str(Path(result["manifest_path"]).resolve().parent / "artifacts"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    payload = run_cli(
        study=str(args.study),
        phase=str(args.phase),
        balance_mode=str(args.balance_mode),
        feature_export=str(args.feature_export),
        resume_run_id=None if args.resume_run_id in {None, ""} else str(args.resume_run_id),
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
