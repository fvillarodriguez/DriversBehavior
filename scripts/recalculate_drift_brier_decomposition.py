#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.drift_bias_variance import enrich_drift_rows_with_bias_variance
from src.drift_detection_app import (
    format_appendix_tables,
    format_appendix_tables_mean,
    summarize_results,
)


DEFAULT_INPUT = Path(
    "Resultados/drift_recalibration_runs/run_6d1e6bd611a2b7a1/"
    "drift_recalibration_optuna_20260409_125138.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _max_abs_additive_residual(df: pd.DataFrame) -> float | None:
    required = {"brier_score", "bias2", "variance", "noise"}
    if df.empty or not required <= set(df.columns):
        return None
    values = df[list(required)].apply(pd.to_numeric, errors="coerce")
    residual = values["brier_score"] - values["bias2"] - values["variance"] - values["noise"]
    residual = residual.replace([np.inf, -np.inf], np.nan).dropna()
    if residual.empty:
        return None
    return float(residual.abs().max())


def _nonzero_count(df: pd.DataFrame, column: str, *, tol: float = 1e-12) -> int:
    if df.empty or column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return int((values.abs() > tol).sum())


def _max_abs_delta(before: pd.DataFrame, after: pd.DataFrame, column: str) -> float | None:
    if before.empty or after.empty or column not in before.columns or column not in after.columns:
        return None
    n = min(len(before), len(after))
    if n <= 0:
        return None
    left = pd.to_numeric(before[column].iloc[:n], errors="coerce")
    right = pd.to_numeric(after[column].iloc[:n], errors="coerce")
    delta = (right - left).replace([np.inf, -np.inf], np.nan).dropna()
    if delta.empty:
        return None
    return float(delta.abs().max())


def _report_for_table(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(after)),
        "old_nonzero_variance_rows": _nonzero_count(before, "variance"),
        "new_nonzero_variance_rows": _nonzero_count(after, "variance"),
        "max_abs_brier_delta": _max_abs_delta(before, after, "brier_score"),
        "max_abs_bias2_delta": _max_abs_delta(before, after, "bias2"),
        "max_abs_variance_delta": _max_abs_delta(before, after, "variance"),
        "max_abs_additive_residual": _max_abs_additive_residual(after),
    }


def recalculate(input_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = _load_json(input_path)
    roc_payload = list(payload.get("roc_payload") or [])

    yearly_before = pd.DataFrame(payload.get("yearly_results") or [])
    adaptive_before = pd.DataFrame(payload.get("adaptive_results") or [])

    yearly_after = pd.DataFrame(
        enrich_drift_rows_with_bias_variance(
            yearly_before.to_dict(orient="records"),
            roc_payload,
            yearly=True,
            overwrite_existing=True,
        )
    )
    adaptive_after = pd.DataFrame(
        enrich_drift_rows_with_bias_variance(
            adaptive_before.to_dict(orient="records"),
            roc_payload,
            yearly=False,
            overwrite_existing=True,
        )
    )

    summary_after = summarize_results(yearly_after, adaptive_after)
    appendix_after = format_appendix_tables(yearly_after, adaptive_after)
    appendix_mean_after = format_appendix_tables_mean(yearly_after, adaptive_after)

    _write_table(yearly_after, output_dir / "yearly_results_recalculated.csv")
    _write_table(adaptive_after, output_dir / "adaptive_results_recalculated.csv")
    _write_table(summary_after, output_dir / "summary_recalculated.csv")

    for key, table in appendix_after.items():
        _write_table(table, output_dir / f"{key.replace('.', '')}_recalculated.csv")
    for key, table in appendix_mean_after.items():
        _write_table(table, output_dir / f"{key.replace('.', '')}_mean_recalculated.csv")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_json": str(input_path),
        "output_dir": str(output_dir),
        "repetition_seeds": (payload.get("run_manifest") or {}).get("repetition_seeds"),
        "roc_payload_rows": int(len(roc_payload)),
        "yearly_results": _report_for_table(yearly_before, yearly_after),
        "adaptive_results": _report_for_table(adaptive_before, adaptive_after),
        "summary": {
            "rows": int(len(summary_after)),
            "new_nonzero_variance_rows": _nonzero_count(summary_after, "variance"),
            "max_abs_additive_residual": _max_abs_additive_residual(summary_after),
        },
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalculate Drift Brier bias-variance-noise tables from saved ROC payloads."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input.resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (Path("Resultados/Drift") / f"brier_decomposition_recalculated_{stamp}").resolve()
    else:
        output_dir = args.output_dir.resolve()

    report = recalculate(input_path, output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
