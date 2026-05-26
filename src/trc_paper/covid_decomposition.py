#!/usr/bin/env python3
"""
Exp. 1+3 — COVID-19 natural experiment decomposition.

Inputs:
  --p-per-split  parquet from script 04 with P_{ij}^{(s)} per phase.
  --h-15min      parquet from script 02 with H_{p,τ} at 15 min granularity.

Outputs:
  --output-timeline   weekly time series of cluster shares (population-level).
  --output-result     JSON with per-phase π, asymmetry index, and a
                      summary of structural change vs the pre-COVID baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trc_paper.lib import (  # noqa: E402
    load_yaml_config,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p-per-split", required=True, type=Path)
    p.add_argument("--h-15min", required=True, type=Path)
    p.add_argument("--output-result", required=True, type=Path)
    p.add_argument("--output-timeline", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default.yaml")
    return p.parse_args()


def long_to_matrices(per_split_df: pd.DataFrame) -> tuple[Dict[str, np.ndarray], Dict[int, int]]:
    states = sorted({int(s) for s in per_split_df["from_state"].unique()})
    idx = {s: i for i, s in enumerate(states)}
    K = len(states)
    out: Dict[str, np.ndarray] = {}
    for split in per_split_df["split"].unique():
        P = np.zeros((K, K))
        sub = per_split_df[per_split_df["split"] == split]
        for _, row in sub.iterrows():
            P[idx[int(row["from_state"])], idx[int(row["to_state"])]] = float(row["P_ij"])
        rs = P.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        out[split] = P / rs
    return out, idx


def stationary(P: np.ndarray) -> np.ndarray:
    ev, vec = np.linalg.eig(P.T)
    i = int(np.argmin(np.abs(ev - 1.0)))
    v = np.real(vec[:, i])
    v = v / v.sum()
    return np.clip(v, 0.0, 1.0)


def main() -> int:
    args = parse_args()
    args.output_result.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_config(args.config)
    splits_cfg = {s["label"]: s for s in cfg["markov"]["homogeneity_test"]["splits"]}
    phase_labels = [
        "pre_covid", "lockdown_1", "paso_a_paso", "lockdown_2", "post_covid"
    ]

    per_split = pd.read_parquet(args.p_per_split)
    P_dict, state_index = long_to_matrices(per_split)
    inverse = {v: k for k, v in state_index.items()}

    # Per-phase π
    pi_per_phase: Dict[str, Dict[str, float]] = {}
    for ph in phase_labels:
        if ph not in P_dict:
            continue
        pi = stationary(P_dict[ph])
        pi_per_phase[ph] = {str(inverse[i]): float(pi[i]) for i in range(len(pi))}

    # Structural change vs pre_covid
    structural_change: Dict[str, Dict[str, float]] = {}
    if "pre_covid" in pi_per_phase:
        baseline = pi_per_phase["pre_covid"]
        for ph, pi_d in pi_per_phase.items():
            tv = 0.5 * sum(abs(pi_d[k] - baseline.get(k, 0.0)) for k in pi_d)
            kl = sum(
                pi_d[k] * np.log(max(pi_d[k], 1e-12) / max(baseline.get(k, 0.0), 1e-12))
                for k in pi_d
            )
            structural_change[ph] = {"total_variation_vs_pre_covid": tv, "kl_vs_pre_covid": float(kl)}

    # Weekly timeline from H_{p,τ} (aggregated cluster shares)
    h_df = pd.read_parquet(args.h_15min)
    h_df["week"] = pd.to_datetime(h_df["tau"]).dt.to_period("W").dt.start_time
    share_cols = [c for c in h_df.columns if c.startswith("share_")]
    weekly = h_df.groupby("week")[share_cols].mean().reset_index()
    weekly.to_parquet(args.output_timeline, index=False)

    write_json_atomic(args.output_result, {
        "phase_labels": phase_labels,
        "stationary_per_phase": pi_per_phase,
        "structural_change_vs_pre_covid": structural_change,
        "timeline_path": str(args.output_timeline),
    })
    print(f"  per-phase π estimated for {len(pi_per_phase)} phases")
    if structural_change:
        print("  Structural change vs pre_covid (TV):")
        for ph, vals in structural_change.items():
            print(f"    {ph:15s}  TV={vals['total_variation_vs_pre_covid']:.4f}  KL={vals['kl_vs_pre_covid']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
