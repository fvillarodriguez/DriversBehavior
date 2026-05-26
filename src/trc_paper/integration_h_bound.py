#!/usr/bin/env python3
"""
Phase 3 — Verify the H-bound relating the entropy of the stationary
distribution π and the spatial-temporal mean of macroscopic entropy H_{p,τ}.

Theorem (informal):
    H(π) ≤ E_{p,τ}[H_{p,τ}] + E_{p,τ}[ KL(share_{p,τ} || π) ]

This script:
  1. Loads π from the stationary JSON (script 05).
  2. Loads H_{p,τ} parquet (script 02).
  3. Reconstructs share_{p,τ,k} from H_{p,τ} columns (share_k columns persisted).
  4. Computes bar H, R (mean KL), and reports whether H(π) ≤ bar H + R.
  5. Reports the residual signed gap to inspect tightness.
"""

from __future__ import annotations

import argparse
import json
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
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stationary", required=True, type=Path)
    p.add_argument("--h-15min", required=True, type=Path)
    p.add_argument("--homogeneity", required=True, type=Path)
    p.add_argument("--output-result", required=True, type=Path)
    p.add_argument("--output-crosstab", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_result.parent.mkdir(parents=True, exist_ok=True)

    stationary = json.loads(args.stationary.read_text())
    pi_named: Dict[str, float] = stationary["stationary_pi"]
    H_pi = float(stationary["entropy_pi"])

    h_df = pd.read_parquet(args.h_15min)
    share_cols = [c for c in h_df.columns if c.startswith("share_")]
    if not share_cols:
        raise RuntimeError("H_{p,τ} parquet missing share_* columns.")

    # Match share columns to stationary states by k
    def share_key(col: str) -> str:
        return col.replace("share_", "")
    state_to_share_col = {share_key(c): c for c in share_cols}

    # Reorder the π vector to share columns
    pi_vec = np.array([
        float(pi_named.get(k, 0.0)) for k in state_to_share_col.keys()
    ])
    pi_vec = pi_vec / max(pi_vec.sum(), 1e-12)
    share_matrix = h_df[list(state_to_share_col.values())].to_numpy()

    # Compute KL(share || π) per row, mean, and bar H
    p_safe = np.clip(share_matrix, 1e-12, 1.0)
    pi_safe = np.clip(pi_vec, 1e-12, 1.0)
    kl_row = np.sum(p_safe * (np.log(p_safe) - np.log(pi_safe)), axis=1)
    R = float(np.nanmean(kl_row))
    bar_H = float(np.nanmean(h_df["H"]))
    upper_bound = bar_H + R

    # Crosstab: per-portico aggregates
    crosstab = h_df.groupby("portico").agg(
        H_mean=("H", "mean"),
        H_p95=("H", lambda x: x.quantile(0.95)),
        n_buckets=("H", "size"),
    ).reset_index()
    crosstab["KL_to_pi_mean"] = h_df.assign(_kl=kl_row).groupby("portico")["_kl"].mean().values
    crosstab.to_parquet(args.output_crosstab, index=False)

    # Whether the bound holds (theoretical guarantee but worth empirical check)
    holds = bool(H_pi <= upper_bound + 1e-6)

    result = {
        "H_pi": H_pi,
        "bar_H": bar_H,
        "R_mean_KL": R,
        "upper_bound": upper_bound,
        "bound_holds": holds,
        "gap_upper_minus_Hpi": float(upper_bound - H_pi),
        "n_buckets_considered": int(len(h_df)),
    }
    write_json_atomic(args.output_result, result)
    print(f"  H(π)        = {H_pi:.4f}")
    print(f"  E[H_{{p,τ}}] = {bar_H:.4f}")
    print(f"  R = E[KL]   = {R:.4f}")
    print(f"  bar_H + R   = {upper_bound:.4f}")
    print(f"  bound H(π) ≤ bar_H + R holds: {holds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
