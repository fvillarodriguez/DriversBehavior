#!/usr/bin/env python3
"""
Exp. 1 — Stationary distribution π and Kolmogorov detailed-balance test.

For the global P_{ij} estimate (from script 03) we compute:

  • π = left eigenvector of P with eigenvalue 1 (normalized to sum 1).
  • Mixing time τ_mix ≈ 1 / (1 - |λ_2|), where λ_2 is the second eigenvalue.
  • Kolmogorov criterion for reversibility: for every cycle i → j → k → i,
    P_{ij} · P_{jk} · P_{ki} = P_{ik} · P_{kj} · P_{ji}.
  • Detailed-balance asymmetry indicator: A_{ij} = P_{ij} π_i − P_{ji} π_j.

A_{ij} ≠ 0 ⇒ irreversibility along the pair (i,j). The output captures the
top-K asymmetric pairs which form the central narrative figure of Section 6.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

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
    p.add_argument("--p-global", required=True, type=Path)
    p.add_argument("--output-result", required=True, type=Path)
    p.add_argument("--output-pairs", required=True, type=Path)
    p.add_argument("--top-k-pairs", type=int, default=20)
    return p.parse_args()


def long_to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, Dict[int, int]]:
    states = sorted({int(s) for s in df["from_state"].unique()})
    idx = {s: i for i, s in enumerate(states)}
    K = len(states)
    P = np.zeros((K, K))
    for _, row in df.iterrows():
        P[idx[int(row["from_state"])], idx[int(row["to_state"])]] = float(row["P_ij"])
    # Ensure rows sum to 1 (numerical safety)
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P = P / rs
    return P, idx


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the left eigenvector associated with eigenvalue 1."""
    eig_vals, eig_vecs = np.linalg.eig(P.T)
    one_idx = int(np.argmin(np.abs(eig_vals - 1.0)))
    v = np.real(eig_vecs[:, one_idx])
    v = v / v.sum()
    return np.clip(v, 0.0, 1.0)


def mixing_time(P: np.ndarray) -> float:
    eig_vals = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    if len(eig_vals) < 2:
        return float("inf")
    lam2 = eig_vals[1]
    if lam2 >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - lam2))


def kolmogorov_cycle_test(P: np.ndarray, max_cycle_len: int = 3) -> Dict[str, Any]:
    """Compute violations of Kolmogorov's reversibility criterion over cycles."""
    K = P.shape[0]
    violations = []
    for cycle in combinations(range(K), 3):
        i, j, k = cycle
        forward = P[i, j] * P[j, k] * P[k, i]
        backward = P[i, k] * P[k, j] * P[j, i]
        if forward + backward > 0:
            ratio = forward / max(backward, 1e-12)
            violations.append({
                "cycle": [i, j, k], "forward": forward, "backward": backward, "ratio": ratio
            })
    sorted_violations = sorted(violations, key=lambda v: abs(np.log(v["ratio"] + 1e-12)), reverse=True)
    return {
        "n_cycles": len(violations),
        "top_5_violations": sorted_violations[:5],
        "max_log_ratio": max((abs(np.log(v["ratio"] + 1e-12)) for v in violations), default=0.0),
    }


def asymmetry_pairs(
    P: np.ndarray, pi: np.ndarray, state_inv: Dict[int, int], top_k: int
) -> pd.DataFrame:
    K = P.shape[0]
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            A = P[i, j] * pi[i] - P[j, i] * pi[j]
            rows.append({
                "from_state": state_inv[i],
                "to_state": state_inv[j],
                "asymmetry_A_ij": float(A),
                "abs_A": float(abs(A)),
                "Pij": float(P[i, j]),
                "Pji": float(P[j, i]),
                "pi_i": float(pi[i]),
                "pi_j": float(pi[j]),
            })
    df = pd.DataFrame(rows).sort_values("abs_A", ascending=False)
    return df.head(top_k)


def main() -> int:
    args = parse_args()
    args.output_result.parent.mkdir(parents=True, exist_ok=True)

    p_df = pd.read_parquet(args.p_global)
    P, state_index = long_to_matrix(p_df)
    inverse = {v: k for k, v in state_index.items()}

    print("Computing stationary distribution …")
    pi = stationary_distribution(P)
    print("Computing mixing time …")
    tau_mix = mixing_time(P)
    print("Running Kolmogorov reversibility test …")
    kolmo = kolmogorov_cycle_test(P)
    print("Ranking asymmetric pairs …")
    pairs_df = asymmetry_pairs(P, pi, inverse, args.top_k_pairs)
    pairs_df.to_parquet(args.output_pairs, index=False)

    pi_entropy = float(-np.sum(pi * np.log(np.clip(pi, 1e-12, 1.0))))
    pi_named = {str(inverse[i]): float(pi[i]) for i in range(len(pi))}

    write_json_atomic(args.output_result, {
        "state_index": {str(k): int(v) for k, v in state_index.items()},
        "stationary_pi": pi_named,
        "entropy_pi": pi_entropy,
        "mixing_time": tau_mix,
        "kolmogorov_test": kolmo,
        "top_asymmetric_pairs_path": str(args.output_pairs),
    })
    print(f"  π  = {pi_named}")
    print(f"  H(π) = {pi_entropy:.4f},  τ_mix = {tau_mix:.2f} weeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
