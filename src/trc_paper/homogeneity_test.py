#!/usr/bin/env python3
"""
Exp. 1 — Temporal homogeneity test of the soft Markov matrix.

For each predefined split in config.markov.homogeneity_test.splits we estimate
a per-split P_{ij}^{(s)} restricted to pairs whose left endpoint window_end is
inside split s. We then test the global null

    H0: P^{(s_1)} = P^{(s_2)} = … = P^{(s_S)}

via a robust chi-square statistic adapted to multiple sample paths
(Leskelä 2026). We also report pairwise total-variation distances
||P^{(s)} - P^{(s')}||_TV and a Frobenius-distance ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

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
    p.add_argument("--assignments-db", required=True, type=Path)
    p.add_argument("--p-global", required=True, type=Path)
    p.add_argument("--output-result", required=True, type=Path)
    p.add_argument("--output-p-per-split", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default.yaml")
    return p.parse_args()


def estimate_p_for_split(
    con: duckdb.DuckDBPyConnection,
    split_start: str,
    split_end: str,
    state_index: Dict[int, int],
) -> np.ndarray:
    schema = con.execute("DESCRIBE asgn_db.dynamic_assignments").fetchdf()
    cols = schema["column_name"].tolist()
    prob_cols = sorted(
        c for c in cols
        if c.startswith("cluster_prob_") and c[len("cluster_prob_"):].lstrip("-").isdigit()
    )
    prob_cols_select = ", ".join(prob_cols) if prob_cols else ""
    next_lead = (
        ", " + ", ".join(
            f"LEAD({c}) OVER (PARTITION BY plate ORDER BY window_end) AS next_{c}"
            for c in prob_cols
        )
        if prob_cols else ""
    )
    df = con.execute(
        f"""
        WITH base AS (
            SELECT plate, window_end,
                   CAST(cluster_label AS INTEGER) AS state
                   {',' + prob_cols_select if prob_cols_select else ''}
            FROM asgn_db.dynamic_assignments
            WHERE window_end >= ?::TIMESTAMP AND window_end <= ?::TIMESTAMP
        ),
        ordered AS (
            SELECT *,
                LEAD(window_end) OVER (PARTITION BY plate ORDER BY window_end) AS next_window_end,
                LEAD(state)      OVER (PARTITION BY plate ORDER BY window_end) AS next_state
                {next_lead}
            FROM base
        )
        SELECT *
        FROM ordered
        WHERE next_window_end IS NOT NULL
          AND date_diff('second', window_end, next_window_end) BETWEEN 518400 AND 691200
        """,
        [split_start, split_end],
    ).fetchdf()

    K = len(state_index)
    P = np.zeros((K, K))
    denom = np.zeros(K)
    if df.empty:
        return P, denom, prob_cols

    if prob_cols:
        labels = sorted(int(c[len("cluster_prob_"):]) for c in prob_cols)
        rt = df[[f"cluster_prob_{k}" for k in labels]].fillna(0.0).to_numpy()
        rt1 = df[[f"next_cluster_prob_{k}" for k in labels]].fillna(0.0).to_numpy()
        idx = np.array([state_index[k] for k in labels])
        block = rt.T @ rt1
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                P[i_global, j_global] += block[i_local, j_local]
            denom[i_global] += rt[:, i_local].sum()
        # Unknown (-1) contributions
        is_u = df["state"] == -1
        if is_u.any():
            u = state_index[-1]
            for k_, col in zip(labels, [f"next_cluster_prob_{k}" for k in labels]):
                P[u, state_index[k_]] += df.loc[is_u, col].fillna(0.0).sum()
            denom[u] += int(is_u.sum())
    else:
        for s_from, g_from in state_index.items():
            sub = df[df["state"] == s_from]
            denom[g_from] = len(sub)
            for s_to, g_to in state_index.items():
                P[g_from, g_to] = int((sub["next_state"] == s_to).sum())

    Pnorm = np.divide(P, denom[:, None], out=np.zeros_like(P), where=denom[:, None] > 0)
    return Pnorm, denom, prob_cols


def robust_chi2_homogeneity(
    P_per_split: Dict[str, np.ndarray],
    denom_per_split: Dict[str, np.ndarray],
    P_pooled: np.ndarray,
) -> Dict[str, Any]:
    """Asymptotic chi-square test for H0: all P^{(s)} equal P_pooled.

    Reference: Leskelä (2026) Statistica Neerlandica, adapting the
    Anderson–Goodman statistic to multiple sample paths.
    """
    K = P_pooled.shape[0]
    stat = 0.0
    df_total = 0
    for split, P_s in P_per_split.items():
        denom = denom_per_split[split]
        for i in range(K):
            if denom[i] <= 0:
                continue
            for j in range(K):
                if P_pooled[i, j] <= 0:
                    continue
                expected = denom[i] * P_pooled[i, j]
                observed = denom[i] * P_s[i, j]
                stat += (observed - expected) ** 2 / max(expected, 1e-12)
                df_total += 1
    # Degrees of freedom: (n_splits - 1) * K * (K - 1)
    df_used = max((len(P_per_split) - 1) * K * (K - 1), 1)
    p_value = 1.0 - stats.chi2.cdf(stat, df_used)
    return {
        "statistic": float(stat),
        "degrees_of_freedom": int(df_used),
        "p_value": float(p_value),
    }


def total_variation(P1: np.ndarray, P2: np.ndarray) -> float:
    return 0.5 * float(np.abs(P1 - P2).sum(axis=1).max())


def main() -> int:
    args = parse_args()
    args.output_result.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_config(args.config)
    splits = cfg["markov"]["homogeneity_test"]["splits"]

    # Recover state index from the global P parquet
    p_global_long = pd.read_parquet(args.p_global)
    states_global = sorted(set(p_global_long["from_state"].unique()))
    state_index = {int(s): i for i, s in enumerate(states_global)}
    K = len(state_index)
    print(f"State index: {state_index}")

    P_global = np.zeros((K, K))
    for _, row in p_global_long.iterrows():
        i = state_index[int(row["from_state"])]
        j = state_index[int(row["to_state"])]
        P_global[i, j] = row["P_ij"]

    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{args.assignments_db.as_posix()}' AS asgn_db (READ_ONLY)")

    P_per_split: Dict[str, np.ndarray] = {}
    denom_per_split: Dict[str, np.ndarray] = {}
    per_split_long_rows: List[Dict[str, Any]] = []
    for s in splits:
        label = s["label"]
        print(f"  estimating P for split {label}: {s['start']} → {s['end']}")
        P_s, denom_s, _ = estimate_p_for_split(con, s["start"], s["end"], state_index)
        P_per_split[label] = P_s
        denom_per_split[label] = denom_s
        for fr, i in state_index.items():
            for to, j in state_index.items():
                per_split_long_rows.append({
                    "split": label, "from_state": fr, "to_state": to, "P_ij": float(P_s[i, j])
                })
    con.close()

    pd.DataFrame(per_split_long_rows).to_parquet(args.output_p_per_split, index=False)

    # Robust chi-square against the global P
    chi2 = robust_chi2_homogeneity(P_per_split, denom_per_split, P_global)

    # Pairwise total-variation matrix
    labels = list(P_per_split.keys())
    tv = {a: {b: total_variation(P_per_split[a], P_per_split[b]) for b in labels} for a in labels}

    result = {
        "splits": list(labels),
        "chi2_test": chi2,
        "pairwise_total_variation": tv,
        "state_index": {str(k): int(v) for k, v in state_index.items()},
    }
    write_json_atomic(args.output_result, result)
    print(f"  H0 (all P equal) chi² = {chi2['statistic']:.2f}, df = {chi2['degrees_of_freedom']}, p = {chi2['p_value']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
