#!/usr/bin/env python3
"""
Exp. 1 — Estimate soft Markov transition matrix P_{ij}^{(Δt)}.

For each pair of consecutive Markov steps (default: 1 week apart) and each
plate m in the chosen subpopulation, we collect (r_{m,t}, r_{m,t+Δt}) where
r_{m,t} is the GMM soft-membership vector of plate m at time t. The transition
estimator is

    P_{ij} = (Σ_m Σ_t r_{m,t,i} · r_{m,t+Δt,j}) / (Σ_m Σ_t r_{m,t,i})

When soft probabilities are unavailable the script falls back to the hard
estimator P_{ij} = #{i→j} / #{i→·}. The unknown class (-1) is included as an
explicit state "U" if present.

Bootstrap confidence intervals are produced by cluster_by_plate resampling
(1000 replicas by default), which is the correct resampling unit when we
treat each plate's trajectory as an independent sample.

Outputs:
  --output-global    P_{ij} point estimate, one row per (from_state, to_state)
  --output-bootstrap one row per bootstrap replica × pair
  --output-summary   JSON with shape, n_replicas, KL divergence to stationary, etc.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trc_paper.lib import (  # noqa: E402
    write_json_atomic,
)

PANDAS_STEP_ALIASES = {
    "1D": pd.Timedelta(days=1),
    "1W": pd.Timedelta(weeks=1),
    "1M": pd.Timedelta(days=30),  # canonical step for monthly approximation
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assignments-db", required=True, type=Path)
    p.add_argument("--step", default="1W")
    p.add_argument("--subpopulation", choices=("frequent", "all", "stratified"), default="frequent")
    p.add_argument("--bootstrap-replicas", type=int, default=1000)
    p.add_argument("--output-global", required=True, type=Path)
    p.add_argument("--output-bootstrap", required=True, type=Path)
    p.add_argument("--output-summary", required=True, type=Path)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_assignments_with_pairs(
    con: duckdb.DuckDBPyConnection, step: pd.Timedelta, subpopulation: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Load consecutive (t, t+Δ) assignment pairs per plate in long format."""
    schema = con.execute("DESCRIBE asgn_db.dynamic_assignments").fetchdf()
    cols = schema["column_name"].tolist()
    prob_cols = sorted(
        c for c in cols
        if c.startswith("cluster_prob_") and c[len("cluster_prob_"):].lstrip("-").isdigit()
    )

    # Subpopulation filter — frequent drivers
    if subpopulation == "frequent":
        plate_filter = """
            plate IN (
                SELECT plate FROM (
                    SELECT plate, COUNT(*) AS n
                    FROM asgn_db.dynamic_assignments
                    GROUP BY plate
                ) WHERE n >= 20
            )
        """
    else:
        plate_filter = "TRUE"

    select_prob_cols = ", ".join(f"{c}" for c in prob_cols)
    step_seconds = int(step.total_seconds())

    # We pre-sort by plate, window_end and emit (current, next) pairs using LAG/LEAD.
    query = f"""
        WITH base AS (
            SELECT
                plate,
                window_end,
                CAST(cluster_label AS INTEGER) AS state
                {',' + select_prob_cols if select_prob_cols else ''}
            FROM asgn_db.dynamic_assignments
            WHERE {plate_filter}
        ),
        ordered AS (
            SELECT *,
                LEAD(window_end) OVER (PARTITION BY plate ORDER BY window_end) AS next_window_end,
                LEAD(state)      OVER (PARTITION BY plate ORDER BY window_end) AS next_state
                {','.join([''] + [f"LEAD({c}) OVER (PARTITION BY plate ORDER BY window_end) AS next_{c}" for c in prob_cols]) if prob_cols else ''}
            FROM base
        )
        SELECT *
        FROM ordered
        WHERE next_window_end IS NOT NULL
          AND date_diff('second', window_end, next_window_end) BETWEEN {step_seconds - 86400} AND {step_seconds + 86400}
    """
    df = con.execute(query).fetchdf()
    return df, prob_cols


def compute_p_matrix(
    df: pd.DataFrame, prob_cols: List[str], state_index: Dict[int, int]
) -> np.ndarray:
    K = len(state_index)
    numer = np.zeros((K, K), dtype=np.float64)
    denom = np.zeros(K, dtype=np.float64)

    if prob_cols:
        # Soft estimator
        # Vectorize: each row contributes outer(r_t, r_{t+1}) to numer.
        # We compute via einsum on a matrix view.
        labels = sorted(int(c[len("cluster_prob_"):]) for c in prob_cols)
        rt_cols = [f"cluster_prob_{k}" for k in labels]
        rt1_cols = [f"next_cluster_prob_{k}" for k in labels]

        # Index in P for each label
        idx = np.array([state_index[k] for k in labels])

        rt = df[rt_cols].fillna(0.0).to_numpy()
        rt1 = df[rt1_cols].fillna(0.0).to_numpy()

        block = rt.T @ rt1  # (K_labels, K_labels)
        for i_local, i_global in enumerate(idx):
            for j_local, j_global in enumerate(idx):
                numer[i_global, j_global] += block[i_local, j_local]
            denom[i_global] += rt[:, i_local].sum()

        # Add hard contribution for unknown states (no probability vector)
        if "U" in state_index:
            u_idx = state_index["U"] if "U" in state_index else None
        # If hard label is -1 it had no prob vector; treat as state U.
        is_unknown_t = df["state"] == -1
        is_unknown_t1 = df["next_state"] == -1
        if is_unknown_t.any():
            u = state_index[-1]
            for k, kcol in zip(labels, rt1_cols):
                numer[u, state_index[k]] += df.loc[is_unknown_t, kcol].fillna(0.0).sum()
            numer[u, state_index[-1]] += int((is_unknown_t & is_unknown_t1).sum())
            denom[u] += int(is_unknown_t.sum())
        if is_unknown_t1.any():
            for k, kcol in zip(labels, rt_cols):
                numer[state_index[k], state_index[-1]] += df.loc[is_unknown_t1, kcol].fillna(0.0).sum()
    else:
        # Hard estimator
        for s_from, g_from in state_index.items():
            sub = df[df["state"] == s_from]
            denom[g_from] = len(sub)
            for s_to, g_to in state_index.items():
                numer[g_from, g_to] = int((sub["next_state"] == s_to).sum())

    P = np.divide(numer, denom[:, None], out=np.zeros_like(numer), where=denom[:, None] > 0)
    return P


def bootstrap_p(
    df: pd.DataFrame,
    prob_cols: List[str],
    state_index: Dict[int, int],
    n_replicas: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    plates = df["plate"].unique()
    n_plates = len(plates)
    K = len(state_index)
    cube = np.zeros((n_replicas, K, K), dtype=np.float64)

    plate_to_rows = df.groupby("plate", sort=False).indices

    for b in range(n_replicas):
        sample = rng.choice(plates, size=n_plates, replace=True)
        idx = np.concatenate([plate_to_rows[p] for p in sample])
        sub = df.iloc[idx]
        cube[b] = compute_p_matrix(sub, prob_cols, state_index)
        if (b + 1) % 100 == 0:
            print(f"    bootstrap {b+1}/{n_replicas}")
    return cube


def long_format(P: np.ndarray, state_index: Dict[int, int]) -> pd.DataFrame:
    inverse = {v: k for k, v in state_index.items()}
    rows = []
    K = P.shape[0]
    for i in range(K):
        for j in range(K):
            rows.append({
                "from_state": inverse[i],
                "to_state": inverse[j],
                "P_ij": float(P[i, j]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.output_global.parent.mkdir(parents=True, exist_ok=True)

    if args.step not in PANDAS_STEP_ALIASES:
        raise ValueError(f"Unsupported --step value: {args.step}")
    step = PANDAS_STEP_ALIASES[args.step]

    print("=" * 60)
    print(f"Markov soft transition P_{{ij}}  step={args.step}")
    print(f"  subpopulation: {args.subpopulation}")
    print(f"  bootstrap: {args.bootstrap_replicas} replicas")
    print("=" * 60)

    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{args.assignments_db.as_posix()}' AS asgn_db (READ_ONLY)")

    print("Loading consecutive (t, t+Δ) pairs …")
    t0 = time.time()
    df, prob_cols = load_assignments_with_pairs(con, step, args.subpopulation)
    print(f"  pairs loaded: {len(df):,}  ({time.time()-t0:.1f}s)  prob_cols={len(prob_cols)}")
    con.close()

    if df.empty:
        raise RuntimeError("No pairs available — check assignments DB and step.")

    # Build the state index, always including -1 (Unknown) as state "U"
    labels_present = sorted({int(s) for s in df["state"].dropna().unique()})
    state_index: Dict[Any, int] = {s: i for i, s in enumerate(labels_present)}
    print(f"  state index: {state_index}")

    print("Estimating global P_{ij} …")
    t0 = time.time()
    P = compute_p_matrix(df, prob_cols, state_index)
    print(f"  done ({time.time()-t0:.1f}s)")

    long_format(P, state_index).to_parquet(args.output_global, index=False)

    print("Running cluster bootstrap …")
    t0 = time.time()
    cube = bootstrap_p(df, prob_cols, state_index, args.bootstrap_replicas, args.random_state)
    print(f"  bootstrap done ({(time.time()-t0)/60.0:.1f}min)")

    # Long-format bootstrap output
    inverse = {v: k for k, v in state_index.items()}
    boot_rows = []
    for b in range(cube.shape[0]):
        for i in range(cube.shape[1]):
            for j in range(cube.shape[2]):
                boot_rows.append({
                    "replica": b,
                    "from_state": inverse[i],
                    "to_state": inverse[j],
                    "P_ij": float(cube[b, i, j]),
                })
    pd.DataFrame(boot_rows).to_parquet(args.output_bootstrap, index=False)

    # Summary
    summary = {
        "step": args.step,
        "subpopulation": args.subpopulation,
        "n_pairs": int(len(df)),
        "n_plates": int(df["plate"].nunique()),
        "state_index": {str(k): int(v) for k, v in state_index.items()},
        "P_global_shape": list(P.shape),
        "n_bootstrap_replicas": int(cube.shape[0]),
        "soft_membership": bool(prob_cols),
    }
    write_json_atomic(args.output_summary, summary)
    print(f"  outputs: {args.output_global} | {args.output_bootstrap} | {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
