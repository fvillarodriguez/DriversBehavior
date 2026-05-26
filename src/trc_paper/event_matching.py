#!/usr/bin/env python3
"""
Exp. 2 — Match macroscopic entropy fluctuations against operational events.

For each event e at portico p_e and time t_e we extract:
  • H_{p,τ} for p = p_e, τ in [t_e − Δ_pre, t_e + Δ_post] (pre/post slices).
  • Matched random control intervals at the same portico, same DoW × hour.

We report:
  • Mean entropy pre vs post per event type.
  • Wilcoxon signed-rank test of H_post − H_pre per event type.
  • Cohen's d effect size.

This is the validation step for the entropy-as-safety-indicator claim of Exp. 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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
    load_porticos_geometry,
    normalize_portico_id,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h-15min", required=True, type=Path)
    p.add_argument("--events-db", required=True, type=Path)
    p.add_argument("--porticos-csv", required=True, type=Path)
    p.add_argument("--output-matched", required=True, type=Path)
    p.add_argument("--output-summary", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "default.yaml")
    return p.parse_args()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def main() -> int:
    args = parse_args()
    args.output_matched.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_config(args.config)
    event_cfg = cfg["event_matching"]
    types_include = set(event_cfg["event_types_include"])
    pre_min = int(event_cfg["temporal_window_pre_minutes"])
    post_min = int(event_cfg["temporal_window_post_minutes"])
    matches_per_event = int(event_cfg.get("matches_per_event", 4))

    porticos = load_porticos_geometry(args.porticos_csv)

    h_df = pd.read_parquet(args.h_15min)
    h_df["portico_norm"] = h_df["portico"].astype(str).map(normalize_portico_id)
    h_df["tau"] = pd.to_datetime(h_df["tau"])
    h_df = h_df.sort_values(["portico_norm", "tau"]).reset_index(drop=True)

    con = duckdb.connect(str(args.events_db), read_only=True)
    events = con.execute(
        "SELECT evento_time, tipo_evento, portico_inicio FROM eventos"
    ).fetchdf()
    con.close()
    events["tipo_lower"] = events["tipo_evento"].astype(str).str.lower()
    events = events[events["tipo_lower"].apply(
        lambda t: any(target in t for target in types_include)
    )]
    events["portico_norm"] = events["portico_inicio"].astype(str).map(normalize_portico_id)
    events["evento_time"] = pd.to_datetime(events["evento_time"])
    print(f"Events of interest: {len(events):,}")

    matched_rows: List[Dict] = []
    rng = np.random.default_rng(42)
    for _, ev in events.iterrows():
        sub = h_df[h_df["portico_norm"] == ev["portico_norm"]]
        if sub.empty:
            continue
        # Pre / post around the event
        pre_mask = (sub["tau"] >= ev["evento_time"] - pd.Timedelta(minutes=pre_min)) & (sub["tau"] < ev["evento_time"])
        post_mask = (sub["tau"] >= ev["evento_time"]) & (sub["tau"] <= ev["evento_time"] + pd.Timedelta(minutes=post_min))
        H_pre = sub.loc[pre_mask, "H"].mean()
        H_post = sub.loc[post_mask, "H"].mean()
        if np.isnan(H_pre) or np.isnan(H_post):
            continue

        # Matched random controls: same hour-of-day, same DoW, different date
        ev_hour = ev["evento_time"].hour
        ev_dow = ev["evento_time"].dayofweek
        candidate_mask = (
            (sub["tau"].dt.hour == ev_hour)
            & (sub["tau"].dt.dayofweek == ev_dow)
            & (sub["tau"].dt.date != ev["evento_time"].date())
        )
        candidates = sub.loc[candidate_mask, "H"].dropna().to_numpy()
        if len(candidates) >= matches_per_event:
            control_sample = rng.choice(candidates, size=matches_per_event, replace=False)
            H_control = float(control_sample.mean())
        else:
            H_control = float("nan")

        matched_rows.append({
            "event_time": ev["evento_time"],
            "event_type": ev["tipo_evento"],
            "portico": ev["portico_norm"],
            "H_pre": float(H_pre),
            "H_post": float(H_post),
            "H_control": H_control,
        })

    matched = pd.DataFrame(
        matched_rows,
        columns=["event_time", "event_type", "portico", "H_pre", "H_post", "H_control"],
    )
    matched.to_parquet(args.output_matched, index=False)
    print(f"Matched pairs written: {len(matched):,}")

    # Per-type summary (skip groupby on empty frame to avoid KeyError)
    summary: Dict[str, Dict] = {}
    groups = matched.groupby("event_type") if not matched.empty else []
    for et, group in groups:
        H_pre = group["H_pre"].to_numpy()
        H_post = group["H_post"].to_numpy()
        H_ctrl = group["H_control"].dropna().to_numpy()
        # Wilcoxon for pre vs post (within-event)
        if len(H_pre) >= 5:
            try:
                w_stat, w_p = stats.wilcoxon(H_post, H_pre)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
        else:
            w_stat, w_p = float("nan"), float("nan")
        summary[str(et)] = {
            "n_events": int(len(group)),
            "H_pre_mean": float(H_pre.mean()) if len(H_pre) else float("nan"),
            "H_post_mean": float(H_post.mean()) if len(H_post) else float("nan"),
            "H_control_mean": float(H_ctrl.mean()) if len(H_ctrl) else float("nan"),
            "delta_post_minus_pre": float((H_post - H_pre).mean()) if len(H_pre) else float("nan"),
            "delta_post_minus_control": float((H_post.mean() - H_ctrl.mean())) if len(H_ctrl) else float("nan"),
            "cohens_d_post_vs_pre": cohens_d(H_post, H_pre),
            "cohens_d_post_vs_control": cohens_d(H_post, H_ctrl),
            "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else None,
        }
    write_json_atomic(args.output_summary, {"event_types": summary, "n_total_matches": int(len(matched))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
