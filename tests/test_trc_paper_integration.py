"""
Integration tests for the TRC paper pipeline.

These tests build a tiny synthetic universe (a small flujos.duckdb +
dynamic_assignments.duckdb + Porticos.csv + eventos.duckdb) and exercise the
real scripts end-to-end via their main() functions. Each test verifies that:

  • The script exits with code 0.
  • The expected output artifact is created and non-empty.
  • The output payload has the expected schema / numeric invariants.

The synthetic dataset is deliberately small (a few thousand rows) so the
whole integration suite runs in under a minute on a laptop.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"trc_paper_int_{name}",
        REPO_ROOT / "src" / "trc_paper" / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic universe fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_universe(tmp_path_factory) -> dict:
    """
    Build a small but realistic synthetic dataset:

      • 3 pórticos (P01, P02, P03)
      • 200 plates × ~15 windows = 3000 dynamic_assignments
      • ~50k detections across 30 days
      • 6 events distributed across pórticos and types
    """
    root = tmp_path_factory.mktemp("trc_universe")
    rng = np.random.default_rng(0)
    K = 3

    porticos = ["P01", "P02", "P03"]
    plates = [f"PLT{i:04d}" for i in range(200)]
    date_start = pd.Timestamp("2022-01-01")
    n_days = 30
    n_windows = 15

    # ----------------- flujos.duckdb -----------------
    # Dense sampling so 5/15/60 min buckets each contain >= 10 detections
    flow_rows = []
    for day in range(n_days):
        day_start = date_start + pd.Timedelta(days=day)
        # 480 timestamps per day (one every 3 min) × 3 pórticos × 3 detections ≈ 13k rows/day
        for minute in range(0, 24 * 60, 3):
            ts = day_start + pd.Timedelta(minutes=minute)
            for p in porticos:
                for _ in range(3):
                    plate = rng.choice(plates)
                    speed = float(rng.normal(85, 8))
                    flow_rows.append({
                        "FECHA": ts + pd.Timedelta(seconds=int(rng.integers(0, 60))),
                        "VELOCIDAD": speed,
                        "CATEGORIA": int(rng.integers(1, 4)),
                        "MATRICULA": plate,
                        "PORTICO": p,
                        "CARRIL": str(int(rng.integers(1, 4))),
                    })
    flow_df = pd.DataFrame(flow_rows)

    flow_db = root / "flujos.duckdb"
    con = duckdb.connect(str(flow_db))
    try:
        con.register("tmp", flow_df)
        con.execute("CREATE TABLE flujos_duckdb AS SELECT * FROM tmp")
    finally:
        con.close()

    # ----------------- Porticos.csv -----------------
    porticos_csv = root / "Porticos.csv"
    porticos_csv.write_text(
        "cod_portico;Km;Calzada;Orden;Eje;lat;lon\n"
        "P01;10.0;Oriente;1;RUTA 5 SUR;-33.45;-70.66\n"
        "P02;20.0;Oriente;2;RUTA 5 SUR;-33.55;-70.70\n"
        "P03;30.0;Oriente;3;RUTA 5 SUR;-33.62;-70.80\n"
    )

    # ----------------- eventos.duckdb -----------------
    event_rows = []
    event_types = ["accidente", "averia_mayor", "objeto_calzada"]
    for i in range(6):
        ev_time = date_start + pd.Timedelta(days=int(rng.integers(0, n_days)))
        event_rows.append({
            "evento_time": ev_time,
            "tipo_evento": event_types[i % len(event_types)],
            "eje": "RUTA 5 SUR",
            "calzada": "Oriente",
            "km": float(rng.choice([10.0, 20.0, 30.0])),
            "ultimo_portico": rng.choice(porticos),
            "portico_inicio": rng.choice(porticos),
            "portico_fin": rng.choice(porticos),
            "Descripcion": "synthetic",
            "SubTipo": "",
            "lat": -33.5,
            "lon": -70.7,
        })
    events_db = root / "eventos.duckdb"
    con = duckdb.connect(str(events_db))
    try:
        con.register("ev", pd.DataFrame(event_rows))
        con.execute("CREATE TABLE eventos AS SELECT * FROM ev")
    finally:
        con.close()

    # ----------------- dynamic_assignments.duckdb -----------------
    assign_rows = []
    for plate in plates:
        for w in range(n_windows):
            window_end = date_start + pd.Timedelta(days=2 * w + 1)
            window_start = window_end - pd.Timedelta(days=7)
            probs = rng.dirichlet(np.ones(K))
            row = {
                "run_id": "synthetic",
                "window_index": w + 1,
                "window_label": f"{window_start:%Y-%m-%d}_to_{window_end:%Y-%m-%d}",
                "window_start": window_start,
                "window_end": window_end,
                "plate": plate,
                "raw_cluster_label": int(np.argmax(probs)),
                "cluster_label": int(np.argmax(probs)),
                "confidence_score": float(probs.max()),
                "assignment_status": "assigned",
                "is_low_support": False,
                "soft_entropy": float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))),
            }
            for k in range(K):
                row[f"cluster_prob_{k}"] = float(probs[k])
            assign_rows.append(row)
    assignments_db = root / "dynamic_assignments.duckdb"
    con = duckdb.connect(str(assignments_db))
    try:
        con.register("a", pd.DataFrame(assign_rows))
        con.execute("CREATE TABLE dynamic_assignments AS SELECT * FROM a")
    finally:
        con.close()

    return {
        "root": root,
        "flow_db": flow_db,
        "porticos_csv": porticos_csv,
        "events_db": events_db,
        "assignments_db": assignments_db,
        "K": K,
        "date_start": date_start,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# Step 0 — validate_data
# ---------------------------------------------------------------------------


def test_validate_data_reports_ready(synthetic_universe: dict, tmp_path: Path) -> None:
    mod = _load_script("validate_data")
    out = tmp_path / "validation.json"

    argv = [
        "validate_data.py",
        "--flow-db", str(synthetic_universe["flow_db"]),
        "--porticos-csv", str(synthetic_universe["porticos_csv"]),
        "--events-db", str(synthetic_universe["events_db"]),
        "--date-start", "2022-01-01",
        "--date-end", "2022-01-30",
        "--output", str(out),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = mod.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["ready_for_phase_1"] is True
    assert payload["flow_db_check"]["ok"]
    assert payload["events_db_check"]["ok"]
    assert payload["porticos_check"]["ok"]


# ---------------------------------------------------------------------------
# Step 2 — compute_entropy
# ---------------------------------------------------------------------------


def test_compute_entropy_produces_parquet_with_H(synthetic_universe: dict, tmp_path: Path) -> None:
    mod = _load_script("compute_entropy")
    out_15 = tmp_path / "H_15.parquet"
    out_5 = tmp_path / "H_5.parquet"
    out_60 = tmp_path / "H_60.parquet"
    summary = tmp_path / "H_summary.json"

    argv = [
        "compute_entropy.py",
        "--assignments-db", str(synthetic_universe["assignments_db"]),
        "--flow-db", str(synthetic_universe["flow_db"]),
        "--output-15min", str(out_15),
        "--output-5min", str(out_5),
        "--output-60min", str(out_60),
        "--output-summary", str(summary),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = mod.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    df = pd.read_parquet(out_15)
    assert "H" in df.columns
    assert "portico" in df.columns
    assert "tau" in df.columns
    # Entropy non-negative
    assert (df["H"] >= -1e-9).all()
    # n_eff = exp(H) is positive
    assert (df["n_eff"] > 0).all()
    # Summary JSON has the three deltas
    s = json.loads(summary.read_text())
    assert set(s) == {"delta_5min", "delta_15min", "delta_60min"}


# ---------------------------------------------------------------------------
# Step 3 — markov_matrix
# ---------------------------------------------------------------------------


def test_markov_matrix_row_stochastic(synthetic_universe: dict, tmp_path: Path) -> None:
    mod = _load_script("markov_matrix")
    p_global = tmp_path / "P_global.parquet"
    p_boot = tmp_path / "P_boot.parquet"
    summary = tmp_path / "P_summary.json"

    argv = [
        "markov_matrix.py",
        "--assignments-db", str(synthetic_universe["assignments_db"]),
        "--step", "1D",  # synthetic windows are 2 days apart → 1D step lets pairs match
        "--subpopulation", "all",
        "--bootstrap-replicas", "5",
        "--output-global", str(p_global),
        "--output-bootstrap", str(p_boot),
        "--output-summary", str(summary),
    ]
    # The synthetic windows are 2 days apart; widen the matching window so we capture pairs.
    # The script enforces step ± 1 day, and step=1D allows window gaps of 0-2 days.
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = mod.main()
    finally:
        sys.argv = old_argv
    assert rc == 0

    global_df = pd.read_parquet(p_global)
    assert set(global_df.columns) == {"from_state", "to_state", "P_ij"}
    # Reconstruct matrix per row and check stochasticity
    K = global_df["from_state"].nunique()
    pivot = global_df.pivot(index="from_state", columns="to_state", values="P_ij").fillna(0.0)
    row_sums = pivot.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    # Bootstrap parquet has the expected schema
    boot_df = pd.read_parquet(p_boot)
    assert set(boot_df.columns) == {"replica", "from_state", "to_state", "P_ij"}
    assert boot_df["replica"].nunique() == 5

    s = json.loads(summary.read_text())
    assert s["step"] == "1D"
    assert s["n_bootstrap_replicas"] == 5


# ---------------------------------------------------------------------------
# Step 5 — stationary_asymmetry
# ---------------------------------------------------------------------------


def test_stationary_asymmetry_after_markov(synthetic_universe: dict, tmp_path: Path) -> None:
    # First, regenerate the Markov P (re-use the previous test's logic in isolation)
    markov_mod = _load_script("markov_matrix")
    p_global = tmp_path / "P.parquet"
    argv = [
        "markov_matrix.py",
        "--assignments-db", str(synthetic_universe["assignments_db"]),
        "--step", "1D",
        "--subpopulation", "all",
        "--bootstrap-replicas", "2",
        "--output-global", str(p_global),
        "--output-bootstrap", str(tmp_path / "boot.parquet"),
        "--output-summary", str(tmp_path / "sum.json"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        assert markov_mod.main() == 0
    finally:
        sys.argv = old_argv

    # Then stationary_asymmetry on top of that P
    stat_mod = _load_script("stationary_asymmetry")
    result = tmp_path / "stationary.json"
    pairs = tmp_path / "asym.parquet"
    argv = [
        "stationary_asymmetry.py",
        "--p-global", str(p_global),
        "--output-result", str(result),
        "--output-pairs", str(pairs),
        "--top-k-pairs", "5",
    ]
    sys.argv = argv
    try:
        rc = stat_mod.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    payload = json.loads(result.read_text())
    pi = payload["stationary_pi"]
    assert abs(sum(pi.values()) - 1.0) < 1e-6
    assert payload["entropy_pi"] >= 0
    assert "kolmogorov_test" in payload
    pairs_df = pd.read_parquet(pairs)
    assert len(pairs_df) <= 5
    assert set(pairs_df.columns) >= {"from_state", "to_state", "asymmetry_A_ij", "abs_A"}


# ---------------------------------------------------------------------------
# Step 7 — event_matching
# ---------------------------------------------------------------------------


def test_event_matching_produces_summary(synthetic_universe: dict, tmp_path: Path) -> None:
    # First compute_entropy (cheap on synthetic data)
    entropy_mod = _load_script("compute_entropy")
    out_15 = tmp_path / "H_15.parquet"
    argv = [
        "compute_entropy.py",
        "--assignments-db", str(synthetic_universe["assignments_db"]),
        "--flow-db", str(synthetic_universe["flow_db"]),
        "--output-15min", str(out_15),
        "--output-5min", str(tmp_path / "H_5.parquet"),
        "--output-60min", str(tmp_path / "H_60.parquet"),
        "--output-summary", str(tmp_path / "H_summary.json"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        assert entropy_mod.main() == 0
    finally:
        sys.argv = old_argv

    # Now event_matching using the synthetic config
    config_yaml = tmp_path / "cfg.yaml"
    config_yaml.write_text(
        "event_matching:\n"
        "  event_types_include: ['accidente', 'averia_mayor', 'objeto_calzada']\n"
        "  spatial_window_km: 1.0\n"
        "  temporal_window_pre_minutes: 30\n"
        "  temporal_window_post_minutes: 60\n"
        "  matches_per_event: 2\n"
    )

    events_mod = _load_script("event_matching")
    matched = tmp_path / "matched.parquet"
    summary = tmp_path / "event_summary.json"
    argv = [
        "event_matching.py",
        "--h-15min", str(out_15),
        "--events-db", str(synthetic_universe["events_db"]),
        "--porticos-csv", str(synthetic_universe["porticos_csv"]),
        "--output-matched", str(matched),
        "--output-summary", str(summary),
        "--config", str(config_yaml),
    ]
    sys.argv = argv
    try:
        rc = events_mod.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    assert matched.exists()
    payload = json.loads(summary.read_text())
    assert "event_types" in payload
    assert "n_total_matches" in payload
