#python -m pytest tests/test_cluster_batch_consistency.py -q 
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    # Ensure src/utils.py is importable during tests.
    sys.path.insert(0, str(SRC_DIR))

import clustering  # noqa: E402
from clustering import Clusterization, _clusterize_in_batches  # noqa: E402
from utils import (  # noqa: E402
    FLOW_TABLE_NAME,
    FlowColumns,
    get_flow_db_summary,
    load_flujos_range,
)

try:
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover
    duckdb = None

WINDOWS = [
    pd.Timedelta(days=14),
    pd.Timedelta(days=7),
    pd.Timedelta(days=3),
    pd.Timedelta(days=1),
    pd.Timedelta(hours=12),
    pd.Timedelta(hours=6),
    pd.Timedelta(hours=3),
    pd.Timedelta(hours=1),
]
MAX_ROWS = 100_000_000


def _pick_sample_range(
    summary,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if duckdb is None:
        pytest.skip("duckdb is not installed.")
    if summary.row_count == 0:
        pytest.skip("Flow DB is empty.")
    if summary.min_timestamp is None or summary.max_timestamp is None:
        pytest.skip("Flow DB has no timestamp range.")

    start = summary.min_timestamp
    conn = duckdb.connect(str(summary.db_path), read_only=True)
    try:
        fallback_end: pd.Timestamp | None = None
        for window in WINDOWS:
            end = min(start + window, summary.max_timestamp)
            end_exclusive = end + pd.Timedelta(nanoseconds=1)
            count = conn.execute(
                f"SELECT COUNT(*) FROM {FLOW_TABLE_NAME} "
                "WHERE FECHA >= ? AND FECHA < ?",
                [start, end_exclusive],
            ).fetchone()[0]
            if count == 0:
                continue
            fallback_end = end
            if count <= MAX_ROWS:
                return start, end
        return (start, fallback_end) if fallback_end is not None else None
    finally:
        conn.close()


def test_clusterization_batch_vs_full(tmp_path, monkeypatch):
    if duckdb is None:
        pytest.skip("duckdb is not installed.")
    try:
        summary = get_flow_db_summary()
    except ImportError as exc:
        pytest.skip(str(exc))

    selection = _pick_sample_range(summary)
    if selection is None:
        pytest.skip("No rows available for the sample range.")
    start, end = selection
    end_exclusive = end + pd.Timedelta(nanoseconds=1)

    flujos_df = load_flujos_range(start, end_exclusive)
    if flujos_df.empty:
        pytest.skip("Sample range returned no rows.")

    flow_cols = FlowColumns()
    full_df = Clusterization(
        flujos_df,
        flow_cols,
        ttc_max_map=None,
        monthly_weighting=False,
        overlap_col=None,
        include_counts=True,
        progress=None,
        outlier_action="none",
    )
    if full_df.empty:
        pytest.skip("Full clusterization returned no rows.")

    monkeypatch.setattr(clustering, "ROOT_DIR", tmp_path)
    batch_df, _paths = _clusterize_in_batches(
        flow_cols,
        ttc_max_map=None,
        batch_mode="week",
        monthly_weighting=False,
        date_start=start,
        date_end=end,
        outlier_action="none",
    )
    if batch_df.empty:
        pytest.skip("Batch clusterization returned no rows.")

    compare_cols = [
        "total_passes",
        "avg_speed_kmh",
        "avg_relative_speed",
        "avg_headway_s",
        "conflict_rate",
        "lane_prop_1",
        "lane_prop_2",
        "lane_prop_3",
        "lane_changes",
        "lane_change_rate",
    ]
    missing_full = set(compare_cols) - set(full_df.columns)
    missing_batch = set(compare_cols) - set(batch_df.columns)
    assert not missing_full, f"Missing columns in full results: {sorted(missing_full)}"
    assert not missing_batch, f"Missing columns in batch results: {sorted(missing_batch)}"

    full = full_df.set_index("plate").sort_index()
    batch = batch_df.set_index("plate").sort_index()
    assert set(full.index) == set(
        batch.index
    ), "Mismatch in plate IDs between batch and full results."

    full = full.loc[batch.index, compare_cols]
    batch = batch.loc[full.index, compare_cols]

    int_cols = ["total_passes"]
    float_cols = [col for col in compare_cols if col not in int_cols]

    for col in int_cols:
        pd.testing.assert_series_equal(
            full[col], batch[col], check_dtype=False, check_names=False
        )

    for col in float_cols:
        left = pd.to_numeric(full[col], errors="coerce").fillna(0).to_numpy()
        right = pd.to_numeric(batch[col], errors="coerce").fillna(0).to_numpy()
        np.testing.assert_allclose(
            left, right, rtol=1e-3, atol=5.0, equal_nan=True, err_msg=f"Column failed: {col}"
        )


def test_weekly_rollup_matches_monthly_batches(tmp_path, monkeypatch):
    if duckdb is None:
        pytest.skip("duckdb is not installed.")
    try:
        summary = get_flow_db_summary()
    except ImportError as exc:
        pytest.skip(str(exc))

    selection = _pick_sample_range(summary)
    if selection is None:
        pytest.skip("No rows available for the sample range.")
    start, end = selection

    flow_cols = FlowColumns()
    monkeypatch.setattr(clustering, "ROOT_DIR", tmp_path)
    month_df, _paths = _clusterize_in_batches(
        flow_cols,
        ttc_max_map=None,
        batch_mode="month",
        monthly_weighting=True,
        date_start=start,
        date_end=end,
        outlier_action="none",
    )
    if month_df.empty:
        pytest.skip("Monthly batch clusterization returned no rows.")

    week_df, _paths = _clusterize_in_batches(
        flow_cols,
        ttc_max_map=None,
        batch_mode="week",
        monthly_weighting=True,
        date_start=start,
        date_end=end,
        outlier_action="none",
    )
    if week_df.empty:
        pytest.skip("Weekly rollup clusterization returned no rows.")

    compare_cols = [
        "total_passes",
        "avg_speed_kmh",
        "avg_relative_speed",
        "avg_headway_s",
        "conflict_rate",
        "lane_prop_1",
        "lane_prop_2",
        "lane_prop_3",
        "lane_changes",
        "lane_change_rate",
    ]
    missing_month = set(compare_cols) - set(month_df.columns)
    missing_week = set(compare_cols) - set(week_df.columns)
    assert not missing_month, f"Missing columns in monthly results: {sorted(missing_month)}"
    assert not missing_week, f"Missing columns in weekly rollup results: {sorted(missing_week)}"

    month = month_df.set_index("plate").sort_index()
    week = week_df.set_index("plate").sort_index()
    assert set(month.index) == set(
        week.index
    ), "Mismatch in plate IDs between weekly rollup and monthly results."

    month = month.loc[week.index, compare_cols]
    week = week.loc[month.index, compare_cols]

    int_cols = ["total_passes"]
    float_cols = [col for col in compare_cols if col not in int_cols]

    for col in int_cols:
        pd.testing.assert_series_equal(
            month[col], week[col], check_dtype=False, check_names=False
        )

    for col in float_cols:
        left = pd.to_numeric(month[col], errors="coerce").fillna(0).to_numpy()
        right = pd.to_numeric(week[col], errors="coerce").fillna(0).to_numpy()
        np.testing.assert_allclose(
            left, right, rtol=1e-3, atol=5.0, equal_nan=True, err_msg=f"Column failed: {col}"
        )
