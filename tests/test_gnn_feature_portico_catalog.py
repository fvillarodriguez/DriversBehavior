import pandas as pd
import polars as pl
import pytest

from src.graph_builder_app import (
    _duckdb_portico_filter_clause,
    _feature_portico_sequence_from_source,
    _filter_flow_frame_to_porticos,
    _resolve_generation_porticos_filter,
)


def _porticos_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portico": ["1", "2", "3"],
            "autopista": ["C", "C", "C"],
            "calzada": ["Norte", "Norte", "Norte"],
            "eje": ["E1", "E1", "E1"],
            "orden": [1, 2, 3],
            "km": [10.0, 11.0, 12.0],
        }
    )


def test_feature_portico_sequence_discards_values_outside_porticos_csv(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "features.duckdb"
    features = pd.DataFrame(
        {
            "portico": [1, 2, 901, 901],
            "ts_min": [100, 100, 100, 105],
            "speed_mean": [80.0, 82.0, 0.0, 0.0],
        }
    )

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE features AS SELECT * FROM features")
    finally:
        con.close()

    info = _feature_portico_sequence_from_source(
        {
            "duckdb_path": str(db_path),
            "duckdb_table": "features",
            "params": {
                "duckdb_path": str(db_path),
                "duckdb_table": "features",
            },
        },
        df_port=_porticos_catalog(),
    )

    assert info["porticos"] == ["1", "2"]
    assert info["raw_portico_count"] == 3
    assert info["discarded_portico_count"] == 1
    assert info["discarded_porticos"] == ["901"]


def test_generation_porticos_filter_intersects_requested_with_catalog():
    selected = _resolve_generation_porticos_filter(
        _porticos_catalog(), ["2.0", "901"]
    )

    assert selected == ["2"]


def test_duckdb_portico_filter_clause_normalizes_numeric_codes():
    duckdb = pytest.importorskip("duckdb")
    clause, params = _duckdb_portico_filter_clause(["PORTICO"], ["1", "2.0"])

    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE flows(PORTICO VARCHAR)")
        con.executemany(
            "INSERT INTO flows VALUES (?)",
            [("1",), ("2.0",), ("901",)],
        )
        rows = con.execute(
            f"SELECT PORTICO FROM flows WHERE {clause} ORDER BY PORTICO",
            params,
        ).fetchall()
    finally:
        con.close()

    assert rows == [("1",), ("2.0",)]


def test_filter_flow_frame_to_porticos_filters_polars_input():
    flows = pl.DataFrame(
        {
            "PORTICO": ["1", "2.0", "901"],
            "FECHA": ["2024-01-01"] * 3,
        }
    )

    filtered = _filter_flow_frame_to_porticos(flows, ["1", "2"])

    assert filtered["PORTICO"].to_list() == ["1", "2.0"]
