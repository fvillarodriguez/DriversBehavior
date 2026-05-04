from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("optuna")

from src import cluster_accident_app as app
from src.utils import compute_flow_features


def _register_duckdb_tables(con, flows: pd.DataFrame, assignments: pd.DataFrame) -> None:
    con.register("flows_df", flows)
    con.execute("CREATE TABLE flows AS SELECT * FROM flows_df")
    con.unregister("flows_df")
    con.register("assignments_df", assignments)
    con.execute(
        f"CREATE TABLE {app.DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME} AS "
        "SELECT * FROM assignments_df"
    )
    con.unregister("assignments_df")


def _sort_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "interval_start" in result.columns:
        result["interval_start"] = pd.to_datetime(result["interval_start"])
    return result.sort_values(["portico", "interval_start"]).reset_index(drop=True)


def test_dynamic_gmm_membership_ui_message_describes_soft_membership():
    message = app._dynamic_gmm_membership_ui_message(
        ["cluster_prob_0", "cluster_prob_1"]
    )

    assert "2 columnas cluster_prob_*" in message
    assert "membresia soft" in message
    assert "window_end <= FECHA" in message
    assert "cluster_share_*" in message
    assert "cluster_flow_*" in message
    assert "cluster_speed_*" in message
    assert "cluster_density_*" in message
    assert "cluster_entropy" in message
    assert "cluster_label" in message


def test_dynamic_gmm_membership_ui_message_describes_hard_label_fallback():
    message = app._dynamic_gmm_membership_ui_message([])

    assert "No se detectaron columnas cluster_prob_*" in message
    assert "etiquetas hard cluster_label" in message


def test_dynamic_gmm_cluster_features_use_causal_soft_membership():
    flows = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(
                [
                    "2024-01-01 12:00:00",
                    "2024-01-03 10:00:00",
                    "2024-01-03 10:00:00",
                    "2024-01-05 10:00:00",
                ]
            ),
            "PORTICO": ["P1", "P1", "P2", "P1"],
            "MATRICULA": ["A1", "A1", "B1", "A1"],
            "VELOCIDAD": [70.0, 80.0, 90.0, 100.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "plate": ["A1", "A1", "B1"],
            "window_end": pd.to_datetime(
                ["2024-01-02", "2024-01-04", "2024-01-10"]
            ),
            "cluster_label": [0, 1, 1],
            "cluster_prob_0": [0.8, 0.1, 0.0],
            "cluster_prob_1": [0.2, 0.9, 1.0],
        }
    )

    result = app._compute_dynamic_gmm_cluster_features(
        flows,
        assignments,
        include_counts=True,
        include_speed=True,
        lanes=1,
    )

    assert result.attrs["dynamic_gmm_total_flow_rows"] == 4
    assert result.attrs["dynamic_gmm_matched_flow_rows"] == 2
    assert set(result["portico"]) == {"P1"}
    assert pd.Timestamp("2024-01-01 12:00:00") not in set(result["interval_start"])
    first = result.loc[
        result["interval_start"].eq(pd.Timestamp("2024-01-03 10:00:00"))
    ].iloc[0]
    second = result.loc[
        result["interval_start"].eq(pd.Timestamp("2024-01-05 10:00:00"))
    ].iloc[0]
    assert first["cluster_share_0"] == pytest.approx(0.8)
    assert first["cluster_share_1"] == pytest.approx(0.2)
    assert first["cluster_flow_0"] == pytest.approx(9.6)
    assert first["cluster_flow_1"] == pytest.approx(2.4)
    assert second["cluster_share_0"] == pytest.approx(0.1)
    assert second["cluster_share_1"] == pytest.approx(0.9)
    assert second["cluster_speed_0"] == pytest.approx(100.0)
    assert second["cluster_speed_1"] == pytest.approx(100.0)


def test_dynamic_gmm_cluster_features_fall_back_to_hard_labels():
    flows = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(
                ["2024-01-03 10:00:00", "2024-01-05 10:00:00"]
            ),
            "PORTICO": ["P1", "P1"],
            "MATRICULA": ["A1", "A1"],
            "VELOCIDAD": [80.0, 100.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "plate": ["A1", "A1"],
            "window_end": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "cluster_label": [0, 1],
        }
    )

    result = app._compute_dynamic_gmm_cluster_features(
        flows,
        assignments,
        include_counts=True,
        include_speed=True,
        lanes=1,
    )

    first = result.loc[
        result["interval_start"].eq(pd.Timestamp("2024-01-03 10:00:00"))
    ].iloc[0]
    second = result.loc[
        result["interval_start"].eq(pd.Timestamp("2024-01-05 10:00:00"))
    ].iloc[0]
    assert first["cluster_share_0"] == pytest.approx(1.0)
    assert first["cluster_flow_0"] == pytest.approx(12.0)
    assert first["cluster_speed_0"] == pytest.approx(80.0)
    assert second["cluster_share_1"] == pytest.approx(1.0)
    assert second["cluster_flow_1"] == pytest.approx(12.0)
    assert second["cluster_speed_1"] == pytest.approx(100.0)


def test_dynamic_gmm_external_duckdb_is_resolved_and_loaded(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "dynamic_gmm_external.duckdb"
    assignments = pd.DataFrame(
        {
            "plate": ["A1"],
            "window_end": [pd.Timestamp("2024-01-02")],
            "cluster_label": [0],
            "cluster_prob_0": [0.7],
            "cluster_prob_1": [0.3],
        }
    )
    metadata = pd.DataFrame(
        {
            "key": ["assignment_scope"],
            "value_json": ['"prevalent"'],
        }
    )
    con = duckdb.connect(str(db_path))
    try:
        con.register("assignments", assignments)
        con.execute(
            f"CREATE TABLE {app.DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME} AS "
            "SELECT * FROM assignments"
        )
        con.register("metadata", metadata)
        con.execute(
            f"CREATE TABLE {app.DYNAMIC_GMM_METADATA_TABLE_NAME} AS "
            "SELECT * FROM metadata"
        )
    finally:
        con.close()

    candidates, error = app._list_dynamic_gmm_db_paths(tmp_path)
    info = app._inspect_dynamic_gmm_duckdb(db_path)
    loaded = app._load_dynamic_gmm_assignments(db_path)

    assert error is None
    assert candidates == [db_path.resolve()]
    assert info["rows"] == 1
    assert info["probability_cols"] == ["cluster_prob_0", "cluster_prob_1"]
    assert info["metadata"]["assignment_scope"] == "prevalent"
    assert loaded.loc[0, "plate"] == "A1"
    assert loaded.loc[0, "cluster_prob_0"] == pytest.approx(0.7)


def test_dynamic_gmm_duckdb_batch_matches_pandas_soft_membership():
    duckdb = pytest.importorskip("duckdb")
    flows = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(
                [
                    "2024-01-01 12:00:00",
                    "2024-01-03 10:00:00",
                    "2024-01-03 10:00:00",
                    "2024-01-05 10:00:00",
                ]
            ),
            "PORTICO": ["P1", "P1", "P2", "P1"],
            "MATRICULA": ["A1", "A1", "B1", "A1"],
            "VELOCIDAD": [70.0, 80.0, 90.0, 100.0],
            "CATEGORIA": [1, 1, 2, 1],
        }
    )
    assignments = pd.DataFrame(
        {
            "plate": ["A1", "A1", "B1"],
            "window_end": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-10"]),
            "cluster_label": [0, 1, 1],
            "cluster_prob_0": [0.8, 0.1, 0.0],
            "cluster_prob_1": [0.2, 0.9, 1.0],
        }
    )
    con = duckdb.connect(":memory:")
    _register_duckdb_tables(con, flows, assignments)

    expected = app._compute_dynamic_gmm_cluster_features(
        flows,
        assignments,
        include_counts=True,
        include_speed=True,
        include_density=True,
        include_entropy=True,
        lanes=1,
    )
    result = app._compute_dynamic_gmm_cluster_features_duckdb_batch(
        con,
        "flows",
        app.DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME,
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-06"),
        include_counts=True,
        include_speed=True,
        include_density=True,
        include_entropy=True,
        lanes=1,
        probability_cols=["cluster_prob_0", "cluster_prob_1"],
    )

    pd.testing.assert_frame_equal(
        _sort_feature_frame(result),
        _sort_feature_frame(expected),
        check_dtype=False,
        rtol=1e-9,
        atol=1e-9,
    )
    assert result.attrs["dynamic_gmm_total_flow_rows"] == 4
    assert result.attrs["dynamic_gmm_matched_flow_rows"] == 2


def test_dynamic_gmm_duckdb_batch_matches_pandas_hard_labels():
    duckdb = pytest.importorskip("duckdb")
    flows = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2024-01-03 10:00:00", "2024-01-05 10:00:00"]),
            "PORTICO": ["P1", "P1"],
            "MATRICULA": ["A1", "A1"],
            "VELOCIDAD": [80.0, 100.0],
            "CATEGORIA": [1, 1],
        }
    )
    assignments = pd.DataFrame(
        {
            "plate": ["A1", "A1"],
            "window_end": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "cluster_label": [0, 1],
        }
    )
    con = duckdb.connect(":memory:")
    _register_duckdb_tables(con, flows, assignments)

    expected = app._compute_dynamic_gmm_cluster_features(
        flows,
        assignments,
        include_counts=True,
        include_speed=True,
        lanes=1,
    )
    result = app._compute_dynamic_gmm_cluster_features_duckdb_batch(
        con,
        "flows",
        app.DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME,
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-06"),
        include_counts=True,
        include_speed=True,
        lanes=1,
        probability_cols=[],
    )

    pd.testing.assert_frame_equal(
        _sort_feature_frame(result),
        _sort_feature_frame(expected),
        check_dtype=False,
    )


def test_dynamic_gmm_duckdb_batch_keeps_expected_columns_without_matches():
    duckdb = pytest.importorskip("duckdb")
    flows = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(["2024-01-01 10:00:00"]),
            "PORTICO": ["P1"],
            "MATRICULA": ["A1"],
            "VELOCIDAD": [80.0],
            "CATEGORIA": [1],
        }
    )
    assignments = pd.DataFrame(
        {
            "plate": ["A1"],
            "window_end": pd.to_datetime(["2024-01-02"]),
            "cluster_label": [0],
            "cluster_prob_0": [0.7],
            "cluster_prob_1": [0.3],
        }
    )
    con = duckdb.connect(":memory:")
    _register_duckdb_tables(con, flows, assignments)

    result = app._compute_dynamic_gmm_cluster_features_duckdb_batch(
        con,
        "flows",
        app.DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME,
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02"),
        include_counts=True,
        include_speed=True,
        lanes=1,
        probability_cols=["cluster_prob_0", "cluster_prob_1"],
    )

    assert result.empty
    assert {
        "cluster_share_0",
        "cluster_share_1",
        "cluster_flow_0",
        "cluster_flow_1",
        "cluster_speed_0",
        "cluster_speed_1",
    }.issubset(result.columns)
    assert result.attrs["dynamic_gmm_total_flow_rows"] == 1
    assert result.attrs["dynamic_gmm_matched_flow_rows"] == 0


def test_flow_features_duckdb_batch_matches_polars_reference():
    duckdb = pytest.importorskip("duckdb")
    flows = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(
                [
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:01:00",
                    "2024-01-01 10:05:00",
                    "2024-01-01 10:06:00",
                ]
            ),
            "PORTICO": ["P1", "P1", "P1", "P2"],
            "MATRICULA": ["A1", "A2", "A1", "B1"],
            "VELOCIDAD": [80.0, 90.0, 100.0, 70.0],
            "CATEGORIA": [1, 2, 1, 4],
        }
    )
    con = duckdb.connect(":memory:")
    con.register("flows_df", flows)
    con.execute("CREATE TABLE flows AS SELECT * FROM flows_df")
    con.unregister("flows_df")

    expected = compute_flow_features(
        flows,
        interval_minutes=5,
        lanes=1,
        metrics=["flow", "speed", "speed_std", "density", "delta_speed", "delta_density"],
        categories=["Light", "Heavy", "Motorcycles"],
    )
    result = app._compute_flow_features_duckdb_batch(
        con,
        "flows",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-02"),
        interval_minutes=5,
        lanes=1,
        metrics=["flow", "speed", "speed_std", "density", "delta_speed", "delta_density"],
        categories=["Light", "Heavy", "Motorcycles"],
    )

    pd.testing.assert_frame_equal(
        _sort_feature_frame(result),
        _sort_feature_frame(expected),
        check_dtype=False,
        rtol=1e-9,
        atol=1e-9,
    )
    assert result.attrs["input_rows"] == 4


def test_feature_table_profile_and_preview_return_expected_rows(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "features.duckdb"
    features = pd.DataFrame(
        {
            "portico_last": ["P1", "P1", "P2"],
            "portico_next": ["P2", "P2", "P3"],
            "interval_start": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "last_flow_light": [1.0, 2.0, 3.0],
        }
    )
    con = duckdb.connect(str(db_path))
    con.register("features_df", features)
    con.execute("CREATE TABLE flow_features AS SELECT * FROM features_df")
    con.close()

    profile = app._profile_feature_table_duckdb(db_path)
    preview = app._load_feature_preview_duckdb(db_path, limit=2)

    assert profile["rows"] == 3
    assert profile["cols"] == 4
    assert len(preview) == 2


def test_feature_memory_policy_loads_full_result_when_requested(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "features_policy.duckdb"
    features = pd.DataFrame(
        {
            "portico_last": ["P1", "P1", "P2"],
            "portico_next": ["P2", "P2", "P3"],
            "interval_start": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "last_flow_light": [1.0, 2.0, 3.0],
        }
    )
    con = duckdb.connect(str(db_path))
    con.register("features_df", features)
    con.execute("CREATE TABLE flow_features AS SELECT * FROM features_df")
    con.close()

    result = app._load_feature_table_with_memory_policy(
        db_path,
        load_full=True,
    )

    assert result["loaded_full"] is True
    assert len(result["full_df"]) == 3
    assert result["row_count"] == 3
    assert len(result["preview_df"]) == 3
