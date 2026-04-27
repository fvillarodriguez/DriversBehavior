from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clustering  # noqa: E402
import clustering_tabs_app as clustering_app  # noqa: E402
import experiments_live_app as live_app  # noqa: E402


class IdentityScaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)


class ThresholdGmm:
    n_components = 2

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        first = x[:, 0]
        probs = np.where(first[:, None] < 5.0, [[0.8, 0.2]], [[0.25, 0.75]])
        return probs.astype(float)


def _patch_dynamic_gmm_dependencies(monkeypatch):
    monkeypatch.setattr(
        clustering,
        "fit_gmm_cluster_model",
        lambda *args, **kwargs: (ThresholdGmm(), IdentityScaler()),
    )
    monkeypatch.setattr(
        clustering,
        "load_flujos_range",
        lambda start, end: pd.DataFrame({"dummy": [1]}),
    )

    def fake_clusterization(*args, **kwargs):
        return pd.DataFrame(
            {
                "plate": ["A", "B"],
                "feature": [1.0, 6.0],
                "total_passes": [6, 3],
            }
        )

    monkeypatch.setattr(clustering, "Clusterization", fake_clusterization)


def test_build_dynamic_gmm_windows_uses_full_daily_sliding_windows():
    windows = clustering.build_dynamic_gmm_windows(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-05"),
        window_days=3,
    )

    assert [label for *_rest, label in windows] == [
        "2024-01-01_to_2024-01-03",
        "2024-01-02_to_2024-01-04",
        "2024-01-03_to_2024-01-05",
    ]
    assert windows[0][0] == pd.Timestamp("2024-01-01")
    assert windows[0][1] == pd.Timestamp("2024-01-04")


def test_resolve_dynamic_gmm_output_dir_accepts_custom_directory(tmp_path):
    target = tmp_path / "dynamic_outputs"

    resolved, error = clustering_app._resolve_dynamic_gmm_output_dir(target)

    assert error is None
    assert resolved == target.resolve()


def test_resolve_dynamic_gmm_output_dir_rejects_files(tmp_path):
    target = tmp_path / "not_a_directory.txt"
    target.write_text("content", encoding="utf-8")

    resolved, error = clustering_app._resolve_dynamic_gmm_output_dir(target)

    assert resolved == target.resolve()
    assert error is not None
    assert "no es una carpeta" in error


def test_build_prevalent_plate_selection_ranks_and_selects_exactly():
    features = pd.DataFrame(
        {
            "plate": ["A", "B", "C", "D", "E"],
            "feature": [1, 2, 3, 4, 5],
            "n_months_active": [3, 3, 2, 1, 1],
            "n_years_active": [1, 2, 2, 2, 1],
            "total_passes": [100, 20, 200, 10, 500],
        }
    )

    selection = clustering.build_prevalent_plate_selection(
        features,
        fraction_pct=40,
        feature_cols=["feature"],
    )

    assert selection["selected_count"] == 2
    assert selection["total_valid_plates"] == 5
    assert selection["plates"] == ["B", "A"]
    assert selection["ranked"]["plate"].tolist() == ["B", "A", "C", "D", "E"]


def test_predict_gmm_cluster_membership_marks_low_confidence_and_support():
    class FixedGmm:
        n_components = 2

        def predict_proba(self, x):
            return np.array(
                [
                    [0.8, 0.2],
                    [0.55, 0.45],
                    [0.1, 0.9],
                ],
                dtype=float,
            )

    df = pd.DataFrame(
        {
            "plate": ["A", "B", "C"],
            "total_passes": [10, 10, 2],
            "feature": [1.0, 2.0, 3.0],
        }
    )

    result = clustering.predict_gmm_cluster_membership(
        df,
        ["feature"],
        FixedGmm(),
        IdentityScaler(),
        confidence_threshold_proba=0.7,
        min_window_passes=5,
    )

    assert result["cluster_label"].tolist() == [0, -1, -1]
    assert result["raw_cluster_label"].tolist() == [0, 0, 1]
    assert result["assignment_status"].tolist() == [
        "assigned",
        "low_confidence",
        "low_support",
    ]
    assert {"cluster_prob_0", "cluster_prob_1", "soft_entropy"}.issubset(
        result.columns
    )


def test_run_dynamic_gmm_clustering_uses_fixed_model_across_windows(monkeypatch):
    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C"],
            "feature": [1.0, 6.0, 2.0],
            "total_passes": [20, 20, 20],
            "n_days_active": [3, 3, 3],
            "n_months_active": [1, 1, 1],
        }
    )

    _patch_dynamic_gmm_dependencies(monkeypatch)
    events = []

    result = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=False,
        window_callback=events.append,
    )

    assignments = result["assignments"]
    assert len(result["windows"]) == 2
    assert len(assignments) == 4
    assert assignments["window_index"].tolist() == [1, 1, 2, 2]
    assert assignments.loc[assignments["plate"] == "A", "cluster_label"].eq(0).all()
    assert assignments.loc[assignments["plate"] == "B", "cluster_label"].eq(-1).all()
    assert assignments.loc[assignments["plate"] == "B", "is_low_support"].all()
    assert result["window_summary"]["window_index"].tolist() == [1, 2]
    assert not result["window_summary"].empty
    assert not result["driver_summary"].empty
    assert [event["window_index"] for event in events] == [1, 2]
    assert [event["status"] for event in events] == ["completed", "completed"]
    assert events[0]["assignments"]["window_index"].tolist() == [1, 1]


def test_run_dynamic_gmm_clustering_prevalent_scope_filters_window_plates(monkeypatch):
    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C", "D", "E"],
            "feature": [1.0, 6.0, 2.0, 7.0, 3.0],
            "total_passes": [20, 20, 20, 20, 20],
            "n_days_active": [3, 3, 3, 3, 3],
            "n_months_active": [5, 1, 4, 2, 3],
            "n_years_active": [2, 1, 2, 1, 1],
        }
    )
    monkeypatch.setattr(
        clustering,
        "fit_gmm_cluster_model",
        lambda *args, **kwargs: (ThresholdGmm(), IdentityScaler()),
    )
    monkeypatch.setattr(
        clustering,
        "load_flujos_range",
        lambda start, end: pd.DataFrame(
            {
                "FECHA": [pd.Timestamp(start)] * 5,
                "MATRICULA": ["A", "B", "C", "D", "E"],
                "VELOCIDAD": [80, 81, 82, 83, 84],
                "PORTICO": ["P1"] * 5,
                "CARRIL": [1] * 5,
            }
        ),
    )
    seen_window_plates = []

    def fake_clusterization(flujos_df, flow_cols, *args, **kwargs):
        plates = sorted(flujos_df[flow_cols.plate_id].astype(str).unique().tolist())
        seen_window_plates.append(plates)
        feature_by_plate = {"A": 1.0, "B": 6.0, "C": 2.0, "D": 7.0, "E": 3.0}
        return pd.DataFrame(
            {
                "plate": plates,
                "feature": [feature_by_plate[plate] for plate in plates],
                "total_passes": [6] * len(plates),
            }
        )

    monkeypatch.setattr(clustering, "Clusterization", fake_clusterization)

    result = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=False,
        assignment_scope="prevalent",
        prevalent_fraction_pct=40,
    )

    assignments = result["assignments"]
    assert seen_window_plates == [["A", "C"], ["A", "C"]]
    assert assignments["plate"].tolist() == ["A", "C", "A", "C"]
    assert result["metadata"]["assignment_scope"] == "prevalent"
    assert result["metadata"]["prevalent_fraction_pct"] == 40.0
    assert result["metadata"]["prevalent_plate_count"] == 2
    assert result["metadata"]["prevalent_valid_plate_count"] == 5
    assert result["metadata"]["prevalent_source"] == "historical_features"


def test_run_dynamic_gmm_clustering_can_omit_membership_probabilities(monkeypatch):
    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C"],
            "feature": [1.0, 6.0, 2.0],
            "total_passes": [20, 20, 20],
            "n_days_active": [3, 3, 3],
            "n_months_active": [1, 1, 1],
        }
    )
    _patch_dynamic_gmm_dependencies(monkeypatch)

    result = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=False,
        include_membership_probabilities=False,
    )

    assignments = result["assignments"]
    assert not any(col.startswith("cluster_prob_") for col in assignments.columns)
    assert "soft_entropy" in assignments.columns
    assert result["metadata"]["include_membership_probabilities"] is False


def test_run_dynamic_gmm_clustering_parallel_jobs_matches_sequential(monkeypatch):
    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C"],
            "feature": [1.0, 6.0, 2.0],
            "total_passes": [20, 20, 20],
            "n_days_active": [3, 3, 3],
            "n_months_active": [1, 1, 1],
        }
    )
    _patch_dynamic_gmm_dependencies(monkeypatch)

    sequential = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=False,
        parallel_jobs=1,
    )

    workers_seen = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers):
            workers_seen.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, payload):
            return FakeFuture(fn(payload))

    monkeypatch.setattr(clustering, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(clustering, "as_completed", lambda futures: list(futures))

    parallel = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=False,
        parallel_jobs=2,
    )

    cols = ["window_index", "plate", "cluster_label", "assignment_status"]
    pd.testing.assert_frame_equal(
        sequential["assignments"][cols].reset_index(drop=True),
        parallel["assignments"][cols].reset_index(drop=True),
    )
    assert workers_seen == [2]


def test_dynamic_gmm_ui_helpers_page_plates_in_order():
    assignments = pd.DataFrame(
        {
            "plate": [f"P{i:02d}" for i in range(12)] + ["P01"],
            "cluster_label": [0] * 13,
            "window_label": ["w1"] * 13,
            "window_start": [pd.Timestamp("2024-01-01")] * 13,
            "window_end": [pd.Timestamp("2024-01-08")] * 13,
        }
    )

    assert clustering_app._dynamic_gmm_plate_order(assignments) == [
        f"P{i:02d}" for i in range(12)
    ]
    selected, start, end, total, page = clustering_app._dynamic_gmm_selected_plate_page(
        assignments,
        page_index=0,
    )
    assert selected == [f"P{i:02d}" for i in range(10)]
    assert (start, end, total, page) == (0, 10, 12, 0)

    selected, start, end, total, page = clustering_app._dynamic_gmm_selected_plate_page(
        assignments,
        page_index=1,
    )
    assert selected == ["P10", "P11"]
    assert (start, end, total, page) == (10, 12, 12, 1)
    assert clustering_app._next_dynamic_gmm_plate_page_index(1, 12) == 0


def test_dynamic_gmm_ui_helpers_reconstruct_window_index_for_old_results():
    assignments = pd.DataFrame(
        {
            "plate": ["A", "A", "B"],
            "cluster_label": [1, 0, -1],
            "window_label": [
                "2024-01-08_to_2024-01-14",
                "2024-01-01_to_2024-01-07",
                "2024-01-08_to_2024-01-14",
            ],
            "window_start": [
                pd.Timestamp("2024-01-08"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-08"),
            ],
            "window_end": [
                pd.Timestamp("2024-01-15"),
                pd.Timestamp("2024-01-08"),
                pd.Timestamp("2024-01-15"),
            ],
            "confidence_score": [0.8, 0.7, 0.6],
            "assignment_status": ["assigned", "assigned", "low_confidence"],
        }
    )

    normalized = clustering_app._ensure_dynamic_window_index(assignments)
    by_label = normalized.groupby("window_label")["window_index"].first().to_dict()
    assert by_label["2024-01-01_to_2024-01-07"] == 1
    assert by_label["2024-01-08_to_2024-01-14"] == 2

    plot_df = clustering_app._dynamic_gmm_assignment_plot_frame(normalized, ["A"])
    assert plot_df["week_label"].tolist() == ["Semana 1", "Semana 2"]


def test_dynamic_gmm_probability_helper_is_empty_without_probability_columns():
    assignments = pd.DataFrame(
        {
            "plate": ["A"],
            "window_index": [1],
            "cluster_label": [0],
            "window_label": ["w1"],
        }
    )

    assert clustering_app._dynamic_gmm_probability_columns(assignments) == []
    assert clustering_app._dynamic_gmm_probability_plot_frame(
        assignments,
        ["A"],
        "cluster_prob_0",
    ).empty


def test_save_and_load_dynamic_gmm_results_duckdb(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("joblib")

    assignments = pd.DataFrame(
        {
            "window_index": [1],
            "window_label": ["2024-01-01_to_2024-01-02"],
            "window_start": [pd.Timestamp("2024-01-01")],
            "window_end": [pd.Timestamp("2024-01-03")],
            "plate": ["A"],
            "cluster_label": [0],
            "confidence_score": [0.8],
            "assignment_status": ["assigned"],
            "is_low_support": [False],
            "cluster_prob_0": [0.8],
            "cluster_prob_1": [0.2],
            "soft_entropy": [0.72],
        }
    )
    window_summary = clustering.build_dynamic_gmm_window_summary(assignments)
    metadata = {"method": "gmm_dynamic", "k": 2, "feature_cols": ["feature"]}

    model_path, db_path = clustering.save_dynamic_gmm_results(
        assignments,
        window_summary,
        model={"fake": "model"},
        scaler={"fake": "scaler"},
        metadata=metadata,
        output_dir=tmp_path,
        stem="dynamic_gmm_test",
    )
    loaded_assignments, loaded_summary, loaded_metadata = (
        clustering.load_dynamic_gmm_results_duckdb(db_path)
    )

    assert model_path.exists()
    assert db_path.exists()
    assert loaded_assignments["plate"].tolist() == ["A"]
    assert loaded_assignments["window_index"].tolist() == [1]
    assert int(loaded_summary.loc[0, "window_index"]) == 1
    assert int(loaded_summary.loc[0, "assigned_rows"]) == 1
    assert loaded_metadata["method"] == "gmm_dynamic"
    assert loaded_metadata["k"] == 2


def test_run_dynamic_gmm_clustering_persists_incrementally(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("joblib")

    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C"],
            "feature": [1.0, 6.0, 2.0],
            "total_passes": [20, 20, 20],
            "n_days_active": [3, 3, 3],
            "n_months_active": [1, 1, 1],
        }
    )
    _patch_dynamic_gmm_dependencies(monkeypatch)

    result = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=True,
        checkpoint_enabled=True,
        output_dir=tmp_path,
        load_final_result=False,
        run_id="dynamic_gmm_incremental_test",
    )

    assert result["assignments"].empty
    assert result["duckdb_path"].exists()
    loaded_assignments, loaded_summary, loaded_metadata = (
        clustering.load_dynamic_gmm_results_duckdb(result["duckdb_path"])
    )
    assert len(loaded_assignments) == 4
    assert loaded_assignments["window_index"].tolist() == [1, 1, 2, 2]
    assert loaded_summary["window_index"].tolist() == [1, 2]
    assert loaded_metadata["status"] == "completed"
    assert loaded_metadata["n_assignments"] == 4

    conn = duckdb.connect(str(result["duckdb_path"]), read_only=True)
    try:
        checkpoint = conn.execute(
            f"SELECT status, attempts FROM {clustering.DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME} "
            "ORDER BY window_index"
        ).fetchall()
        run_status = conn.execute(
            f"SELECT status, assignment_rows FROM {clustering.DYNAMIC_GMM_RUN_STATUS_TABLE_NAME}"
        ).fetchone()
    finally:
        conn.close()
    assert checkpoint == [("completed", 1), ("completed", 1)]
    assert run_status == ("completed", 4)


def test_dynamic_gmm_append_df_aligns_variable_window_summary_schema(tmp_path):
    duckdb = pytest.importorskip("duckdb")

    db_path = tmp_path / "dynamic_gmm_schema_test.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        first = pd.DataFrame(
            {
                "run_id": ["run"],
                "window_index": [1],
                "window_label": ["w1"],
                "rows": [10],
                "cluster_count_-1": [2],
                "cluster_share_-1": [0.2],
                "cluster_count_0": [8],
                "cluster_share_0": [0.8],
            }
        )
        second = pd.DataFrame(
            {
                "run_id": ["run"],
                "window_index": [2],
                "window_label": ["w2"],
                "rows": [5],
                "cluster_count_1": [5],
                "cluster_share_1": [1.0],
            }
        )
        clustering._dynamic_gmm_append_df(
            conn,
            clustering.DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME,
            first,
        )
        clustering._dynamic_gmm_append_df(
            conn,
            clustering.DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME,
            second,
        )
        loaded = conn.execute(
            f"SELECT * FROM {clustering.DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME} "
            "ORDER BY window_index"
        ).df()
    finally:
        conn.close()

    assert loaded["window_index"].tolist() == [1, 2]
    assert loaded.loc[1, "cluster_count_-1"] == 0
    assert loaded.loc[1, "cluster_share_-1"] == 0.0
    assert loaded.loc[0, "cluster_count_1"] == 0
    assert loaded.loc[0, "cluster_share_1"] == 0.0


def test_run_dynamic_gmm_clustering_resume_skips_completed_windows(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("joblib")

    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C"],
            "feature": [1.0, 6.0, 2.0],
            "total_passes": [20, 20, 20],
            "n_days_active": [3, 3, 3],
            "n_months_active": [1, 1, 1],
        }
    )
    _patch_dynamic_gmm_dependencies(monkeypatch)

    first = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=True,
        checkpoint_enabled=True,
        output_dir=tmp_path,
        load_final_result=False,
        run_id="dynamic_gmm_resume_test",
    )
    db_path = first["duckdb_path"]
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            f"UPDATE {clustering.DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME} "
            "SET status = 'failed', error = 'forced retry' "
            "WHERE window_index = 2"
        )
        conn.execute(
            f"DELETE FROM {clustering.DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME} "
            "WHERE window_index = 2"
        )
        conn.execute(
            f"DELETE FROM {clustering.DYNAMIC_GMM_WINDOW_SUMMARY_TABLE_NAME} "
            "WHERE window_index = 2"
        )
    finally:
        conn.close()

    resumed = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=True,
        checkpoint_enabled=True,
        incremental_db_path=db_path,
        resume_existing=True,
        load_final_result=False,
    )

    assert resumed["duckdb_path"] == db_path
    loaded_assignments, _summary, _metadata = clustering.load_dynamic_gmm_results_duckdb(
        db_path
    )
    assert len(loaded_assignments) == 4
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        attempts = conn.execute(
            f"SELECT window_index, status, attempts "
            f"FROM {clustering.DYNAMIC_GMM_WINDOW_CHECKPOINT_TABLE_NAME} "
            "ORDER BY window_index"
        ).fetchall()
    finally:
        conn.close()
    assert attempts == [(1, "completed", 1), (2, "completed", 2)]


def test_run_dynamic_gmm_clustering_rejects_incompatible_resume(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    pytest.importorskip("joblib")

    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C"],
            "feature": [1.0, 6.0, 2.0],
            "total_passes": [20, 20, 20],
            "n_days_active": [3, 3, 3],
            "n_months_active": [1, 1, 1],
        }
    )
    _patch_dynamic_gmm_dependencies(monkeypatch)

    first = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=True,
        checkpoint_enabled=True,
        output_dir=tmp_path,
        load_final_result=False,
        run_id="dynamic_gmm_incompatible_test",
    )

    with pytest.raises(ValueError) as exc_info:
        clustering.run_dynamic_gmm_clustering(
            base_features_df=base_features,
            feature_cols=["feature"],
            flow_cols=clustering.FlowColumns(),
            ttc_max_map=None,
            k=2,
            confidence_threshold_proba=0.6,
            window_days=2,
            date_start=pd.Timestamp("2024-01-01"),
            date_end=pd.Timestamp("2024-01-03"),
            min_window_passes=5,
            train_params={
                "min_total_passes": 1,
                "min_days_active": 1,
                "min_months_active": 1,
            },
            persist=True,
            checkpoint_enabled=True,
            incremental_db_path=first["duckdb_path"],
            resume_existing=True,
            load_final_result=False,
        )
    message = str(exc_info.value)
    assert "parametros no coinciden" in message
    assert "Variables diferentes" in message
    assert "confidence_threshold_proba" in message
    assert "checkpoint=0.7" in message
    assert "actual=0.6" in message


def test_run_dynamic_gmm_clustering_rejects_prevalent_fraction_resume_mismatch(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("duckdb")
    pytest.importorskip("joblib")

    base_features = pd.DataFrame(
        {
            "plate": ["A", "B", "C", "D", "E"],
            "feature": [1.0, 6.0, 2.0, 7.0, 3.0],
            "total_passes": [20, 20, 20, 20, 20],
            "n_days_active": [3, 3, 3, 3, 3],
            "n_months_active": [5, 1, 4, 2, 3],
            "n_years_active": [2, 1, 2, 1, 1],
        }
    )
    _patch_dynamic_gmm_dependencies(monkeypatch)

    first = clustering.run_dynamic_gmm_clustering(
        base_features_df=base_features,
        feature_cols=["feature"],
        flow_cols=clustering.FlowColumns(),
        ttc_max_map=None,
        k=2,
        confidence_threshold_proba=0.7,
        window_days=2,
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        min_window_passes=5,
        train_params={
            "min_total_passes": 1,
            "min_days_active": 1,
            "min_months_active": 1,
        },
        persist=True,
        checkpoint_enabled=True,
        output_dir=tmp_path,
        load_final_result=False,
        run_id="dynamic_gmm_prevalent_incompatible_test",
        assignment_scope="prevalent",
        prevalent_fraction_pct=40,
    )

    with pytest.raises(ValueError) as exc_info:
        clustering.run_dynamic_gmm_clustering(
            base_features_df=base_features,
            feature_cols=["feature"],
            flow_cols=clustering.FlowColumns(),
            ttc_max_map=None,
            k=2,
            confidence_threshold_proba=0.7,
            window_days=2,
            date_start=pd.Timestamp("2024-01-01"),
            date_end=pd.Timestamp("2024-01-03"),
            min_window_passes=5,
            train_params={
                "min_total_passes": 1,
                "min_days_active": 1,
                "min_months_active": 1,
            },
            persist=True,
            checkpoint_enabled=True,
            incremental_db_path=first["duckdb_path"],
            resume_existing=True,
            load_final_result=False,
            assignment_scope="prevalent",
            prevalent_fraction_pct=20,
        )

    message = str(exc_info.value)
    assert "parametros no coinciden" in message
    assert "prevalent_fraction_pct" in message


def test_estimate_dynamic_gmm_parallelism_limits_by_memory_and_cpu(tmp_path, monkeypatch):
    duckdb = pytest.importorskip("duckdb")

    flow_db = tmp_path / "flows.duckdb"
    flows = pd.DataFrame(
        {
            "FECHA": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ]
        }
    )
    conn = duckdb.connect(str(flow_db))
    try:
        conn.register("flows_df", flows)
        conn.execute(
            f"CREATE TABLE {clustering.FLOW_TABLE_NAME} AS SELECT * FROM flows_df"
        )
    finally:
        conn.close()

    class Summary:
        db_path = flow_db

    monkeypatch.setattr(clustering, "ensure_flow_db_summary", lambda: Summary())
    monkeypatch.setattr(clustering, "_dynamic_gmm_available_memory_bytes", lambda: 5000)

    estimate = clustering.estimate_dynamic_gmm_parallelism(
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-03"),
        window_days=2,
        memory_fraction=1.0,
        bytes_per_flow_row=100,
        worker_overhead_bytes=1000,
        max_cpu_count=4,
    )

    assert estimate["n_windows"] == 2
    assert estimate["max_window_rows"] == 4
    assert estimate["estimated_worker_bytes"] == 1400
    assert estimate["max_parallel_jobs_by_memory"] == 3
    assert estimate["max_parallel_jobs_by_cpu"] == 3
    assert estimate["recommended_parallel_jobs"] == 3


def test_dynamic_gmm_experiments_live_reads_plate_pages_from_duckdb(tmp_path):
    duckdb = pytest.importorskip("duckdb")

    db_path = tmp_path / "dynamic_gmm_live_test.duckdb"
    assignments = pd.DataFrame(
        {
            "window_index": [1] * 12 + [2],
            "window_label": ["w1"] * 12 + ["w2"],
            "window_start": [pd.Timestamp("2024-01-01")] * 13,
            "window_end": [pd.Timestamp("2024-01-08")] * 13,
            "plate": [f"P{i:02d}" for i in range(12)] + ["P01"],
            "cluster_label": [0] * 13,
            "confidence_score": [0.8] * 13,
            "assignment_status": ["assigned"] * 13,
            "cluster_prob_0": [0.8] * 13,
            "cluster_prob_1": [0.2] * 13,
            "total_passes": [10, 10, 10, 50, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        }
    )
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("assignments_df", assignments)
        conn.execute(
            f"CREATE TABLE {clustering.DYNAMIC_GMM_ASSIGNMENT_TABLE_NAME} AS "
            "SELECT * FROM assignments_df"
        )
        plate_options = live_app._dynamic_gmm_read_plate_options(conn)
        selected, total, page = live_app._dynamic_gmm_read_plate_page(conn, 0)
        selected_2, total_2, page_2 = live_app._dynamic_gmm_read_plate_page(conn, 1)
        selected_assignments = live_app._dynamic_gmm_read_assignments_for_plates(
            conn,
            selected_2,
        )
        single_plate_assignments = live_app._dynamic_gmm_read_assignments_for_plates(
            conn,
            ["P01"],
        )
    finally:
        conn.close()

    assert plate_options == ["P03", "P01"] + [
        plate for plate in [f"P{i:02d}" for i in range(12)] if plate not in {"P01", "P03"}
    ]
    assert live_app._dynamic_gmm_resolve_selected_plate(plate_options, "P10") == "P10"
    assert live_app._dynamic_gmm_resolve_selected_plate(plate_options, "missing") == "P03"
    assert live_app._dynamic_gmm_resolve_selected_plate([], "P00") is None
    assert selected == [f"P{i:02d}" for i in range(10)]
    assert (total, page) == (12, 0)
    assert selected_2 == ["P10", "P11"]
    assert (total_2, page_2) == (12, 1)
    assert selected_assignments["plate"].drop_duplicates().tolist() == ["P10", "P11"]
    assert single_plate_assignments["plate"].drop_duplicates().tolist() == ["P01"]
    assert len(single_plate_assignments) == 2
