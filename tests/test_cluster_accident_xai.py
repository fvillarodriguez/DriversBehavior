from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import src.cluster_accident_app as app
from src.model_training import train_model
from src.model_xai import compute_xai_report
from tests.pipeline_helpers import build_synthetic_base_df


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}


def test_record_experiment_history_persists_base_cluster_xai_bundle(
    tmp_path, monkeypatch
):
    pytest.importorskip("streamlit")
    pytest.importorskip("shap")
    base_df, feature_cols, base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()

    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "flow_features_path": str(tmp_path / "flow_features.duckdb"),
            "flow_features_source": "duckdb",
            "cluster_features_path": str(tmp_path / "cluster_features.duckdb"),
            "cluster_features_source": "calculadas",
            "cluster_features_df": features_df,
            "cluster_choice": "cluster_kmeans_k2.csv",
            "selected_features": feature_cols,
            "feature_selection_store": {},
            "feature_importances_df": None,
            "optuna_results_store": {},
            "optuna_active_key": None,
            "balanced_base_df": None,
            "balanced_cluster_df": None,
            "balance_last_stats": None,
            "balance_last_params": None,
            "flow_features_tramo": None,
            "flow_features_tramo_label": None,
            "acc_flow_metrics": ["flow", "speed", "density"],
            "acc_flow_categories": ["Light", "Heavy"],
            "acc_flow_lanes": 2,
            "acc_flow_include_cluster_vars": True,
            "acc_flow_cluster_vars": ["share", "flow", "speed"],
            "history_entries": [],
        }
    )

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(app, "MODELS_DIR", tmp_path / "model_history")
    monkeypatch.setattr(app, "HISTORY_PATH", tmp_path / "experiment_history.jsonl")

    base_result = train_model(
        base_df,
        base_cols,
        "Random Forest",
        {"n_estimators": 20, "max_depth": 3},
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        random_state=42,
    )
    cluster_result = train_model(
        base_df,
        feature_cols,
        "Random Forest",
        {"n_estimators": 20, "max_depth": 3},
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        random_state=42,
    )

    entry = app._record_experiment_history(
        base_df=base_df,
        features_df=features_df,
        balanced_df=None,
        base_feature_cols=base_cols,
        base_result=base_result,
        cluster_feature_cols=feature_cols,
        cluster_result=cluster_result,
        model_choice="Random Forest",
        model_params_base={"n_estimators": 20, "max_depth": 3},
        model_params_cluster={"n_estimators": 20, "max_depth": 3},
        random_state=42,
        test_size=0.2,
        val_size=0.2,
        far_target=0.2,
        use_balanced=False,
    )

    cluster_entry = entry["models"]["Base + Cluster"]
    bundle_path = Path(cluster_entry["xai_bundle_path"])
    assert bundle_path.exists()
    assert (bundle_path / "manifest.json").exists()
    assert entry["xai"]["base_cluster"]["available"] is True

    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cluster_features_path"] == str(tmp_path / "cluster_features.duckdb")
    assert manifest["cluster_features_source"] == "calculadas"
    assert manifest["feature_cols"] == feature_cols
    report = compute_xai_report(bundle_path)
    assert isinstance(report["beeswarm_points"], pd.DataFrame)
    assert not report["beeswarm_points"].empty


def test_resolve_base_cluster_xai_info_handles_legacy_entry():
    bundle_path, bundle_error = app._resolve_base_cluster_xai_info(
        {"run_id": "legacy", "models": {"Base + Cluster": {"model_path": "legacy.joblib"}}}
    )
    assert bundle_path is None
    assert bundle_error is None


def test_apply_optuna_model_params_to_state_targets_base_cluster(tmp_path, monkeypatch):
    base_df, feature_cols, _base_cols, _cluster_cols = build_synthetic_base_df(tmp_path)
    features_df = base_df.drop(columns=["target"]).copy()
    features_path = str(tmp_path / "flow_features.duckdb")
    feature_key = app._feature_selection_key(features_path, "duckdb", features_df)
    optuna_key = app._optuna_result_key(feature_key, feature_cols)
    optuna_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 3.6,
        "reg_lambda": 8.3,
        "gamma": 0.0,
    }
    fake_st = _FakeStreamlit()
    fake_st.session_state.update(
        {
            "flow_features_path": features_path,
            "flow_features_source": "duckdb",
            "selected_features": feature_cols,
            "optuna_best_model_params": optuna_params,
            "optuna_best_model_choice": "XGBoost",
            "optuna_active_key": optuna_key,
            "optuna_model_params_applied_signature": None,
        }
    )
    monkeypatch.setattr(app, "st", fake_st)

    status = app._apply_optuna_model_params_to_state(
        model_choice="XGBoost",
        base_df=base_df,
        features_df=features_df,
    )

    assert status == "Parametros Optuna cargados en los selectores."
    assert fake_st.session_state["cluster_model_xgb_n_estimators"] == 200
    assert fake_st.session_state["cluster_model_xgb_max_depth"] == 4
    assert fake_st.session_state["cluster_model_xgb_learning_rate"] == pytest.approx(0.01)
    assert fake_st.session_state["cluster_model_xgb_subsample"] == pytest.approx(0.9)
    assert fake_st.session_state["cluster_model_xgb_colsample"] == pytest.approx(0.8)
    assert fake_st.session_state["cluster_model_xgb_reg_alpha"] == pytest.approx(3.6)
    assert fake_st.session_state["cluster_model_xgb_reg_lambda"] == pytest.approx(8.3)
    assert "cluster_model_xgb_n_jobs" not in fake_st.session_state
    assert "base_model_xgb_n_estimators" not in fake_st.session_state
    assert fake_st.session_state["optuna_model_params_applied_signature"]


def test_controlled_feature_date_bounds_and_loader_filter_interval(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    features_path = tmp_path / "features.duckdb"
    raw_df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                ["2024-01-01 08:00:00", "2024-01-02 08:00:00", "2024-01-03 08:00:00"]
            ),
            "eje": ["N", "N", "N"],
            "calzada": ["Oriente", "Oriente", "Oriente"],
            "portico_last": ["1", "1", "1"],
            "portico_next": ["2", "2", "2"],
            "flow_light": [10.0, 11.0, 12.0],
            "cluster_count_0": [1.0, 2.0, 3.0],
        }
    )
    con = duckdb.connect(str(features_path))
    try:
        con.register("features_view", raw_df)
        con.execute("CREATE TABLE flow_features AS SELECT * FROM features_view")
    finally:
        con.close()

    bounds = app._controlled_feature_timestamp_bounds(features_path)
    assert bounds is not None
    assert bounds[0] == pd.Timestamp("2024-01-01 08:00:00")
    assert bounds[1] == pd.Timestamp("2024-01-03 08:00:00")

    filtered = app._load_controlled_features_df(
        features_path,
        ("N", "Oriente", "1", "2"),
        date_start=pd.Timestamp("2024-01-02 00:00:00"),
        date_end=pd.Timestamp("2024-01-02 23:59:59"),
    )

    assert len(filtered) == 1
    assert filtered["interval_start"].iloc[0] == pd.Timestamp("2024-01-02 08:00:00")
    assert filtered["flow_light"].iloc[0] == pytest.approx(11.0)


def test_controlled_comparison_metric_options_include_extended_metrics():
    df = pd.DataFrame(
        [
            {
                "val_objective_score": 0.81,
                "test_objective_score": 0.79,
                "test_accuracy": 0.95,
                "test_pr_auc": 0.22,
                "test_mcc": 0.11,
                "test_false_negatives": 3,
                "decision_threshold": 0.45,
            }
        ]
    )

    options = app._controlled_comparison_metric_options(
        df,
        objective_label="ROC-AUC",
    )
    labels_by_col = {column: label for label, column in options}

    assert labels_by_col["val_objective_score"] == "Validación ROC-AUC (objetivo)"
    assert labels_by_col["test_accuracy"] == "Test Accuracy"
    assert labels_by_col["test_pr_auc"] == "Test PR-AUC"
    assert labels_by_col["test_mcc"] == "Test MCC"
    assert labels_by_col["test_false_negatives"] == "Test Falsos Negativos"
    assert labels_by_col["decision_threshold"] == "Threshold de decisión"


def test_prepare_controlled_comparison_detail_display_formats_all_values():
    detail_df = pd.DataFrame(
        [
            {
                "model_name": "XGBoost",
                "feature_set": "Base",
                "balance_mode": "none",
                "k": 25,
                "status": "completed",
                "selected_features": '["f1","f2"]',
                "best_params": '{"max_depth": 4}',
                "smote_params": '{"sampling_strategy": 0.5}',
                "test_confusion_matrix": "[[10, 2], [3, 4]]",
                "val_confusion_matrix": [[8, 1], [2, 5]],
                "test_accuracy": 0.91,
            }
        ]
    )

    display_df = app._prepare_controlled_comparison_detail_display(detail_df)

    assert list(display_df.columns[:4]) == [
        "model_name",
        "feature_set",
        "balance_mode",
        "k",
    ]
    row = display_df.iloc[0]
    assert row["selected_features"] == "f1, f2"
    assert "max_depth" in row["best_params"]
    assert "sampling_strategy" in row["smote_params"]
    assert row["test_confusion_matrix"] == "[[10, 2], [3, 4]]"
    assert row["val_confusion_matrix"] == "[[8, 1], [2, 5]]"


def test_prepare_controlled_comparison_detail_display_keeps_ablation_columns():
    detail_df = pd.DataFrame(
        [
            {
                "model_name": "XGBoost",
                "feature_set": "Base + Cluster",
                "ablation_phase": "cross_eval",
                "params_source_feature_set": "Base",
                "target_feature_set": "Base + Cluster",
                "source_combo_id": "ablation_source",
                "frozen_tuning": True,
                "threshold_freeze_policy": "recalibrate_per_target",
                "balance_mode": "smote",
                "threshold_protocol": "conservative",
                "k": 2,
                "effective_k": 2,
                "selected_base_feature_count": 1,
                "selected_cluster_feature_count": 1,
                "selected_features": '["flow","cluster_count_0"]',
                "best_params": '{"source": "Base"}',
                "effective_model_params": '{"source": "Base", "n_jobs": 1}',
                "smote_params": '{"sampling_strategy": 0.5}',
                "status": "completed",
            }
        ]
    )

    display_df = app._prepare_controlled_comparison_detail_display(detail_df)
    expected_cols = [
        "model_name",
        "feature_set",
        "ablation_phase",
        "params_source_feature_set",
        "target_feature_set",
        "source_combo_id",
        "frozen_tuning",
        "threshold_freeze_policy",
    ]

    assert list(display_df.columns[: len(expected_cols)]) == expected_cols
    row = display_df.iloc[0]
    assert row["selected_features"] == "flow, cluster_count_0"
    assert "source" in row["best_params"]
    assert "n_jobs" in row["effective_model_params"]
    assert "sampling_strategy" in row["smote_params"]


def test_seed_controlled_comparison_live_db_backfills_checkpoint_results(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(app, "RESULTS_DIR", tmp_path)

    db_path = app._init_experiment_db(
        "Controlled comparison",
        {"dataset_name": "events.csv", "features_name": "features.duckdb"},
    )
    assert db_path is not None

    checkpoint_run_dir = tmp_path / "controlled_run"
    results_dir = checkpoint_run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "combo_id": "combo__rf__base__none__k5",
                "run_id": "controlled_001",
                "model_name": "Random Forest",
                "feature_set": "Base",
                "balance_mode": "none",
                "k": 5,
                "status": "completed",
                "val_objective_score": 0.73,
            }
        ]
    ).to_csv(results_dir / "grid_results.csv", index=False)

    seeded = app._seed_controlled_comparison_live_db(
        db_path,
        checkpoint_run_dir=checkpoint_run_dir,
        dataset_name="events.csv",
        features_name="features.duckdb",
        segment_info={"segment_label": "P1 -> P2"},
    )

    assert seeded == 1

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT payload_json FROM results ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    assert len(rows) == 1
    payload = json.loads(rows[0][0])
    assert payload["experiment"] == "Controlled comparison"
    assert payload["dataset_name"] == "events.csv"
    assert payload["features_name"] == "features.duckdb"
    assert payload["segment_info"]["segment_label"] == "P1 -> P2"
    assert payload["combo_id"] == "combo__rf__base__none__k5"


def test_load_controlled_comparison_result_frames_reads_paths(tmp_path):
    summary_df = pd.DataFrame([{"model_name": "Random Forest", "k_optimo": 5}])
    curves_df = pd.DataFrame([{"model_name": "Random Forest", "k": 5}])
    detail_df = pd.DataFrame([{"combo_id": "combo__rf__base__none__k5"}])
    deltas_df = pd.DataFrame([{"effect_type": "feature_effect", "k": 5}])

    summary_path = tmp_path / "summary.csv"
    curves_path = tmp_path / "curves.csv"
    detail_path = tmp_path / "detail.csv"
    deltas_path = tmp_path / "deltas.csv"
    summary_df.to_csv(summary_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    deltas_df.to_csv(deltas_path, index=False)

    loaded_summary, loaded_curves, loaded_detail, loaded_deltas = (
        app._load_controlled_comparison_result_frames(
            {
                "summary_path": str(summary_path),
                "curves_path": str(curves_path),
                "detail_path": str(detail_path),
                "ablation_deltas_path": str(deltas_path),
            }
        )
    )

    pd.testing.assert_frame_equal(loaded_summary, summary_df)
    pd.testing.assert_frame_equal(loaded_curves, curves_df)
    pd.testing.assert_frame_equal(loaded_detail, detail_df)
    pd.testing.assert_frame_equal(loaded_deltas, deltas_df)


def test_load_controlled_comparison_result_frames_tolerates_legacy_without_deltas(tmp_path):
    summary_df = pd.DataFrame([{"model_name": "Random Forest", "k_optimo": 5}])
    summary_path = tmp_path / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    loaded_summary, loaded_curves, loaded_detail, loaded_deltas = (
        app._load_controlled_comparison_result_frames(
            {
                "summary_path": str(summary_path),
            }
        )
    )

    pd.testing.assert_frame_equal(loaded_summary, summary_df)
    assert loaded_curves.empty
    assert loaded_detail.empty
    assert loaded_deltas.empty
