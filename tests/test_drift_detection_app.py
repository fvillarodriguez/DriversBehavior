import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

import src.drift_detection_app as drift_app

from src.drift_detection_app import (
    ARF_VARIANT_NAMES,
    BALANCE_MODE_NONE,
    BALANCE_MODE_NOT_APPLICABLE,
    BALANCE_MODE_SMOTE,
    DRIFT_FOCUS_PORTICOS,
    DRIFT_FOCUS_TRAMO,
    KSWIN_VARIANT_NAMES,
    _evaluate_split,
    _add_interval_accident_target,
    _build_batch_ranges,
    _build_feature_selection_payload,
    _count_positive_target_rows,
    _compute_batched_flow_features_to_duckdb,
    _compute_drift_article_features,
    _describe_tramo_selection,
    _filter_accidents_for_allowed_porticos,
    _import_external_xgboost,
    _load_feature_payload_from_duckdb,
    _load_feature_df_from_duckdb,
    _load_feature_selection_payload_from_duckdb,
    _normalize_portico_series,
    _porticos_in_tramo,
    _rebuild_preparation_artifacts_from_payload,
    _write_feature_payload_to_duckdb,
    SimpleADWIN,
    apply_missing_data_policy,
    article_coverage_percentage,
    build_article_coverage_matrix,
    build_average_roc_curves,
    build_python_migration_review_plan,
    drop_highly_correlated_features,
    filter_long_zero_accident_runs,
    generate_synthetic_article_dataset,
    parse_repetition_seeds,
    run_arf_strategy,
    run_adaptive_strategy,
    run_configurable_preparation_pipeline,
    run_kswin_strategy,
    run_recalibration_experiments,
    run_yearly_strategy,
    train_model_with_internal_validation,
    youden_threshold,
    compute_classification_metrics,
)


def _make_kswin_shift_dataset(rows_per_year: int = 360, random_state: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    yearly_shift = {2018: 0.0, 2019: 4.0, 2020: -3.5}

    for year, shift in yearly_shift.items():
        ts = pd.date_range(f"{year}-01-01 00:00:00", periods=rows_per_year, freq="5min")
        idx = np.arange(rows_per_year, dtype=float)

        x1 = rng.normal(loc=shift, scale=0.25, size=rows_per_year)
        x2 = rng.normal(loc=shift * 0.9, scale=0.3, size=rows_per_year)
        x3 = rng.normal(loc=0.0, scale=1.0, size=rows_per_year)

        flow_light = 100.0 + 15.0 * x1 + 4.0 * np.sin(idx / 18.0)
        flow_heavy = 45.0 + 8.0 * x2 + 2.5 * np.cos(idx / 22.0)
        speed_light = 80.0 - 3.0 * x1 + rng.normal(0.0, 0.8, size=rows_per_year)
        speed_heavy = 67.0 - 2.5 * x2 + rng.normal(0.0, 0.8, size=rows_per_year)
        density_light = 20.0 + 2.0 * x1 + rng.normal(0.0, 0.5, size=rows_per_year)
        density_heavy = 10.0 + 1.5 * x2 + rng.normal(0.0, 0.4, size=rows_per_year)
        periodic_signal = np.sin(idx / 12.0) + 0.35 * np.cos(idx / 25.0)
        logits = periodic_signal + 0.25 * x1 - 0.20 * x2 + rng.normal(0.0, 0.35, size=rows_per_year)
        target = (logits > 0.0).astype(int)

        for i in range(rows_per_year):
            rows.append(
                {
                    "interval_start": ts[i],
                    "portico": "15",
                    "flow_light": flow_light[i],
                    "flow_heavy": flow_heavy[i],
                    "speed_light": speed_light[i],
                    "speed_heavy": speed_heavy[i],
                    "density_light": density_light[i],
                    "density_heavy": density_heavy[i],
                    "x1": x1[i],
                    "x2": x2[i],
                    "x3": x3[i],
                    "target": int(target[i]),
                }
            )

    return pd.DataFrame(rows)


def test_count_positive_target_rows_uses_final_target_column():
    clean_df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=6, freq="5min"),
            "target": [0, 1, "1", None, 0, 1],
        }
    )

    assert _count_positive_target_rows(clean_df) == 3
    assert _count_positive_target_rows(pd.DataFrame({"x": [1, 2, 3]})) == 0


def test_import_external_xgboost_exposes_classifier():
    xgb = _import_external_xgboost()

    assert hasattr(xgb, "XGBClassifier")
    assert "site-packages/xgboost" in str(getattr(xgb, "__file__", ""))


def test_article_coverage_is_100_percent():
    matrix = build_article_coverage_matrix()

    # 21 sections + 7 figures + 9 tables
    assert len(matrix) == 37
    assert matrix["implemented"].all()
    assert article_coverage_percentage(matrix) == 100.0


def test_python_migration_review_plan_mentions_log_and_sequential_review():
    plan = build_python_migration_review_plan()

    assert not plan.empty
    assert "step" in plan.columns
    joined = " ".join(plan.astype(str).stack().tolist()).lower()
    assert "log" in joined
    assert "secu" in joined or "sequential" in joined


def test_apply_missing_data_policy_removes_columns_and_rows():
    df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "target": [0, 1, 0, 1, 0],
            "f1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "f2": [np.nan, np.nan, 3.0, np.nan, np.nan],
            "f3": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    clean, info = apply_missing_data_policy(
        df,
        feature_cols=["f1", "f2", "f3"],
        missing_threshold=0.4,
        target_col="target",
        time_col="interval_start",
    )

    # f2 has 80% missing and must be removed.
    assert "f2" in info["removed_cols"]

    # f1 remains and row 3 has missing f1 -> should be dropped.
    assert len(clean) == 4
    assert set(info["remaining_features"]) == {"f1", "f3"}


def test_filter_long_zero_accident_runs():
    time_idx = pd.date_range("2024-01-01", periods=80, freq="1h")
    target = np.zeros(len(time_idx), dtype=int)
    target[30] = 1

    df = pd.DataFrame({"interval_start": time_idx, "target": target, "x": np.arange(len(time_idx))})

    filtered, runs = filter_long_zero_accident_runs(
        df,
        target_col="target",
        time_col="interval_start",
        min_days=1,
        interval_minutes=60,
    )

    # Two long zero-runs exist (before and after the single positive).
    assert len(runs) == 2
    assert len(filtered) < len(df)
    assert filtered["target"].sum() == 1


def test_drop_highly_correlated_features():
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 200)
    x2 = 2.0 * x1 + rng.normal(0, 0.001, 200)
    x3 = rng.normal(0, 1, 200)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    kept, dropped, corr = drop_highly_correlated_features(df, ["x1", "x2", "x3"], threshold=0.95)

    assert not corr.empty
    assert len(dropped) == 1
    assert len(kept) == 2
    assert not ({"x1", "x2"} <= set(kept))


def test_youden_threshold_and_metrics():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.10, 0.40, 0.35, 0.80])

    info = youden_threshold(y_true, scores)
    metrics = compute_classification_metrics(y_true, scores, threshold=info["threshold"])

    assert 0.0 <= info["threshold"] <= 1.0
    assert metrics["auc"] > 0.5
    assert 0.0 <= metrics["error_rate"] <= 1.0


def test_simple_adwin_detects_shift():
    detector = SimpleADWIN(delta=0.01, min_window=80, min_subwindow=20)
    stream = [0.0] * 120 + [1.0] * 180

    detections = 0
    for value in stream:
        is_drift, _info = detector.update(value)
        if is_drift:
            detections += 1

    assert detections >= 1


def test_simple_adwin_uses_canonical_hoeffding_bound():
    detector = SimpleADWIN(delta=0.002, min_window=80, min_subwindow=20)

    n0 = 40
    n1 = 60
    n = n0 + n1

    expected = np.sqrt((1.0 / (2.0 * (1.0 / ((1.0 / n0) + (1.0 / n1))))) * np.log(4.0 * n / detector.delta))

    assert detector._hoeffding_cut_threshold(n0, n1, n) == pytest.approx(expected)


def test_run_yearly_strategy_static_random_forest():
    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=280, random_state=7)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    results, roc_payload = run_yearly_strategy(
        df,
        strategy="static",
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=13,
        fast_mode=True,
        grid_limit=2,
    )

    assert not results.empty
    assert set(results["prediction_year"].astype(int).unique()) == {2019, 2020}
    assert (results["strategy"] == "static").all()
    assert len(roc_payload) == len(results)


def test_run_adaptive_strategy_random_forest():
    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=260, random_state=19)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    adaptive_df, roc_payload = run_adaptive_strategy(
        df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=5,
        fast_mode=True,
        grid_limit=2,
        adwin_delta=0.01,
        min_window=120,
    )

    assert not adaptive_df.empty
    assert (adaptive_df["strategy"] == "adaptive_adwin").all()
    assert len(roc_payload) >= 1


def test_run_yearly_strategy_emits_stream_record_callback_payloads(monkeypatch):
    df = _make_kswin_shift_dataset(rows_per_year=36, random_state=321)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    class _FakeModel:
        def predict_proba(self, X):
            X_np = np.asarray(X, dtype=float)
            base = np.clip(0.35 + 0.03 * np.nan_to_num(X_np[:, 0], nan=0.0), 0.05, 0.95)
            return np.column_stack([1.0 - base, base])

    def fake_train_model_with_internal_validation(*args, **kwargs):
        return {
            "model": _FakeModel(),
            "threshold": 0.5,
            "operating_threshold": 0.5,
            "raw_youden_threshold": 0.5,
            "calibration_model": None,
            "calibration_metadata": None,
            "feature_transform": None,
            "fill_values": {},
            "best_params": {},
            "tuning_artifact": {"best_params": {}},
        }

    callbacks: list[dict] = []
    monkeypatch.setattr(drift_app, "train_model_with_internal_validation", fake_train_model_with_internal_validation)

    yearly_df, roc_payload = run_yearly_strategy(
        df,
        strategy="cumulative",
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=42,
        fast_mode=True,
        grid_limit=1,
        stream_record_callback=lambda payload: callbacks.append(dict(payload)),
    )

    assert not yearly_df.empty
    assert roc_payload
    assert callbacks
    assert {int(row["prediction_year"]) for row in callbacks} == set(yearly_df["prediction_year"].astype(int))
    assert any(str(row["action_taken"]) == "retrain" for row in callbacks)
    assert all(pd.notna(row["timestamp"]) for row in callbacks)
    assert {"timestamp", "score", "prediction", "action_taken"} <= set(callbacks[0].keys())


def test_run_adaptive_strategy_emits_stream_record_callback_for_each_stream_row(monkeypatch):
    df = _make_kswin_shift_dataset(rows_per_year=30, random_state=654)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    class _FakeModel:
        def predict_proba(self, X):
            X_np = np.asarray(X, dtype=float)
            base = np.clip(0.30 + 0.02 * np.nan_to_num(X_np[:, 0], nan=0.0), 0.05, 0.95)
            return np.column_stack([1.0 - base, base])

    def fake_train_model_with_internal_validation(*args, **kwargs):
        return {
            "model": _FakeModel(),
            "threshold": 0.5,
            "operating_threshold": 0.5,
            "raw_youden_threshold": 0.5,
            "calibration_model": None,
            "calibration_metadata": None,
            "feature_transform": None,
            "fill_values": {},
            "best_params": {},
            "tuning_artifact": {"best_params": {}},
        }

    callbacks: list[dict] = []
    expected_stream_rows = int(pd.to_datetime(df["interval_start"]).dt.year.gt(2018).sum())
    monkeypatch.setattr(drift_app, "train_model_with_internal_validation", fake_train_model_with_internal_validation)

    adaptive_df, roc_payload = run_adaptive_strategy(
        df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=42,
        fast_mode=True,
        grid_limit=1,
        min_window=10_000,
        stream_record_callback=lambda payload: callbacks.append(dict(payload)),
    )

    assert len(callbacks) == expected_stream_rows
    assert callbacks[0]["record_index"] == 0
    assert callbacks[-1]["record_index"] == expected_stream_rows - 1
    assert all(pd.notna(row["timestamp"]) for row in callbacks)
    assert set(str(row["action_taken"]) for row in callbacks).issubset({"none", "retrain"})
    assert roc_payload or adaptive_df.empty


def test_run_arf_strategy_variants():
    pytest.importorskip("river")

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=240, random_state=29)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    adaptive_df, roc_payload = run_arf_strategy(
        df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        arf_variants=["ARFmoderate", "ARFmaj"],
        validation_size=0.2,
        random_state=7,
    )

    assert not adaptive_df.empty
    assert (adaptive_df["strategy"] == "adaptive_arf").all()
    assert set(adaptive_df["model"].unique()) == {"ARFmoderate", "ARFmaj"}
    assert set(adaptive_df["prediction_year"].astype(int).unique()) == {2019, 2020}
    assert {"segment_rows", "n_internal_drifts", "n_internal_warnings"} <= set(adaptive_df.columns)
    assert len(roc_payload) == 4


def test_run_kswin_strategy_variants():
    pytest.importorskip("river")

    df = _make_kswin_shift_dataset(rows_per_year=360, random_state=41)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    adaptive_df, roc_payload = run_kswin_strategy(
        df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        kswin_variants=["KSWINpaper", "KSWINseasonal"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=1,
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
    )

    assert not adaptive_df.empty
    assert (adaptive_df["strategy"] == "adaptive_kswin").all()
    assert set(adaptive_df["detector_variant"].unique()) == {"KSWINpaper", "KSWINseasonal"}
    assert set(adaptive_df["base_model"].unique()) == {"Random Forest"}
    assert adaptive_df["vote_count"].max() >= 1
    assert adaptive_df["retrain_rows"].max() >= 80
    assert any("KSWINpaper" in label for label in adaptive_df["model"].tolist())
    assert len(roc_payload) >= 2


def test_build_model_uses_extratrees_classifier_for_extratrees_splitrule():
    model = drift_app._build_model(
        "Random Forest",
        {"splitrule": "extratrees", "mtry": 2, "min_node_size": 1},
        random_state=13,
        n_features=5,
    )

    assert isinstance(model, ExtraTreesClassifier)


def test_build_model_bounds_random_forest_n_jobs(monkeypatch):
    monkeypatch.delenv("SUMO_RF_N_JOBS", raising=False)
    default_model = drift_app._build_model(
        "Random Forest",
        {"splitrule": "gini", "mtry": 2, "min_node_size": 1},
        random_state=13,
        n_features=5,
    )

    monkeypatch.setenv("SUMO_RF_N_JOBS", "1")
    overridden_model = drift_app._build_model(
        "Random Forest",
        {"splitrule": "gini", "mtry": 2, "min_node_size": 1},
        random_state=13,
        n_features=5,
    )

    assert 1 <= int(default_model.n_jobs) <= 2
    assert int(overridden_model.n_jobs) == 1


def test_build_model_full_memory_mode_uses_all_visible_cpus(monkeypatch):
    monkeypatch.delenv("SUMO_RF_N_JOBS", raising=False)

    model = drift_app._build_model(
        "Random Forest",
        {"splitrule": "gini", "mtry": 2, "min_node_size": 1},
        random_state=13,
        n_features=5,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY,
    )

    assert int(model.n_jobs) == max(1, int(os.cpu_count() or 1))


def test_resolve_restricted_resource_policy_applies_overrides():
    cpu_count = max(1, int(os.cpu_count() or 1))
    policy = drift_app._resolve_experiment_resource_policy(
        drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        overrides={
            "estimator_n_jobs": {
                "Random Forest": 99,
                "XGBoost": 1,
            },
            "parallel_cv_enabled": False,
        },
    )

    assert int((policy.get("estimator_n_jobs") or {}).get("Random Forest", 0)) == min(2, cpu_count)
    assert int((policy.get("estimator_n_jobs") or {}).get("XGBoost", 0)) == 1
    assert list(policy.get("parallel_cv_models") or []) == []
    assert bool(policy.get("parallel_cv_enabled")) is False


@pytest.mark.parametrize(
    ("splitrule", "expected_type"),
    [
        ("gini", "RandomForestClassifier"),
        ("extratrees", "ExtraTreesClassifier"),
    ],
)
def test_build_model_disables_class_weight_when_smote(splitrule, expected_type):
    smote_model = drift_app._build_model(
        "Random Forest",
        {"splitrule": splitrule, "mtry": 2, "min_node_size": 1},
        random_state=13,
        n_features=5,
        balance_mode=BALANCE_MODE_SMOTE,
    )
    default_model = drift_app._build_model(
        "Random Forest",
        {"splitrule": splitrule, "mtry": 2, "min_node_size": 1},
        random_state=13,
        n_features=5,
        balance_mode=BALANCE_MODE_NONE,
    )

    assert type(smote_model).__name__ == expected_type
    assert smote_model.class_weight is None
    assert default_model.class_weight in {"balanced_subsample", "balanced"}


def test_recalibration_run_id_is_deterministic():
    df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 00:05:00",
                    "2019-01-01 00:00:00",
                    "2019-01-01 00:05:00",
                ]
            ),
            "x": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        }
    )
    feature_selection_context = {"selected_features": ["x"], "feature_count": 1}

    first = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        arf_variants=["ARFmoderate"],
        kswin_variants=["KSWINpaper"],
        balance_modes=[BALANCE_MODE_NONE, BALANCE_MODE_SMOTE],
        repetition_seeds=[11],
        base_year=2018,
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        min_retrain_size=20,
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )
    second = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        arf_variants=["ARFmoderate"],
        kswin_variants=["KSWINpaper"],
        balance_modes=[BALANCE_MODE_NONE, BALANCE_MODE_SMOTE],
        repetition_seeds=[11],
        base_year=2018,
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        min_retrain_size=20,
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )
    different = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        arf_variants=["ARFmoderate"],
        kswin_variants=["KSWINpaper"],
        balance_modes=[BALANCE_MODE_NONE],
        repetition_seeds=[11],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        min_retrain_size=20,
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )

    assert first == second
    assert first != different


def test_preview_recalibration_checkpoint_detects_resumable_manifest(tmp_path):
    df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 00:05:00",
                    "2019-01-01 00:00:00",
                    "2019-01-01 00:05:00",
                ]
            ),
            "x": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        }
    )
    feature_selection_context = {"selected_features": ["x"], "feature_count": 1}
    preview = drift_app._preview_recalibration_checkpoint(
        df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        min_retrain_size=20,
        arf_variants=["ARFmoderate"],
        kswin_variants=["KSWINpaper"],
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        repetition_seeds=[11],
        balance_modes=[BALANCE_MODE_NONE],
        feature_selection_context=feature_selection_context,
        checkpoint_root=tmp_path / "runs",
    )

    assert preview["checkpoint_available"] is False

    run_dir = Path(preview["run_dir"])
    paths = drift_app._recalibration_run_paths(run_dir)
    drift_app._ensure_recalibration_run_dirs(paths)
    manifest = drift_app._initial_recalibration_manifest(
        run_id=preview["run_id"],
        run_manifest={"total_progress_units": 2},
        feature_selection_context=feature_selection_context,
        experiment_blocks=[{"strategy": "static", "model": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        tuning_tasks=[{"model_name": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        preflight={},
    )
    manifest["status"] = "running"
    manifest["completed_block_ids"] = ["block_demo"]
    manifest["tuning_index"] = {
        "tuning_demo": {"status": "completed", "filename": "tuning_demo.json"}
    }
    manifest["smote_index"] = {
        "smote_demo": {"status": "available", "filename": "smote_demo.duckdb"}
    }
    drift_app._update_manifest_progress(
        manifest,
        completed_units=2.0,
        pending_block_ids=["block_pending"],
    )
    drift_app._persist_manifest(paths["manifest"], manifest)

    resumed = drift_app._preview_recalibration_checkpoint(
        df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        min_retrain_size=20,
        arf_variants=["ARFmoderate"],
        kswin_variants=["KSWINpaper"],
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        repetition_seeds=[11],
        balance_modes=[BALANCE_MODE_NONE],
        feature_selection_context=feature_selection_context,
        checkpoint_root=tmp_path / "runs",
    )

    assert resumed["checkpoint_available"] is True
    assert resumed["can_resume"] is True
    assert resumed["can_load_completed"] is False
    assert resumed["can_recompute_trainings"] is True
    assert resumed["status"] == "running"
    assert resumed["completed_tuning_tasks"] == 1
    assert resumed["completed_blocks"] == 1
    assert resumed["smote_artifacts"] == 1
    assert resumed["manifest_path"].endswith("manifest.json")


def test_preview_recalibration_checkpoint_separates_resource_modes(tmp_path):
    df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 00:05:00",
                    "2019-01-01 00:00:00",
                    "2019-01-01 00:05:00",
                ]
            ),
            "x": [0.1, 0.2, 0.3, 0.4],
            "target": [0, 1, 0, 1],
        }
    )
    feature_selection_context = {"selected_features": ["x"], "feature_count": 1}

    restricted = drift_app._preview_recalibration_checkpoint(
        df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        min_retrain_size=20,
        arf_variants=["ARFmoderate"],
        kswin_variants=["KSWINpaper"],
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        repetition_seeds=[11],
        balance_modes=[BALANCE_MODE_NONE],
        feature_selection_context=feature_selection_context,
        checkpoint_root=tmp_path / "runs",
    )

    run_dir = Path(restricted["run_dir"])
    paths = drift_app._recalibration_run_paths(run_dir)
    drift_app._ensure_recalibration_run_dirs(paths)
    manifest = drift_app._initial_recalibration_manifest(
        run_id=restricted["run_id"],
        run_manifest={"total_progress_units": 1, "resource_mode": drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED},
        feature_selection_context=feature_selection_context,
        experiment_blocks=[{"strategy": "static", "model": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        tuning_tasks=[{"model_name": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        preflight={},
    )
    drift_app._persist_manifest(paths["manifest"], manifest)

    full_memory = drift_app._preview_recalibration_checkpoint(
        df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        min_retrain_size=20,
        arf_variants=["ARFmoderate"],
        kswin_variants=["KSWINpaper"],
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        repetition_seeds=[11],
        balance_modes=[BALANCE_MODE_NONE],
        feature_selection_context=feature_selection_context,
        checkpoint_root=tmp_path / "runs",
    )

    assert restricted["run_id"] != full_memory["run_id"]
    assert full_memory["checkpoint_available"] is False


def test_preview_recalibration_checkpoint_run_reads_explicit_run_id(tmp_path):
    run_dir = drift_app._recalibration_run_dir("run_explicit_preview", checkpoint_root=tmp_path / "runs")
    paths = drift_app._recalibration_run_paths(run_dir)
    drift_app._ensure_recalibration_run_dirs(paths)
    manifest = drift_app._initial_recalibration_manifest(
        run_id="run_explicit_preview",
        run_manifest={"total_progress_units": 3},
        feature_selection_context={},
        experiment_blocks=[],
        tuning_tasks=[],
        preflight={},
    )
    manifest["status"] = "running"
    manifest["progress"]["completed_tuning_tasks"] = 2
    manifest["progress"]["total_tuning_tasks"] = 3
    manifest["progress"]["completed_blocks"] = 1
    manifest["progress"]["total_blocks"] = 2
    drift_app._atomic_write_json(paths["manifest"], manifest)

    preview = drift_app._preview_recalibration_checkpoint_run(
        "run_explicit_preview",
        checkpoint_root=tmp_path / "runs",
    )

    assert preview["checkpoint_available"] is True
    assert preview["run_id"] == "run_explicit_preview"
    assert preview["status"] == "running"
    assert preview["can_resume"] is True
    assert preview["can_recompute_trainings"] is True


def test_recalibration_checkpoint_archive_roundtrip(tmp_path):
    source_root = tmp_path / "runs_source"
    run_dir = drift_app._recalibration_run_dir("run_demo_checkpoint", checkpoint_root=source_root)
    paths = drift_app._recalibration_run_paths(run_dir)
    drift_app._ensure_recalibration_run_dirs(paths)

    manifest = drift_app._initial_recalibration_manifest(
        run_id="run_demo_checkpoint",
        run_manifest={"total_progress_units": 2},
        feature_selection_context={"selected_features": ["x"], "feature_count": 1},
        experiment_blocks=[{"strategy": "static", "model": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        tuning_tasks=[{"model_name": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        preflight={},
    )
    manifest["status"] = "completed"
    manifest["completed_block_ids"] = ["block_demo"]
    manifest["block_index"] = {
        "block_demo": {"status": "completed", "filename": "block_demo.json"}
    }
    manifest["tuning_index"] = {
        "tuning_demo": {"status": "completed", "filename": "tuning_demo.json", "study_id": "study_demo"}
    }
    manifest["smote_index"] = {
        "smote_demo": {"status": "available", "filename": "smote_demo.duckdb"}
    }
    drift_app._update_manifest_progress(
        manifest,
        completed_units=2.0,
        pending_block_ids=[],
    )
    drift_app._persist_manifest(paths["manifest"], manifest)
    drift_app._persist_tuning_artifact(
        paths["tuning_dir"] / "tuning_demo.json",
        {"tuning_key": "tuning_demo", "study_id": "study_demo", "best_params": {"mtry": 2}},
    )
    drift_app._persist_recalibration_block(
        paths["blocks_dir"] / "block_demo.json",
        {
            "block_id": "block_demo",
            "block": {"strategy": "static", "model": "Random Forest", "balance_mode": BALANCE_MODE_NONE},
            "run_seed": 11,
            "run_order": 1,
            "yearly_rows": [],
            "adaptive_rows": [],
            "roc_payload": [],
            "execution_log": [],
            "tuning_refs": ["tuning_demo"],
            "smote_refs": ["smote_demo"],
        },
    )
    (paths["smote_dir"] / "smote_demo.duckdb").write_text("stub", encoding="utf-8")

    archive_bytes = drift_app._build_recalibration_checkpoint_archive(run_dir)
    imported = drift_app._import_recalibration_checkpoint_archive(
        archive_bytes,
        checkpoint_root=tmp_path / "runs_imported",
    )

    imported_run_dir = Path(imported["run_dir"])
    imported_paths = drift_app._recalibration_run_paths(imported_run_dir)
    imported_manifest = json.loads(imported_paths["manifest"].read_text(encoding="utf-8"))

    assert imported["run_id"] == "run_demo_checkpoint"
    assert imported["status"] == "completed"
    assert imported_paths["blocks_dir"].joinpath("block_demo.json").exists()
    assert imported_paths["tuning_dir"].joinpath("tuning_demo.json").exists()
    assert imported_paths["smote_dir"].joinpath("smote_demo.duckdb").exists()
    assert imported_manifest["status"] == "completed"
    assert imported_manifest["completed_block_ids"] == ["block_demo"]


def test_recalibration_checkpoint_archive_import_requires_overwrite_for_existing_run(tmp_path):
    source_root = tmp_path / "runs_source"
    run_dir = drift_app._recalibration_run_dir("run_existing_checkpoint", checkpoint_root=source_root)
    paths = drift_app._recalibration_run_paths(run_dir)
    drift_app._ensure_recalibration_run_dirs(paths)
    drift_app._persist_manifest(
        paths["manifest"],
        drift_app._initial_recalibration_manifest(
            run_id="run_existing_checkpoint",
            run_manifest={"total_progress_units": 1},
            feature_selection_context={"selected_features": ["x"], "feature_count": 1},
            experiment_blocks=[],
            tuning_tasks=[],
            preflight={},
        ),
    )

    archive_bytes = drift_app._build_recalibration_checkpoint_archive(run_dir)
    target_root = tmp_path / "runs_imported"
    drift_app._import_recalibration_checkpoint_archive(archive_bytes, checkpoint_root=target_root)

    with pytest.raises(FileExistsError):
        drift_app._import_recalibration_checkpoint_archive(archive_bytes, checkpoint_root=target_root)


def test_list_recalibration_checkpoints_sorts_by_updated_at_desc(tmp_path):
    checkpoint_root = tmp_path / "runs"

    older_dir = drift_app._recalibration_run_dir("run_older", checkpoint_root=checkpoint_root)
    older_paths = drift_app._recalibration_run_paths(older_dir)
    drift_app._ensure_recalibration_run_dirs(older_paths)
    older_manifest = drift_app._initial_recalibration_manifest(
        run_id="run_older",
        run_manifest={"models": ["Random Forest"], "strategies": ["static"], "feature_count": 9},
        feature_selection_context={"selected_features": ["x1", "x2"], "feature_count": 2},
        experiment_blocks=[],
        tuning_tasks=[],
        preflight={},
    )
    older_manifest["updated_at"] = "2026-03-27T10:00:00"
    older_manifest["status"] = "completed"
    drift_app._atomic_write_json(older_paths["manifest"], older_manifest)

    newer_dir = drift_app._recalibration_run_dir("run_newer", checkpoint_root=checkpoint_root)
    newer_paths = drift_app._recalibration_run_paths(newer_dir)
    drift_app._ensure_recalibration_run_dirs(newer_paths)
    newer_manifest = drift_app._initial_recalibration_manifest(
        run_id="run_newer",
        run_manifest={"models": ["AdaBoost"], "strategies": ["adaptive_adwin"], "feature_count": 7},
        feature_selection_context={"selected_features": ["x3"], "feature_count": 1},
        experiment_blocks=[],
        tuning_tasks=[],
        preflight={},
    )
    newer_manifest["updated_at"] = "2026-03-27T11:00:00"
    newer_manifest["status"] = "running"
    drift_app._atomic_write_json(newer_paths["manifest"], newer_manifest)

    checkpoints = drift_app._list_recalibration_checkpoints(checkpoint_root=checkpoint_root)

    assert [item["run_id"] for item in checkpoints[:2]] == ["run_newer", "run_older"]
    assert checkpoints[0]["status"] == "running"
    assert checkpoints[1]["status"] == "completed"
    assert checkpoints[0]["feature_selection_context"]["selected_features"] == ["x3"]
    assert checkpoints[1]["feature_selection_context"]["selected_features"] == ["x1", "x2"]


def test_checkpoint_form_state_from_run_manifest_maps_supported_fields():
    run_manifest = {
        "resource_mode": drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY,
        "models": ["Random Forest", "UnknownModel"],
        "strategies": ["static", "adaptive_kswin", "invalid_strategy"],
        "fast_mode": False,
        "optuna_trials": 17,
        "folds": 4,
        "validation_size": 0.25,
        "adwin_delta": 0.003,
        "min_window": 1234,
        "min_retrain_size": 321,
        "arf_variants": ["ARFmoderate", "missing_variant"],
        "kswin_variants": ["KSWINpaper", "missing_variant"],
        "kswin_top_k_features": 99,
        "kswin_vote_threshold": 99,
        "kswin_retrain_days": 45,
        "kswin_min_retrain_rows": 222,
        "balance_modes": [BALANCE_MODE_NONE, "invalid_mode"],
        "repetition_seeds": [7, 11],
        "continue_on_block_error": True,
        "target_col": "target",
        "time_col": "interval_start",
        "feature_selection_context": {
            "selected_features": ["x1", "x_missing"],
            "candidate_features": ["x1", "target", "x_missing"],
            "feature_export_path": "/tmp/checkpoint_features.duckdb",
        },
    }

    updates = drift_app._checkpoint_form_state_from_run_manifest(
        run_manifest,
        available_columns=["interval_start", "target", "x1"],
        feature_count=8,
    )

    assert updates["drift_exp_resource_mode"] == drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY
    assert updates["drift_exp_models"] == ["Random Forest"]
    assert updates["drift_exp_strategies"] == ["static", "adaptive_kswin"]
    assert updates["drift_exp_fast_mode"] is False
    assert updates["drift_exp_optuna_trials"] == 17
    assert updates["drift_exp_folds"] == 4
    assert updates["drift_exp_validation_size"] == pytest.approx(0.25)
    assert updates["drift_exp_adwin_delta"] == pytest.approx(0.003)
    assert updates["drift_exp_min_window"] == 1234
    assert updates["drift_exp_min_retrain_size"] == 321
    assert updates["drift_exp_arf_variants"] == ["ARFmoderate"]
    assert updates["drift_exp_kswin_variants"] == ["KSWINpaper"]
    assert updates["drift_exp_kswin_top_k_features"] == 8
    assert updates["drift_exp_kswin_vote_threshold"] == 8
    assert updates["drift_exp_kswin_retrain_days"] == 45
    assert updates["drift_exp_kswin_min_retrain_rows"] == 222
    assert updates["drift_exp_balance_modes"] == [BALANCE_MODE_NONE]
    assert updates["drift_exp_seed_text"] == "7, 11"
    assert updates["drift_exp_continue_on_block_error"] is True
    assert updates["drift_exp_target_col"] == "target"
    assert updates["drift_exp_time_col"] == "interval_start"
    assert updates["drift_feature_cols"] == ["x1"]
    assert updates["drift_candidate_feature_cols"] == ["x1", "target"]
    assert updates["drift_feature_export_path"] == "/tmp/checkpoint_features.duckdb"
    assert updates["drift_feature_selection_meta"]["selected_features"] == ["x1", "x_missing"]


def test_checkpoint_form_state_from_run_manifest_restores_restricted_resource_controls():
    run_manifest = {
        "resource_mode": drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        "resource_policy": {
            "estimator_n_jobs": {
                "Random Forest": 1,
                "XGBoost": 3,
            },
            "parallel_cv_models": [],
        },
    }

    updates = drift_app._checkpoint_form_state_from_run_manifest(
        run_manifest,
        available_columns=["interval_start", "target", "x1"],
        feature_count=3,
    )

    assert updates["drift_exp_resource_mode"] == drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED
    assert updates["drift_exp_restricted_rf_n_jobs"] == 1
    assert updates["drift_exp_restricted_xgb_n_jobs"] == min(3, max(1, int(os.cpu_count() or 1)))
    assert updates["drift_exp_restricted_parallel_cv"] is False


def test_checkpoint_form_state_from_run_manifest_resolves_windows_feature_export_path(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)
    feature_path = tmp_path / "checkpoint_features.duckdb"
    drift_app._write_feature_payload_to_duckdb(
        feature_path,
        raw_df=pd.DataFrame(
            {
                "interval_start": pd.date_range("2024-01-01", periods=2, freq="h"),
                "target": [0, 1],
                "x1": [1.0, 2.0],
            }
        ),
        clean_df=pd.DataFrame(
            {
                "interval_start": pd.date_range("2024-01-01", periods=2, freq="h"),
                "target": [0, 1],
                "x1": [1.0, 2.0],
            }
        ),
    )
    run_manifest = {
        "feature_selection_context": {
            "selected_features": ["x1"],
            "feature_export_path": r"C:\Users\usuario\Resultados\checkpoint_features.duckdb",
        }
    }

    updates = drift_app._checkpoint_form_state_from_run_manifest(
        run_manifest,
        available_columns=["interval_start", "target", "x1"],
        feature_count=1,
    )

    assert updates["drift_feature_export_path"] == str(feature_path.resolve())
    assert updates["drift_feature_selection_meta"]["feature_export_path"] == str(feature_path.resolve())


def test_execution_config_from_checkpoint_run_manifest_restores_hidden_tuning_fields():
    current_config = {
        "feature_cols": ["flow_light", "x1"],
        "target_col": "target",
        "time_col": "interval_start",
        "model_names": ["Random Forest"],
        "strategies": ["static"],
        "validation_size": 0.2,
        "folds": 3,
        "base_year": None,
        "random_state": 42,
        "fast_mode": True,
        "resource_mode": drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        "grid_limit": 30,
        "adwin_delta": 0.002,
        "min_window": 45000,
        "min_retrain_size": 4500,
        "arf_variants": list(drift_app.ARF_DEFAULT_VARIANTS),
        "kswin_variants": list(drift_app.KSWIN_DEFAULT_VARIANTS),
        "kswin_top_k_features": 2,
        "kswin_vote_threshold": 2,
        "kswin_retrain_days": 30,
        "kswin_min_retrain_rows": 50,
        "repetition_seeds": (42,),
        "balance_modes": [BALANCE_MODE_SMOTE],
        "feature_selection_context": {"selected_features": ["flow_light", "x1"]},
        "custom_grids": {},
        "continue_on_block_error": False,
    }
    run_manifest = {
        "models": ["NNet", "Random Forest"],
        "strategies": ["static", "adaptive_kswin"],
        "arf_variants": ["ARFmoderate"],
        "kswin_variants": ["KSWINpaper"],
        "balance_modes": [BALANCE_MODE_NONE],
        "repetition_seeds": [11, 13],
        "validation_size": 0.35,
        "folds": 5,
        "base_year": 2018,
        "random_state_fallback": 99,
        "fast_mode": False,
        "resource_mode": drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY,
        "optuna_trials": 17,
        "adwin_delta": 0.003,
        "min_window": 1234,
        "min_retrain_size": 321,
        "kswin_top_k_features": 9,
        "kswin_vote_threshold": 9,
        "kswin_retrain_days": 45,
        "kswin_min_retrain_rows": 222,
        "target_col": "target",
        "time_col": "interval_start",
        "continue_on_block_error": True,
        "custom_grids": {"NNet": {"hidden_layer_sizes": [[32, 16]]}},
    }
    feature_selection_context = {
        "selected_features": ["x1", "x2", "missing_feature"],
        "feature_export_path": "/tmp/checkpoint_features.duckdb",
    }

    execution_config = drift_app._execution_config_from_checkpoint_run_manifest(
        run_manifest,
        available_columns=["interval_start", "target", "x1", "x2"],
        current_config=current_config,
        feature_selection_context=feature_selection_context,
    )

    assert execution_config["feature_cols"] == ["x1", "x2"]
    assert execution_config["model_names"] == ["NNet", "Random Forest"]
    assert execution_config["strategies"] == ["static", "adaptive_kswin"]
    assert execution_config["balance_modes"] == [BALANCE_MODE_NONE]
    assert execution_config["repetition_seeds"] == (11, 13)
    assert execution_config["validation_size"] == pytest.approx(0.35)
    assert execution_config["folds"] == 5
    assert execution_config["base_year"] == 2018
    assert execution_config["random_state"] == 99
    assert execution_config["fast_mode"] is False
    assert execution_config["resource_mode"] == drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY
    assert execution_config["grid_limit"] == 17
    assert execution_config["custom_grids"] == {"NNet": {"hidden_layer_sizes": [[32, 16]]}}
    assert execution_config["continue_on_block_error"] is True


def test_execution_config_from_checkpoint_run_manifest_restores_restricted_resource_overrides():
    current_config = {
        "feature_cols": ["x1"],
        "target_col": "target",
        "time_col": "interval_start",
        "model_names": ["Random Forest"],
        "strategies": ["static"],
        "validation_size": 0.2,
        "folds": 3,
        "base_year": None,
        "random_state": 42,
        "fast_mode": True,
        "resource_mode": drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        "resource_policy_overrides": {},
        "grid_limit": 30,
        "adwin_delta": 0.002,
        "min_window": 45000,
        "min_retrain_size": 4500,
        "arf_variants": list(drift_app.ARF_DEFAULT_VARIANTS),
        "kswin_variants": list(drift_app.KSWIN_DEFAULT_VARIANTS),
        "kswin_top_k_features": 1,
        "kswin_vote_threshold": 1,
        "kswin_retrain_days": 30,
        "kswin_min_retrain_rows": 50,
        "repetition_seeds": (42,),
        "balance_modes": [BALANCE_MODE_NONE],
        "feature_selection_context": {"selected_features": ["x1"]},
        "custom_grids": {},
        "continue_on_block_error": False,
    }
    run_manifest = {
        "resource_mode": drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        "resource_policy": {
            "estimator_n_jobs": {
                "Random Forest": 1,
                "XGBoost": 2,
            },
            "parallel_cv_models": [],
        },
    }

    execution_config = drift_app._execution_config_from_checkpoint_run_manifest(
        run_manifest,
        available_columns=["interval_start", "target", "x1"],
        current_config=current_config,
    )

    assert execution_config["resource_mode"] == drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED
    assert execution_config["resource_policy_overrides"] == {
        "estimator_n_jobs": {
            "Random Forest": 1,
            "XGBoost": 2,
        },
        "parallel_cv_enabled": False,
    }


def test_load_checkpoint_feature_bundle_resolves_windows_export_path(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)
    feature_path = tmp_path / "checkpoint_features.duckdb"
    raw_df = pd.DataFrame(
        {
            "interval_start": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "portico": ["15", "15"],
            "target": [0, 1],
            "x1": [1.0, 2.0],
        }
    )
    clean_df = raw_df.copy()
    drift_app._write_feature_payload_to_duckdb(feature_path, raw_df=raw_df, clean_df=clean_df)

    bundle = drift_app._load_checkpoint_feature_bundle(
        run_manifest={
            "feature_selection_context": {
                "selected_features": ["x1"],
                "feature_export_path": r"C:\Users\usuario\Resultados\checkpoint_features.duckdb",
            }
        }
    )

    assert bundle is not None
    assert bundle["resolved_path"] == str(feature_path.resolve())
    assert pd.api.types.is_datetime64_any_dtype(bundle["clean_df"]["interval_start"])
    assert bundle["clean_df"]["target"].tolist() == [0, 1]


def test_cv_auc_parallel_uses_sklearn_parallel_without_warning():
    rows = 40
    X = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, rows),
            "x2": np.tile([0.0, 1.0], rows // 2),
        }
    )
    y = pd.Series(([0, 1] * (rows // 2)), name="target")
    params = {"iter": 5, "maxdepth": 1, "nu": 0.1}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        auc = drift_app._cv_auc(
            X,
            y,
            model_name="AdaBoost",
            params=params,
            folds=4,
            random_state=11,
            balance_mode=BALANCE_MODE_NONE,
        )

    assert np.isfinite(float(auc))
    assert not any(
        "sklearn.utils.parallel.delayed" in str(item.message)
        for item in caught
    )


def test_streamlit_arrow_safe_df_casts_mixed_object_columns_to_string():
    df = pd.DataFrame(
        {
            "training_year": [2018, "[<= 2019]", None],
            "metric": ["auc", "auc", "auc"],
            "value": [0.81, 0.79, 0.77],
        }
    )

    safe_df = drift_app._streamlit_arrow_safe_df(df)

    assert str(safe_df["training_year"].dtype) == "string"
    assert safe_df["training_year"].tolist()[:2] == ["2018", "[<= 2019]"]


def test_build_experiment_evolution_records_supports_yearly_and_adaptive():
    yearly_df = pd.DataFrame(
        [
            {
                "strategy": "static",
                "model": "Random Forest",
                "balance_mode": BALANCE_MODE_NONE,
                "prediction_year": 2019,
                "auc": 0.71,
                "threshold": 0.42,
                "run_seed": 11,
                "run_order": 1,
            }
        ]
    )
    adaptive_df = pd.DataFrame(
        [
            {
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": BALANCE_MODE_NONE,
                "drift": 1,
                "drift_date": "2019-06-01T00:00:00",
                "auc": 0.69,
                "remaining_periods": 12,
                "run_seed": 11,
                "run_order": 1,
            }
        ]
    )

    yearly_records = drift_app._build_experiment_evolution_records(
        yearly_df,
        source="yearly",
        metric_candidates=drift_app.YEARLY_EVOLUTION_METRICS,
    )
    adaptive_records = drift_app._build_experiment_evolution_records(
        adaptive_df,
        source="adaptive",
        metric_candidates=drift_app.ADAPTIVE_EVOLUTION_METRICS,
    )

    assert not yearly_records.empty
    assert not adaptive_records.empty
    assert {"metric", "value", "event_label", "event_order"} <= set(yearly_records.columns)
    assert {"metric", "value", "event_label", "event_order"} <= set(adaptive_records.columns)
    assert set(yearly_records["metric"]) >= {"auc", "threshold"}
    assert set(adaptive_records["metric"]) >= {"auc", "remaining_periods"}


def test_train_model_with_internal_validation_refits_final_model_on_full_training_rows(monkeypatch):
    fit_sizes = []

    class RecordingModel:
        def fit(self, X, y):
            fit_sizes.append(len(X))
            return self

        def predict_proba(self, X):
            values = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0.0).astype(float).to_numpy()
            scores = 1.0 / (1.0 + np.exp(-values))
            return np.column_stack([1.0 - scores, scores])

    def fake_tune_hyperparameters(*args, **kwargs):
        return (
            {"splitrule": "gini", "mtry": 1, "min_node_size": 1},
            pd.DataFrame(),
            {"study_id": "unit-test-study"},
        )

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(
        drift_app,
        "_build_model",
        lambda *args, **kwargs: RecordingModel(),
    )

    train_df = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )

    bundle = train_model_with_internal_validation(
        train_df,
        feature_cols=["x"],
        target_col="target",
        model_name="Random Forest",
        validation_size=0.33,
        folds=2,
        random_state=7,
        fast_mode=True,
        grid_limit=1,
    )

    assert fit_sizes[-1] == len(train_df)
    assert len(fit_sizes) >= 3
    assert 0.0 <= float(bundle["threshold"]) <= 1.0
    assert bundle["operating_threshold"] == pytest.approx(bundle["threshold"])
    assert bundle["calibration_method"] == drift_app.DEFAULT_CALIBRATION_METHOD
    assert bundle["threshold_policy"] == drift_app.DEFAULT_THRESHOLD_POLICY
    assert bundle["fn_cost"] == pytest.approx(drift_app.DEFAULT_THRESHOLD_FN_COST)
    assert bundle["fp_cost"] == pytest.approx(drift_app.DEFAULT_THRESHOLD_FP_COST)
    assert "calibration_metadata" in bundle
    assert "raw_youden_threshold" in bundle
    assert "calibrated_youden_threshold" in bundle
    assert "val_metrics_before_calibration" in bundle
    assert "val_metrics_after_calibration" in bundle


def test_stabilize_probability_calibration_rejects_negative_platt_slope():
    y_true = np.asarray([0, 0, 1, 1], dtype=int)
    raw_scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=float)
    raw_youden = youden_threshold(y_true, raw_scores)

    stabilized_scores, stabilized_youden, metadata, calibration_model = drift_app._stabilize_probability_calibration(
        y_true,
        raw_scores,
        raw_youden=raw_youden,
        calibrated_scores=np.asarray([0.52, 0.51, 0.50, 0.49], dtype=float),
        calibration_metadata={
            "method": drift_app.DEFAULT_CALIBRATION_METHOD,
            "kind": "platt",
            "applied": True,
            "coef": -0.1,
            "intercept": 0.0,
        },
        calibration_model=object(),
    )

    assert np.allclose(stabilized_scores, raw_scores)
    assert stabilized_youden["threshold"] == pytest.approx(raw_youden["threshold"])
    assert metadata["kind"] == "identity"
    assert metadata["reason"] == "rejected_non_positive_platt_slope"
    assert calibration_model is None


def test_stabilize_probability_calibration_rejects_worse_youden_than_raw():
    y_true = np.asarray([0, 0, 1, 1], dtype=int)
    raw_scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=float)
    raw_youden = youden_threshold(y_true, raw_scores)

    stabilized_scores, stabilized_youden, metadata, calibration_model = drift_app._stabilize_probability_calibration(
        y_true,
        raw_scores,
        raw_youden=raw_youden,
        calibrated_scores=np.asarray([0.48, 0.51, 0.49, 0.52], dtype=float),
        calibration_metadata={
            "method": drift_app.DEFAULT_CALIBRATION_METHOD,
            "kind": "platt",
            "applied": True,
            "coef": 0.1,
            "intercept": 0.0,
        },
        calibration_model=object(),
    )

    assert np.allclose(stabilized_scores, raw_scores)
    assert stabilized_youden["threshold"] == pytest.approx(raw_youden["threshold"])
    assert metadata["kind"] == "identity"
    assert metadata["reason"] == "rejected_worse_youden_than_raw"
    assert calibration_model is None


def test_train_model_with_internal_validation_rejects_degenerate_platt_calibration(monkeypatch):
    class RecordingModel:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            values = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0.0).astype(float).to_numpy()
            scores = 1.0 / (1.0 + np.exp(-values))
            return np.column_stack([1.0 - scores, scores])

    def fake_tune_hyperparameters(*args, **kwargs):
        return (
            {"splitrule": "gini", "mtry": 1, "min_node_size": 1},
            pd.DataFrame(),
            {"study_id": "unit-test-study"},
        )

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "_build_model", lambda *args, **kwargs: RecordingModel())
    monkeypatch.setattr(
        drift_app,
        "_cross_validated_scores",
        lambda *args, **kwargs: np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=float),
    )
    monkeypatch.setattr(
        drift_app,
        "_fit_platt_calibrator",
        lambda *args, **kwargs: (
            np.asarray([0.52, 0.51, 0.50, 0.49, 0.48, 0.47], dtype=float),
            {
                "method": drift_app.DEFAULT_CALIBRATION_METHOD,
                "kind": "platt",
                "applied": True,
                "coef": -0.05,
                "intercept": 0.0,
            },
            object(),
        ),
    )

    train_df = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )

    bundle = train_model_with_internal_validation(
        train_df,
        feature_cols=["x"],
        target_col="target",
        model_name="Random Forest",
        validation_size=0.33,
        folds=2,
        random_state=7,
        fast_mode=True,
        grid_limit=1,
    )

    assert bundle["threshold"] == pytest.approx(bundle["raw_youden_threshold"])
    assert bundle["calibration_model"] is None
    assert bundle["calibration_metadata"]["kind"] == "identity"
    assert bundle["calibration_metadata"]["reason"] == "rejected_non_positive_platt_slope"
    assert bundle["val_metrics_before_calibration"] == bundle["val_metrics_after_calibration"]


def test_fit_nnet_warning_safe_suppresses_convergence_warning():
    class WarningModel:
        n_iter_ = 200

        def fit(self, X, y):
            warnings.warn(
                "Maximum iterations (200) reached and the optimization hasn't converged yet.",
                ConvergenceWarning,
            )
            return self

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = drift_app._fit_nnet_warning_safe(
            WarningModel(),
            pd.DataFrame({"x": [0.0, 1.0]}),
            np.asarray([0, 1]),
        )

    assert diagnostics["converged"] is False
    assert diagnostics["fit_warning_count"] == 1
    assert diagnostics["n_iter_final"] == 200
    assert not any(issubclass(item.category, ConvergenceWarning) for item in caught)


def test_fit_model_with_diagnostics_retries_nnet_once(monkeypatch):
    built_params = []

    class DummyModel:
        pass

    diagnostics_sequence = iter(
        [
            {
                "converged": False,
                "fit_warning_count": 1,
                "n_iter_final": 300,
                "warning_messages": ["first"],
            },
            {
                "converged": True,
                "fit_warning_count": 0,
                "n_iter_final": 600,
                "warning_messages": [],
            },
        ]
    )

    def fake_build_model(model_name, params, **kwargs):
        built_params.append(dict(params))
        return DummyModel()

    monkeypatch.setattr(drift_app, "_build_model", fake_build_model)
    monkeypatch.setattr(
        drift_app,
        "_fit_nnet_warning_safe",
        lambda model, X, y: dict(next(diagnostics_sequence)),
    )

    _model, diagnostics, params_used = drift_app._fit_model_with_diagnostics(
        "NNet",
        {"size": 5, "decay": 0.001, "maxit": 300},
        random_state=7,
        n_features=1,
        X_fit=pd.DataFrame({"x": [0.0, 1.0]}),
        y_fit=np.asarray([0, 1]),
        allow_retry=True,
    )

    assert built_params[0]["maxit"] == 300
    assert built_params[1]["maxit"] == 600
    assert diagnostics["fit_retry_applied"] is True
    assert diagnostics["converged"] is True
    assert params_used["maxit"] == 600


def test_train_model_with_internal_validation_builds_feature_transform_for_nnet(monkeypatch):
    class PredictiveModel:
        def predict_proba(self, X):
            values = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0.0).astype(float).to_numpy()
            scores = 1.0 / (1.0 + np.exp(-values))
            return np.column_stack([1.0 - scores, scores])

    def fake_tune_hyperparameters(*args, **kwargs):
        params = {"size": 5, "decay": 0.001, "maxit": 300}
        return params, pd.DataFrame(), {"study_id": "nnet-study", "has_valid_trial": True, "trials": []}

    def fake_cross_validated_scores(*args, **kwargs):
        scores = np.linspace(0.2, 0.8, 8, dtype=float)
        return {
            "scores": scores,
            "converged": True,
            "n_non_converged_folds": 0,
            "n_valid_folds": 2,
            "fold_diagnostics": [],
        }

    def fake_fit_model_with_diagnostics(*args, **kwargs):
        params = dict(kwargs.get("model_params", {}) or args[1])
        return (
            PredictiveModel(),
            {
                "converged": True,
                "fit_warning_count": 0,
                "n_iter_final": int(params.get("maxit", 300)),
                "fit_retry_applied": False,
            },
            params,
        )

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "_cross_validated_scores", fake_cross_validated_scores)
    monkeypatch.setattr(drift_app, "_fit_model_with_diagnostics", fake_fit_model_with_diagnostics)

    train_df = pd.DataFrame(
        {
            "x1": [10.0, 11.0, 9.5, 12.0, -9.0, -10.0, -11.0, -12.0],
            "x2": [100.0, 99.0, 98.0, 101.0, -95.0, -96.0, -97.0, -98.0],
            "target": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )

    bundle = train_model_with_internal_validation(
        train_df,
        feature_cols=["x1", "x2"],
        target_col="target",
        model_name="NNet",
        validation_size=0.25,
        folds=2,
        random_state=5,
        fast_mode=True,
        grid_limit=1,
    )

    assert bundle["feature_transform"]["kind"] == "standard_scaler"
    assert bundle["feature_transform"]["policy_version"] == drift_app.NNET_TRAINING_POLICY_VERSION
    assert len(bundle["feature_transform"]["columns"]) == 2
    assert bundle["calibration_method"] == drift_app.DEFAULT_CALIBRATION_METHOD
    assert bundle["operating_threshold"] == pytest.approx(bundle["threshold"])
    assert bundle["calibrated_youden_threshold"] == pytest.approx(bundle["threshold"])

    eval_payload = _evaluate_split(
        bundle["model"],
        train_df,
        feature_cols=["x1", "x2"],
        target_col="target",
        threshold=float(bundle["threshold"]),
        raw_threshold=float(bundle["raw_youden_threshold"]),
        fill_values=bundle.get("fill_values"),
        feature_transform=bundle.get("feature_transform"),
        calibration_model=bundle.get("calibration_model"),
        calibration_metadata=bundle.get("calibration_metadata"),
    )

    assert len(eval_payload["scores"]) == len(train_df)
    assert len(eval_payload["calibrated_scores"]) == len(train_df)
    assert "metrics_before_calibration" in eval_payload
    assert "metrics_after_calibration" in eval_payload


def test_tune_hyperparameters_caps_adaboost_trials_to_search_space():
    pytest.importorskip("optuna")

    train_df = pd.DataFrame(
        {
            "x1": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0] * 5,
            "x2": [1.0, 0.5, 0.2, 0.1, -0.1, -0.2, -0.5, -1.0] * 5,
            "target": [0, 0, 0, 0, 1, 1, 1, 1] * 5,
        }
    )
    X, y = drift_app._prepare_xy_raw(train_df, ["x1", "x2"], "target")

    _, search_df, artifact = drift_app.tune_hyperparameters(
        X,
        y,
        model_name="AdaBoost",
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=40,
        balance_mode=BALANCE_MODE_NONE,
    )

    assert artifact["requested_trials"] == 40
    assert artifact["search_space_size"] == 8
    assert artifact["n_trials"] == 8
    assert len(search_df) == 8


def test_cross_validated_scores_without_diagnostics_skips_fold_serialization(monkeypatch):
    train_df = pd.DataFrame(
        {
            "x1": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0] * 5,
            "x2": [1.0, 0.5, 0.2, 0.1, -0.1, -0.2, -0.5, -1.0] * 5,
            "target": [0, 0, 0, 0, 1, 1, 1, 1] * 5,
        }
    )
    X, y = drift_app._prepare_xy_raw(train_df, ["x1", "x2"], "target")
    params = drift_app._default_search_params(drift_app.FAST_HYPERPARAM_GRIDS["AdaBoost"])

    def fail_json_safe(_value):
        raise AssertionError("Fold diagnostics should not be serialized when only scores are requested.")

    monkeypatch.setattr(drift_app, "_to_json_safe", fail_json_safe)

    scores = drift_app._cross_validated_scores(
        X,
        y,
        model_name="AdaBoost",
        params=params,
        folds=2,
        random_state=11,
        balance_mode=BALANCE_MODE_NONE,
        return_diagnostics=False,
    )

    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(X)
    assert np.isfinite(scores).sum() > 0
    assert np.isnan(scores).sum() > 0


def test_temporal_cv_splits_use_past_for_training_only():
    y = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], name="target")

    splits = drift_app._temporal_cv_splits(y, folds=3)

    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert int(np.max(train_idx)) < int(np.min(val_idx))


def test_recalibration_experiments_skip_nnet_blocks_without_valid_tuning_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019), rows_per_year=60, random_state=101)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    def fake_tune_hyperparameters(X, y, **kwargs):
        params = {"size": 5, "decay": 0.001, "maxit": 300}
        search_df = pd.DataFrame(
            [
                {
                    "trial_number": 0,
                    "model": "NNet",
                    "cv_auc": np.nan,
                    "params": json.dumps(params, sort_keys=True),
                    "state": "COMPLETE",
                    "balance_mode": kwargs["balance_mode"],
                    "converged": False,
                    "n_non_converged_folds": 2,
                    "n_valid_folds": 0,
                }
            ]
        )
        artifact = {
            "study_id": "nnet-invalid",
            "model_name": "NNet",
            "balance_mode": kwargs["balance_mode"],
            "best_value": None,
            "best_params": params,
            "model_params": params,
            "smote_params": {},
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": {"maxit": [300]},
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": None,
                    "params": params,
                    "model_params": params,
                    "smote_params": {},
                    "converged": False,
                    "n_non_converged_folds": 2,
                    "n_valid_folds": 0,
                }
            ],
            "has_valid_trial": False,
            "invalid_trial_count": 1,
            "policy_version": drift_app.NNET_TRAINING_POLICY_VERSION,
        }
        return params, search_df, artifact

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(
        drift_app,
        "run_yearly_strategy",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Yearly strategy should be skipped for invalid NNet tuning")),
    )

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["NNet"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE,),
        checkpoint_root=tmp_path / "runs",
    )

    assert outputs["yearly_results"].empty
    assert outputs["adaptive_results"].empty
    assert not outputs["execution_log"].empty
    assert outputs["execution_log"]["phase"].eq("nnet_trial_invalid").any()
    assert len(outputs["checkpoint_manifest"]["completed_block_ids"]) == 1


def test_nnet_policy_version_changes_run_and_tuning_keys(monkeypatch):
    df = generate_synthetic_article_dataset(years=(2018, 2019), rows_per_year=40, random_state=131)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]
    feature_selection_context = {"selected_features": features}
    canonical_train_df = df.loc[pd.to_datetime(df["interval_start"], errors="coerce").dt.year.eq(2018)].copy()

    run_id_before = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["NNet"],
        strategies=["static"],
        arf_variants=[],
        kswin_variants=[],
        balance_modes=[BALANCE_MODE_NONE],
        repetition_seeds=[42],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=30,
        adwin_delta=0.002,
        min_window=45_000,
        min_retrain_size=None,
        kswin_top_k_features=10,
        kswin_vote_threshold=2,
        kswin_retrain_days=90,
        kswin_min_retrain_rows=100,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )
    tuning_key_before = drift_app._build_global_tuning_key(
        model_name="NNet",
        balance_mode=BALANCE_MODE_NONE,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        canonical_train_df=canonical_train_df,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=30,
        custom_grid=None,
    )

    monkeypatch.setattr(drift_app, "NNET_TRAINING_POLICY_VERSION", "scaled_early_stopping_v2")

    run_id_after = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["NNet"],
        strategies=["static"],
        arf_variants=[],
        kswin_variants=[],
        balance_modes=[BALANCE_MODE_NONE],
        repetition_seeds=[42],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=30,
        adwin_delta=0.002,
        min_window=45_000,
        min_retrain_size=None,
        kswin_top_k_features=10,
        kswin_vote_threshold=2,
        kswin_retrain_days=90,
        kswin_min_retrain_rows=100,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )
    tuning_key_after = drift_app._build_global_tuning_key(
        model_name="NNet",
        balance_mode=BALANCE_MODE_NONE,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        canonical_train_df=canonical_train_df,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=30,
        custom_grid=None,
    )

    assert run_id_before != run_id_after
    assert tuning_key_before != tuning_key_after


def test_resource_mode_changes_run_and_tuning_keys():
    df = generate_synthetic_article_dataset(years=(2018, 2019), rows_per_year=40, random_state=211)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]
    feature_selection_context = {"selected_features": features}
    canonical_train_df = df.loc[pd.to_datetime(df["interval_start"], errors="coerce").dt.year.eq(2018)].copy()

    restricted_run_id = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        arf_variants=[],
        kswin_variants=[],
        balance_modes=[BALANCE_MODE_NONE],
        repetition_seeds=[42],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        grid_limit=30,
        adwin_delta=0.002,
        min_window=45_000,
        min_retrain_size=None,
        kswin_top_k_features=10,
        kswin_vote_threshold=2,
        kswin_retrain_days=90,
        kswin_min_retrain_rows=100,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )
    full_run_id = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        arf_variants=[],
        kswin_variants=[],
        balance_modes=[BALANCE_MODE_NONE],
        repetition_seeds=[42],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY,
        grid_limit=30,
        adwin_delta=0.002,
        min_window=45_000,
        min_retrain_size=None,
        kswin_top_k_features=10,
        kswin_vote_threshold=2,
        kswin_retrain_days=90,
        kswin_min_retrain_rows=100,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )

    restricted_tuning_key = drift_app._build_global_tuning_key(
        model_name="Random Forest",
        balance_mode=BALANCE_MODE_NONE,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        canonical_train_df=canonical_train_df,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        grid_limit=30,
        custom_grid=None,
    )
    full_tuning_key = drift_app._build_global_tuning_key(
        model_name="Random Forest",
        balance_mode=BALANCE_MODE_NONE,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        canonical_train_df=canonical_train_df,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_FULL_MEMORY,
        grid_limit=30,
        custom_grid=None,
    )

    assert restricted_run_id != full_run_id
    assert restricted_tuning_key != full_tuning_key


def test_restricted_resource_overrides_change_run_and_tuning_keys():
    df = generate_synthetic_article_dataset(years=(2018, 2019), rows_per_year=40, random_state=211)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]
    feature_selection_context = {"selected_features": features}
    canonical_train_df = df.loc[pd.to_datetime(df["interval_start"], errors="coerce").dt.year.eq(2018)].copy()
    custom_overrides = {
        "estimator_n_jobs": {
            "Random Forest": 1,
            "XGBoost": 2,
        },
        "parallel_cv_enabled": False,
    }

    default_run_id = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        arf_variants=[],
        kswin_variants=[],
        balance_modes=[BALANCE_MODE_NONE],
        repetition_seeds=[42],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        grid_limit=30,
        adwin_delta=0.002,
        min_window=45_000,
        min_retrain_size=None,
        kswin_top_k_features=10,
        kswin_vote_threshold=2,
        kswin_retrain_days=90,
        kswin_min_retrain_rows=100,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
    )
    custom_run_id = drift_app._build_recalibration_run_id(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        arf_variants=[],
        kswin_variants=[],
        balance_modes=[BALANCE_MODE_NONE],
        repetition_seeds=[42],
        base_year=2018,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        grid_limit=30,
        adwin_delta=0.002,
        min_window=45_000,
        min_retrain_size=None,
        kswin_top_k_features=10,
        kswin_vote_threshold=2,
        kswin_retrain_days=90,
        kswin_min_retrain_rows=100,
        custom_grids=None,
        feature_selection_context=feature_selection_context,
        resource_policy_overrides=custom_overrides,
    )

    default_tuning_key = drift_app._build_global_tuning_key(
        model_name="Random Forest",
        balance_mode=BALANCE_MODE_NONE,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        canonical_train_df=canonical_train_df,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        grid_limit=30,
        custom_grid=None,
    )
    custom_tuning_key = drift_app._build_global_tuning_key(
        model_name="Random Forest",
        balance_mode=BALANCE_MODE_NONE,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        canonical_train_df=canonical_train_df,
        validation_size=0.2,
        folds=3,
        random_state=42,
        fast_mode=True,
        resource_mode=drift_app.EXPERIMENT_RESOURCE_MODE_RESTRICTED,
        grid_limit=30,
        custom_grid=None,
        resource_policy_overrides=custom_overrides,
    )

    assert default_run_id != custom_run_id
    assert default_tuning_key != custom_tuning_key


def test_batch_tuning_tasks_scope_by_window_and_dedupe_identical_train_signatures():
    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=40, random_state=223)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    tasks = drift_app._batch_tuning_tasks(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static", "period_aligned", "cumulative", drift_app.ADAPTIVE_ADWIN_STRATEGY],
        balance_modes=[BALANCE_MODE_NONE],
        base_year=2018,
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=1,
        custom_grids=None,
    )

    assert len(tasks) == 3
    assert len({task["tuning_key"] for task in tasks}) == 3
    assert any(task["window_kind"] == "base_year" for task in tasks)
    assert any(task["window_kind"] == "period_aligned" and task["training_year_label"] == "2019" for task in tasks)
    assert any(task["window_kind"] == "cumulative" and task["prediction_year"] == 2020 for task in tasks)


def test_load_or_create_smote_artifact_reuses_exact_signature(tmp_path, monkeypatch):
    pytest.importorskip("imblearn")

    train_df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2018-01-01 00:00:00", periods=8, freq="5min"),
            "x1": [0.0, 0.2, 0.1, -0.1, 1.0, 1.2, 1.1, 0.9],
            "x2": [1.0, 0.9, 1.1, 1.2, -0.2, -0.1, -0.3, -0.4],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    X = train_df[["x1", "x2"]].copy()
    y = train_df["target"].astype(int).to_numpy()

    X_first, y_first, fit_info_first = drift_app._load_or_create_smote_artifact(
        run_id="run_unit",
        train_df=train_df,
        X=X,
        y=y,
        feature_cols=["x1", "x2"],
        target_col="target",
        time_col="interval_start",
        sampling_strategy=1.0,
        k_neighbors=3,
        random_state=11,
        smote_cache_dir=tmp_path,
        smote_cache_registry={},
    )

    monkeypatch.setattr(
        drift_app,
        "_apply_smote_balance",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SMOTE should be loaded from cache")),
    )
    X_second, y_second, fit_info_second = drift_app._load_or_create_smote_artifact(
        run_id="run_unit",
        train_df=train_df,
        X=X,
        y=y,
        feature_cols=["x1", "x2"],
        target_col="target",
        time_col="interval_start",
        sampling_strategy=1.0,
        k_neighbors=3,
        random_state=11,
        smote_cache_dir=tmp_path,
        smote_cache_registry={},
    )

    assert fit_info_first["artifact_key"] == fit_info_second["artifact_key"]
    assert fit_info_second["artifact_reused"] is True
    assert X_first.equals(X_second)
    assert np.array_equal(y_first, y_second)


def test_train_model_with_internal_validation_smote_refits_on_full_training_data():
    pytest.importorskip("imblearn")

    df = generate_synthetic_article_dataset(years=(2018, 2019), rows_per_year=260, random_state=43)
    year_mask = pd.to_datetime(df["interval_start"], errors="coerce").dt.year.eq(2018)
    train_df = df.loc[year_mask].copy().reset_index(drop=True)
    test_df = df.loc[~year_mask].copy().reset_index(drop=True)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    positive_idx = train_df.index[train_df["target"].astype(int).eq(1)].tolist()
    kept_positive_idx = set(positive_idx[: max(10, len(positive_idx) // 5)])
    train_df.loc[
        train_df.index.isin([idx for idx in positive_idx if idx not in kept_positive_idx]),
        "target",
    ] = 0

    bundle = train_model_with_internal_validation(
        train_df,
        feature_cols=features,
        target_col="target",
        model_name="Random Forest",
        validation_size=0.2,
        folds=2,
        random_state=13,
        fast_mode=True,
        grid_limit=1,
        custom_grid={
            "mtry": [2],
            "splitrule": ["gini"],
            "min_node_size": [1],
            "sampling_strategy": [0.1],
            "k_neighbors": [9],
        },
        balance_mode=BALANCE_MODE_SMOTE,
    )

    assert bundle["balance_mode"] == BALANCE_MODE_SMOTE
    assert bundle["smote_fit_info"]["applied"] is True
    assert bundle["smote_fit_info"]["train_rows"] == len(train_df)
    assert bundle["smote_fit_info"]["balanced_rows"] > bundle["smote_fit_info"]["train_rows"]
    assert bundle["smote_params"]
    assert bundle["tuning_artifact"]["smote_params"] == bundle["smote_params"]
    assert bundle["search_df"]["balance_mode"].eq(BALANCE_MODE_SMOTE).all()
    assert bundle["calibration_method"] == drift_app.DEFAULT_CALIBRATION_METHOD
    assert bundle["threshold_policy"] == drift_app.DEFAULT_THRESHOLD_POLICY

    eval_payload = _evaluate_split(
        bundle["model"],
        test_df,
        feature_cols=features,
        target_col="target",
        threshold=float(bundle["threshold"]),
        raw_threshold=float(bundle["raw_youden_threshold"]),
        fill_values=bundle.get("fill_values"),
        feature_transform=bundle.get("feature_transform"),
        calibration_model=bundle.get("calibration_model"),
        calibration_metadata=bundle.get("calibration_metadata"),
    )

    assert len(eval_payload["y_true"]) == len(test_df)
    assert len(eval_payload["scores"]) == len(test_df)
    assert len(eval_payload["calibrated_scores"]) == len(test_df)
    assert "metrics_before_calibration" in eval_payload
    assert "metrics_after_calibration" in eval_payload


def test_train_arf_with_internal_validation_uses_warm_block_fill_values(monkeypatch):
    class RecordingOnlineModel:
        def __init__(self):
            self.prediction_inputs = []

        def predict_proba_one(self, x_dict):
            self.prediction_inputs.append(dict(x_dict))
            return {1: 1.0 if float(x_dict["x"]) > 0 else 0.0}

        def learn_one(self, x_dict, y):
            return self

    model = RecordingOnlineModel()
    monkeypatch.setattr(
        drift_app,
        "_build_arf_classifier",
        lambda variant_name, random_state: model,
    )

    train_df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2018-01-01 00:00:00", periods=5, freq="5min"),
            "x": [0.0, 0.0, 100.0, np.nan, 100.0],
            "target": [0, 1, 0, 0, 1],
        }
    )

    bundle = drift_app.train_arf_with_internal_validation(
        train_df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        variant_name=ARF_VARIANT_NAMES[0],
        validation_size=0.4,
        random_state=5,
    )

    assert bundle["fill_values"]["x"] == pytest.approx(0.0)
    assert model.prediction_inputs[0]["x"] == pytest.approx(0.0)


def test_run_arf_strategy_uses_base_fill_values_for_stream(monkeypatch):
    class RecordingOnlineModel:
        n_drifts_detected = 0
        n_warnings_detected = 0

        def __init__(self):
            self.prediction_inputs = []

        def predict_proba_one(self, x_dict):
            self.prediction_inputs.append(dict(x_dict))
            return {1: 0.5}

        def learn_one(self, x_dict, y):
            return self

    model = RecordingOnlineModel()

    def fake_train_arf_with_internal_validation(*args, **kwargs):
        return {
            "model": model,
            "threshold": 0.5,
            "youden": {"threshold": 0.5},
            "val_metrics": {},
            "best_params": {"variant": ARF_VARIANT_NAMES[0]},
            "fill_values": {"x": 7.0},
            "warm_rows": 2,
            "validation_rows": 1,
        }

    monkeypatch.setattr(
        drift_app,
        "train_arf_with_internal_validation",
        fake_train_arf_with_internal_validation,
    )

    df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 00:05:00",
                    "2019-01-01 00:00:00",
                    "2019-01-01 00:05:00",
                ]
            ),
            "x": [0.0, 1.0, np.nan, 100.0],
            "target": [0, 1, 0, 1],
        }
    )

    adaptive_df, roc_payload = run_arf_strategy(
        df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        arf_variants=[ARF_VARIANT_NAMES[0]],
        validation_size=0.5,
        random_state=11,
    )

    assert not adaptive_df.empty
    assert len(roc_payload) == 1
    assert model.prediction_inputs[0]["x"] == pytest.approx(7.0)


def test_end_to_end_recalibration_and_average_roc():
    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=220, random_state=23)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static", "period_aligned", "cumulative", "adaptive_adwin"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=2,
        adwin_delta=0.01,
        min_window=100,
        repetition_seeds=(11,),
    )

    assert "yearly_results" in outputs
    assert "adaptive_results" in outputs
    assert "average_roc" in outputs
    assert "appendix_tables" in outputs
    assert "appendix_tables_mean" in outputs
    assert "execution_log" in outputs
    assert "run_manifest" in outputs

    roc_df = outputs["average_roc"]
    assert isinstance(roc_df, pd.DataFrame)
    assert not roc_df.empty
    assert outputs["summary"]["n_repetitions"].eq(1).all()
    assert outputs["run_manifest"]["repetition_seeds"] == [11]
    assert outputs["run_manifest"]["resource_mode"] == drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE
    assert not outputs["execution_log"].empty
    assert "run_seed" in outputs["yearly_results"].columns
    assert {"pr_auc", "brier_score", "bias2", "variance", "noise", "f1"} <= set(outputs["yearly_results"].columns)
    assert {"pr_auc", "brier_score", "bias2", "variance", "noise", "f1"} <= set(outputs["adaptive_results"].columns)
    assert {"pr_auc", "brier_score", "bias2", "variance", "noise", "f1"} <= set(outputs["appendix_tables"]["A.6"].columns)
    assert {"pr_auc", "brier_score", "bias2", "variance", "noise", "f1"} <= set(outputs["appendix_tables"]["A.9"].columns)

    # Also validate standalone ROC builder path.
    payload = [
        {
            "strategy": "static",
            "model": "Random Forest",
            "segment": "2019",
            "y_true": np.array([0, 0, 1, 1]),
            "scores": np.array([0.1, 0.4, 0.35, 0.8]),
        }
    ]
    roc2 = build_average_roc_curves(payload)
    assert not roc2.empty
    assert set(["strategy", "model", "fpr", "tpr", "label"]).issubset(set(roc2.columns))


def test_recalibration_experiments_compare_balance_modes_end_to_end():
    pytest.importorskip("imblearn")

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=180, random_state=67)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static", "adaptive_adwin"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=40,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE, BALANCE_MODE_SMOTE),
    )

    assert set(outputs["yearly_results"]["balance_mode"].unique()) == {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}
    assert set(outputs["adaptive_results"]["balance_mode"].unique()) == {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}
    assert set(outputs["summary"]["balance_mode"].unique()) == {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}
    assert outputs["run_manifest"]["balance_modes"] == [BALANCE_MODE_NONE, BALANCE_MODE_SMOTE]

    log_modes = set(outputs["execution_log"]["balance_mode"].dropna().astype(str).unique())
    assert {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}.issubset(log_modes)


def test_recalibration_experiments_emit_progress_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=160, random_state=31)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    progress_events = []
    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        progress_callback=progress_events.append,
    )

    assert not outputs["summary"].empty
    assert len(progress_events) >= 3
    assert progress_events[0]["completed_units"] == 0
    assert any(
        0 < float(event["completed_units"]) < float(event["total_units"])
        and abs(float(event["completed_units"]) - round(float(event["completed_units"]))) > 1e-9
        for event in progress_events[1:-1]
    )
    assert progress_events[-1]["completed_units"] == progress_events[-1]["total_units"] == 5


def test_recalibration_experiments_support_adaptive_arf(tmp_path, monkeypatch):
    pytest.importorskip("river")
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=180, random_state=37)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    progress_events = []
    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest", "NNet"],
        strategies=["adaptive_arf"],
        arf_variants=["ARFmoderate", "ARFfast"],
        validation_size=0.2,
        random_state=17,
        repetition_seeds=(17,),
        progress_callback=progress_events.append,
    )

    adaptive = outputs["adaptive_results"]
    summary = outputs["summary"]
    roc_df = outputs["average_roc"]

    assert outputs["yearly_results"].empty
    assert not adaptive.empty
    assert not summary.empty
    assert not roc_df.empty
    assert set(adaptive["model"].unique()) == {"ARFmoderate", "ARFfast"}
    assert set(summary["strategy"].unique()) == {"adaptive_arf"}
    assert outputs["run_manifest"]["arf_variants"] == ["ARFmoderate", "ARFfast"]
    assert ARF_VARIANT_NAMES[:2] == ["ARFmoderate", "ARFmaj"]
    assert progress_events[-1]["completed_units"] == progress_events[-1]["total_units"] == 2


def test_recalibration_experiments_support_adaptive_kswin(tmp_path, monkeypatch):
    pytest.importorskip("river")
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = _make_kswin_shift_dataset(rows_per_year=320, random_state=59)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    progress_events = []
    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["adaptive_kswin"],
        kswin_variants=["KSWINpaper", "KSWINseasonal"],
        validation_size=0.2,
        folds=2,
        random_state=19,
        fast_mode=True,
        grid_limit=1,
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=80,
        repetition_seeds=(19,),
        progress_callback=progress_events.append,
    )

    adaptive = outputs["adaptive_results"]
    summary = outputs["summary"]
    roc_df = outputs["average_roc"]

    assert outputs["yearly_results"].empty
    assert not adaptive.empty
    assert not summary.empty
    assert not roc_df.empty
    assert set(adaptive["detector_variant"].unique()) == {"KSWINpaper", "KSWINseasonal"}
    assert set(adaptive["balance_mode"].unique()) == {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}
    assert outputs["run_manifest"]["kswin_variants"] == ["KSWINpaper", "KSWINseasonal"]
    assert KSWIN_VARIANT_NAMES == ["KSWINpaper", "KSWINseasonal"]
    assert progress_events[-1]["completed_units"] == progress_events[-1]["total_units"] == 7


def test_recalibration_experiments_passes_model_and_balance_to_block_tuning_resolver(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = _make_kswin_shift_dataset(rows_per_year=120, random_state=73)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    resolver_calls = []

    def fake_resolve_or_create_tuning_artifact(**kwargs):
        resolver_calls.append(
            (
                str(kwargs.get("model_name")),
                str(kwargs.get("balance_mode")),
                str((kwargs.get("tuning_context") or {}).get("stage")),
            )
        )
        return {
            "tuning_key": "fake_tuning",
            "best_params": {},
            "has_valid_trial": True,
        }

    def fake_run_kswin_strategy(
        df,
        *,
        model_names,
        kswin_variants,
        balance_mode,
        tuning_resolver=None,
        run_seed=None,
        run_order=None,
        **kwargs,
    ):
        assert tuning_resolver is not None
        tuning_artifact = tuning_resolver(
            train_df=df.iloc[:40].copy(),
            model_name=str(model_names[0]),
            balance_mode=str(balance_mode),
            strategy_context={
                "stage": "kswin_base",
                "strategy": "adaptive_kswin",
            },
        )
        assert tuning_artifact["tuning_key"] == "fake_tuning"
        adaptive_df = pd.DataFrame(
            [
                {
                    "strategy": "adaptive_kswin",
                    "drift": 1,
                    "drift_date": pd.Timestamp("2020-01-15"),
                    "prediction_year": 2020,
                    "model": f"{model_names[0]} | {kswin_variants[0]}",
                    "base_model": str(model_names[0]),
                    "detector_variant": str(kswin_variants[0]),
                    "balance_mode": str(balance_mode),
                    "auc": 0.71,
                    "sensitivity": 0.63,
                    "specificity": 0.77,
                    "error_rate": 0.24,
                    "training_time_sec": 0.5,
                    "threshold": 0.4,
                    "segment_rows": 12,
                    "run_seed": int(run_seed or 0),
                    "run_order": int(run_order or 0),
                }
            ]
        )
        return adaptive_df, []

    monkeypatch.setattr(drift_app, "_resolve_or_create_tuning_artifact", fake_resolve_or_create_tuning_artifact)
    monkeypatch.setattr(drift_app, "run_kswin_strategy", fake_run_kswin_strategy)

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["adaptive_kswin"],
        kswin_variants=["KSWINpaper"],
        balance_modes=["none"],
        validation_size=0.2,
        folds=2,
        random_state=31,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(31,),
    )

    assert not outputs["adaptive_results"].empty
    assert ("Random Forest", "none", "kswin_base") in resolver_calls


def test_recalibration_experiments_passes_model_and_balance_to_yearly_block_tuning_resolver(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=74)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    resolver_calls = []

    def fake_resolve_or_create_tuning_artifact(**kwargs):
        resolver_calls.append(
            (
                str(kwargs.get("model_name")),
                str(kwargs.get("balance_mode")),
                str((kwargs.get("tuning_context") or {}).get("stage")),
            )
        )
        return {
            "tuning_key": "fake_tuning",
            "best_params": {},
            "has_valid_trial": True,
        }

    def fake_run_yearly_strategy(
        df,
        *,
        strategy,
        model_names,
        balance_mode,
        tuning_resolver=None,
        run_seed=None,
        run_order=None,
        **kwargs,
    ):
        assert tuning_resolver is not None
        tuning_artifact = tuning_resolver(
            train_df=df.iloc[:40].copy(),
            model_name=str(model_names[0]),
            balance_mode=str(balance_mode),
            strategy_context={
                "stage": "yearly_train",
                "strategy": str(strategy),
            },
        )
        assert tuning_artifact["tuning_key"] == "fake_tuning"
        yearly_df = pd.DataFrame(
            [
                {
                    "strategy": str(strategy),
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": str(model_names[0]),
                    "balance_mode": str(balance_mode),
                    "auc": 0.74,
                    "sensitivity": 0.61,
                    "specificity": 0.79,
                    "error_rate": 0.22,
                    "training_time_sec": 0.4,
                    "threshold": 0.45,
                    "n_train": 40,
                    "n_test": 20,
                    "run_seed": int(run_seed or 0),
                    "run_order": int(run_order or 0),
                }
            ]
        )
        return yearly_df, []

    monkeypatch.setattr(drift_app, "_resolve_or_create_tuning_artifact", fake_resolve_or_create_tuning_artifact)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        balance_modes=["none"],
        validation_size=0.2,
        folds=2,
        random_state=31,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(31,),
    )

    assert not outputs["yearly_results"].empty
    assert ("Random Forest", "none", "yearly_train") in resolver_calls


def test_recalibration_experiments_smote_rerank_reuses_window_tuning_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=75)
    features = ["x1", "x2"]
    feature_selection_context = {
        "selected_features": features,
        "candidate_features": ["x1", "x2", "x3"],
        "smote_rerank_enabled": True,
    }

    tune_feature_cols = []
    tune_balance_modes = []
    tune_fixed_model_params = []
    tune_fixed_smote_params = []
    tune_custom_grids = []
    tune_smote_search_spaces = []
    rerank_calls = []
    rerank_smote_params = []

    def fake_rerank_features_after_smote(train_df, **kwargs):
        rerank_calls.append(sorted(pd.to_datetime(train_df["interval_start"]).dt.year.unique().tolist()))
        rerank_smote_params.append(
            (
                float(kwargs.get("sampling_strategy", 0.0)),
                int(kwargs.get("k_neighbors", 0)),
            )
        )
        return (
            ["x3", "x1"],
            pd.DataFrame(
                {
                    "feature": ["x3", "x1"],
                    "importance": [0.9, 0.8],
                    "rank": [1, 2],
                }
            ),
        )

    def fake_tune_hyperparameters(X, y, **kwargs):
        tune_feature_cols.append(list(X.columns))
        tune_balance_modes.append(str(kwargs.get("balance_mode")))
        tune_fixed_model_params.append(dict(kwargs.get("fixed_model_params") or {}))
        tune_fixed_smote_params.append(dict(kwargs.get("fixed_smote_params") or {}))
        tune_custom_grids.append(dict(kwargs.get("custom_grid") or {}))
        tune_smote_search_spaces.append(dict(kwargs.get("smote_search_space") or {}))
        base_params = {"mtry": 2, "splitrule": "gini", "min_node_size": 1}
        balance_mode = str(kwargs.get("balance_mode"))
        fixed_model_params = dict(kwargs.get("fixed_model_params") or {})
        fixed_smote_params = dict(kwargs.get("fixed_smote_params") or {})
        if balance_mode == drift_app.BALANCE_MODE_NONE:
            params = dict(base_params)
            search_space = {"mtry": [2]}
            smote_params = {}
            model_params = dict(base_params)
        elif fixed_model_params:
            assert fixed_model_params == base_params
            assert dict(kwargs.get("smote_search_space") or {}) == dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE)
            params = {
                **dict(base_params),
                "sampling_strategy": 0.02,
                "k_neighbors": 5,
            }
            search_space = dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE)
            smote_params = {
                "sampling_strategy": 0.02,
                "k_neighbors": 5,
            }
            model_params = dict(base_params)
        elif fixed_smote_params:
            assert fixed_smote_params == {
                "sampling_strategy": 0.02,
                "k_neighbors": 5,
            }
            assert dict(kwargs.get("smote_search_space") or {}) == {}
            params = {
                **dict(base_params),
                **dict(fixed_smote_params),
            }
            search_space = {"mtry": [2]}
            smote_params = dict(fixed_smote_params)
            model_params = dict(base_params)
        else:
            params = {
                **dict(base_params),
                "sampling_strategy": 0.3,
                "k_neighbors": 5,
            }
            search_space = {"mtry": [2]}
            smote_params = {
                "sampling_strategy": 0.3,
                "k_neighbors": 5,
            }
            model_params = dict(base_params)
        artifact = {
            "study_id": "study_smote_rerank",
            "model_name": kwargs["model_name"],
            "balance_mode": kwargs["balance_mode"],
            "best_value": 0.74,
            "best_params": params,
            "model_params": model_params,
            "smote_params": smote_params,
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": search_space,
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": 0.74,
                    "params": params,
                    "model_params": model_params,
                    "smote_params": smote_params,
                }
            ],
        }
        return params, pd.DataFrame(), artifact

    def fake_run_yearly_strategy(
        df,
        *,
        strategy,
        model_names,
        balance_mode,
        tuning_resolver=None,
        run_seed=None,
        run_order=None,
        **kwargs,
    ):
        assert tuning_resolver is not None
        train_df = df.loc[pd.to_datetime(df["interval_start"]).dt.year.eq(2018)].copy()
        tuning_artifact = tuning_resolver(
            train_df=train_df,
            model_name=str(model_names[0]),
            balance_mode=str(balance_mode),
            strategy_context={
                "stage": "yearly_train",
                "strategy": str(strategy),
                "window_kind": "base_year",
                "training_year": "2018",
                "prediction_year": 2019,
            },
        )
        assert tuning_artifact["selected_feature_cols"] == ["x3", "x1"]
        yearly_df = pd.DataFrame(
            [
                {
                    "strategy": str(strategy),
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": str(model_names[0]),
                    "balance_mode": str(balance_mode),
                    "auc": 0.74,
                    "sensitivity": 0.61,
                    "specificity": 0.79,
                    "error_rate": 0.22,
                    "training_time_sec": 0.4,
                    "threshold": 0.45,
                    "n_train": int(len(train_df)),
                    "n_test": 20,
                    "run_seed": int(run_seed or 0),
                    "run_order": int(run_order or 0),
                }
            ]
        )
        return yearly_df, []

    monkeypatch.setattr(drift_app, "_rerank_features_after_smote", fake_rerank_features_after_smote)
    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        balance_modes=[BALANCE_MODE_SMOTE],
        validation_size=0.2,
        folds=2,
        random_state=31,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(31,),
        feature_selection_context=feature_selection_context,
    )

    assert not outputs["yearly_results"].empty
    assert tune_feature_cols == [["x1", "x2"], ["x1", "x2"], ["x3", "x1"]]
    assert tune_balance_modes == ["none", drift_app.BALANCE_MODE_SMOTE, drift_app.BALANCE_MODE_SMOTE]
    assert tune_fixed_model_params == [{}, {"mtry": 2, "splitrule": "gini", "min_node_size": 1}, {}]
    assert tune_fixed_smote_params == [{}, {}, {"sampling_strategy": 0.02, "k_neighbors": 5}]
    assert tune_custom_grids == [{}, dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE), {}]
    assert tune_smote_search_spaces == [{}, dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE), {}]
    assert rerank_calls == [[2018]]
    assert rerank_smote_params == [(0.02, 5)]


def test_recalibration_experiments_smote_pre_tuning_keeps_original_ranking_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=76)
    features = ["x1", "x2"]
    feature_selection_context = {
        "selected_features": features,
        "candidate_features": ["x1", "x2", "x3"],
    }

    tune_feature_cols = []
    tune_balance_modes = []
    tune_fixed_model_params = []
    tune_fixed_smote_params = []
    tune_custom_grids = []
    tune_smote_search_spaces = []

    def fail_rerank_features_after_smote(*args, **kwargs):
        raise AssertionError("SMOTE rerank should stay disabled unless explicitly requested")

    def fake_tune_hyperparameters(X, y, **kwargs):
        tune_feature_cols.append(list(X.columns))
        tune_balance_modes.append(str(kwargs.get("balance_mode")))
        tune_fixed_model_params.append(dict(kwargs.get("fixed_model_params") or {}))
        tune_fixed_smote_params.append(dict(kwargs.get("fixed_smote_params") or {}))
        tune_custom_grids.append(dict(kwargs.get("custom_grid") or {}))
        tune_smote_search_spaces.append(dict(kwargs.get("smote_search_space") or {}))
        base_params = {"mtry": 2, "splitrule": "gini", "min_node_size": 1}
        balance_mode = str(kwargs.get("balance_mode"))
        fixed_model_params = dict(kwargs.get("fixed_model_params") or {})
        fixed_smote_params = dict(kwargs.get("fixed_smote_params") or {})
        if balance_mode == drift_app.BALANCE_MODE_NONE:
            params = dict(base_params)
            search_space = {"mtry": [2]}
            smote_params = {}
            model_params = dict(base_params)
        elif fixed_model_params:
            assert fixed_model_params == base_params
            assert dict(kwargs.get("smote_search_space") or {}) == dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE)
            params = {
                **dict(base_params),
                "sampling_strategy": 0.02,
                "k_neighbors": 10,
            }
            search_space = dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE)
            smote_params = {
                "sampling_strategy": 0.02,
                "k_neighbors": 10,
            }
            model_params = dict(base_params)
        elif fixed_smote_params:
            assert fixed_smote_params == {
                "sampling_strategy": 0.02,
                "k_neighbors": 10,
            }
            assert dict(kwargs.get("smote_search_space") or {}) == {}
            params = {
                **dict(base_params),
                **dict(fixed_smote_params),
            }
            search_space = {"mtry": [2]}
            smote_params = dict(fixed_smote_params)
            model_params = dict(base_params)
        else:
            raise AssertionError("Unexpected tuning stage for SMOTE original ranking path")
        artifact = {
            "study_id": "study_smote_original_ranking",
            "model_name": kwargs["model_name"],
            "balance_mode": kwargs["balance_mode"],
            "best_value": 0.74,
            "best_params": params,
            "model_params": model_params,
            "smote_params": smote_params,
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": search_space,
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": 0.74,
                    "params": params,
                    "model_params": model_params,
                    "smote_params": smote_params,
                }
            ],
        }
        return params, pd.DataFrame(), artifact

    def fake_run_yearly_strategy(
        df,
        *,
        strategy,
        model_names,
        balance_mode,
        tuning_resolver=None,
        run_seed=None,
        run_order=None,
        **kwargs,
    ):
        assert tuning_resolver is not None
        train_df = df.loc[pd.to_datetime(df["interval_start"]).dt.year.eq(2018)].copy()
        tuning_artifact = tuning_resolver(
            train_df=train_df,
            model_name=str(model_names[0]),
            balance_mode=str(balance_mode),
            strategy_context={
                "stage": "yearly_train",
                "strategy": str(strategy),
                "window_kind": "base_year",
                "training_year": "2018",
                "prediction_year": 2019,
            },
        )
        assert tuning_artifact["selected_feature_cols"] == features
        assert tuning_artifact["smote_rerank_info"]["applied"] is False
        assert tuning_artifact["smote_rerank_info"]["reason"] == "disabled"
        yearly_df = pd.DataFrame(
            [
                {
                    "strategy": str(strategy),
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": str(model_names[0]),
                    "balance_mode": str(balance_mode),
                    "auc": 0.74,
                    "sensitivity": 0.61,
                    "specificity": 0.79,
                    "error_rate": 0.22,
                    "training_time_sec": 0.4,
                    "threshold": 0.45,
                    "n_train": int(len(train_df)),
                    "n_test": 20,
                    "run_seed": int(run_seed or 0),
                    "run_order": int(run_order or 0),
                }
            ]
        )
        return yearly_df, []

    monkeypatch.setattr(drift_app, "_rerank_features_after_smote", fail_rerank_features_after_smote)
    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        balance_modes=[BALANCE_MODE_SMOTE],
        validation_size=0.2,
        folds=2,
        random_state=31,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(31,),
        feature_selection_context=feature_selection_context,
    )

    assert not outputs["yearly_results"].empty
    assert tune_feature_cols == [["x1", "x2"], ["x1", "x2"], ["x1", "x2"]]
    assert tune_balance_modes == ["none", drift_app.BALANCE_MODE_SMOTE, drift_app.BALANCE_MODE_SMOTE]
    assert tune_fixed_model_params == [{}, {"mtry": 2, "splitrule": "gini", "min_node_size": 1}, {}]
    assert tune_fixed_smote_params == [{}, {}, {"sampling_strategy": 0.02, "k_neighbors": 10}]
    assert tune_custom_grids == [{}, dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE), {}]
    assert tune_smote_search_spaces == [{}, dict(drift_app.SMOTE_PRE_TUNING_SEARCH_SPACE), {}]


def test_recalibration_experiments_persist_optuna_json(tmp_path, monkeypatch):
    pytest.importorskip("imblearn")

    features_duckdb = tmp_path / "feature_selection.duckdb"
    features_duckdb.write_text("", encoding="utf-8")
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=180, random_state=71)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]
    feature_selection_context = {
        "corr_threshold": 0.85,
        "top_n": len(features),
        "selected_features": features,
        "feature_count": len(features),
        "feature_export_path": str(features_duckdb),
    }

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE, BALANCE_MODE_SMOTE),
        feature_selection_context=feature_selection_context,
    )

    json_path = Path(outputs["optuna_json_path"])
    assert json_path.exists()
    assert json_path.parent == tmp_path
    assert outputs["run_manifest"]["optuna_json_path"] == str(json_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_manifest"]["optuna_json_path"] == str(json_path)
    assert payload["feature_selection_context"]["selected_features"] == features
    assert payload["feature_selection_context"]["feature_export_path"] == str(features_duckdb)
    assert payload["studies"]
    assert any("best_params" in study for study in payload["studies"])
    assert any("trials" in study for study in payload["studies"])
    json.dumps(payload)


def test_recalibration_experiments_persist_live_status_and_events(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=91)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]
    progress_updates = []

    def fake_tune_hyperparameters(X, y, **kwargs):
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback(
                {
                    "ratio": 0.5,
                    "label": "Sintonizando hiperparametros...",
                    "detail": "Trial 1 / 1 | Mejor AUC CV 0.73",
                }
            )
        params = {"mtry": 2, "splitrule": "gini", "min_node_size": 1}
        artifact = {
            "study_id": "study_rf_none",
            "model_name": kwargs["model_name"],
            "balance_mode": kwargs["balance_mode"],
            "best_value": 0.73,
            "best_params": params,
            "model_params": params,
            "smote_params": {},
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": {"mtry": [2]},
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": 0.73,
                    "params": params,
                    "model_params": params,
                    "smote_params": {},
                }
            ],
        }
        search_df = pd.DataFrame(
            [
                {
                    "trial_number": 0,
                    "model": kwargs["model_name"],
                    "cv_auc": 0.73,
                    "params": "{}",
                    "state": "COMPLETE",
                    "balance_mode": kwargs["balance_mode"],
                }
            ]
        )
        return params, search_df, artifact

    def fake_run_yearly_strategy(*args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback({"ratio": 0.35, "label": "Entrenando modelo...", "detail": "Ano 2019"})
            callback({"ratio": 1.0, "label": "Bloque listo.", "detail": "Ano 2020"})
        rows = pd.DataFrame(
            [
                {
                    "strategy": kwargs["strategy"],
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": kwargs["model_names"][0],
                    "balance_mode": kwargs["balance_mode"],
                    "auc": 0.82,
                    "sensitivity": 0.71,
                    "specificity": 0.77,
                    "error_rate": 0.18,
                    "threshold": 0.5,
                    "training_time_sec": 0.1,
                    "n_train": 10,
                    "n_test": 10,
                    "best_params": "{}",
                    "run_seed": kwargs["run_seed"],
                    "run_order": kwargs["run_order"],
                }
            ]
        )
        roc_payload = [
            {
                "strategy": kwargs["strategy"],
                "model": kwargs["model_names"][0],
                "balance_mode": kwargs["balance_mode"],
                "segment": "2019",
                "y_true": np.asarray([0, 1]),
                "scores": np.asarray([0.1, 0.9]),
                "run_seed": kwargs["run_seed"],
                "run_order": kwargs["run_order"],
            }
        ]
        return rows, roc_payload

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    checkpoint_root = tmp_path / "runs"
    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE,),
        checkpoint_root=checkpoint_root,
        progress_callback=progress_updates.append,
    )

    manifest_path = Path(outputs["checkpoint_manifest_path"])
    run_dir = manifest_path.parent
    live_status_path = run_dir / "live_status.json"
    live_events_path = run_dir / "live_events.jsonl"

    assert live_status_path.exists()
    assert live_events_path.exists()

    live_status = json.loads(live_status_path.read_text(encoding="utf-8"))
    live_events = [
        json.loads(line)
        for line in live_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert live_status["status"] == "completed"
    assert live_status["context"]["phase"] == "run_complete"
    assert live_status["progress_ratio"] == pytest.approx(1.0)
    assert any(event["context"].get("phase") == "window_tuning" for event in live_events)
    assert any(event["context"].get("phase") == "block_running" for event in live_events)
    assert any(event["context"].get("phase") == "block_complete" for event in live_events)
    assert progress_updates


def test_recalibration_experiments_resume_from_failed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=71)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    tuning_calls = []
    block_calls = []
    fail_once = {"enabled": True}

    def fake_tune_hyperparameters(X, y, **kwargs):
        tuning_calls.append((kwargs["model_name"], kwargs["balance_mode"]))
        params = {"mtry": 2, "splitrule": "gini", "min_node_size": 1}
        artifact = {
            "study_id": f"study_{kwargs['model_name']}_{kwargs['balance_mode']}",
            "model_name": kwargs["model_name"],
            "balance_mode": kwargs["balance_mode"],
            "best_value": 0.75,
            "best_params": params,
            "model_params": params,
            "smote_params": {},
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": {"mtry": [2]},
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": 0.75,
                    "params": params,
                    "model_params": params,
                    "smote_params": {},
                }
            ],
        }
        return params, pd.DataFrame([{"trial_number": 0, "model": kwargs["model_name"], "cv_auc": 0.75, "params": "{}", "state": "COMPLETE", "balance_mode": kwargs["balance_mode"]}]), artifact

    def fake_run_yearly_strategy(*args, **kwargs):
        block_calls.append((kwargs["balance_mode"], kwargs["run_seed"], kwargs["run_order"]))
        if kwargs["balance_mode"] == BALANCE_MODE_SMOTE and fail_once["enabled"]:
            raise RuntimeError("forced block failure")
        rows = pd.DataFrame(
            [
                {
                    "strategy": kwargs["strategy"],
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": kwargs["model_names"][0],
                    "balance_mode": kwargs["balance_mode"],
                    "auc": 0.8,
                    "sensitivity": 0.7,
                    "specificity": 0.75,
                    "error_rate": 0.2,
                    "threshold": 0.5,
                    "training_time_sec": 0.1,
                    "n_train": 10,
                    "n_test": 10,
                    "best_params": "{}",
                    "run_seed": kwargs["run_seed"],
                    "run_order": kwargs["run_order"],
                }
            ]
        )
        roc_payload = [
            {
                "strategy": kwargs["strategy"],
                "model": kwargs["model_names"][0],
                "balance_mode": kwargs["balance_mode"],
                "segment": "2019",
                "y_true": np.asarray([0, 1]),
                "scores": np.asarray([0.1, 0.9]),
                "run_seed": kwargs["run_seed"],
                "run_order": kwargs["run_order"],
            }
        ]
        return rows, roc_payload

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    checkpoint_root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="forced block failure"):
        run_recalibration_experiments(
            df,
            feature_cols=features,
            model_names=["Random Forest"],
            strategies=["static"],
            validation_size=0.2,
            folds=2,
            random_state=11,
            fast_mode=True,
            grid_limit=1,
            repetition_seeds=(11,),
            balance_modes=(BALANCE_MODE_NONE, BALANCE_MODE_SMOTE),
            checkpoint_root=checkpoint_root,
        )

    manifests = sorted(checkpoint_root.glob("run_*/manifest.json"))
    assert len(manifests) == 1
    failed_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    assert len(failed_manifest["completed_block_ids"]) == 1
    assert tuning_calls == [
        ("Random Forest", BALANCE_MODE_NONE),
        ("Random Forest", BALANCE_MODE_SMOTE),
        ("Random Forest", BALANCE_MODE_SMOTE),
    ]

    fail_once["enabled"] = False
    block_calls.clear()
    tuning_calls.clear()

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE, BALANCE_MODE_SMOTE),
        checkpoint_root=checkpoint_root,
    )

    assert outputs["auto_resumed"] is True
    assert len(block_calls) == 1
    assert block_calls[0][0] == BALANCE_MODE_SMOTE
    assert tuning_calls == []
    assert set(outputs["yearly_results"]["balance_mode"].unique()) == {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}
    assert outputs["checkpoint_manifest"]["status"] == "completed"

    block_calls.clear()
    tuning_calls.clear()

    fresh_outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE, BALANCE_MODE_SMOTE),
        checkpoint_root=checkpoint_root,
        auto_resume=False,
    )

    assert fresh_outputs["auto_resumed"] is False
    assert len(block_calls) == 2
    assert {call[0] for call in block_calls} == {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}
    assert tuning_calls == [
        ("Random Forest", BALANCE_MODE_NONE),
        ("Random Forest", BALANCE_MODE_SMOTE),
        ("Random Forest", BALANCE_MODE_SMOTE),
    ]


def test_recalibration_experiments_can_load_explicit_checkpoint_run_id_override(tmp_path):
    checkpoint_root = tmp_path / "runs"
    run_dir = drift_app._recalibration_run_dir("run_selected_checkpoint", checkpoint_root=checkpoint_root)
    paths = drift_app._recalibration_run_paths(run_dir)
    drift_app._ensure_recalibration_run_dirs(paths)

    manifest = drift_app._initial_recalibration_manifest(
        run_id="run_selected_checkpoint",
        run_manifest={"total_progress_units": 1},
        feature_selection_context={"feature_export_path": "/tmp/original.duckdb", "selected_features": ["x1", "x2"]},
        experiment_blocks=[{"strategy": "static", "model": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        tuning_tasks=[],
        preflight={},
    )
    manifest["status"] = "completed"
    manifest["completed_block_ids"] = ["block_demo"]
    manifest["block_index"] = {
        "block_demo": {"status": "completed", "filename": "block_demo.json"}
    }
    drift_app._update_manifest_progress(
        manifest,
        completed_units=1.0,
        pending_block_ids=[],
    )
    drift_app._atomic_write_json(paths["manifest"], manifest)
    drift_app._persist_recalibration_block(
        paths["blocks_dir"] / "block_demo.json",
        {
            "block_id": "block_demo",
            "block": {"strategy": "static", "model": "Random Forest", "balance_mode": BALANCE_MODE_NONE},
            "run_seed": 11,
            "run_order": 1,
            "yearly_rows": [
                {
                    "strategy": "static",
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": "Random Forest",
                    "balance_mode": BALANCE_MODE_NONE,
                    "auc": 0.81,
                    "sensitivity": 0.7,
                    "specificity": 0.8,
                    "error_rate": 0.2,
                    "threshold": 0.5,
                    "training_time_sec": 0.1,
                    "n_train": 10,
                    "n_test": 10,
                    "run_seed": 11,
                    "run_order": 1,
                }
            ],
            "adaptive_rows": [],
            "roc_payload": [
                {
                    "strategy": "static",
                    "model": "Random Forest",
                    "balance_mode": BALANCE_MODE_NONE,
                    "segment": "2019",
                    "y_true": [0, 1],
                    "scores": [0.1, 0.9],
                    "run_seed": 11,
                    "run_order": 1,
                }
            ],
            "execution_log": [],
            "tuning_refs": [],
            "smote_refs": [],
        },
    )

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=60, random_state=17)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE,),
        checkpoint_root=checkpoint_root,
        checkpoint_run_id_override="run_selected_checkpoint",
        feature_selection_context={"feature_export_path": "/tmp/different.duckdb", "selected_features": ["x1"]},
    )

    assert outputs["auto_resumed"] is True
    assert outputs["checkpoint_manifest_path"].endswith("run_selected_checkpoint/manifest.json")
    assert outputs["run_manifest"]["checkpoint_run_id_override"] == "run_selected_checkpoint"
    assert outputs["run_manifest"]["computed_run_id"] != "run_selected_checkpoint"
    assert not outputs["yearly_results"].empty


def test_recalibration_experiments_can_recompute_trainings_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=77)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    tuning_calls = []
    block_calls = []
    block_run_counter = {"value": 0}

    def fake_tune_hyperparameters(X, y, **kwargs):
        tuning_calls.append((kwargs["model_name"], kwargs["balance_mode"]))
        params = {"mtry": 2, "splitrule": "gini", "min_node_size": 1}
        artifact = {
            "study_id": f"study_{kwargs['model_name']}_{kwargs['balance_mode']}",
            "model_name": kwargs["model_name"],
            "balance_mode": kwargs["balance_mode"],
            "best_value": 0.75,
            "best_params": params,
            "model_params": params,
            "smote_params": {},
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": {"mtry": [2]},
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": 0.75,
                    "params": params,
                    "model_params": params,
                    "smote_params": {},
                }
            ],
        }
        return params, pd.DataFrame([{"trial_number": 0, "model": kwargs["model_name"], "cv_auc": 0.75, "params": "{}", "state": "COMPLETE", "balance_mode": kwargs["balance_mode"]}]), artifact

    def fake_run_yearly_strategy(*args, **kwargs):
        block_run_counter["value"] += 1
        auc_value = 0.80 + 0.05 * float(block_run_counter["value"])
        block_calls.append((kwargs["balance_mode"], kwargs["run_seed"], kwargs["run_order"], auc_value))
        rows = pd.DataFrame(
            [
                {
                    "strategy": kwargs["strategy"],
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": kwargs["model_names"][0],
                    "balance_mode": kwargs["balance_mode"],
                    "auc": auc_value,
                    "sensitivity": 0.7,
                    "specificity": 0.75,
                    "error_rate": 0.2,
                    "threshold": 0.5,
                    "training_time_sec": 0.1,
                    "n_train": 10,
                    "n_test": 10,
                    "best_params": "{}",
                    "run_seed": kwargs["run_seed"],
                    "run_order": kwargs["run_order"],
                }
            ]
        )
        roc_payload = [
            {
                "strategy": kwargs["strategy"],
                "model": kwargs["model_names"][0],
                "balance_mode": kwargs["balance_mode"],
                "segment": "2019",
                "y_true": np.asarray([0, 1]),
                "scores": np.asarray([0.1, 0.9]),
                "run_seed": kwargs["run_seed"],
                "run_order": kwargs["run_order"],
            }
        ]
        return rows, roc_payload

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    checkpoint_root = tmp_path / "runs"
    first_outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE,),
        checkpoint_root=checkpoint_root,
    )

    preview = drift_app._preview_recalibration_checkpoint(
        df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=[11],
        balance_modes=[BALANCE_MODE_NONE],
        checkpoint_root=checkpoint_root,
    )

    assert first_outputs["checkpoint_manifest"]["status"] == "completed"
    assert len(tuning_calls) == 1
    assert len(block_calls) == 1
    assert preview["can_load_completed"] is True
    assert preview["can_recompute_trainings"] is True

    tuning_calls.clear()
    block_calls.clear()

    second_outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE,),
        checkpoint_root=checkpoint_root,
        recompute_blocks_from_checkpoint=True,
    )

    assert second_outputs["auto_resumed"] is True
    assert tuning_calls == []
    assert len(block_calls) == 1
    assert second_outputs["checkpoint_manifest"]["resume"]["checkpoint_mode"] == "recompute_blocks"
    assert second_outputs["checkpoint_manifest"]["status"] == "completed"
    assert second_outputs["yearly_results"]["auc"].iloc[0] == pytest.approx(0.90)


def test_recalibration_experiments_recompute_override_fails_instead_of_retuning_on_feature_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=91)
    checkpoint_features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]
    mismatched_features = checkpoint_features[:-1]

    def fake_tune_hyperparameters(X, y, **kwargs):
        params = {"mtry": 2, "splitrule": "gini", "min_node_size": 1}
        artifact = {
            "study_id": f"study_{kwargs['model_name']}_{kwargs['balance_mode']}",
            "model_name": kwargs["model_name"],
            "balance_mode": kwargs["balance_mode"],
            "best_value": 0.75,
            "best_params": params,
            "model_params": params,
            "smote_params": {},
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": {"mtry": [2]},
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": 0.75,
                    "params": params,
                    "model_params": params,
                    "smote_params": {},
                }
            ],
        }
        search_df = pd.DataFrame(
            [
                {
                    "trial_number": 0,
                    "model": kwargs["model_name"],
                    "cv_auc": 0.75,
                    "params": "{}",
                    "state": "COMPLETE",
                    "balance_mode": kwargs["balance_mode"],
                }
            ]
        )
        return params, search_df, artifact

    def fake_run_yearly_strategy(*args, **kwargs):
        rows = pd.DataFrame(
            [
                {
                    "strategy": kwargs["strategy"],
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": kwargs["model_names"][0],
                    "balance_mode": kwargs["balance_mode"],
                    "auc": 0.81,
                    "sensitivity": 0.7,
                    "specificity": 0.75,
                    "error_rate": 0.2,
                    "threshold": 0.5,
                    "training_time_sec": 0.1,
                    "n_train": 10,
                    "n_test": 10,
                    "best_params": "{}",
                    "run_seed": kwargs["run_seed"],
                    "run_order": kwargs["run_order"],
                }
            ]
        )
        roc_payload = [
            {
                "strategy": kwargs["strategy"],
                "model": kwargs["model_names"][0],
                "balance_mode": kwargs["balance_mode"],
                "segment": "2019",
                "y_true": np.asarray([0, 1]),
                "scores": np.asarray([0.1, 0.9]),
                "run_seed": kwargs["run_seed"],
                "run_order": kwargs["run_order"],
            }
        ]
        return rows, roc_payload

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    checkpoint_root = tmp_path / "runs"
    first_outputs = run_recalibration_experiments(
        df,
        feature_cols=checkpoint_features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE,),
        checkpoint_root=checkpoint_root,
    )

    run_id = str(first_outputs["run_manifest"]["run_id"])

    def fail_if_re_tuned(*args, **kwargs):
        pytest.fail("recompute-only mode should not create new tuning artifacts")

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fail_if_re_tuned)

    with pytest.raises(RuntimeError, match="missing persisted tuning artifacts"):
        run_recalibration_experiments(
            df,
            feature_cols=mismatched_features,
            model_names=["Random Forest"],
            strategies=["static"],
            validation_size=0.2,
            folds=2,
            random_state=11,
            fast_mode=True,
            grid_limit=1,
            repetition_seeds=(11,),
            balance_modes=(BALANCE_MODE_NONE,),
            checkpoint_root=checkpoint_root,
            checkpoint_run_id_override=run_id,
            recompute_blocks_from_checkpoint=True,
        )


def test_recalibration_experiments_recompute_accepts_legacy_tuning_key_with_same_train_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=92)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]
    checkpoint_root = tmp_path / "runs"
    run_id = "run_legacy_tuning"
    run_dir = drift_app._recalibration_run_dir(run_id, checkpoint_root=checkpoint_root)
    paths = drift_app._recalibration_run_paths(run_dir)
    drift_app._ensure_recalibration_run_dirs(paths)

    run_manifest = {
        "run_id": run_id,
        "models": ["Random Forest"],
        "strategies": ["static"],
        "balance_modes": [BALANCE_MODE_NONE],
        "repetition_seeds": [11],
        "base_year": 2018,
        "validation_size": 0.2,
        "folds": 2,
        "random_state_fallback": 11,
        "fast_mode": True,
        "resource_mode": drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        "optuna_trials": 1,
        "adwin_delta": 0.002,
        "min_window": 45000,
        "min_retrain_size": 4500,
        "kswin_top_k_features": 3,
        "kswin_vote_threshold": 2,
        "kswin_retrain_days": 30,
        "kswin_min_retrain_rows": 50,
        "continue_on_block_error": False,
        "feature_count": len(features),
        "feature_cols": list(features),
        "target_col": "target",
        "time_col": "interval_start",
        "input_rows": int(len(df)),
        "total_progress_units": 2,
        "total_tuning_tasks": 1,
        "total_block_units": 1,
        "preflight": {},
    }
    feature_selection_context = {"selected_features": list(features), "feature_count": len(features)}
    manifest = drift_app._initial_recalibration_manifest(
        run_id=run_id,
        run_manifest=run_manifest,
        feature_selection_context=feature_selection_context,
        experiment_blocks=[{"strategy": "static", "model": "Random Forest", "balance_mode": BALANCE_MODE_NONE}],
        tuning_tasks=[],
        preflight={},
    )

    tuning_task = drift_app._batch_tuning_tasks(
        df=df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        strategies=["static"],
        balance_modes=[BALANCE_MODE_NONE],
        base_year=2018,
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        resource_mode=drift_app.DEFAULT_EXPERIMENT_RESOURCE_MODE,
        grid_limit=1,
    )[0]
    train_df = drift_app._materialize_tuning_train_df(
        df,
        time_col="interval_start",
        task=tuning_task,
    )
    signature_cols = [col for col in ["interval_start", "target"] + list(features) if col in train_df.columns]
    train_signature = drift_app._frame_signature(train_df, columns=signature_cols, include_index=True)
    legacy_tuning_key = "tuning_legacy_match"
    legacy_filename = f"{legacy_tuning_key}.json"
    drift_app._persist_tuning_artifact(
        paths["tuning_dir"] / legacy_filename,
        {
            "tuning_key": legacy_tuning_key,
            "study_id": "legacy_study",
            "model_name": "Random Forest",
            "balance_mode": BALANCE_MODE_NONE,
            "best_params": {"mtry": 2},
            "has_valid_trial": True,
            "train_signature": train_signature,
            "strategy_context": {"stage": "window_tuning", "window_kind": "base_year", "training_year": "2018"},
        },
    )
    manifest["tuning_index"] = {
        legacy_tuning_key: {
            "filename": legacy_filename,
            "status": "completed",
            "model_name": "Random Forest",
            "balance_mode": BALANCE_MODE_NONE,
        }
    }
    drift_app._persist_manifest(paths["manifest"], manifest)

    def fail_if_re_tuned(*args, **kwargs):
        pytest.fail("legacy-compatible checkpoint artifact should be reused without retuning")

    def fake_run_yearly_strategy(*args, **kwargs):
        rows = pd.DataFrame(
            [
                {
                    "strategy": kwargs["strategy"],
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": kwargs["model_names"][0],
                    "balance_mode": kwargs["balance_mode"],
                    "auc": 0.81,
                    "sensitivity": 0.7,
                    "specificity": 0.75,
                    "error_rate": 0.2,
                    "threshold": 0.5,
                    "training_time_sec": 0.1,
                    "n_train": 10,
                    "n_test": 10,
                    "best_params": "{}",
                    "run_seed": kwargs["run_seed"],
                    "run_order": kwargs["run_order"],
                }
            ]
        )
        return rows, []

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fail_if_re_tuned)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11,),
        balance_modes=(BALANCE_MODE_NONE,),
        checkpoint_root=checkpoint_root,
        checkpoint_run_id_override=run_id,
        recompute_blocks_from_checkpoint=True,
    )

    assert outputs["checkpoint_manifest"]["status"] == "completed"
    current_tuning_key = str(tuning_task["tuning_key"])
    assert outputs["checkpoint_manifest"]["tuning_index"][current_tuning_key]["matched_legacy_tuning_key"] == legacy_tuning_key


def test_recalibration_experiments_can_skip_failed_blocks_and_continue(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=19)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    block_calls = []

    def fake_tune_hyperparameters(X, y, **kwargs):
        params = {"mtry": 2, "splitrule": "gini", "min_node_size": 1}
        artifact = {
            "study_id": f"study_{kwargs['model_name']}_{kwargs['balance_mode']}",
            "model_name": kwargs["model_name"],
            "balance_mode": kwargs["balance_mode"],
            "best_value": 0.75,
            "best_params": params,
            "model_params": params,
            "smote_params": {},
            "n_trials": 1,
            "requested_trials": 1,
            "search_space_size": 1,
            "search_space": {"mtry": [2]},
            "trials": [
                {
                    "trial_number": 0,
                    "state": "COMPLETE",
                    "cv_auc": 0.75,
                    "params": params,
                    "model_params": params,
                    "smote_params": {},
                }
            ],
        }
        return params, pd.DataFrame([{"trial_number": 0, "model": kwargs["model_name"], "cv_auc": 0.75, "params": "{}", "state": "COMPLETE", "balance_mode": kwargs["balance_mode"]}]), artifact

    def fake_run_yearly_strategy(*args, **kwargs):
        block_calls.append((kwargs["balance_mode"], kwargs["run_seed"], kwargs["run_order"]))
        if kwargs["balance_mode"] == BALANCE_MODE_SMOTE and kwargs["run_seed"] == 11:
            raise RuntimeError("forced block failure")
        rows = pd.DataFrame(
            [
                {
                    "strategy": kwargs["strategy"],
                    "iteration": 1,
                    "training_year": "2018",
                    "prediction_year": 2019,
                    "model": kwargs["model_names"][0],
                    "balance_mode": kwargs["balance_mode"],
                    "auc": 0.8,
                    "sensitivity": 0.7,
                    "specificity": 0.75,
                    "error_rate": 0.2,
                    "threshold": 0.5,
                    "training_time_sec": 0.1,
                    "n_train": 10,
                    "n_test": 10,
                    "best_params": "{}",
                    "run_seed": kwargs["run_seed"],
                    "run_order": kwargs["run_order"],
                }
            ]
        )
        roc_payload = [
            {
                "strategy": kwargs["strategy"],
                "model": kwargs["model_names"][0],
                "balance_mode": kwargs["balance_mode"],
                "segment": "2019",
                "y_true": np.asarray([0, 1]),
                "scores": np.asarray([0.1, 0.9]),
                "run_seed": kwargs["run_seed"],
                "run_order": kwargs["run_order"],
            }
        ]
        return rows, roc_payload

    monkeypatch.setattr(drift_app, "tune_hyperparameters", fake_tune_hyperparameters)
    monkeypatch.setattr(drift_app, "run_yearly_strategy", fake_run_yearly_strategy)

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(11, 12),
        balance_modes=(BALANCE_MODE_NONE, BALANCE_MODE_SMOTE),
        checkpoint_root=tmp_path / "runs",
        continue_on_block_error=True,
    )

    manifest = outputs["checkpoint_manifest"]
    assert outputs["auto_resumed"] is False
    assert manifest["status"] == "completed"
    assert len(manifest["completed_block_ids"]) == 3
    assert len(manifest["skipped_failed_block_ids"]) == 1
    assert len(manifest["nonfatal_block_errors"]) == 1
    assert manifest["progress"]["skipped_failed_blocks"] == 1
    assert len(outputs["yearly_results"]) == 3
    assert len(block_calls) == 4
    skipped_block_id = manifest["skipped_failed_block_ids"][0]
    assert manifest["block_index"][skipped_block_id]["status"] == "skipped_failed"


def test_adaptive_adwin_optuna_studies_capture_base_and_retrains_with_balance_mode():
    pytest.importorskip("imblearn")

    df = _make_kswin_shift_dataset(rows_per_year=220, random_state=41)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    studies = []
    adaptive_df, roc_payload = run_adaptive_strategy(
        df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=5,
        fast_mode=True,
        grid_limit=1,
        adwin_delta=0.05,
        min_window=120,
        min_retrain_size=20,
        balance_mode=BALANCE_MODE_SMOTE,
        study_collector=studies,
    )

    assert not adaptive_df.empty
    assert len(roc_payload) >= 1
    assert {"adaptive_base", "adaptive_retrain"} <= {study["stage"] for study in studies}
    assert all(study["balance_mode"] == BALANCE_MODE_SMOTE for study in studies)
    assert len({study["study_id"] for study in studies}) == len(studies)


def test_run_adaptive_strategy_uses_separate_min_retrain_size(monkeypatch):
    fit_sizes = []

    class FakeBatchModel:
        def predict_proba(self, X):
            values = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0.0).astype(float).to_numpy()
            scores = np.where(values > 0, 0.8, 0.2)
            return np.column_stack([1.0 - scores, scores])

    class FakeADWIN:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def update(self, value):
            self.calls += 1
            if self.calls == 30:
                return True, {"n": 60.0, "n0": 30.0, "n1": 30.0}
            return False, None

        def __len__(self):
            return self.calls

    def fake_train_model_with_internal_validation(train_df, **kwargs):
        fit_sizes.append(len(train_df))
        return {
            "model": FakeBatchModel(),
            "threshold": 0.5,
            "best_params": {"splitrule": "gini"},
            "fill_values": {"x": 0.0},
            "tuning_artifact": {"study_id": f"study_{len(fit_sizes)}"},
        }

    monkeypatch.setattr(drift_app, "SimpleADWIN", FakeADWIN)
    monkeypatch.setattr(
        drift_app,
        "train_model_with_internal_validation",
        fake_train_model_with_internal_validation,
    )

    base_times = pd.date_range("2018-01-01 00:00:00", periods=10, freq="5min")
    stream_times = pd.date_range("2019-01-01 00:00:00", periods=40, freq="5min")
    df = pd.DataFrame(
        {
            "interval_start": base_times.append(stream_times),
            "x": [(-1.0) ** idx for idx in range(50)],
            "target": [idx % 2 for idx in range(50)],
        }
    )

    adaptive_df, roc_payload = run_adaptive_strategy(
        df,
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=3,
        fast_mode=True,
        grid_limit=1,
        adwin_delta=0.01,
        min_window=1000,
        min_retrain_size=30,
    )

    assert fit_sizes == [10, 30]
    assert not adaptive_df.empty
    assert len(roc_payload) >= 1


def test_run_yearly_strategy_reuses_static_training_bundle(monkeypatch):
    fit_calls = []

    class FakeModel:
        def predict_proba(self, X):
            values = pd.to_numeric(X.iloc[:, 0], errors="coerce").fillna(0.0).astype(float).to_numpy()
            scores = np.where(values > 0, 0.8, 0.2)
            return np.column_stack([1.0 - scores, scores])

    def fake_train_model_with_internal_validation(train_df, **kwargs):
        fit_calls.append(len(train_df))
        return {
            "model": FakeModel(),
            "threshold": 0.5,
            "best_params": {"splitrule": "gini"},
            "fill_values": {"x": 0.0},
            "tuning_artifact": {"study_id": "cached-static-study"},
        }

    monkeypatch.setattr(
        drift_app,
        "train_model_with_internal_validation",
        fake_train_model_with_internal_validation,
    )

    df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 00:05:00",
                    "2019-01-01 00:00:00",
                    "2019-01-01 00:05:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                ]
            ),
            "x": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    yearly_df, roc_payload = run_yearly_strategy(
        df,
        strategy="static",
        feature_cols=["x"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=13,
        fast_mode=True,
        grid_limit=1,
    )

    assert fit_calls == [2]
    assert set(yearly_df["prediction_year"].astype(int)) == {2019, 2020}
    assert len(roc_payload) == 2


def test_run_yearly_strategy_uses_selected_feature_cols_from_tuning_artifact(monkeypatch):
    fit_feature_cols = []
    eval_feature_cols = []

    class FakeModel:
        def predict_proba(self, X):
            scores = np.full(len(X), 0.8, dtype=float)
            return np.column_stack([1.0 - scores, scores])

    def fake_train_model_with_internal_validation(train_df, **kwargs):
        fit_feature_cols.append(list(kwargs["feature_cols"]))
        return {
            "model": FakeModel(),
            "threshold": 0.5,
            "best_params": {"splitrule": "gini"},
            "fill_values": {"x2": 0.0},
            "feature_transform": None,
            "tuning_artifact": {"study_id": "study_selected_cols"},
        }

    def fake_evaluate_split(model, test_df, **kwargs):
        eval_feature_cols.append(list(kwargs["feature_cols"]))
        y_true = np.asarray([0, 1], dtype=int)
        scores = np.asarray([0.2, 0.8], dtype=float)
        metrics = {
            "auc": 0.75,
            "pr_auc": 0.76,
            "f1": 0.67,
            "sensitivity": 0.7,
            "specificity": 0.8,
            "error_rate": 0.2,
        }
        return {
            "metrics": metrics,
            "metrics_before_calibration": metrics,
            "metrics_after_calibration": metrics,
            "y_true": y_true,
            "scores": scores,
            "calibrated_scores": scores,
        }

    monkeypatch.setattr(
        drift_app,
        "train_model_with_internal_validation",
        fake_train_model_with_internal_validation,
    )
    monkeypatch.setattr(drift_app, "_evaluate_split", fake_evaluate_split)

    df = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                [
                    "2018-01-01 00:00:00",
                    "2018-01-01 00:05:00",
                    "2019-01-01 00:00:00",
                    "2019-01-01 00:05:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:05:00",
                ]
            ),
            "x1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "x2": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    def fake_tuning_resolver(**kwargs):
        return {
            "tuning_key": "selected_cols_tuning",
            "best_params": {},
            "has_valid_trial": True,
            "selected_feature_cols": ["x2"],
        }

    yearly_df, roc_payload = run_yearly_strategy(
        df,
        strategy="static",
        feature_cols=["x1"],
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        validation_size=0.2,
        folds=2,
        random_state=13,
        fast_mode=True,
        grid_limit=1,
        tuning_resolver=fake_tuning_resolver,
    )

    assert fit_feature_cols == [["x2"]]
    assert eval_feature_cols == [["x2"], ["x2"]]
    assert not yearly_df.empty
    assert len(roc_payload) == 2


def test_kswin_optuna_studies_capture_base_and_retrains_with_balance_mode():
    pytest.importorskip("river")
    pytest.importorskip("imblearn")

    df = _make_kswin_shift_dataset(rows_per_year=220, random_state=41)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    studies = []
    adaptive_df, roc_payload = run_kswin_strategy(
        df,
        feature_cols=features,
        target_col="target",
        time_col="interval_start",
        model_names=["Random Forest"],
        kswin_variants=["KSWINpaper"],
        validation_size=0.2,
        folds=2,
        random_state=11,
        fast_mode=True,
        grid_limit=1,
        kswin_top_k_features=3,
        kswin_vote_threshold=1,
        kswin_retrain_days=30,
        kswin_min_retrain_rows=60,
        balance_mode=BALANCE_MODE_SMOTE,
        study_collector=studies,
    )

    assert not adaptive_df.empty
    assert len(roc_payload) >= 1
    assert {"kswin_base", "kswin_retrain"} <= {study["stage"] for study in studies}
    assert all(study["balance_mode"] == BALANCE_MODE_SMOTE for study in studies)
    assert len({study["study_id"] for study in studies}) == len(studies)


def test_recalibration_experiments_keep_adaptive_arf_outside_balance_duplication():
    pytest.importorskip("river")

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=180, random_state=37)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static", "adaptive_arf"],
        arf_variants=["ARFmoderate"],
        validation_size=0.2,
        folds=2,
        random_state=17,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(17,),
        balance_modes=(BALANCE_MODE_NONE, BALANCE_MODE_SMOTE),
    )

    yearly = outputs["yearly_results"]
    arf_rows = outputs["adaptive_results"].loc[outputs["adaptive_results"]["strategy"] == "adaptive_arf"].copy()
    arf_summary = outputs["summary"].loc[outputs["summary"]["strategy"] == "adaptive_arf"].copy()
    arf_log = outputs["execution_log"].loc[outputs["execution_log"]["strategy"] == "adaptive_arf"].copy()

    assert set(yearly["balance_mode"].unique()) == {BALANCE_MODE_NONE, BALANCE_MODE_SMOTE}
    assert not arf_rows.empty
    assert set(arf_rows["balance_mode"].unique()) == {BALANCE_MODE_NOT_APPLICABLE}
    assert set(arf_rows["prediction_year"].astype(int).unique()) == {2019, 2020}
    assert len(arf_rows) == 2
    assert arf_summary["balance_mode"].eq(BALANCE_MODE_NOT_APPLICABLE).all()
    assert arf_log["balance_mode"].dropna().eq(BALANCE_MODE_NOT_APPLICABLE).all()


def test_parse_repetition_seeds_and_multi_seed_logging(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    assert parse_repetition_seeds("") == [42, 52, 62]
    assert parse_repetition_seeds("7, 9, 7 11") == [7, 9, 11]

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=180, random_state=31)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=13,
        fast_mode=True,
        grid_limit=2,
        repetition_seeds=(3, 5),
    )

    yearly = outputs["yearly_results"]
    log_df = outputs["execution_log"]
    appendix_mean = outputs["appendix_tables_mean"]

    assert set(yearly["run_seed"].astype(int).unique()) == {3, 5}
    assert set(yearly["run_order"].astype(int).unique()) == {1, 2}
    assert outputs["summary"]["n_repetitions"].eq(2).all()
    assert not log_df.empty
    assert {"run_start", "run_complete"} <= set(log_df["phase"])
    assert not appendix_mean["A.6"].empty
    assert appendix_mean["A.6"]["n_repetitions"].eq(2).all()
    assert appendix_mean["A.6"]["seed_list"].str.contains("3").all()
    assert {"pr_auc", "brier_score", "bias2", "variance", "noise", "f1"} <= set(appendix_mean["A.6"].columns)


def test_persisted_recalibration_json_keeps_full_records(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_app, "RESULTS_DIR", tmp_path)

    df = generate_synthetic_article_dataset(years=(2018, 2019, 2020), rows_per_year=120, random_state=91)
    features = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
        "x1",
        "x2",
        "x3",
    ]

    outputs = run_recalibration_experiments(
        df,
        feature_cols=features,
        model_names=["Random Forest"],
        strategies=["static"],
        validation_size=0.2,
        folds=2,
        random_state=19,
        fast_mode=True,
        grid_limit=1,
        repetition_seeds=(19,),
    )

    out_path = Path(str(outputs["optuna_json_path"]))
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert {"appendix_tables", "appendix_tables_mean", "execution_log", "roc_payload"} <= set(payload.keys())
    assert payload["execution_log"]
    assert payload["roc_payload"]
    assert payload["yearly_results"]


def test_batch_range_builder_modes():
    start = pd.Timestamp("2024-01-01 00:00:00")
    end = pd.Timestamp("2024-01-10 00:00:00")

    daily = _build_batch_ranges(start, end, "Diario")
    weekly = _build_batch_ranges(start, end, "Semanal")
    monthly = _build_batch_ranges(start, end, "Mensual")

    assert len(daily) == 9
    assert len(weekly) == 2
    assert len(monthly) == 1
    assert daily[0][0] == start
    assert daily[-1][1] == end


def test_focus_porticos_special_selection():
    assert _porticos_in_tramo(DRIFT_FOCUS_TRAMO) == DRIFT_FOCUS_PORTICOS
    assert _describe_tramo_selection(DRIFT_FOCUS_TRAMO) == "Porticos fijos: 11, 12, 14, 15"


def test_filter_accidents_uses_consecutive_corridor_segments_only():
    accidents_df = pd.DataFrame(
        {
            "accidente_time": pd.date_range("2024-01-01", periods=6, freq="5min"),
            "ultimo_portico": ["15", "14", "12", "18", "11", "15"],
            "proximo_portico": ["14", "12", "11", "15", "9", "12"],
        }
    )

    filtered = _filter_accidents_for_allowed_porticos(accidents_df, DRIFT_FOCUS_PORTICOS)

    assert list(filtered["ultimo_portico"]) == ["15", "14", "12"]
    assert list(filtered["proximo_portico"]) == ["14", "12", "11"]


def test_drift_article_features_cover_table4_and_target_shift():
    flows_df = pd.DataFrame(
        [
            {"FECHA": "2024-01-01 00:00:10", "VELOCIDAD": 100.0, "CATEGORIA": 1, "MATRICULA": "AAA111", "PORTICO": "15", "CARRIL": "1"},
            {"FECHA": "2024-01-01 00:02:40", "VELOCIDAD": 98.0, "CATEGORIA": 1, "MATRICULA": "AAA111", "PORTICO": "14", "CARRIL": "2"},
            {"FECHA": "2024-01-01 00:05:40", "VELOCIDAD": 97.0, "CATEGORIA": 1, "MATRICULA": "AAA111", "PORTICO": "12", "CARRIL": "2"},
            {"FECHA": "2024-01-01 00:09:20", "VELOCIDAD": 96.0, "CATEGORIA": 1, "MATRICULA": "AAA111", "PORTICO": "11", "CARRIL": "3"},
            {"FECHA": "2024-01-01 00:01:00", "VELOCIDAD": 80.0, "CATEGORIA": 2, "MATRICULA": "BBB222", "PORTICO": "15", "CARRIL": "2"},
            {"FECHA": "2024-01-01 00:03:30", "VELOCIDAD": 79.0, "CATEGORIA": 2, "MATRICULA": "BBB222", "PORTICO": "14", "CARRIL": "2"},
            {"FECHA": "2024-01-01 00:06:00", "VELOCIDAD": 104.0, "CATEGORIA": 4, "MATRICULA": "CCC333", "PORTICO": "15", "CARRIL": "3"},
            {"FECHA": "2024-01-01 00:08:30", "VELOCIDAD": 103.0, "CATEGORIA": 4, "MATRICULA": "CCC333", "PORTICO": "14", "CARRIL": "1"},
            {"FECHA": "2024-01-01 00:05:20", "VELOCIDAD": 101.0, "CATEGORIA": 1, "MATRICULA": "DDD444", "PORTICO": "14", "CARRIL": "1"},
            {"FECHA": "2024-01-01 00:07:40", "VELOCIDAD": 99.0, "CATEGORIA": 1, "MATRICULA": "DDD444", "PORTICO": "12", "CARRIL": "1"},
            {"FECHA": "2024-01-01 00:10:40", "VELOCIDAD": 95.0, "CATEGORIA": 1, "MATRICULA": "DDD444", "PORTICO": "11", "CARRIL": "2"},
        ]
    )

    features_df = _compute_drift_article_features(
        flows_df,
        interval_minutes=5,
        allowed_porticos=DRIFT_FOCUS_PORTICOS,
    )

    expected_cols = {
        "flow_light_15",
        "vel_heavy_15",
        "sd_motorcycle_15",
        "den_light_14",
        "delta_flow_light_15",
        "delta_vel_light_15",
        "delta_den_light_14",
        "delta_sd_light_15",
        "ft_motorcycle_15",
        "ft_heavy_15",
        "flow_light_15_14",
        "vel_light_15_14",
        "sd_light_15_14",
        "den_light_15_14",
        "delta_flow_light_15_14",
        "delta_vel_light_15_14",
        "delta_den_light_15_14",
        "cl_light_15_14",
        "vel_15_14",
        "flow_15_14",
        "sd_15_14",
        "den_15_14",
        "delta_flow_15_14",
        "delta_den_15_14",
        "flow_lane1_15_14",
        "vel_lane2_15_14",
        "den_lane1_14_12",
        "delta_flow_lane1_14_12",
    }
    assert expected_cols.issubset(set(features_df.columns))
    assert features_df["interval_start"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:05:00"),
        pd.Timestamp("2024-01-01 00:10:00"),
    ]

    accidents_df = pd.DataFrame(
        {
            "accidente_time": [pd.Timestamp("2024-01-01 00:06:00")],
            "ultimo_portico": ["14"],
            "proximo_portico": ["12"],
        }
    )
    corridor_accidents = _filter_accidents_for_allowed_porticos(accidents_df, DRIFT_FOCUS_PORTICOS)
    base_df = _add_interval_accident_target(
        features_df,
        corridor_accidents,
        interval_minutes=5,
    )
    targets = base_df.set_index("interval_start")["target"].to_dict()
    assert targets[pd.Timestamp("2024-01-01 00:00:00")] == 1
    assert targets[pd.Timestamp("2024-01-01 00:05:00")] == 0


def test_drift_batched_flow_features_duckdb_consistency(tmp_path):
    duckdb = pytest.importorskip("duckdb")

    def _trip_rows(
        ts: str,
        *,
        plate: str,
        category: int,
        start_portico: str,
        end_portico: str,
        start_lane: str,
        end_lane: str,
        start_speed: float,
        end_speed: float,
        travel_minutes: float,
    ) -> list[dict[str, object]]:
        start_ts = pd.Timestamp(ts)
        end_ts = start_ts + pd.Timedelta(minutes=travel_minutes)
        return [
            {
                "FECHA": start_ts,
                "VELOCIDAD": float(start_speed),
                "CATEGORIA": int(category),
                "MATRICULA": plate,
                "PORTICO": start_portico,
                "CARRIL": start_lane,
            },
            {
                "FECHA": end_ts,
                "VELOCIDAD": float(end_speed),
                "CATEGORIA": int(category),
                "MATRICULA": plate,
                "PORTICO": end_portico,
                "CARRIL": end_lane,
            },
        ]

    def _gate_row(ts: str, *, plate: str, portico: str, speed: float, category: int, lane: str) -> dict[str, object]:
        return {
            "FECHA": pd.Timestamp(ts),
            "VELOCIDAD": float(speed),
            "CATEGORIA": int(category),
            "MATRICULA": plate,
            "PORTICO": portico,
            "CARRIL": lane,
        }

    rows: list[dict[str, object]] = []
    rows.extend(
        _trip_rows(
            "2024-01-31 23:55:00",
            plate="JAN_A",
            category=1,
                start_portico="15",
                end_portico="14",
            start_lane="1",
            end_lane="2",
            start_speed=70.0,
            end_speed=68.0,
            travel_minutes=3.0,
        )
    )
    rows.append(_gate_row("2024-01-31 23:55:20", plate="JAN_B", portico="12", speed=66.0, category=2, lane="2"))

    rows.extend(
        _trip_rows(
            "2024-02-01 00:00:00",
            plate="FEB_A",
            category=1,
                start_portico="15",
                end_portico="14",
            start_lane="1",
            end_lane="1",
            start_speed=75.0,
            end_speed=73.0,
            travel_minutes=3.0,
        )
    )
    rows.extend(
        _trip_rows(
            "2024-02-01 00:00:30",
            plate="FEB_B",
            category=2,
                start_portico="14",
                end_portico="12",
            start_lane="2",
            end_lane="3",
            start_speed=72.0,
            end_speed=71.0,
            travel_minutes=4.0,
        )
    )

    rows.extend(
        _trip_rows(
            "2024-02-01 00:05:00",
            plate="FEB_C",
            category=1,
                start_portico="15",
                end_portico="14",
            start_lane="1",
            end_lane="2",
            start_speed=80.0,
            end_speed=78.0,
            travel_minutes=3.0,
        )
    )
    rows.extend(
        _trip_rows(
            "2024-02-01 00:05:20",
            plate="FEB_D",
            category=4,
                start_portico="14",
                end_portico="12",
            start_lane="3",
            end_lane="1",
            start_speed=82.0,
            end_speed=81.0,
            travel_minutes=4.0,
        )
    )
    rows.append(_gate_row("2024-02-01 00:05:40", plate="FEB_E", portico="11", speed=79.0, category=1, lane="1"))

    rows.extend(
        _trip_rows(
            "2024-02-28 23:55:00",
            plate="FEBZ_A",
            category=1,
                start_portico="15",
                end_portico="14",
            start_lane="2",
            end_lane="2",
            start_speed=85.0,
            end_speed=83.0,
            travel_minutes=3.0,
        )
    )
    rows.append(_gate_row("2024-02-28 23:55:20", plate="FEBZ_B", portico="12", speed=82.0, category=2, lane="3"))

    rows.extend(
        _trip_rows(
            "2024-03-01 00:00:00",
            plate="MAR_A",
            category=1,
                start_portico="15",
                end_portico="14",
            start_lane="1",
            end_lane="1",
            start_speed=90.0,
            end_speed=89.0,
            travel_minutes=3.0,
        )
    )
    rows.extend(
        _trip_rows(
            "2024-03-01 00:00:30",
            plate="MAR_B",
            category=2,
                start_portico="14",
                end_portico="12",
            start_lane="2",
            end_lane="3",
            start_speed=88.0,
            end_speed=87.0,
            travel_minutes=4.0,
        )
    )

    rows.extend(
        _trip_rows(
            "2024-03-01 00:05:00",
            plate="MAR_C",
            category=1,
                start_portico="15",
                end_portico="14",
            start_lane="2",
            end_lane="3",
            start_speed=95.0,
            end_speed=94.0,
            travel_minutes=3.0,
        )
    )
    rows.extend(
        _trip_rows(
            "2024-03-01 00:05:20",
            plate="MAR_D",
            category=4,
                start_portico="14",
                end_portico="12",
            start_lane="1",
            end_lane="1",
            start_speed=97.0,
            end_speed=96.0,
            travel_minutes=4.0,
        )
    )
    rows.append(_gate_row("2024-03-01 00:05:40", plate="MAR_E", portico="11", speed=93.0, category=1, lane="2"))
    flows_df = pd.DataFrame(rows)

    flow_db_path = tmp_path / "flows.duckdb"
    out_path = tmp_path / "drift_batched_features.duckdb"
    con = duckdb.connect(str(flow_db_path))
    try:
        con.register("flows_view", flows_df)
        con.execute("CREATE TABLE flujos_duckdb AS SELECT * FROM flows_view")
    finally:
        con.close()

    full_features = _compute_drift_article_features(
        flows_df,
        interval_minutes=5,
        allowed_porticos=DRIFT_FOCUS_PORTICOS,
    ).sort_values("interval_start").reset_index(drop=True)

    meta = _compute_batched_flow_features_to_duckdb(
        flow_db_path=str(flow_db_path),
        flow_table_name="flujos_duckdb",
        out_path=out_path,
        batch_mode="month",
        interval_minutes=5,
        lanes=1,
        metrics=[],
        categories=["Light", "Heavy", "Motorcycle"],
        allowed_porticos=DRIFT_FOCUS_PORTICOS,
    )

    assert out_path.exists()
    assert len(meta["batch_ranges"]) == 3
    assert meta["input_rows"] == len(flows_df)

    batched_df = _load_feature_df_from_duckdb(out_path)
    batched_df["interval_start"] = pd.to_datetime(batched_df["interval_start"], errors="coerce")
    batched_df = batched_df.sort_values("interval_start").reset_index(drop=True)

    static_cols = [c for c in full_features.columns if not c.startswith("delta_")]
    pd.testing.assert_frame_equal(
        full_features[static_cols],
        batched_df[static_cols],
        obj="Static features for Drift detection batching",
    )

    boundary_starts = {
        pd.Timestamp("2024-02-01 00:00:00"),
        pd.Timestamp("2024-03-01 00:00:00"),
    }
    safe_mask = ~full_features["interval_start"].isin(boundary_starts)
    pd.testing.assert_frame_equal(
        full_features.loc[safe_mask].reset_index(drop=True),
        batched_df.loc[safe_mask].reset_index(drop=True),
        obj="Delta features away from month boundaries",
    )

    boundary_batch = batched_df.loc[batched_df["interval_start"].isin(boundary_starts)]
    delta_cols = [c for c in batched_df.columns if c.startswith("delta_")]
    assert not boundary_batch.empty
    for col in delta_cols:
        assert ((boundary_batch[col] == 0) | boundary_batch[col].isna()).all()


def test_normalize_portico_series():
    series = pd.Series([" 101.0 ", "ac-07", "None", np.nan, "12,5"])
    out = _normalize_portico_series(series)
    vals = out.astype("string").tolist()
    assert vals[0] == "101"
    assert vals[1] == "AC-07"
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])
    assert vals[4] == "12.5"


def test_configurable_preparation_pipeline_respects_stage_switches():
    df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=72, freq="1h"),
            "target": [0] * 35 + [1] + [0] * 36,
            "f1": [1.0] * 72,
            "f2": [np.nan] * 5 + [2.0] * 67,
        }
    )

    clean_no_stages, summary_no_stages, _, runs_no_stages, steps_no_stages = run_configurable_preparation_pipeline(
        df,
        feature_cols=["f1", "f2"],
        apply_stage1=False,
        apply_stage2=False,
        min_zero_days=1,
        interval_minutes=60,
    )
    assert len(clean_no_stages) == len(df)
    assert summary_no_stages.removed_high_missing_features == 0
    assert runs_no_stages.empty
    assert "omitido" in " ".join(steps_no_stages).lower()

    clean_with_stages, summary_with_stages, stage1_info, runs_with_stages, _ = run_configurable_preparation_pipeline(
        df,
        feature_cols=["f1", "f2"],
        apply_stage1=True,
        missing_threshold=0.01,
        apply_stage2=True,
        min_zero_days=1,
        interval_minutes=60,
    )
    assert "f2" in stage1_info["removed_cols"]
    assert len(clean_with_stages) < len(df)
    assert summary_with_stages.final_rows == len(clean_with_stages)
    assert len(runs_with_stages) >= 1


def test_rebuild_preparation_artifacts_from_payload_matches_pipeline_counts():
    df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=72, freq="1h"),
            "target": [0] * 35 + [1] + [0] * 36,
            "f1": [1.0] * 72,
            "f2": [np.nan] * 5 + [2.0] * 67,
        }
    )

    clean_df, summary, stage1_info, zero_runs, _ = run_configurable_preparation_pipeline(
        df,
        feature_cols=["f1", "f2"],
        apply_stage1=True,
        missing_threshold=0.01,
        apply_stage2=True,
        min_zero_days=1,
        interval_minutes=60,
    )

    rebuilt_summary, rebuilt_stage1_info, rebuilt_zero_runs = _rebuild_preparation_artifacts_from_payload(
        df,
        clean_df,
        min_zero_days=1,
    )

    assert rebuilt_summary.removed_high_missing_features == summary.removed_high_missing_features
    assert rebuilt_summary.rows_after_missing_drop == summary.rows_after_missing_drop
    assert rebuilt_summary.removed_zero_run_rows == summary.removed_zero_run_rows
    assert rebuilt_summary.final_rows == summary.final_rows
    assert rebuilt_stage1_info["removed_cols"] == stage1_info["removed_cols"]
    assert len(rebuilt_zero_runs) == len(zero_runs)


def test_load_feature_payload_from_duckdb(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "drift_features_test.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE TABLE raw_features AS SELECT "
            "TIMESTAMP '2024-01-01 00:00:00' AS interval_start, "
            "'AC-07' AS portico, 1 AS target, 10.0 AS flow_light"
        )
        con.execute(
            "CREATE TABLE clean_features AS SELECT "
            "TIMESTAMP '2024-01-01 00:00:00' AS interval_start, "
            "'AC-07' AS portico, 1 AS target, 10.0 AS flow_light"
        )
    finally:
        con.close()

    raw_df, clean_df = _load_feature_payload_from_duckdb(db_path)
    assert not raw_df.empty
    assert not clean_df.empty
    assert set(["interval_start", "portico", "target", "flow_light"]).issubset(set(raw_df.columns))


def test_feature_selection_payload_persists_with_feature_duckdb(tmp_path):
    pytest.importorskip("duckdb")

    db_path = tmp_path / "drift_features_with_selection.duckdb"
    raw_df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=3, freq="5min"),
            "portico": ["15", "14", "12"],
            "target": [0, 1, 0],
            "f1": [1.0, 2.0, 3.0],
            "f2": [1.1, 2.1, 3.1],
            "f3": [5.0, 4.0, 3.0],
        }
    )
    clean_df = raw_df[["interval_start", "portico", "target", "f1", "f3"]].copy()
    corr_pairs = pd.Series(
        [0.981, 0.744],
        index=pd.MultiIndex.from_tuples([("f1", "f2"), ("f1", "f3")]),
        dtype=float,
    )
    importance_df = pd.DataFrame(
        {
            "feature": ["f1", "f3"],
            "importance": [0.73, 0.27],
            "rank": [1, 2],
        }
    )
    selection_payload = _build_feature_selection_payload(
        candidate_features=["f1", "f2", "f3"],
        selected_features=["f1", "f3"],
        corr_pairs=corr_pairs,
        importance_df=importance_df,
        config={
            "method": "correlation_plus_random_forest",
            "corr_threshold": 0.95,
            "top_n": 2,
            "generated_at": "2026-03-20T12:00:00",
        },
    )

    _write_feature_payload_to_duckdb(
        db_path,
        raw_df=raw_df,
        clean_df=clean_df,
        selection_payload=selection_payload,
    )

    loaded_raw_df, loaded_clean_df = _load_feature_payload_from_duckdb(db_path)
    loaded_selection = _load_feature_selection_payload_from_duckdb(db_path)

    assert loaded_selection is not None
    assert list(loaded_raw_df.columns) == list(raw_df.columns)
    assert list(loaded_clean_df.columns) == list(clean_df.columns)
    assert loaded_selection["candidate_features"] == ["f1", "f2", "f3"]
    assert loaded_selection["selected_features"] == ["f1", "f3"]
    assert loaded_selection["config"]["top_n"] == 2
    assert loaded_selection["config"]["method"] == "correlation_plus_random_forest"
    assert list(loaded_selection["importance_df"]["feature"]) == ["f1", "f3"]
    assert loaded_selection["corr_pairs"].loc[("f1", "f2")] == pytest.approx(0.981)

    _write_feature_payload_to_duckdb(
        db_path,
        raw_df=raw_df,
        clean_df=clean_df,
        selection_payload=None,
    )

    assert _load_feature_selection_payload_from_duckdb(db_path) is None


def test_main_mounts_neural_drift_tab_and_passes_context(monkeypatch: pytest.MonkeyPatch):
    class _FakeTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {
                "drift_clean_df": pd.DataFrame(
                    {
                        "interval_start": pd.date_range("2024-01-01 00:00:00", periods=3, freq="5min"),
                        "flow_light": [1.0, 2.0, 3.0],
                        "target": [0, 1, 0],
                    }
                ),
                "drift_raw_df": pd.DataFrame(
                    {
                        "interval_start": pd.date_range("2024-01-01 00:00:00", periods=3, freq="5min"),
                        "flow_light": [1.0, 2.0, 3.0],
                        "target": [0, 1, 0],
                    }
                ),
                "drift_feature_cols": ["flow_light"],
                "drift_feature_export_path": "/tmp/drift_features.duckdb",
                "drift_feature_selection_meta": {"method": "rf"},
                "drift_candidate_feature_cols": ["flow_light"],
            }
            self.tab_labels = []

        def set_page_config(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def caption(self, *args, **kwargs):
            return None

        def tabs(self, labels):
            self.tab_labels = list(labels)
            return [_FakeTab() for _ in labels]

    fake_st = FakeStreamlit()
    init_calls: list[str] = []
    render_calls: list[dict] = []

    monkeypatch.setattr(drift_app, "st", fake_st)
    monkeypatch.setattr(drift_app, "_init_state", lambda: None)
    monkeypatch.setattr(drift_app, "_render_events_tab", lambda: None)
    monkeypatch.setattr(drift_app, "_render_feature_engineering_tab", lambda: None)
    monkeypatch.setattr(drift_app, "_render_feature_selection_tab", lambda: None)
    monkeypatch.setattr(drift_app, "_render_experiments_tab", lambda: None)
    monkeypatch.setattr(drift_app.neural_drift_app, "init_state", lambda: init_calls.append("init"))
    monkeypatch.setattr(
        drift_app.neural_drift_app,
        "render_tab",
        lambda context: render_calls.append(dict(context)),
    )

    drift_app.main(set_page_config=False, show_exit_button=False)

    assert "Neural drift" in fake_st.tab_labels
    assert init_calls == ["init"]
    assert len(render_calls) == 1
    assert render_calls[0]["feature_cols"] == ["flow_light"]
    assert render_calls[0]["feature_export_path"] == "/tmp/drift_features.duckdb"
    assert render_calls[0]["selection_metadata"]["selected_features"] == ["flow_light"]
