import numpy as np
import pandas as pd
import pytest

from src.drift_detection_app import (
    _build_batch_ranges,
    _load_feature_payload_from_duckdb,
    _normalize_portico_series,
    SimpleADWIN,
    apply_missing_data_policy,
    article_coverage_percentage,
    build_article_coverage_matrix,
    build_average_roc_curves,
    drop_highly_correlated_features,
    filter_long_zero_accident_runs,
    generate_synthetic_article_dataset,
    run_adaptive_strategy,
    run_configurable_preparation_pipeline,
    run_recalibration_experiments,
    run_yearly_strategy,
    youden_threshold,
    compute_classification_metrics,
)


def test_article_coverage_is_100_percent():
    matrix = build_article_coverage_matrix()

    # 21 sections + 7 figures + 9 tables
    assert len(matrix) == 37
    assert matrix["implemented"].all()
    assert article_coverage_percentage(matrix) == 100.0


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
    )

    assert "yearly_results" in outputs
    assert "adaptive_results" in outputs
    assert "average_roc" in outputs
    assert "appendix_tables" in outputs

    roc_df = outputs["average_roc"]
    assert isinstance(roc_df, pd.DataFrame)
    assert not roc_df.empty

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
