import numpy as np
import pandas as pd
import pytest

from src.drift_detection_app import (
    DRIFT_FOCUS_PORTICOS,
    DRIFT_FOCUS_TRAMO,
    _add_interval_accident_target,
    _build_batch_ranges,
    _count_positive_target_rows,
    _compute_batched_flow_features_to_duckdb,
    _compute_drift_article_features,
    _describe_tramo_selection,
    _filter_accidents_for_allowed_porticos,
    _load_feature_payload_from_duckdb,
    _load_feature_df_from_duckdb,
    _normalize_portico_series,
    _porticos_in_tramo,
    _rebuild_preparation_artifacts_from_payload,
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
    run_adaptive_strategy,
    run_configurable_preparation_pipeline,
    run_recalibration_experiments,
    run_yearly_strategy,
    youden_threshold,
    compute_classification_metrics,
)


def test_count_positive_target_rows_uses_final_target_column():
    clean_df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01", periods=6, freq="5min"),
            "target": [0, 1, "1", None, 0, 1],
        }
    )

    assert _count_positive_target_rows(clean_df) == 3
    assert _count_positive_target_rows(pd.DataFrame({"x": [1, 2, 3]})) == 0


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
    assert not outputs["execution_log"].empty
    assert "run_seed" in outputs["yearly_results"].columns

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


def test_parse_repetition_seeds_and_multi_seed_logging():
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
