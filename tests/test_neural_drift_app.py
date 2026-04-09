from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import src.Neural_drift_app as neural_drift_app


def _feature_bundle(df: pd.DataFrame) -> dict:
    feature_cols = [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
    ]
    return {
        "source": "test",
        "df": df.copy(),
        "feature_cols": feature_cols,
        "selection_metadata": {},
    }


def test_build_window_dataset_aligns_prediction_horizon():
    df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01 00:00:00", periods=6, freq="5min"),
            "flow_light": [10, 11, 12, 13, 14, 15],
            "target": [0, 1, 0, 1, 0, 0],
        }
    )

    dataset = neural_drift_app.build_window_dataset(
        df,
        feature_cols=["flow_light"],
        interval_minutes=5,
        lookback_steps=3,
        horizon_steps=1,
    )

    assert dataset.X.shape == (4, 3)
    assert dataset.y.tolist() == [0, 1, 0, 0]
    assert dataset.metadata.loc[0, "window_end"] == pd.Timestamp("2024-01-01 00:10:00")
    assert dataset.metadata.loc[0, "horizon_end"] == pd.Timestamp("2024-01-01 00:15:00")


def test_build_window_dataset_preserves_temporal_order():
    df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01 00:00:00", periods=5, freq="5min"),
            "flow_light": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 0, 1, 0, 1],
        }
    )

    dataset = neural_drift_app.build_window_dataset(
        df,
        feature_cols=["flow_light"],
        lookback_steps=3,
        horizon_steps=1,
    )

    assert dataset.feature_names == [
        "flow_light[t-2]",
        "flow_light[t-1]",
        "flow_light[t-0]",
    ]
    assert dataset.X[0].tolist() == [10.0, 20.0, 30.0]
    assert dataset.X[1].tolist() == [20.0, 30.0, 40.0]


def test_subset_dataset_by_percentage_uses_most_recent_rows():
    df = pd.DataFrame(
        {
            "interval_start": pd.date_range("2024-01-01 00:00:00", periods=10, freq="5min"),
            "flow_light": np.arange(10, dtype=float),
            "target": [0] * 10,
        }
    )

    subset = neural_drift_app._subset_dataset_by_percentage(df, dataset_percent=30)

    assert len(subset) == 3
    assert subset["interval_start"].tolist() == list(
        pd.date_range("2024-01-01 00:35:00", periods=3, freq="5min")
    )


def test_embedding_drift_score_increases_under_regime_shift():
    rng = np.random.default_rng(42)
    X_ref = rng.normal(0.0, 1.0, size=(32, 8))
    y_ref = np.array([0, 1] * 16)
    scores_ref = np.clip(rng.normal(0.35, 0.08, size=32), 0.01, 0.99)
    embeddings_ref = rng.normal(0.0, 0.2, size=(32, 4))

    artifact = {
        "reference": neural_drift_app._build_reference_stats(
            X_ref=X_ref,
            y_ref=y_ref,
            calibrated_scores=scores_ref,
            embeddings=embeddings_ref,
        )
    }
    detectors = {
        neural_drift_app.DRIFT_INPUT: neural_drift_app.ClassicDriftDetector(rolling_window=8),
        neural_drift_app.DRIFT_SCORE: neural_drift_app.ClassicDriftDetector(rolling_window=8),
        neural_drift_app.DRIFT_ERROR: neural_drift_app.ClassicDriftDetector(rolling_window=8),
    }

    near_payload = neural_drift_app._build_channel_scores(
        artifact=artifact,
        x_row=X_ref[0],
        calibrated_score=float(scores_ref[0]),
        y_true=int(y_ref[0]),
        embeddings=np.array([0.05, -0.03, 0.02, 0.01], dtype=float),
        recent_embedding_history=None,
        selected_channels=[neural_drift_app.DRIFT_EMBEDDING],
        detectors=detectors,
    )
    far_payload = neural_drift_app._build_channel_scores(
        artifact=artifact,
        x_row=X_ref[1],
        calibrated_score=float(scores_ref[1]),
        y_true=int(y_ref[1]),
        embeddings=np.array([4.5, 4.8, 5.0, 4.9], dtype=float),
        recent_embedding_history=None,
        selected_channels=[neural_drift_app.DRIFT_EMBEDDING],
        detectors=detectors,
    )

    assert far_payload["channel_scores"][neural_drift_app.DRIFT_EMBEDDING] > near_payload["channel_scores"][neural_drift_app.DRIFT_EMBEDDING]


def test_torch_mlp_trains_embedding_autoencoder_monitor():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=180, drift_start=110, random_state=17)
    augmented_df, augmented_cols = neural_drift_app.augment_feature_frame(
        df,
        feature_cols=_feature_bundle(df)["feature_cols"],
    )
    dataset = neural_drift_app.build_window_dataset(
        augmented_df,
        feature_cols=augmented_cols,
        lookback_steps=8,
        horizon_steps=1,
    )
    split = neural_drift_app._split_window_dataset(
        dataset,
        train_fraction=0.60,
        validation_fraction=0.20,
        max_stream_rows=32,
    )

    artifact = neural_drift_app._train_torch_mlp(
        split["X_train"],
        split["y_train"],
        split["X_val"],
        split["y_val"],
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "mlp_epochs": 4,
            "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE,
            "drift_monitor_epochs": 6,
            "drift_monitor_hidden_dim": 12,
            "drift_monitor_bottleneck_dim": 4,
        },
    )

    assert artifact["embedding_monitor"] is not None
    assert artifact["monitor_effective_architecture"] == neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE
    assert "embedding_reconstruction_mean" in artifact["reference"]
    assert "embedding_reconstruction_std" in artifact["reference"]


def test_monitor_architecture_explanation_mentions_key_components():
    explanation = neural_drift_app._build_monitor_architecture_explanation(
        {
            "augmented_feature_count": 18,
            "predictor_input_dim": 216,
            "predictor_embedding_dim": 24,
            "monitor_input_dim": 24,
            "monitor_hidden_dim": 16,
            "monitor_bottleneck_dim": 6,
        },
        neural_drift_app.DEFAULT_CONFIG,
        0.65,
    )

    assert "predice riesgo de accidente" in explanation["overview"]
    assert any("embedding[24]" in step[0] for step in explanation["predictor_steps"])
    assert any("bottleneck[6]" in step[0] for step in explanation["monitor_steps"])
    assert "reconstruction_error" in explanation["score_formula"]


def test_drift_monitor_profiles_define_moderate_and_sensitive_presets():
    moderate = neural_drift_app._drift_monitor_profile_preset(
        neural_drift_app.DRIFT_MONITOR_PROFILE_MODERATE
    )
    sensitive = neural_drift_app._drift_monitor_profile_preset(
        neural_drift_app.DRIFT_MONITOR_PROFILE_SENSITIVE
    )

    assert moderate["drift_monitor_bottleneck_dim"] == 6
    assert moderate["drift_monitor_reconstruction_weight"] == 0.65
    assert sensitive["drift_monitor_bottleneck_dim"] < moderate["drift_monitor_bottleneck_dim"]
    assert sensitive["drift_monitor_reconstruction_weight"] > moderate["drift_monitor_reconstruction_weight"]


def test_detector_sensitivity_presets_span_conservative_to_very_sensitive():
    conservative = neural_drift_app._detector_sensitivity_preset_config(
        neural_drift_app.DETECTOR_SENSITIVITY_PRESET_CONSERVATIVE
    )
    moderate = neural_drift_app._detector_sensitivity_preset_config(
        neural_drift_app.DETECTOR_SENSITIVITY_PRESET_MODERATE
    )
    very_sensitive = neural_drift_app._detector_sensitivity_preset_config(
        neural_drift_app.DETECTOR_SENSITIVITY_PRESET_VERY_SENSITIVE
    )

    assert conservative["severity_threshold"] > moderate["severity_threshold"]
    assert conservative["recent_window_size"] > moderate["recent_window_size"]
    assert conservative["detector_adwin_delta"] < moderate["detector_adwin_delta"]
    assert conservative["drift_point_signal_weight"] < moderate["drift_point_signal_weight"]
    assert very_sensitive["severity_threshold"] < moderate["severity_threshold"]
    assert very_sensitive["recent_window_size"] < moderate["recent_window_size"]
    assert very_sensitive["detector_adwin_delta"] > moderate["detector_adwin_delta"]


def test_run_signature_changes_when_monitor_profile_changes():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=48, random_state=21)
    bundle = _feature_bundle(df)
    bundle["feature_export_path"] = "/tmp/example.duckdb"

    moderate_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "drift_monitor_profile": neural_drift_app.DRIFT_MONITOR_PROFILE_MODERATE,
        },
    )
    sensitive_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "drift_monitor_profile": neural_drift_app.DRIFT_MONITOR_PROFILE_SENSITIVE,
            **neural_drift_app._drift_monitor_profile_preset(
                neural_drift_app.DRIFT_MONITOR_PROFILE_SENSITIVE
            ),
        },
    )

    assert moderate_signature != sensitive_signature


def test_run_signature_changes_when_monitor_architecture_changes():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=48, random_state=23)
    bundle = _feature_bundle(df)

    classic_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE,
        },
    )
    attention_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
            "drift_monitor_sequence_length": 10,
        },
    )

    assert classic_signature != attention_signature


def test_run_backtest_pipeline_returns_expected_strategy_rows():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=220, drift_start=140, random_state=7)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_TORCH_MLP],
        "strategies": [
            neural_drift_app.STRATEGY_FIXED,
            neural_drift_app.STRATEGY_RECALIBRATION,
            neural_drift_app.STRATEGY_FINE_TUNING,
            neural_drift_app.STRATEGY_RETRAIN,
        ],
        "recent_window_size": 24,
        "recalibration_min_rows": 16,
        "retrain_min_rows": 24,
        "max_stream_rows": 48,
        "rolling_metric_window": 12,
        "mlp_epochs": 4,
        "fine_tune_epochs": 2,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    summary = results["summary"]
    assert not summary.empty
    assert set(summary["strategy"].astype(str)) == {
        neural_drift_app.STRATEGY_FIXED,
        neural_drift_app.STRATEGY_RECALIBRATION,
        neural_drift_app.STRATEGY_FINE_TUNING,
        neural_drift_app.STRATEGY_RETRAIN,
    }
    assert not results["stream_metrics"].empty


def test_run_backtest_pipeline_excludes_fine_tuning_for_xgboost():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=210, drift_start=130, random_state=11)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [
            neural_drift_app.STRATEGY_FIXED,
            neural_drift_app.STRATEGY_FINE_TUNING,
            neural_drift_app.STRATEGY_RETRAIN,
        ],
        "max_stream_rows": 40,
        "xgb_estimators": 12,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    summary = results["summary"]
    assert neural_drift_app.STRATEGY_FINE_TUNING not in set(summary["strategy"].astype(str))
    assert set(summary["strategy"].astype(str)) == {
        neural_drift_app.STRATEGY_FIXED,
        neural_drift_app.STRATEGY_RETRAIN,
    }
    assert set(summary["monitor_effective_architecture"].astype(str)) == {"not_available"}


def test_xgboost_classic_channels_detect_drift_on_shifted_dataset():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=260, drift_start=140, random_state=19)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [neural_drift_app.STRATEGY_RECALIBRATION],
        "xgb_estimators": 12,
        "max_stream_rows": 80,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    drift_events = results["drift_events"]
    summary = results["summary"]
    assert not drift_events.empty
    assert "max_channel_score" in drift_events.columns
    assert float(summary.loc[summary["strategy"].eq(neural_drift_app.STRATEGY_RECALIBRATION), "n_drift_events"].iloc[0]) > 0


def test_shifted_dataset_produces_multiple_drift_events_for_both_models():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=220, drift_start=140, random_state=7)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_TORCH_MLP, neural_drift_app.MODEL_XGBOOST],
        "strategies": [neural_drift_app.STRATEGY_RECALIBRATION],
        "recent_window_size": 24,
        "recalibration_min_rows": 16,
        "max_stream_rows": 48,
        "mlp_epochs": 2,
        "drift_monitor_epochs": 2,
        "xgb_estimators": 8,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)
    summary = results["summary"].set_index("model")

    assert int(summary.loc[neural_drift_app.MODEL_TORCH_MLP, "n_drift_events"]) > 5
    assert int(summary.loc[neural_drift_app.MODEL_XGBOOST, "n_drift_events"]) > 5


def test_resolve_dataset_from_context_falls_back_to_duckdb(tmp_path: Path):
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=32, random_state=5)
    db_path = tmp_path / "neural_drift_features.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.register("clean_features_view", df)
        con.execute("CREATE TABLE clean_features AS SELECT * FROM clean_features_view")
    finally:
        con.close()

    bundle = neural_drift_app.resolve_dataset_from_context(
        {
            "clean_df": None,
            "raw_df": None,
            "feature_cols": [
                "flow_light",
                "flow_heavy",
                "speed_light",
                "speed_heavy",
                "density_light",
                "density_heavy",
            ],
            "feature_export_path": str(db_path),
            "selection_metadata": {"from_test": True},
        }
    )

    assert bundle["source"] == "duckdb_export"
    assert len(bundle["df"]) == len(df)
    assert bundle["feature_cols"] == [
        "flow_light",
        "flow_heavy",
        "speed_light",
        "speed_heavy",
        "density_light",
        "density_heavy",
    ]


def test_list_feature_engineering_duckdb_artifacts_filters_clean_features(tmp_path: Path):
    valid_db = tmp_path / "valid_features.duckdb"
    invalid_db = tmp_path / "other.duckdb"
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=24, random_state=9)

    con = duckdb.connect(str(valid_db))
    try:
        con.register("clean_features_view", df)
        con.execute("CREATE TABLE clean_features AS SELECT * FROM clean_features_view")
    finally:
        con.close()

    con = duckdb.connect(str(invalid_db))
    try:
        con.execute("CREATE TABLE something_else AS SELECT 1 AS value")
    finally:
        con.close()

    artifacts = neural_drift_app.list_feature_engineering_duckdb_artifacts(tmp_path)

    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "valid_features.duckdb"
    assert int(artifacts[0]["row_count"]) == len(df)


def test_build_dataset_context_for_source_selection_uses_duckdb_selected_features(tmp_path: Path):
    current_df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=32, random_state=13)
    selected_df = current_df.rename(columns={"speed_light": "speed_selected"}).copy()
    selected_df["speed_selected"] = pd.to_numeric(selected_df["speed_selected"], errors="coerce")

    db_path = tmp_path / "selected_features.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.register("raw_view", selected_df)
        con.execute("CREATE TABLE raw_features AS SELECT * FROM raw_view")
        con.register("clean_view", selected_df)
        con.execute("CREATE TABLE clean_features AS SELECT * FROM clean_view")
        con.execute("CREATE TABLE feature_selection_selected(feature VARCHAR, selected_rank INTEGER)")
        con.execute("INSERT INTO feature_selection_selected VALUES ('speed_selected', 1)")
        con.execute("CREATE TABLE feature_selection_candidates(feature VARCHAR, candidate_rank INTEGER)")
        con.execute("INSERT INTO feature_selection_candidates VALUES ('speed_selected', 1)")
    finally:
        con.close()

    effective_context = neural_drift_app.build_dataset_context_for_source_selection(
        {
            "clean_df": current_df,
            "raw_df": current_df,
            "feature_cols": ["flow_light"],
            "feature_export_path": None,
            "selection_metadata": {"from_session": True},
        },
        selected_feature_export_path=str(db_path),
    )
    bundle = neural_drift_app.resolve_dataset_from_context(effective_context)

    assert bundle["source"] == "duckdb_export"
    assert bundle["feature_cols"] == ["speed_selected"]
    assert bundle["selection_metadata"]["feature_export_path"] == str(db_path)


def test_streamlit_arrow_safe_df_casts_mixed_object_columns():
    df = pd.DataFrame(
        {
            "model": ["Torch MLP", "XGBoost"],
            "metadata": ["default", 0.5],
        }
    )

    safe_df = neural_drift_app._streamlit_arrow_safe_df(df)

    assert str(safe_df["metadata"].dtype) == "string"
    assert safe_df["metadata"].tolist() == ["default", "0.5"]


def test_optimize_decision_threshold_prefers_lower_cutoff_for_rare_events():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    scores = np.array([0.01, 0.02, 0.03, 0.03, 0.04, 0.05, 0.07, 0.16, 0.22, 0.28], dtype=float)

    info = neural_drift_app._optimize_decision_threshold(y_true, scores, beta=2.0)

    assert 0.0 <= info["threshold"] <= 1.0
    assert info["threshold"] < 0.5
    assert info["recall"] >= 0.5


def test_baseline_uses_optimized_decision_threshold():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=220, drift_start=140, random_state=31)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [neural_drift_app.STRATEGY_FIXED],
        "xgb_estimators": 12,
        "max_stream_rows": 40,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    baseline = results["baseline"]
    assert not baseline.empty
    assert float(baseline.loc[0, "threshold"]) != 0.5


def test_run_backtest_pipeline_uses_selected_dataset_percentage():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=120, drift_start=80, random_state=53)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "dataset_percent": 50,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [neural_drift_app.STRATEGY_FIXED],
        "xgb_estimators": 8,
        "max_stream_rows": 24,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    assert len(results["dataset"].augmented_df) == 60


def test_build_embedding_monitor_sequences_aligns_history_and_target():
    embeddings = np.arange(30, dtype=float).reshape(10, 3)

    X_seq, y_target, target_indices = neural_drift_app._build_embedding_monitor_sequences(
        embeddings,
        sequence_length=4,
        stride=1,
    )

    assert X_seq.shape == (6, 4, 3)
    assert y_target.shape == (6, 3)
    assert target_indices.tolist() == [4, 5, 6, 7, 8, 9]
    assert X_seq[0].tolist() == embeddings[0:4].tolist()
    assert y_target[0].tolist() == embeddings[4].tolist()


def test_temporal_attention_monitor_returns_normalized_attention_weights():
    rng = np.random.default_rng(17)
    embeddings = rng.normal(0.0, 1.0, size=(24, 6))

    monitor, reconstruction_errors = neural_drift_app._fit_embedding_monitor(
        embeddings,
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
            "drift_monitor_sequence_length": 6,
            "drift_monitor_attention_hidden_dim": 12,
            "drift_monitor_epochs": 3,
            "drift_monitor_batch_size": 8,
        },
    )

    assert monitor is not None
    assert monitor["monitor_effective_architecture"] == neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION
    assert reconstruction_errors is not None
    assert len(reconstruction_errors) > 0

    details = neural_drift_app._predict_embedding_monitor_details(
        monitor,
        embeddings=embeddings[10].reshape(1, -1),
        recent_embeddings=embeddings[4:10],
    )

    attention_summary = details["attention_summary"]
    assert details["warmup"] is False
    assert attention_summary is not None
    assert len(attention_summary["temporal_attention_mean"]) == 6
    assert np.isclose(np.sum(attention_summary["temporal_attention_mean"]), 1.0, atol=1e-5)


def test_temporal_attention_monitor_warmup_without_enough_history():
    rng = np.random.default_rng(18)
    embeddings = rng.normal(0.0, 1.0, size=(20, 5))
    monitor, _ = neural_drift_app._fit_embedding_monitor(
        embeddings,
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
            "drift_monitor_sequence_length": 8,
            "drift_monitor_epochs": 2,
        },
    )

    assert monitor is not None
    details = neural_drift_app._predict_embedding_monitor_details(
        monitor,
        embeddings=embeddings[7].reshape(1, -1),
        recent_embeddings=embeddings[:6],
    )

    assert details["warmup"] is True
    assert details["reconstruction_error"] is None


def test_temporal_attention_monitor_falls_back_to_classic_for_small_dataset():
    rng = np.random.default_rng(19)
    embeddings = rng.normal(0.0, 1.0, size=(10, 5))
    monitor, reconstruction_errors = neural_drift_app._fit_embedding_monitor(
        embeddings,
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
            "drift_monitor_sequence_length": 12,
            "drift_monitor_epochs": 2,
        },
    )

    assert monitor is not None
    assert monitor["requested_architecture"] == neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION
    assert monitor["monitor_effective_architecture"] == neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE
    assert reconstruction_errors is not None


def test_backtest_with_temporal_attention_detector_returns_detector_attention_outputs():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=220, drift_start=140, random_state=59)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_TORCH_MLP],
        "strategies": [neural_drift_app.STRATEGY_RECALIBRATION],
        "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
        "drift_monitor_sequence_length": 8,
        "drift_monitor_attention_hidden_dim": 16,
        "recent_window_size": 24,
        "recalibration_min_rows": 16,
        "max_stream_rows": 48,
        "mlp_epochs": 2,
        "drift_monitor_epochs": 2,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    summary = results["summary"]
    detector_temporal = results["detector_attention_temporal_summary"]

    assert not summary.empty
    assert summary.loc[0, "monitor_effective_architecture"] == neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION
    assert not detector_temporal.empty
    assert "lag_1" in set(detector_temporal["time_step"].astype(str))


def test_backtest_with_predictor_attention_and_detector_attention_reports_shift():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=220, drift_start=140, random_state=61)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_TORCH_MLP_ATTENTION],
        "strategies": [neural_drift_app.STRATEGY_RECALIBRATION],
        "lookback_steps": 8,
        "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_TEMPORAL_ATTENTION,
        "drift_monitor_sequence_length": 8,
        "recent_window_size": 24,
        "recalibration_min_rows": 16,
        "max_stream_rows": 48,
        "mlp_epochs": 2,
        "drift_monitor_epochs": 2,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    summary = results["summary"]
    detector_shift = results["detector_attention_drift_shift_summary"]

    assert int(summary.loc[0, "n_drift_events"]) > 0
    assert not detector_shift.empty
    assert float(detector_shift["abs_delta_attention"].max()) > 0.0


def test_torch_attention_model_returns_normalized_attention_summaries():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=180, drift_start=110, random_state=41)
    augmented_df, augmented_cols = neural_drift_app.augment_feature_frame(
        df,
        feature_cols=_feature_bundle(df)["feature_cols"],
    )
    dataset = neural_drift_app.build_window_dataset(
        augmented_df,
        feature_cols=augmented_cols,
        lookback_steps=6,
        horizon_steps=1,
    )
    split = neural_drift_app._split_window_dataset(
        dataset,
        train_fraction=0.60,
        validation_fraction=0.20,
        max_stream_rows=24,
    )

    artifact = neural_drift_app._train_torch_mlp(
        split["X_train"],
        split["y_train"],
        split["X_val"],
        split["y_val"],
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "lookback_steps": 6,
            "mlp_epochs": 3,
            "drift_monitor_epochs": 3,
            "attention_feature_hidden_dim": 24,
            "attention_temporal_hidden_dim": 20,
        },
        model_name=neural_drift_app.MODEL_TORCH_MLP_ATTENTION,
        feature_metadata={
            "lookback_steps": 6,
            "base_feature_cols": _feature_bundle(df)["feature_cols"],
            "augmented_feature_cols": augmented_cols,
            "feature_count": len(augmented_cols),
        },
    )

    details = neural_drift_app._predict_torch_model_details(artifact, split["X_val"][:4])
    attention_summary = details["attention_summary"]

    assert artifact["model_family"] == "torch_mlp_attention"
    assert attention_summary is not None
    assert len(attention_summary["feature_attention_mean"]) == len(augmented_cols)
    assert len(attention_summary["temporal_attention_mean"]) == 6
    assert np.isclose(np.sum(attention_summary["feature_attention_mean"]), 1.0, atol=1e-5)
    assert np.isclose(np.sum(attention_summary["temporal_attention_mean"]), 1.0, atol=1e-5)


def test_plain_torch_mlp_remains_compatible_without_attention_metadata():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=160, drift_start=100, random_state=29)
    augmented_df, augmented_cols = neural_drift_app.augment_feature_frame(
        df,
        feature_cols=_feature_bundle(df)["feature_cols"],
    )
    dataset = neural_drift_app.build_window_dataset(
        augmented_df,
        feature_cols=augmented_cols,
        lookback_steps=8,
        horizon_steps=1,
    )
    split = neural_drift_app._split_window_dataset(
        dataset,
        train_fraction=0.60,
        validation_fraction=0.20,
        max_stream_rows=24,
    )

    artifact = neural_drift_app._train_torch_mlp(
        split["X_train"],
        split["y_train"],
        split["X_val"],
        split["y_val"],
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "mlp_epochs": 3,
            "drift_monitor_epochs": 3,
        },
    )

    assert artifact["model_family"] == "torch_mlp"
    assert artifact["attention_summary_reference"] is None


def test_attention_backtest_returns_attention_outputs_and_labels():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=220, drift_start=140, random_state=23)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_TORCH_MLP_ATTENTION],
        "strategies": [
            neural_drift_app.STRATEGY_FIXED,
            neural_drift_app.STRATEGY_RECALIBRATION,
            neural_drift_app.STRATEGY_FINE_TUNING,
            neural_drift_app.STRATEGY_RETRAIN,
        ],
        "lookback_steps": 8,
        "recent_window_size": 24,
        "recalibration_min_rows": 16,
        "retrain_min_rows": 24,
        "max_stream_rows": 48,
        "rolling_metric_window": 12,
        "mlp_epochs": 2,
        "fine_tune_epochs": 1,
        "drift_monitor_epochs": 2,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    summary = results["summary"]
    feature_summary = results["attention_feature_summary"]
    temporal_summary = results["attention_temporal_summary"]

    assert not summary.empty
    assert set(summary["strategy"].astype(str)) == {
        neural_drift_app.STRATEGY_FIXED,
        neural_drift_app.STRATEGY_RECALIBRATION,
        neural_drift_app.STRATEGY_FINE_TUNING,
        neural_drift_app.STRATEGY_RETRAIN,
    }
    assert not feature_summary.empty
    assert not temporal_summary.empty
    assert "flow_light" in set(feature_summary["feature"].astype(str))
    assert "t-0" in set(temporal_summary["time_step"].astype(str))


def test_attention_model_detects_drift_and_reports_attention_shift():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=220, drift_start=140, random_state=37)
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_TORCH_MLP_ATTENTION],
        "strategies": [neural_drift_app.STRATEGY_RECALIBRATION],
        "lookback_steps": 8,
        "recent_window_size": 24,
        "recalibration_min_rows": 16,
        "max_stream_rows": 48,
        "mlp_epochs": 2,
        "drift_monitor_epochs": 2,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    summary = results["summary"]
    shift_df = results["attention_drift_shift_summary"]

    assert int(summary.loc[0, "n_drift_events"]) > 0
    assert not shift_df.empty
    assert float(shift_df["abs_delta_attention"].max()) > 0.0
