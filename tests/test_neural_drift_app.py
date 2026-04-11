import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import src.Neural_drift_app as neural_drift_app
import src.neural_drift_experiments as neural_drift_experiments


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


def _mock_checkpoint_runtime(*, strategies: list[str]) -> dict:
    model_name = neural_drift_app.MODEL_XGBOOST
    balance_mode = neural_drift_app.BALANCE_MODE_NONE
    baseline_key = neural_drift_app._build_neural_drift_baseline_key(model_name, balance_mode)
    experiment_specs = [
        {
            "experiment_key": neural_drift_app._build_neural_drift_experiment_key(
                model_name,
                strategy,
                balance_mode,
            ),
            "baseline_key": baseline_key,
            "model": model_name,
            "strategy": strategy,
            "balance_mode": balance_mode,
        }
        for strategy in strategies
    ]
    return {
        "dataset": {"name": "synthetic"},
        "split": {
            "name": "split",
            "X_train": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float),
            "y_train": np.asarray([0, 1], dtype=int),
            "X_val": np.asarray([[0.5, 0.6]], dtype=float),
            "y_val": np.asarray([1], dtype=int),
        },
        "feature_metadata": {},
        "selected_channels": [neural_drift_app.DRIFT_INPUT],
        "baseline_specs": [
            {
                "baseline_key": baseline_key,
                "model": model_name,
                "balance_mode": balance_mode,
                "experiment_specs": experiment_specs,
            }
        ],
        "experiment_specs": experiment_specs,
    }


def _fake_baseline_row(model_name: str, balance_mode: str, canonical_artifact: dict, split: dict) -> dict:
    return {
        "model": str(model_name),
        "balance_mode": str(balance_mode),
        "status": "baseline",
        "pr_auc": 0.74,
        "recall": 0.61,
        "fnr": 0.39,
        "brier": 0.19,
    }


def _fake_finalize_backtest_results(
    *,
    baseline_rows: list[dict],
    stream_rows: list[dict],
    drift_rows: list[dict],
    attention_rows: list[dict],
    detector_attention_rows: list[dict],
    **kwargs,
) -> dict:
    baseline_df = pd.DataFrame(baseline_rows)
    stream_df = pd.DataFrame(stream_rows)
    drift_df = pd.DataFrame(drift_rows)
    if stream_df.empty:
        summary_df = pd.DataFrame()
        rolling_df = pd.DataFrame()
    else:
        stream_row = dict(stream_rows[-1])
        summary_df = pd.DataFrame(
            [
                {
                    "model": str(stream_row["model"]),
                    "strategy": str(stream_row["strategy"]),
                    "balance_mode": str(stream_row["balance_mode"]),
                    "pr_auc": 0.74,
                    "recall": 0.61,
                    "fnr": 0.39,
                    "brier": 0.19,
                    "n_drift_events": int(len(drift_rows)),
                    "monitor_effective_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE,
                }
            ]
        )
        rolling_df = pd.DataFrame(
            [
                {
                    "timestamp": stream_row["timestamp"],
                    "model": str(stream_row["model"]),
                    "strategy": str(stream_row["strategy"]),
                    "balance_mode": str(stream_row["balance_mode"]),
                    "pr_auc": 0.74,
                    "recall": 0.61,
                    "fnr": 0.39,
                    "brier": 0.19,
                    "severity_score": float(stream_row["severity_score"]),
                }
            ]
        )
    return {
        "baseline": baseline_df,
        "summary": summary_df,
        "stream_metrics": stream_df,
        "rolling_metrics": rolling_df,
        "drift_events": drift_df,
        "attention_feature_summary": pd.DataFrame(attention_rows),
        "attention_temporal_summary": pd.DataFrame(),
        "attention_drift_shift_summary": pd.DataFrame(),
        "detector_attention_temporal_summary": pd.DataFrame(detector_attention_rows),
        "detector_attention_drift_shift_summary": pd.DataFrame(),
    }


def _fixture_results(
    *,
    strategy: str = neural_drift_app.STRATEGY_FIXED,
) -> dict:
    baseline_rows = [
        {
            "model": neural_drift_app.MODEL_XGBOOST,
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "status": "baseline",
            "pr_auc": 0.74,
        }
    ]
    timestamp = pd.Timestamp("2024-01-01 00:00:00")
    stream_rows = [
        {
            "timestamp": timestamp,
            "model": neural_drift_app.MODEL_XGBOOST,
            "strategy": strategy,
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "y_true": 1,
            "prediction": 1,
            "score": 0.82,
            "decision_threshold": 0.5,
            "severity_score": 0.71,
            "max_channel_score": 0.66,
            "severity_threshold": 0.6,
            "is_drift_event": True,
            "action_taken": "recalibration" if strategy != neural_drift_app.STRATEGY_FIXED else "none",
        }
    ]
    drift_rows = [
        {
            "timestamp": timestamp,
            "model": neural_drift_app.MODEL_XGBOOST,
            "strategy": strategy,
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "channel": neural_drift_app.DRIFT_INPUT,
            "severity_score": 0.71,
        }
    ]
    return _fake_finalize_backtest_results(
        baseline_rows=baseline_rows,
        stream_rows=stream_rows,
        drift_rows=drift_rows,
        attention_rows=[],
        detector_attention_rows=[],
    )


def _write_persisted_fixture_run(
    root: Path,
    *,
    run_id: str,
    updated_at: str,
    run_signature: str,
    results: dict | None = None,
) -> Path:
    payload = results or _fixture_results()
    run_dir = root / run_id
    artifacts_dir = run_dir / "artifacts"
    experiments_dir = run_dir / "experiments"
    baselines_dir = run_dir / "baselines"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {}
    for key, value in payload.items():
        if not isinstance(value, pd.DataFrame) or value.empty:
            continue
        path = artifacts_dir / f"{key}.csv"
        value.to_csv(path, index=False)
        artifact_paths[key] = str(path)

    baseline_key = neural_drift_app._build_neural_drift_baseline_key(
        neural_drift_app.MODEL_XGBOOST,
        neural_drift_app.BALANCE_MODE_NONE,
    )
    experiment_key = neural_drift_app._build_neural_drift_experiment_key(
        neural_drift_app.MODEL_XGBOOST,
        neural_drift_app.STRATEGY_FIXED,
        neural_drift_app.BALANCE_MODE_NONE,
    )
    baseline_path = baselines_dir / f"{baseline_key}.csv"
    payload["baseline"].to_csv(baseline_path, index=False)
    experiment_dir = experiments_dir / experiment_key
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for key in [
        "summary",
        "stream_metrics",
        "rolling_metrics",
        "drift_events",
    ]:
        if isinstance(payload.get(key), pd.DataFrame) and not payload[key].empty:
            payload[key].to_csv(experiment_dir / f"{key}.csv", index=False)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_signature": run_signature,
        "run_type": neural_drift_app.NEURAL_DRIFT_RUN_TYPE,
        "status": "completed",
        "result_status": "success",
        "created_at": updated_at,
        "updated_at": updated_at,
        "dataset_context": {
            "source": "test",
            "rows_total": 64,
            "rows_used": 48,
            "feature_cols": ["flow_light"],
            "feature_export_path": "",
            "selection_metadata": {},
            "feature_source_choice": "test",
        },
        "config": {
            **neural_drift_app.DEFAULT_CONFIG,
            "models": [neural_drift_app.MODEL_XGBOOST],
            "strategies": [neural_drift_app.STRATEGY_FIXED],
            "balance_modes": [neural_drift_app.BALANCE_MODE_NONE],
        },
        "baseline_index": {
            baseline_key: {
                "baseline_key": baseline_key,
                "model": neural_drift_app.MODEL_XGBOOST,
                "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
                "status": "completed",
                "artifact_paths": {"baseline": str(baseline_path)},
                "error": None,
            }
        },
        "experiment_index": {
            experiment_key: {
                "experiment_key": experiment_key,
                "baseline_key": baseline_key,
                "model": neural_drift_app.MODEL_XGBOOST,
                "strategy": neural_drift_app.STRATEGY_FIXED,
                "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
                "status": "completed",
                "artifact_paths": {
                    key: str(experiment_dir / f"{key}.csv")
                    for key in ["summary", "stream_metrics", "rolling_metrics", "drift_events"]
                },
                "error": None,
            }
        },
        "artifacts": artifact_paths,
        "last_error": None,
        "resume": {
            "auto_resumed": False,
            "checkpoint_status": "completed",
        },
    }
    neural_drift_app._update_neural_drift_manifest_progress(manifest)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(neural_drift_app._to_json_safe(manifest), indent=2),
        encoding="utf-8",
    )
    return manifest_path


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


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected_device"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_resolve_torch_device_prioritizes_cuda_then_mps(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected_device: str,
):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return bool(cuda_available)

    class _FakeMPS:
        @staticmethod
        def is_available() -> bool:
            return bool(mps_available)

    class _FakeBackends:
        mps = _FakeMPS()

    class _FakeTorch:
        cuda = _FakeCuda()
        backends = _FakeBackends()

        @staticmethod
        def device(name: str) -> str:
            return str(name)

    monkeypatch.setattr(neural_drift_app, "torch", _FakeTorch())

    assert neural_drift_app._resolve_torch_device() == expected_device


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


def test_configuration_controls_explanation_mentions_trigger_rule_and_actions():
    explanation = neural_drift_app._build_configuration_controls_explanation(
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "severity_threshold": 0.40,
            "balance_modes": [
                neural_drift_app.BALANCE_MODE_NONE,
                neural_drift_app.BALANCE_MODE_SMOTE,
            ],
            "recent_window_size": 48,
            "models": [neural_drift_app.MODEL_XGBOOST, neural_drift_app.MODEL_TORCH_MLP],
            "strategies": [
                neural_drift_app.STRATEGY_FIXED,
                neural_drift_app.STRATEGY_RECALIBRATION,
                neural_drift_app.STRATEGY_FINE_TUNING,
                neural_drift_app.STRATEGY_RETRAIN,
            ],
            "drift_channels": [neural_drift_app.DRIFT_INPUT, neural_drift_app.DRIFT_EMBEDDING],
        }
    )

    assert "severity_score >= 0.40" in explanation["decision_rule"]
    assert any("Severity trigger = 0.40" in step[0] for step in explanation["sensitivity_steps"])
    assert any("Balance modes" in step[0] and "smote" in step[1] for step in explanation["execution_steps"])
    assert any("XGBoost fine-tuning metric" in step[0] for step in explanation["execution_steps"])
    assert any("recalibration" in step[1] and "retrain" in step[1] for step in explanation["execution_steps"])
    assert any("Ventana adaptativa de XGBoost" in step[0] for step in explanation["adaptation_steps"])
    assert any("action_taken" in step[1] for step in explanation["tuning_guidance"])


def test_render_configuration_subtab_groups_controls_with_internal_tabs(
    monkeypatch: pytest.MonkeyPatch,
):
    _missing = object()

    class _FakeTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeExpander:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeColumn:
        def __init__(self, root):
            self._root = root

        def __getattr__(self, name):
            return getattr(self._root, name)

    class FakeStreamlit:
        def __init__(self):
            self.session_state: dict = {}
            self.tabs_calls: list[list[str]] = []
            self.markdown_calls: list[str] = []
            self.write_calls: list[str] = []
            self.caption_calls: list[str] = []
            self.info_calls: list[str] = []
            self.expander_labels: list[str] = []
            self.number_input_call_kwargs: dict[str, dict[str, object]] = {}
            self.selectbox_call_kwargs: dict[str, dict[str, object]] = {}

        def tabs(self, labels):
            label_list = list(labels)
            self.tabs_calls.append(label_list)
            return [_FakeTab() for _ in label_list]

        def columns(self, spec):
            count = spec if isinstance(spec, int) else len(spec)
            return [_FakeColumn(self) for _ in range(count)]

        def markdown(self, message):
            self.markdown_calls.append(str(message))

        def write(self, message):
            self.write_calls.append(str(message))

        def caption(self, message):
            self.caption_calls.append(str(message))

        def info(self, message):
            self.info_calls.append(str(message))

        def metric(self, label, value, *args, **kwargs):
            return None

        def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **kwargs):
            resolved = value if value is not None else min_value
            if key is not None:
                self.session_state.setdefault(key, resolved)
                return self.session_state[key]
            return resolved

        def number_input(
            self,
            label,
            min_value=None,
            max_value=None,
            value=_missing,
            step=None,
            key=None,
            **kwargs,
        ):
            call_kwargs = {
                "min_value": min_value,
                "max_value": max_value,
                "step": step,
                **kwargs,
            }
            if value is not _missing:
                call_kwargs["value"] = value
            if key is not None:
                self.number_input_call_kwargs[str(key)] = call_kwargs
            resolved = value if value is not _missing else min_value
            if key is not None:
                self.session_state.setdefault(key, resolved)
                return self.session_state[key]
            return resolved

        def selectbox(self, label, options, index=_missing, key=None, **kwargs):
            option_list = list(options)
            call_kwargs = dict(kwargs)
            if index is not _missing:
                call_kwargs["index"] = index
                resolved = option_list[index]
            else:
                resolved = option_list[0]
            if key is not None:
                self.selectbox_call_kwargs[str(key)] = call_kwargs
            if key is not None:
                self.session_state.setdefault(key, resolved)
                return self.session_state[key]
            return resolved

        def multiselect(self, label, options, default=None, key=None, **kwargs):
            resolved = list(default or [])
            if key is not None:
                self.session_state.setdefault(key, resolved)
                return list(self.session_state[key])
            return resolved

        def expander(self, label, expanded=False):
            self.expander_labels.append(str(label))
            return _FakeExpander()

    fake_st = FakeStreamlit()
    bundle = _feature_bundle(
        neural_drift_app.generate_synthetic_neural_drift_dataset(rows=64, random_state=19)
    )
    bundle["feature_export_path"] = "/tmp/neural_drift_selected.duckdb"

    monkeypatch.setattr(neural_drift_app, "st", fake_st)

    config = neural_drift_app._render_configuration_subtab(bundle)

    assert fake_st.tabs_calls[0] == ["General", "ADWIN", "Modelos", "Adaptación y XGBoost"]
    assert fake_st.session_state["neural_drift_config"] == config
    assert "**Como interpretar esta configuracion**" in fake_st.markdown_calls
    assert "Lectura guiada de los controles" in fake_st.expander_labels
    assert any(
        message.startswith("Esta configuracion define que tramo temporal se evalua")
        for message in fake_st.write_calls
    )
    for key in (
        "dataset_percent",
        "lookback_steps",
        "detector_adwin_delta",
        "models",
        "strategies",
        "recent_window_size",
        "xgb_fine_tune_selection_metric",
        "xgb_fine_tune_window_min",
        "xgb_fine_tune_window_max",
        "xgb_fine_tune_rounds_min",
        "xgb_fine_tune_rounds_max",
        "xgb_fine_tune_eta_multiplier_max",
        "xgb_fine_tune_recent_weight_max",
    ):
        assert key in config
    assert config["dataset_percent"] == int(neural_drift_app.DEFAULT_CONFIG["dataset_percent"])
    assert config["lookback_steps"] == int(neural_drift_app.DEFAULT_CONFIG["lookback_steps"])
    assert config["recent_window_size"] == int(neural_drift_app.DEFAULT_CONFIG["recent_window_size"])
    assert config["xgb_fine_tune_selection_metric"] == neural_drift_app.DEFAULT_CONFIG["xgb_fine_tune_selection_metric"]
    assert config["xgb_fine_tune_window_min"] == int(neural_drift_app.DEFAULT_CONFIG["xgb_fine_tune_window_min"])
    assert config["xgb_fine_tune_window_max"] == int(neural_drift_app.DEFAULT_CONFIG["xgb_fine_tune_window_max"])
    assert config["xgb_fine_tune_rounds_min"] == int(neural_drift_app.DEFAULT_CONFIG["xgb_fine_tune_rounds_min"])
    assert config["xgb_fine_tune_rounds_max"] == int(neural_drift_app.DEFAULT_CONFIG["xgb_fine_tune_rounds_max"])
    assert config["xgb_fine_tune_eta_multiplier_max"] == pytest.approx(
        float(neural_drift_app.DEFAULT_CONFIG["xgb_fine_tune_eta_multiplier_max"])
    )
    assert config["xgb_fine_tune_recent_weight_max"] == pytest.approx(
        float(neural_drift_app.DEFAULT_CONFIG["xgb_fine_tune_recent_weight_max"])
    )
    for key in (
        "neural_drift_xgb_fine_tune_window_min",
        "neural_drift_xgb_fine_tune_window_max",
        "neural_drift_xgb_fine_tune_rounds_min",
        "neural_drift_xgb_fine_tune_rounds_max",
        "neural_drift_xgb_fine_tune_eta_multiplier_max",
        "neural_drift_xgb_fine_tune_recent_weight_max",
    ):
        assert "value" not in fake_st.number_input_call_kwargs[key]
    assert "index" not in fake_st.selectbox_call_kwargs["neural_drift_xgb_fine_tune_selection_metric"]


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


def test_run_signature_changes_when_balance_modes_change():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=48, random_state=25)
    bundle = _feature_bundle(df)

    none_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "balance_modes": [neural_drift_app.BALANCE_MODE_NONE],
        },
    )
    smote_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "balance_modes": [
                neural_drift_app.BALANCE_MODE_NONE,
                neural_drift_app.BALANCE_MODE_SMOTE,
            ],
        },
    )

    assert none_signature != smote_signature


def test_run_signature_changes_when_xgb_fine_tune_selection_metric_changes():
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=48, random_state=27)
    bundle = _feature_bundle(df)

    fbeta_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "xgb_fine_tune_selection_metric": neural_drift_app.XGB_FINE_TUNE_SELECTION_F_BETA_RECALL,
        },
    )
    pr_auc_signature = neural_drift_app._build_run_signature(
        bundle,
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "xgb_fine_tune_selection_metric": neural_drift_app.XGB_FINE_TUNE_SELECTION_PR_AUC,
        },
    )

    assert fbeta_signature != pr_auc_signature


def test_severity_intensity_clips_and_grows_monotonically():
    low = neural_drift_app._severity_intensity(0.10, 0.50)
    edge = neural_drift_app._severity_intensity(0.50, 0.50)
    mid = neural_drift_app._severity_intensity(0.75, 0.50)
    high = neural_drift_app._severity_intensity(1.25, 0.50)

    assert low == 0.0
    assert edge == 0.0
    assert mid == pytest.approx(0.5)
    assert high == 1.0
    assert [low, edge, mid, high] == sorted([low, edge, mid, high])


def test_xgb_fine_tune_window_rows_scales_with_severity():
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "xgb_fine_tune_window_min": 32,
        "xgb_fine_tune_window_max": 160,
    }

    assert neural_drift_app._xgb_fine_tune_window_rows(0.0, config=config) == 32
    assert neural_drift_app._xgb_fine_tune_window_rows(0.5, config=config) == 96
    assert neural_drift_app._xgb_fine_tune_window_rows(1.0, config=config) == 160


def test_xgb_recent_sample_weights_increase_linearly():
    weights = neural_drift_app._xgb_recent_sample_weights(5, 4.0)

    assert weights.shape == (5,)
    assert weights[0] == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(4.0)
    assert np.all(np.diff(weights) > 0.0)


def test_classification_metrics_include_balanced_f1_and_mcc():
    y_true = np.array([0, 0, 1, 1], dtype=int)
    scores = np.array([0.10, 0.80, 0.70, 0.20], dtype=float)

    metrics = neural_drift_app._classification_metrics(y_true, scores, threshold=0.5)

    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["balanced_f1"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_train_xgboost_model_imputes_before_smote(monkeypatch: pytest.MonkeyPatch):
    class _FakeEstimator:
        def __init__(self, **params):
            self._params = dict(params)

        def fit(self, X, y, **kwargs):
            X_np = np.asarray(X, dtype=float)
            assert not np.isnan(X_np).any()
            return self

        def predict_proba(self, X):
            X_np = np.asarray(X, dtype=float)
            base = np.clip(0.3 + 0.05 * np.nan_to_num(X_np[:, 0], nan=0.0), 0.05, 0.95)
            return np.column_stack([1.0 - base, base])

        def get_params(self):
            return dict(self._params)

    class _FakeXGBModule:
        XGBClassifier = _FakeEstimator

    smote_calls = {"count": 0}

    def fake_apply_smote_balance(X, y, **kwargs):
        X_np = np.asarray(X, dtype=float)
        smote_calls["count"] += 1
        assert not np.isnan(X_np).any()
        return X_np, np.asarray(y).astype(int), {
            "applied": True,
            "original_rows": int(len(y)),
            "balanced_rows": int(len(y)),
            "sampling_strategy": float(kwargs["sampling_strategy"]),
            "k_neighbors": int(kwargs["k_neighbors"]),
        }

    monkeypatch.setattr(neural_drift_app, "_import_external_xgboost", lambda: _FakeXGBModule())
    monkeypatch.setattr(neural_drift_app, "_apply_smote_balance", fake_apply_smote_balance)

    X_train = np.array(
        [
            [1.0, np.nan],
            [2.0, 3.0],
            [3.0, np.nan],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, np.nan],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, np.nan],
            [10.0, 11.0],
        ],
        dtype=float,
    )
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=int)
    X_val = np.array(
        [
            [1.5, np.nan],
            [2.5, 3.5],
            [3.5, np.nan],
            [4.5, 5.5],
            [5.5, 6.5],
            [6.5, np.nan],
            [7.5, 8.5],
            [8.5, 9.5],
            [9.5, np.nan],
            [10.5, 11.5],
        ],
        dtype=float,
    )
    y_val = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=int)

    artifact = neural_drift_app._train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "xgb_parallel_neural_enabled": False,
        },
        balance_mode=neural_drift_app.BALANCE_MODE_SMOTE,
    )

    assert smote_calls["count"] == 1
    assert artifact["kind"] == "xgboost"


def test_train_xgboost_model_attaches_parallel_neural_branch(monkeypatch: pytest.MonkeyPatch):
    class _FakeEstimator:
        def __init__(self, **params):
            self._params = dict(params)

        def fit(self, X, y, **kwargs):
            return self

        def predict_proba(self, X):
            X_np = np.asarray(X, dtype=float)
            base = np.clip(0.35 + 0.02 * np.nan_to_num(X_np[:, 0], nan=0.0), 0.05, 0.95)
            return np.column_stack([1.0 - base, base])

        def get_params(self):
            return dict(self._params)

    class _FakeXGBModule:
        XGBClassifier = _FakeEstimator

    parallel_branch_calls: list[dict] = []

    def fake_train_parallel_branch(X_train, y_train, X_val, y_val, *, config, balance_mode, smote_params):
        parallel_branch_calls.append(
            {
                "rows_train": int(len(y_train)),
                "rows_val": int(len(y_val)),
                "balance_mode": str(balance_mode),
                "smote_params": dict(smote_params or {}),
            }
        )
        return {
            "kind": "torch_mlp",
            "model_name": neural_drift_app.XGB_PARALLEL_NEURAL_MODEL,
            "balance_mode": str(balance_mode),
            "smote_params": dict(smote_params or {}),
            "embedding_monitor": {"kind": neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE},
            "monitor_effective_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE,
            "attention_summary_reference": None,
            "reference": {
                "score_mean": 0.62,
                "score_std": 0.08,
                "error_mean": 0.12,
                "error_std": 0.03,
                "embedding_centroid": np.asarray([0.1, -0.1], dtype=float),
                "embedding_distance_mean": 0.05,
                "embedding_distance_std": 0.01,
                "embedding_reconstruction_mean": 0.02,
                "embedding_reconstruction_std": 0.01,
            },
        }

    monkeypatch.setattr(neural_drift_app, "_import_external_xgboost", lambda: _FakeXGBModule())
    monkeypatch.setattr(neural_drift_app, "_train_xgb_parallel_neural_branch", fake_train_parallel_branch)

    X_train = np.arange(40, dtype=float).reshape(20, 2)
    y_train = np.tile(np.array([0, 1], dtype=int), 10)
    X_val = np.arange(20, 40, dtype=float).reshape(10, 2)
    y_val = np.tile(np.array([0, 1], dtype=int), 5)

    artifact = neural_drift_app._train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        config=neural_drift_app.DEFAULT_CONFIG,
        balance_mode=neural_drift_app.BALANCE_MODE_NONE,
    )

    assert parallel_branch_calls == [
        {
            "rows_train": 20,
            "rows_val": 10,
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "smote_params": {},
        }
    ]
    assert artifact["parallel_neural_enabled"] is True
    assert artifact["parallel_neural_model"] == neural_drift_app.XGB_PARALLEL_NEURAL_MODEL
    assert artifact["drift_monitor_source"] == neural_drift_app.DRIFT_MONITOR_SOURCE_XGB_PARALLEL_NEURAL_BRANCH
    assert artifact["monitor_effective_architecture"] == neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE
    assert "aux_score_mean" in artifact["reference"]
    assert "embedding_reconstruction_mean" in artifact["reference"]


def test_xgb_base_learning_rate_falls_back_when_model_reports_invalid_overflow():
    class _FakeModel:
        def get_params(self) -> dict:
            return {"learning_rate": np.finfo(np.float32).max}

    artifact = {"model": _FakeModel()}

    assert neural_drift_app._xgb_base_learning_rate(artifact) == pytest.approx(0.05)


def test_live_backtest_chart_frames_include_drift_and_rolling_metrics():
    stream_rows = [
        {
            "timestamp": pd.Timestamp("2024-01-01 00:00:00"),
            "model": neural_drift_app.MODEL_XGBOOST,
            "strategy": neural_drift_app.STRATEGY_FIXED,
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "y_true": 0,
            "prediction": 0,
            "score": 0.20,
            "decision_threshold": 0.50,
            "severity_score": 0.10,
            "max_channel_score": 0.12,
            "severity_threshold": 0.60,
            "is_drift_event": False,
            "action_taken": "none",
        },
        {
            "timestamp": pd.Timestamp("2024-01-01 00:05:00"),
            "model": neural_drift_app.MODEL_XGBOOST,
            "strategy": neural_drift_app.STRATEGY_FIXED,
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "y_true": 1,
            "prediction": 1,
            "score": 0.80,
            "decision_threshold": 0.50,
            "severity_score": 0.75,
            "max_channel_score": 0.82,
            "severity_threshold": 0.60,
            "is_drift_event": True,
            "action_taken": "recalibration",
        },
        {
            "timestamp": pd.Timestamp("2024-01-01 00:10:00"),
            "model": neural_drift_app.MODEL_XGBOOST,
            "strategy": neural_drift_app.STRATEGY_FIXED,
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "y_true": 1,
            "prediction": 0,
            "score": 0.30,
            "decision_threshold": 0.50,
            "severity_score": 0.40,
            "max_channel_score": 0.45,
            "severity_threshold": 0.60,
            "is_drift_event": False,
            "action_taken": "none",
        },
    ]

    drift_chart, metrics_chart = neural_drift_app._live_backtest_chart_frames(
        stream_rows,
        rolling_window=2,
    )

    assert list(drift_chart.columns) == [
        "severity_score",
        "max_channel_score",
        "severity_threshold",
        "drift_event_flag",
        "adaptation_flag",
    ]
    assert drift_chart["drift_event_flag"].tolist() == [0, 1, 0]
    assert drift_chart["adaptation_flag"].tolist() == [0, 1, 0]
    assert list(metrics_chart.columns) == ["recall", "fnr", "brier"]
    assert float(metrics_chart.iloc[-1]["recall"]) == pytest.approx(0.5)
    assert float(metrics_chart.iloc[-1]["fnr"]) == pytest.approx(0.5)
    assert float(metrics_chart.iloc[-1]["brier"]) == pytest.approx(0.265)


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


def test_run_backtest_pipeline_includes_fine_tuning_for_xgboost():
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
    assert set(summary["strategy"].astype(str)) == {
        neural_drift_app.STRATEGY_FIXED,
        neural_drift_app.STRATEGY_FINE_TUNING,
        neural_drift_app.STRATEGY_RETRAIN,
    }
    assert set(summary["monitor_effective_architecture"].astype(str)) != {"not_available"}
    assert set(summary["parallel_neural_enabled"].astype(bool)) == {True}
    assert set(summary["parallel_neural_model"].astype(str)) == {neural_drift_app.XGB_PARALLEL_NEURAL_MODEL}
    assert set(summary["drift_monitor_source"].astype(str)) == {
        neural_drift_app.DRIFT_MONITOR_SOURCE_XGB_PARALLEL_NEURAL_BRANCH
    }


def test_build_channel_scores_combines_xgb_and_parallel_neural_signals():
    artifact = {
        "reference": {
            "feature_mean": np.zeros(2, dtype=float),
            "feature_std": np.ones(2, dtype=float),
            "input_stat_mean": 0.0,
            "input_stat_std": 1.0,
            "score_mean": 0.50,
            "score_std": 0.20,
            "error_mean": 0.10,
            "error_std": 0.20,
            "aux_score_mean": 0.20,
            "aux_score_std": 0.05,
            "aux_error_mean": 0.05,
            "aux_error_std": 0.05,
        },
        "monitor_effective_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE,
        "drift_monitor_source": neural_drift_app.DRIFT_MONITOR_SOURCE_XGB_PARALLEL_NEURAL_BRANCH,
    }
    detectors = {
        neural_drift_app.DRIFT_INPUT: neural_drift_app.ClassicDriftDetector(rolling_window=8),
        neural_drift_app.DRIFT_SCORE: neural_drift_app.ClassicDriftDetector(rolling_window=8),
        neural_drift_app.DRIFT_ERROR: neural_drift_app.ClassicDriftDetector(rolling_window=8),
        "score_neural": neural_drift_app.ClassicDriftDetector(rolling_window=8),
        "error_neural": neural_drift_app.ClassicDriftDetector(rolling_window=8),
    }

    payload = neural_drift_app._build_channel_scores(
        artifact=artifact,
        x_row=np.asarray([0.1, -0.2], dtype=float),
        calibrated_score=0.55,
        auxiliary_calibrated_score=0.95,
        y_true=0,
        embeddings=np.asarray([], dtype=float),
        recent_embedding_history=None,
        selected_channels=[neural_drift_app.DRIFT_SCORE, neural_drift_app.DRIFT_ERROR],
        detectors=detectors,
    )

    assert payload["channel_scores"][neural_drift_app.DRIFT_SCORE] == pytest.approx(1.0)
    assert payload["channel_scores"][neural_drift_app.DRIFT_ERROR] == pytest.approx(1.0)
    assert payload["channel_scores"]["score_components"]["combined"] == pytest.approx(
        payload["channel_scores"][neural_drift_app.DRIFT_SCORE]
    )
    assert payload["channel_scores"]["error_components"]["combined"] == pytest.approx(
        payload["channel_scores"][neural_drift_app.DRIFT_ERROR]
    )
    assert payload["raw_channel_values"]["score_components"]["parallel_neural"] > payload["raw_channel_values"]["score_components"]["xgboost"]
    assert payload["raw_channel_values"]["error_components"]["parallel_neural"] > payload["raw_channel_values"]["error_components"]["xgboost"]
    assert payload["drift_monitor_source"] == neural_drift_app.DRIFT_MONITOR_SOURCE_XGB_PARALLEL_NEURAL_BRANCH


def test_run_backtest_pipeline_compares_none_and_smote_results(monkeypatch: pytest.MonkeyPatch):
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=180, drift_start=110, random_state=71)

    def fake_train_canonical_artifact(model_name, X_train, y_train, X_val, y_val, *, config, balance_mode, feature_metadata=None):
        return {
            "kind": "fake",
            "model_name": str(model_name),
            "balance_mode": str(balance_mode),
            "smote_params": (
                {"sampling_strategy": 1.0, "k_neighbors": 3}
                if str(balance_mode) == neural_drift_app.BALANCE_MODE_SMOTE
                else {}
            ),
            "smote_fit_info": {
                "applied": str(balance_mode) == neural_drift_app.BALANCE_MODE_SMOTE,
                "balanced_rows": int(len(y_train) + (4 if str(balance_mode) == neural_drift_app.BALANCE_MODE_SMOTE else 0)),
            },
            "imputer": np.zeros(X_train.shape[1], dtype=float),
            "calibrator": None,
            "decision_threshold": 0.5,
            "base_threshold": 0.5,
            "monitor_effective_architecture": "not_available",
            "attention_summary_reference": {},
            "embedding_monitor": None,
        }

    def fake_predict_with_artifact_details(artifact, X):
        base_score = 0.65 if artifact["balance_mode"] == neural_drift_app.BALANCE_MODE_SMOTE else 0.35
        return {
            "probs": np.full(len(X), base_score, dtype=float),
            "embeddings": np.empty((len(X), 0), dtype=float),
            "attention_summary": None,
        }

    def fake_build_channel_scores(**kwargs):
        return {
            "channel_scores": {},
            "raw_channel_values": {},
            "detector_flags": {},
            "severity_score": 0.10,
            "max_channel_score": 0.10,
            "severity_label": "leve",
            "detector_attention_summary": None,
            "monitor_warmup": False,
            "monitor_effective_architecture": "not_available",
        }

    monkeypatch.setattr(neural_drift_app, "_train_canonical_artifact", fake_train_canonical_artifact)
    monkeypatch.setattr(neural_drift_app, "_predict_with_artifact_details", fake_predict_with_artifact_details)
    monkeypatch.setattr(neural_drift_app, "_build_channel_scores", fake_build_channel_scores)

    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [
            neural_drift_app.STRATEGY_FIXED,
        ],
        "balance_modes": [
            neural_drift_app.BALANCE_MODE_NONE,
            neural_drift_app.BALANCE_MODE_SMOTE,
        ],
        "max_stream_rows": 32,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    summary = results["summary"]
    baseline = results["baseline"]
    stream_metrics = results["stream_metrics"]
    rolling_metrics = results["rolling_metrics"]
    drift_events = results["drift_events"]

    assert set(summary["balance_mode"].astype(str)) == {
        neural_drift_app.BALANCE_MODE_NONE,
        neural_drift_app.BALANCE_MODE_SMOTE,
    }
    assert set(baseline["balance_mode"].astype(str)) == {
        neural_drift_app.BALANCE_MODE_NONE,
        neural_drift_app.BALANCE_MODE_SMOTE,
    }
    assert set(stream_metrics["balance_mode"].astype(str)) == {
        neural_drift_app.BALANCE_MODE_NONE,
        neural_drift_app.BALANCE_MODE_SMOTE,
    }
    assert set(rolling_metrics["balance_mode"].astype(str)) == {
        neural_drift_app.BALANCE_MODE_NONE,
        neural_drift_app.BALANCE_MODE_SMOTE,
    }
    assert "balance_mode" in drift_events.columns
    assert "smote_sampling_strategy" in baseline.columns


def test_fine_tune_artifact_uses_dynamic_window_and_recency_weights(monkeypatch: pytest.MonkeyPatch):
    class _FakeBooster:
        def __init__(self, rounds: int):
            self._rounds = int(rounds)

        def num_boosted_rounds(self) -> int:
            return int(self._rounds)

    class _FakeModel:
        def __init__(self, rounds: int, learning_rate: float = 0.05):
            self.rounds = int(rounds)
            self.learning_rate = float(learning_rate)

        def get_params(self) -> dict:
            return {"learning_rate": float(self.learning_rate)}

        def get_booster(self) -> _FakeBooster:
            return _FakeBooster(self.rounds)

    train_calls: list[dict] = []
    recalibration_rows: list[int] = []

    def fake_train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        config,
        balance_mode,
        smote_params,
        base_model=None,
        imputer=None,
        n_estimators_override=None,
        learning_rate_override=None,
        sample_weight=None,
        history_training_rows_base=0,
        xgb_fine_tune_metadata=None,
    ):
        train_calls.append(
            {
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)),
                "rounds": int(n_estimators_override),
                "learning_rate_override": float(learning_rate_override),
                "sample_weight": np.asarray(sample_weight, dtype=float),
            }
        )
        total_rounds = int(getattr(base_model, "rounds", 0)) + int(n_estimators_override or 0)
        return {
            "kind": "xgboost",
            "model_name": neural_drift_app.MODEL_XGBOOST,
            "model": _FakeModel(total_rounds, learning_rate_override or 0.05),
            "imputer": np.asarray(imputer, dtype=float),
            "calibrator": None,
            "reference": {},
            "monitor_effective_architecture": "not_available",
            "decision_threshold": 0.55,
            "threshold_info": {"threshold": 0.55, "f_beta": 0.60, "recall": 0.70},
            "base_threshold": 0.55,
            "history_training_rows": int(history_training_rows_base + len(y_train)),
            "xgb_fine_tune_metadata": dict(xgb_fine_tune_metadata or {}),
        }

    def fake_evaluate_candidate(artifact, X_val, y_val, *, selection_metric, rounds_selected):
        primary = 1.0 if int(rounds_selected) == 13 else 0.0
        return {
            "selection_metric": str(selection_metric),
            "selection_score": float(rounds_selected),
            "f_beta": 0.60,
            "recall": 0.70,
            "pr_auc": 0.50,
            "brier": 0.20,
            "sort_key": (primary,),
        }

    def fake_recalibrate_artifact(artifact, X_recent, y_recent, *, config):
        recalibration_rows.append(int(len(y_recent)))
        artifact["decision_threshold"] = 0.42
        artifact["threshold_info"] = {"threshold": 0.42}

    monkeypatch.setattr(neural_drift_app, "_train_xgboost_model", fake_train_xgboost_model)
    monkeypatch.setattr(neural_drift_app, "_evaluate_xgb_fine_tune_candidate", fake_evaluate_candidate)
    monkeypatch.setattr(neural_drift_app, "_recalibrate_artifact", fake_recalibrate_artifact)

    artifact = {
        "kind": "xgboost",
        "model_name": neural_drift_app.MODEL_XGBOOST,
        "model": _FakeModel(10, 0.05),
        "imputer": np.zeros(4, dtype=float),
        "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
        "smote_params": {},
        "history_training_rows": 20,
        "decision_threshold": 0.5,
        "base_threshold": 0.5,
        "xgb_fine_tune_metadata": neural_drift_app._default_xgb_fine_tune_metadata(
            neural_drift_app.DEFAULT_CONFIG
        ),
    }
    X_available = np.arange(120 * 4, dtype=float).reshape(120, 4)
    y_available = np.tile(np.array([0, 1], dtype=int), 60)

    result = neural_drift_app._fine_tune_artifact(
        artifact,
        X_available,
        y_available,
        config=neural_drift_app.DEFAULT_CONFIG,
        severity_intensity=0.5,
    )

    assert bool(result["applied"]) is True
    assert all(call["train_rows"] + call["val_rows"] == 96 for call in train_calls)
    assert {call["rounds"] for call in train_calls} == {9, 11, 13, 15, 17}
    assert all(call["sample_weight"].shape == (76,) for call in train_calls)
    assert train_calls[0]["sample_weight"][0] == pytest.approx(1.0)
    assert train_calls[0]["sample_weight"][-1] == pytest.approx(2.5)
    assert all(call["learning_rate_override"] == pytest.approx(0.05 * 1.375) for call in train_calls)
    assert recalibration_rows == [96]
    assert artifact["model"].get_booster().num_boosted_rounds() == 23
    assert int(artifact["history_training_rows"]) > 20
    assert artifact["decision_threshold"] == pytest.approx(0.42)
    assert artifact["xgb_fine_tune_metadata"]["xgb_adaptation_window_rows"] == 96
    assert artifact["xgb_fine_tune_metadata"]["xgb_fine_tune_rounds_selected"] == 13
    assert artifact["xgb_fine_tune_metadata"]["xgb_fine_tune_eta_multiplier"] == pytest.approx(1.375)
    assert artifact["xgb_fine_tune_metadata"]["xgb_fine_tune_recent_weight_max"] == pytest.approx(2.5)


def test_fine_tune_artifact_sanitizes_overflowing_base_learning_rate(monkeypatch: pytest.MonkeyPatch):
    class _FakeModel:
        def __init__(self, rounds: int, learning_rate: float):
            self.rounds = int(rounds)
            self.learning_rate = float(learning_rate)

        def get_params(self) -> dict:
            return {"learning_rate": float(self.learning_rate)}

    train_calls: list[dict] = []

    def fake_train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        config,
        balance_mode,
        smote_params,
        base_model=None,
        imputer=None,
        n_estimators_override=None,
        learning_rate_override=None,
        sample_weight=None,
        history_training_rows_base=0,
        xgb_fine_tune_metadata=None,
    ):
        del X_train, y_train, X_val, y_val, config, balance_mode, smote_params, base_model, imputer
        del n_estimators_override, sample_weight, history_training_rows_base, xgb_fine_tune_metadata
        train_calls.append({"learning_rate_override": float(learning_rate_override)})
        return {
            "kind": "xgboost",
            "model_name": neural_drift_app.MODEL_XGBOOST,
            "model": _FakeModel(12, float(learning_rate_override)),
            "imputer": np.zeros(4, dtype=float),
            "calibrator": None,
            "reference": {},
            "monitor_effective_architecture": "not_available",
            "decision_threshold": 0.55,
            "threshold_info": {"threshold": 0.55, "f_beta": 0.60, "recall": 0.70},
            "base_threshold": 0.55,
            "history_training_rows": 40,
            "xgb_fine_tune_metadata": {},
        }

    def fake_evaluate_candidate(artifact, X_val, y_val, *, selection_metric, rounds_selected):
        del artifact, X_val, y_val, selection_metric
        return {
            "selection_metric": neural_drift_app.XGB_FINE_TUNE_SELECTION_PR_AUC,
            "selection_score": float(rounds_selected),
            "f_beta": 0.60,
            "recall": 0.70,
            "pr_auc": 0.50,
            "brier": 0.20,
            "sort_key": (1.0 if int(rounds_selected) == 13 else 0.0,),
        }

    monkeypatch.setattr(neural_drift_app, "_train_xgboost_model", fake_train_xgboost_model)
    monkeypatch.setattr(neural_drift_app, "_evaluate_xgb_fine_tune_candidate", fake_evaluate_candidate)
    monkeypatch.setattr(neural_drift_app, "_recalibrate_artifact", lambda *args, **kwargs: None)

    artifact = {
        "kind": "xgboost",
        "model_name": neural_drift_app.MODEL_XGBOOST,
        "model": _FakeModel(10, np.finfo(np.float32).max),
        "imputer": np.zeros(4, dtype=float),
        "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
        "smote_params": {},
        "history_training_rows": 20,
        "decision_threshold": 0.5,
        "base_threshold": 0.5,
        "xgb_fine_tune_metadata": neural_drift_app._default_xgb_fine_tune_metadata(
            neural_drift_app.DEFAULT_CONFIG
        ),
    }
    X_available = np.arange(120 * 4, dtype=float).reshape(120, 4)
    y_available = np.tile(np.array([0, 1], dtype=int), 60)

    result = neural_drift_app._fine_tune_artifact(
        artifact,
        X_available,
        y_available,
        config=neural_drift_app.DEFAULT_CONFIG,
        severity_intensity=0.5,
    )

    assert bool(result["applied"]) is True
    assert train_calls
    assert all(call["learning_rate_override"] == pytest.approx(0.05 * 1.375) for call in train_calls)


def test_xgb_fine_tune_selection_metric_changes_candidate_choice(monkeypatch: pytest.MonkeyPatch):
    class _FakeModel:
        def __init__(self, rounds: int, learning_rate: float = 0.05):
            self.rounds = int(rounds)
            self.learning_rate = float(learning_rate)

        def get_params(self) -> dict:
            return {"learning_rate": float(self.learning_rate)}

    def fake_train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        config,
        balance_mode,
        smote_params,
        base_model=None,
        imputer=None,
        n_estimators_override=None,
        learning_rate_override=None,
        sample_weight=None,
        history_training_rows_base=0,
        xgb_fine_tune_metadata=None,
    ):
        return {
            "kind": "xgboost",
            "model_name": neural_drift_app.MODEL_XGBOOST,
            "model": _FakeModel(int(getattr(base_model, "rounds", 0)) + int(n_estimators_override or 0)),
            "imputer": np.asarray(imputer, dtype=float),
            "calibrator": None,
            "reference": {},
            "monitor_effective_architecture": "not_available",
            "decision_threshold": 0.55,
            "threshold_info": {"threshold": 0.55, "f_beta": 0.60, "recall": 0.70},
            "base_threshold": 0.55,
            "history_training_rows": int(history_training_rows_base + len(y_train)),
            "xgb_fine_tune_metadata": dict(xgb_fine_tune_metadata or {}),
        }

    def fake_evaluate_candidate(artifact, X_val, y_val, *, selection_metric, rounds_selected):
        preferred_round = {
            neural_drift_app.XGB_FINE_TUNE_SELECTION_F_BETA_RECALL: 11,
            neural_drift_app.XGB_FINE_TUNE_SELECTION_PR_AUC: 13,
            neural_drift_app.XGB_FINE_TUNE_SELECTION_BRIER: 9,
            neural_drift_app.XGB_FINE_TUNE_SELECTION_F1: 15,
            neural_drift_app.XGB_FINE_TUNE_SELECTION_ROC_AUC: 17,
            neural_drift_app.XGB_FINE_TUNE_SELECTION_BALANCED_F1: 11,
            neural_drift_app.XGB_FINE_TUNE_SELECTION_MCC: 13,
        }[str(selection_metric)]
        primary = 1.0 if int(rounds_selected) == preferred_round else 0.0
        return {
            "selection_metric": str(selection_metric),
            "selection_score": float(rounds_selected),
            "f_beta": 0.60,
            "recall": 0.70,
            "f1": 0.65,
            "roc_auc": 0.71,
            "balanced_f1": 0.63,
            "mcc": 0.22,
            "pr_auc": 0.50,
            "brier": 0.20,
            "sort_key": (primary,),
        }

    monkeypatch.setattr(neural_drift_app, "_train_xgboost_model", fake_train_xgboost_model)
    monkeypatch.setattr(neural_drift_app, "_evaluate_xgb_fine_tune_candidate", fake_evaluate_candidate)
    monkeypatch.setattr(neural_drift_app, "_recalibrate_artifact", lambda artifact, X_recent, y_recent, config: None)

    X_available = np.arange(120 * 4, dtype=float).reshape(120, 4)
    y_available = np.tile(np.array([0, 1], dtype=int), 60)

    def _run(metric: str) -> int:
        artifact = {
            "kind": "xgboost",
            "model_name": neural_drift_app.MODEL_XGBOOST,
            "model": _FakeModel(10, 0.05),
            "imputer": np.zeros(4, dtype=float),
            "balance_mode": neural_drift_app.BALANCE_MODE_NONE,
            "smote_params": {},
            "history_training_rows": 20,
            "decision_threshold": 0.5,
            "base_threshold": 0.5,
            "xgb_fine_tune_metadata": {},
        }
        neural_drift_app._fine_tune_artifact(
            artifact,
            X_available,
            y_available,
            config={
                **neural_drift_app.DEFAULT_CONFIG,
                "xgb_fine_tune_selection_metric": metric,
            },
            severity_intensity=0.5,
        )
        return int(artifact["xgb_fine_tune_metadata"]["xgb_fine_tune_rounds_selected"])

    assert _run(neural_drift_app.XGB_FINE_TUNE_SELECTION_F_BETA_RECALL) == 11
    assert _run(neural_drift_app.XGB_FINE_TUNE_SELECTION_PR_AUC) == 13
    assert _run(neural_drift_app.XGB_FINE_TUNE_SELECTION_BRIER) == 9
    assert _run(neural_drift_app.XGB_FINE_TUNE_SELECTION_F1) == 15
    assert _run(neural_drift_app.XGB_FINE_TUNE_SELECTION_ROC_AUC) == 17
    assert _run(neural_drift_app.XGB_FINE_TUNE_SELECTION_BALANCED_F1) == 11
    assert _run(neural_drift_app.XGB_FINE_TUNE_SELECTION_MCC) == 13


def test_run_backtest_pipeline_records_xgb_fine_tune_metadata_and_keeps_detector_window(
    monkeypatch: pytest.MonkeyPatch,
):
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=180, drift_start=110, random_state=81)
    detector_window_sizes: list[int] = []
    fine_tune_prefix_lengths: list[int] = []

    def fake_train_canonical_artifact(model_name, X_train, y_train, X_val, y_val, *, config, balance_mode, feature_metadata=None):
        return {
            "kind": "xgboost",
            "model_name": str(model_name),
            "balance_mode": str(balance_mode),
            "smote_params": {},
            "smote_fit_info": {"applied": False, "balanced_rows": int(len(y_train))},
            "imputer": np.zeros(X_train.shape[1], dtype=float),
            "calibrator": None,
            "decision_threshold": 0.5,
            "base_threshold": 0.5,
            "monitor_effective_architecture": "not_available",
            "attention_summary_reference": {},
            "embedding_monitor": None,
            "xgb_fine_tune_metadata": neural_drift_app._default_xgb_fine_tune_metadata(config),
        }

    def fake_predict_with_artifact_details(artifact, X):
        return {
            "probs": np.full(len(X), 0.65, dtype=float),
            "embeddings": np.empty((len(X), 0), dtype=float),
            "attention_summary": None,
        }

    def fake_build_channel_scores(**kwargs):
        detector_window_sizes.append(int(kwargs["recent_window_size"]))
        return {
            "channel_scores": {},
            "raw_channel_values": {},
            "detector_flags": {},
            "severity_score": 0.80,
            "max_channel_score": 0.90,
            "severity_label": "severo",
            "detector_attention_summary": None,
            "monitor_warmup": False,
            "monitor_effective_architecture": "not_available",
        }

    def fake_fine_tune_artifact(artifact, X_recent, y_recent, *, config, severity_intensity=None):
        fine_tune_prefix_lengths.append(int(len(y_recent)))
        return {
            "applied": True,
            "xgb_fine_tune_metadata": {
                "severity_intensity": float(severity_intensity),
                "xgb_adaptation_window_rows": 64,
                "xgb_fine_tune_rounds_selected": 13,
                "xgb_fine_tune_eta_multiplier": 1.60,
                "xgb_fine_tune_recent_weight_max": 3.40,
                "xgb_fine_tune_selection_metric": str(config["xgb_fine_tune_selection_metric"]),
                "xgb_fine_tune_selection_score": 0.77,
                "xgb_fine_tune_skip_reason": None,
            },
        }

    monkeypatch.setattr(neural_drift_app, "_train_canonical_artifact", fake_train_canonical_artifact)
    monkeypatch.setattr(neural_drift_app, "_predict_with_artifact_details", fake_predict_with_artifact_details)
    monkeypatch.setattr(neural_drift_app, "_build_channel_scores", fake_build_channel_scores)
    monkeypatch.setattr(neural_drift_app, "_fine_tune_artifact", fake_fine_tune_artifact)

    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [neural_drift_app.STRATEGY_FINE_TUNING],
        "recent_window_size": 24,
        "severity_threshold": 0.50,
        "max_stream_rows": 40,
        "xgb_fine_tune_selection_metric": neural_drift_app.XGB_FINE_TUNE_SELECTION_PR_AUC,
    }

    results = neural_drift_app.run_backtest_pipeline(_feature_bundle(df), config=config)

    drift_events = results["drift_events"]
    stream_metrics = results["stream_metrics"]

    assert detector_window_sizes and all(size == 24 for size in detector_window_sizes)
    assert fine_tune_prefix_lengths and max(fine_tune_prefix_lengths) > 24
    assert "severity_intensity" in drift_events.columns
    assert "xgb_adaptation_window_rows" in drift_events.columns
    assert "xgb_fine_tune_rounds_selected" in stream_metrics.columns
    assert "xgb_fine_tune_selection_metric" in stream_metrics.columns
    assert int(drift_events["xgb_adaptation_window_rows"].dropna().iloc[0]) == 64
    assert int(stream_metrics["xgb_fine_tune_rounds_selected"].dropna().iloc[0]) == 13
    assert float(drift_events["severity_intensity"].iloc[0]) > 0.0


def test_run_backtest_pipeline_emits_live_update_payloads(monkeypatch: pytest.MonkeyPatch):
    df = neural_drift_app.generate_synthetic_neural_drift_dataset(rows=180, drift_start=110, random_state=82)

    def fake_train_canonical_artifact(model_name, X_train, y_train, X_val, y_val, *, config, balance_mode, feature_metadata=None):
        return {
            "kind": "xgboost",
            "model_name": str(model_name),
            "balance_mode": str(balance_mode),
            "smote_params": {},
            "smote_fit_info": {"applied": False, "balanced_rows": int(len(y_train))},
            "imputer": np.zeros(X_train.shape[1], dtype=float),
            "calibrator": None,
            "decision_threshold": 0.5,
            "base_threshold": 0.5,
            "monitor_effective_architecture": "not_available",
            "attention_summary_reference": {},
            "embedding_monitor": None,
            "xgb_fine_tune_metadata": neural_drift_app._default_xgb_fine_tune_metadata(config),
        }

    def fake_predict_with_artifact_details(artifact, X):
        return {
            "probs": np.full(len(X), 0.72, dtype=float),
            "embeddings": np.empty((len(X), 0), dtype=float),
            "attention_summary": None,
        }

    def fake_build_channel_scores(**kwargs):
        return {
            "channel_scores": {},
            "raw_channel_values": {},
            "detector_flags": {},
            "severity_score": 0.78,
            "max_channel_score": 0.85,
            "severity_label": "severo",
            "detector_attention_summary": None,
            "monitor_warmup": False,
            "monitor_effective_architecture": "not_available",
        }

    monkeypatch.setattr(neural_drift_app, "_train_canonical_artifact", fake_train_canonical_artifact)
    monkeypatch.setattr(neural_drift_app, "_predict_with_artifact_details", fake_predict_with_artifact_details)
    monkeypatch.setattr(neural_drift_app, "_build_channel_scores", fake_build_channel_scores)

    live_events: list[dict] = []
    results = neural_drift_app.run_backtest_pipeline(
        _feature_bundle(df),
        config={
            **neural_drift_app.DEFAULT_CONFIG,
            "models": [neural_drift_app.MODEL_XGBOOST],
            "strategies": [neural_drift_app.STRATEGY_FIXED],
            "balance_modes": [neural_drift_app.BALANCE_MODE_NONE],
            "max_stream_rows": 12,
            "rolling_metric_window": 6,
            "severity_threshold": 0.6,
        },
        live_update_callback=live_events.append,
    )

    assert live_events
    assert live_events[0]["event"] == "simulation_start"
    stream_step_events = [event for event in live_events if event["event"] == "stream_step"]
    assert len(stream_step_events) == len(results["stream_metrics"])
    assert stream_step_events[0]["rolling_metric_window"] == 6
    assert stream_step_events[0]["stream_total_rows"] == len(results["stream_metrics"])
    assert {
        "severity_score",
        "max_channel_score",
        "severity_threshold",
        "is_drift_event",
        "action_taken",
        "score",
        "decision_threshold",
        "y_true",
        "prediction",
        "stream_step_index",
        "stream_total_rows",
    }.issubset(stream_step_events[0].keys())


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


def test_run_backtest_with_checkpoints_persists_manifest_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    bundle = _feature_bundle(
        neural_drift_app.generate_synthetic_neural_drift_dataset(rows=72, random_state=31)
    )
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [neural_drift_app.STRATEGY_FIXED],
        "balance_modes": [neural_drift_app.BALANCE_MODE_NONE],
    }
    runtime = _mock_checkpoint_runtime(strategies=[neural_drift_app.STRATEGY_FIXED])

    def fake_execute_backtest_experiment(
        model_name: str,
        strategy: str,
        balance_mode: str,
        canonical_artifact: dict,
        split: dict,
        config: dict,
        selected_channels: list[str],
        live_update_callback=None,
    ) -> dict:
        timestamp = pd.Timestamp("2024-01-01 00:00:00")
        if live_update_callback is not None:
            live_update_callback(
                {
                    "event": "simulation_start",
                    "model": model_name,
                    "strategy": strategy,
                    "balance_mode": balance_mode,
                    "stream_total_rows": 25,
                }
            )
            live_update_callback(
                {
                    "event": "stream_step",
                    "timestamp": timestamp,
                    "model": model_name,
                    "strategy": strategy,
                    "balance_mode": balance_mode,
                    "stream_step_index": 25,
                    "stream_total_rows": 25,
                    "y_true": 1,
                    "prediction": 1,
                    "score": 0.82,
                    "decision_threshold": 0.5,
                    "severity_score": 0.71,
                    "max_channel_score": 0.66,
                    "severity_threshold": 0.6,
                    "is_drift_event": True,
                    "action_taken": "none",
                    "rolling_metric_window": 24,
                }
            )
        return {
            "stream_rows": [
                {
                    "timestamp": timestamp,
                    "model": model_name,
                    "strategy": strategy,
                    "balance_mode": balance_mode,
                    "y_true": 1,
                    "prediction": 1,
                    "score": 0.82,
                    "decision_threshold": 0.5,
                    "severity_score": 0.71,
                    "max_channel_score": 0.66,
                    "severity_threshold": 0.6,
                    "is_drift_event": True,
                    "action_taken": "none",
                }
            ],
            "drift_rows": [
                {
                    "timestamp": timestamp,
                    "model": model_name,
                    "strategy": strategy,
                    "balance_mode": balance_mode,
                    "channel": neural_drift_app.DRIFT_INPUT,
                    "severity_score": 0.71,
                }
            ],
            "attention_rows": [],
            "detector_attention_rows": [],
        }

    monkeypatch.setattr(neural_drift_app, "_prepare_backtest_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(neural_drift_app, "_train_canonical_artifact", lambda *args, **kwargs: {"kind": "fake"})
    monkeypatch.setattr(neural_drift_app, "_build_baseline_result_row", _fake_baseline_row)
    monkeypatch.setattr(neural_drift_app, "_execute_backtest_experiment", fake_execute_backtest_experiment)
    monkeypatch.setattr(neural_drift_app, "_finalize_backtest_results", _fake_finalize_backtest_results)

    root = tmp_path / "runs"
    results = neural_drift_app.run_backtest_with_checkpoints(
        bundle,
        config=config,
        checkpoint_root=root,
    )

    manifest_path = Path(str(results["manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    live_status = json.loads((manifest_path.parent / "live_status.json").read_text(encoding="utf-8"))
    events = neural_drift_app._read_jsonl_records(manifest_path.parent / "live_events.jsonl")

    assert str(results["run_id"]).startswith("run_")
    assert manifest["run_id"] == results["run_id"]
    assert manifest["status"] == "completed"
    assert manifest["result_status"] == "success"
    assert manifest["run_signature"] == results["run_signature"]
    assert manifest["baseline_index"][runtime["baseline_specs"][0]["baseline_key"]]["status"] == "completed"
    experiment_key = runtime["experiment_specs"][0]["experiment_key"]
    assert manifest["experiment_index"][experiment_key]["status"] == "completed"
    assert Path(manifest["artifacts"]["summary"]).exists()
    assert Path(manifest["artifacts"]["stream_metrics"]).exists()
    assert Path(manifest["baseline_index"][runtime["baseline_specs"][0]["baseline_key"]]["artifact_paths"]["baseline"]).exists()
    assert Path(manifest["experiment_index"][experiment_key]["artifact_paths"]["summary"]).exists()
    assert live_status["status"] == "completed"
    assert live_status["result_status"] == "success"
    assert [event["event"] for event in events] == [
        "run_start",
        "baseline_complete",
        "experiment_start",
        "experiment_complete",
        "run_complete",
    ]
    assert not results["summary"].empty
    assert not results["stream_metrics"].empty


def test_run_backtest_with_checkpoints_resumes_only_pending_experiments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    bundle = _feature_bundle(
        neural_drift_app.generate_synthetic_neural_drift_dataset(rows=72, random_state=33)
    )
    config = {
        **neural_drift_app.DEFAULT_CONFIG,
        "models": [neural_drift_app.MODEL_XGBOOST],
        "strategies": [
            neural_drift_app.STRATEGY_FIXED,
            neural_drift_app.STRATEGY_RETRAIN,
        ],
        "balance_modes": [neural_drift_app.BALANCE_MODE_NONE],
    }
    runtime = _mock_checkpoint_runtime(
        strategies=[
            neural_drift_app.STRATEGY_FIXED,
            neural_drift_app.STRATEGY_RETRAIN,
        ]
    )
    phase = {"value": "first", "calls": []}

    def fake_execute_backtest_experiment(
        model_name: str,
        strategy: str,
        balance_mode: str,
        canonical_artifact: dict,
        split: dict,
        config: dict,
        selected_channels: list[str],
        live_update_callback=None,
    ) -> dict:
        phase["calls"].append((phase["value"], strategy))
        if phase["value"] == "first" and strategy == neural_drift_app.STRATEGY_RETRAIN:
            raise RuntimeError("forced checkpoint failure")
        timestamp = pd.Timestamp("2024-01-01 00:00:00")
        return {
            "stream_rows": [
                {
                    "timestamp": timestamp,
                    "model": model_name,
                    "strategy": strategy,
                    "balance_mode": balance_mode,
                    "y_true": 1,
                    "prediction": 1,
                    "score": 0.82,
                    "decision_threshold": 0.5,
                    "severity_score": 0.71,
                    "max_channel_score": 0.66,
                    "severity_threshold": 0.6,
                    "is_drift_event": True,
                    "action_taken": "retrain" if strategy == neural_drift_app.STRATEGY_RETRAIN else "none",
                }
            ],
            "drift_rows": [
                {
                    "timestamp": timestamp,
                    "model": model_name,
                    "strategy": strategy,
                    "balance_mode": balance_mode,
                    "channel": neural_drift_app.DRIFT_INPUT,
                    "severity_score": 0.71,
                }
            ],
            "attention_rows": [],
            "detector_attention_rows": [],
        }

    monkeypatch.setattr(neural_drift_app, "_prepare_backtest_runtime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(neural_drift_app, "_train_canonical_artifact", lambda *args, **kwargs: {"kind": "fake"})
    monkeypatch.setattr(neural_drift_app, "_build_baseline_result_row", _fake_baseline_row)
    monkeypatch.setattr(neural_drift_app, "_execute_backtest_experiment", fake_execute_backtest_experiment)
    monkeypatch.setattr(neural_drift_app, "_finalize_backtest_results", _fake_finalize_backtest_results)

    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="forced checkpoint failure"):
        neural_drift_app.run_backtest_with_checkpoints(
            bundle,
            config=config,
            checkpoint_root=root,
        )

    [run_dir] = list(root.iterdir())
    manifest_path = run_dir / "manifest.json"
    failed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    fixed_key = runtime["experiment_specs"][0]["experiment_key"]
    retrain_key = runtime["experiment_specs"][1]["experiment_key"]
    assert failed_manifest["experiment_index"][fixed_key]["status"] == "completed"
    assert failed_manifest["experiment_index"][retrain_key]["status"] == "failed"

    phase["value"] = "second"
    results = neural_drift_app.run_backtest_with_checkpoints(
        bundle,
        config=config,
        resume_run_id=run_dir.name,
        checkpoint_root=root,
    )

    resumed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = neural_drift_app._read_jsonl_records(run_dir / "live_events.jsonl")

    assert resumed_manifest["status"] == "completed"
    assert resumed_manifest["result_status"] == "success"
    assert resumed_manifest["experiment_index"][fixed_key]["status"] == "completed"
    assert resumed_manifest["experiment_index"][retrain_key]["status"] == "completed"
    assert [call for call in phase["calls"] if call[0] == "second"] == [
        ("second", neural_drift_app.STRATEGY_RETRAIN)
    ]
    assert any(event["event"] == "resume" for event in events)
    assert not results["summary"].empty


def test_history_listing_loading_and_results_render_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = tmp_path / "runs"
    older_manifest = _write_persisted_fixture_run(
        root,
        run_id="run_old",
        updated_at="2024-01-01T10:00:00",
        run_signature="older-signature",
    )
    newer_manifest = _write_persisted_fixture_run(
        root,
        run_id="run_new",
        updated_at="2024-01-01T12:00:00",
        run_signature="newer-signature",
    )

    listed = neural_drift_app._list_persisted_neural_drift_runs(checkpoint_root=root)
    loaded = neural_drift_app._load_persisted_neural_drift_run(newer_manifest)

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.warning_calls = []
            self.info_calls = []
            self.caption_calls = []
            self.markdown_calls = []
            self.dataframe_calls = []
            self.download_calls = []

        def info(self, message):
            self.info_calls.append(str(message))

        def warning(self, message):
            self.warning_calls.append(str(message))

        def caption(self, message):
            self.caption_calls.append(str(message))

        def markdown(self, message):
            self.markdown_calls.append(str(message))

        def dataframe(self, value, **kwargs):
            self.dataframe_calls.append(value)

        def selectbox(self, label, options, key=None, **kwargs):
            option_list = list(options)
            selected = option_list[0]
            if key is not None:
                self.session_state[key] = selected
            return selected

        def line_chart(self, *args, **kwargs):
            return None

        def download_button(self, label, data, **kwargs):
            self.download_calls.append(str(label))
            return False

    fake_st = FakeStreamlit()
    monkeypatch.setattr(neural_drift_app, "st", fake_st)
    monkeypatch.setattr(neural_drift_app, "_build_run_signature", lambda *args, **kwargs: "current-signature")

    neural_drift_app._apply_persisted_neural_drift_run_to_session_state(loaded)
    neural_drift_app._render_results_subtab(
        _feature_bundle(neural_drift_app.generate_synthetic_neural_drift_dataset(rows=48, random_state=41)),
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "attention_top_k": 3,
            "drift_monitor_architecture": neural_drift_app.DRIFT_MONITOR_ARCH_CLASSIC_AE,
        },
    )

    assert [entry["run_id"] for entry in listed] == ["run_new", "run_old"]
    assert older_manifest.exists()
    assert loaded["run_id"] == "run_new"
    assert not loaded["results"]["summary"].empty
    assert fake_st.session_state["neural_drift_loaded_checkpoint_run_id"] == "run_new"
    assert fake_st.session_state["neural_drift_active_run_id"] == "run_new"
    assert fake_st.session_state["neural_drift_active_manifest_path"] == str(newer_manifest)
    assert fake_st.dataframe_calls
    assert any("no corresponden" in message for message in fake_st.warning_calls)
    assert any("Mostrando checkpoint cargado" in message for message in fake_st.caption_calls)


def test_render_tab_includes_history_tab(monkeypatch: pytest.MonkeyPatch):
    class _FakeTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.tabs_calls = []
            self.caption_calls = []

        def subheader(self, message):
            return None

        def caption(self, message):
            self.caption_calls.append(str(message))

        def warning(self, message):
            return None

        def info(self, message):
            return None

        def tabs(self, labels):
            label_list = list(labels)
            self.tabs_calls.append(label_list)
            return [_FakeTab() for _ in label_list]

    bundle = _feature_bundle(
        neural_drift_app.generate_synthetic_neural_drift_dataset(rows=48, random_state=43)
    )
    fake_st = FakeStreamlit()

    monkeypatch.setattr(neural_drift_app, "st", fake_st)
    monkeypatch.setattr(neural_drift_app, "_render_feature_source_selector", lambda context: context)
    monkeypatch.setattr(neural_drift_app, "resolve_dataset_from_context", lambda context: bundle)
    monkeypatch.setattr(neural_drift_app, "_render_configuration_subtab", lambda dataset_bundle: neural_drift_app.DEFAULT_CONFIG)
    monkeypatch.setattr(neural_drift_app, "_render_monitor_network_subtab", lambda dataset_bundle, config: config)
    monkeypatch.setattr(neural_drift_app, "_render_backtest_subtab", lambda dataset_bundle, config: None)
    monkeypatch.setattr(neural_drift_app, "_render_results_subtab", lambda dataset_bundle, config: None)
    monkeypatch.setattr(neural_drift_app, "_render_history_subtab", lambda dataset_bundle, config: None)
    monkeypatch.setattr(neural_drift_app, "_render_experiments_subtab", lambda dataset_bundle, config: None)

    neural_drift_app.render_tab({"dataset_bundle": bundle})

    assert fake_st.tabs_calls[0] == [
        "Configuración",
        "Red de drift",
        "Backtest",
        "Resultados",
        "Historial",
        "Experimentos",
    ]


def test_split_window_dataset_fixed_dates_uses_2018_base_and_2019_stream():
    prediction_time = pd.to_datetime(
        [
            *pd.date_range("2018-01-01", periods=25, freq="14D").tolist(),
            *pd.date_range("2019-01-01", periods=10, freq="30D").tolist(),
        ]
    )
    y = np.asarray(([0, 1] * ((len(prediction_time) // 2) + 1))[: len(prediction_time)], dtype=int)
    dataset = neural_drift_app.WindowDataset(
        X=np.arange(len(prediction_time) * 2, dtype=float).reshape(len(prediction_time), 2),
        y=y,
        feature_names=["f1", "f2"],
        metadata=pd.DataFrame({"prediction_time": prediction_time}),
        augmented_df=pd.DataFrame(
            {
                "prediction_time": prediction_time,
                "f1": np.linspace(0.0, 1.0, len(prediction_time)),
                "f2": np.linspace(1.0, 2.0, len(prediction_time)),
                "target": y,
            }
        ),
        base_feature_cols=["f1", "f2"],
        augmented_feature_cols=["f1", "f2"],
    )

    split = neural_drift_app._split_window_dataset_fixed_dates(
        dataset,
        base_start="2018-01-01",
        base_end="2018-12-31",
        stream_start="2019-01-01",
        validation_fraction=0.2,
    )

    assert len(split["y_train"]) == 20
    assert len(split["y_val"]) == 5
    assert len(split["y_stream"]) == 10
    assert split["metadata_train"]["prediction_time"].max() < pd.Timestamp("2018-12-31 23:59:59")
    assert split["metadata_val"]["prediction_time"].max() <= pd.Timestamp("2018-12-31 23:59:59")
    assert split["metadata_stream"]["prediction_time"].min() >= pd.Timestamp("2019-01-01")


def test_experiments_balance_mode_requires_single_active_mode():
    status = neural_drift_experiments._balance_mode_selection_status(
        {
            **neural_drift_app.DEFAULT_CONFIG,
            "balance_modes": [
                neural_drift_app.BALANCE_MODE_NONE,
                neural_drift_app.BALANCE_MODE_SMOTE,
            ],
        }
    )

    assert status["configured_balance_modes"] == [
        neural_drift_app.BALANCE_MODE_NONE,
        neural_drift_app.BALANCE_MODE_SMOTE,
    ]
    assert status["has_exactly_one_active"] is False
    assert status["resolved_balance_mode"] == neural_drift_app.BALANCE_MODE_NONE
