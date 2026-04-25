import json
from pathlib import Path

import pandas as pd
import pytest

import src.experiments_live_app as live_app


def test_streamlit_arrow_safe_df_casts_mixed_object_columns_to_string():
    df = pd.DataFrame(
        {
            "training_year": [2018, "[<= 2019]", None],
            "metric": ["auc", "auc", "auc"],
            "value": [0.81, 0.79, 0.77],
        }
    )

    safe_df = live_app._streamlit_arrow_safe_df(df)

    assert str(safe_df["training_year"].dtype) == "string"
    assert safe_df["training_year"].tolist()[:2] == ["2018", "[<= 2019]"]


def test_apply_derived_metrics_computes_brier_from_calibrated_scores():
    row = {"auc": 0.81, "pr_auc": None, "f1": None, "brier_score": None}
    roc_item = {
        "y_true": [0, 1],
        "scores": [0.1, 0.9],
        "calibrated_scores": [0.2, 0.8],
    }

    enriched = live_app._apply_derived_metrics(row, roc_item)

    assert enriched["pr_auc"] == pytest.approx(1.0)
    assert enriched["brier_score"] == pytest.approx(0.04)


def test_apply_derived_metrics_sets_bias_variance_noise_from_lookup():
    row = {
        "auc": 0.81,
        "pr_auc": None,
        "f1": None,
        "brier_score": None,
        "bias2": None,
        "variance": None,
        "noise": None,
    }
    roc_item = {
        "y_true": [0, 1],
        "scores": [0.1, 0.9],
        "calibrated_scores": [0.2, 0.8],
    }

    enriched = live_app._apply_derived_metrics(
        row,
        roc_item,
        decomposition_metrics={"bias2": 0.04, "variance": 0.16, "noise": 0.0},
    )

    assert enriched["bias2"] == pytest.approx(0.04)
    assert enriched["variance"] == pytest.approx(0.16)
    assert enriched["noise"] == pytest.approx(0.0)


def test_enrich_payload_result_rows_matches_adaptive_arf_by_prediction_year():
    rows = [
        {
            "strategy": "adaptive_arf",
            "model": "ARFfast",
            "balance_mode": "not_applicable",
            "drift": 1,
            "prediction_year": 2019,
            "brier_score": 0.25,
            "bias2": None,
            "variance": None,
            "noise": None,
        }
    ]
    roc_payload = [
        {
            "strategy": "adaptive_arf",
            "model": "ARFfast",
            "balance_mode": "not_applicable",
            "segment": "2019",
            "run_seed": 42,
            "run_order": 1,
            "y_true": [0, 1],
            "scores": [0.4, 0.6],
            "calibrated_scores": [0.4, 0.6],
        }
    ]

    enriched = live_app._enrich_payload_result_rows(rows, roc_payload, yearly=False)

    assert enriched[0]["bias2"] == pytest.approx(0.16)
    assert enriched[0]["variance"] == pytest.approx(0.24)
    assert enriched[0]["noise"] == pytest.approx(0.0)


def test_display_cell_handles_infinite_thresholds():
    assert live_app._display_cell(float("inf")) == "inf"
    assert live_app._display_cell(float("-inf")) == "-inf"


def test_format_decimal_es_uses_decimal_comma_and_thin_space():
    assert live_app._format_decimal_es(14238.024) == "14\u202f238,024"
    assert live_app._format_decimal_es(0.5) == "0,5"


def test_calibration_progress_curve_preserves_context_and_best_direction():
    df = pd.DataFrame(
        [
            {
                "status": "completed",
                "val_balanced_f1": 0.40,
                "calibration_method": "sigmoid",
                "threshold_objective": "far",
                "balance_mode": "none",
            },
            {
                "status": "failed",
                "val_balanced_f1": 0.99,
                "calibration_method": "sigmoid",
                "threshold_objective": "far",
                "balance_mode": "smote",
            },
            {
                "status": "completed",
                "val_balanced_f1": 0.45,
                "calibration_method": "isotonic",
                "threshold_objective": "f1",
                "balance_mode": "none",
            },
        ]
    )

    curve_df = live_app._calibration_progress_curve_df(
        df,
        metric_col="val_balanced_f1",
    )

    assert curve_df["combo_index"].tolist() == [1, 2]
    assert curve_df["current_value"].tolist() == pytest.approx([0.40, 0.45])
    assert curve_df["best_so_far"].tolist() == pytest.approx([0.40, 0.45])
    assert curve_df["calibration_method"].tolist() == ["sigmoid", "isotonic"]


def test_calibration_default_progress_metric_prefers_protocol_objective():
    metric_options = {
        "val_pr_auc": "Validación PR-AUC",
        "val_balanced_f1": "Validación Balanced F1",
    }

    selected = live_app._calibration_default_progress_metric(
        {"objective_metrics": ["balanced_f1"]},
        metric_options,
    )

    assert selected == "val_balanced_f1"


def test_controlled_live_best_payload_uses_best_test_fallbacks():
    row = {
        "model_name": "XGBoost",
        "feature_set": "Base",
        "balance_mode": "none",
        "k_optimo": 25,
        "objective_label": "ROC-AUC",
        "val_objective_score": 0.91,
        "test_objective_score": 0.84,
        "best_test_accuracy": 0.97,
        "best_test_recall": 0.40,
        "best_test_sensitivity": 0.40,
        "best_test_f1_global": 0.68,
        "best_test_f1_class_0": 0.99,
        "best_test_f1_class_1": 0.37,
        "best_test_false_negatives": 9,
        "best_test_false_positives": 3,
        "best_test_roc_auc": 0.85,
        "best_test_pr_auc": 0.23,
        "best_test_mcc": 0.31,
        "decision_threshold": 0.5,
        "best_test_confusion_matrix": "[[100, 3], [9, 6]]",
    }

    payload = live_app._controlled_live_best_payload(
        row,
        objective_label="ROC-AUC",
    )

    assert payload["test_accuracy"] == pytest.approx(0.97)
    assert payload["test_recall"] == pytest.approx(0.40)
    assert payload["test_sensitivity"] == pytest.approx(0.40)
    assert payload["test_f1_global"] == pytest.approx(0.68)
    assert payload["test_f1_class_0"] == pytest.approx(0.99)
    assert payload["test_f1_class_1"] == pytest.approx(0.37)
    assert payload["test_false_negatives"] == 9
    assert payload["test_false_positives"] == 3
    assert payload["test_roc_auc"] == pytest.approx(0.85)
    assert payload["test_pr_auc"] == pytest.approx(0.23)
    assert payload["test_mcc"] == pytest.approx(0.31)
    assert live_app._coerce_confusion_matrix_cell(
        row["best_test_confusion_matrix"]
    ) == [[100, 3], [9, 6]]


def test_build_drift_tuning_params_frame_compares_smote_against_none():
    artifacts = [
        {
            "study_id": "study_none",
            "tuning_key": "tk_none",
            "model_name": "Random Forest",
            "balance_mode": "none",
            "stage": "window_tuning",
            "best_value": 0.74,
            "best_params": {"mtry": 4, "min_node_size": 2},
            "n_trials": 8,
            "requested_trials": 8,
            "search_space_size": 16,
            "invalid_trial_count": 0,
            "has_valid_trial": True,
            "n_train": 40,
            "positive_rows": 10,
            "positive_rate": 0.25,
            "train_signature": "same_window",
            "strategy_context": {"strategy": "static", "window_kind": "base_year", "training_year": "2018"},
        },
        {
            "study_id": "study_smote",
            "tuning_key": "tk_smote",
            "model_name": "Random Forest",
            "balance_mode": "smote",
            "stage": "window_tuning",
            "best_value": 0.78,
            "best_params": {"mtry": 6, "min_node_size": 1, "sampling_strategy": 1.0, "k_neighbors": 5},
            "n_trials": 8,
            "requested_trials": 8,
            "search_space_size": 24,
            "invalid_trial_count": 1,
            "has_valid_trial": True,
            "n_train": 40,
            "positive_rows": 10,
            "positive_rate": 0.25,
            "train_signature": "same_window",
            "strategy_context": {"strategy": "static", "window_kind": "base_year", "training_year": "2018"},
        },
    ]

    df = live_app._build_drift_tuning_params_frame(artifacts)

    assert len(df) == 2
    smote_row = df.loc[df["balance_mode"] == "smote"].iloc[0]
    assert smote_row["best_cv_auc_none"] == pytest.approx(0.74)
    assert smote_row["cv_auc_delta_vs_none"] == pytest.approx(0.04)
    assert smote_row["param_sampling_strategy"] == 1.0
    assert smote_row["param_k_neighbors"] == 5
    none_row = df.loc[df["balance_mode"] == "none"].iloc[0]
    assert none_row["cv_auc_delta_vs_none"] == pytest.approx(0.0)


def test_build_live_sources_includes_drift_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(live_app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_app,
        "CALIBRATION_EXPERIMENTS_DIR",
        tmp_path / "calibration_experiment_runs",
    )
    monkeypatch.setattr(
        live_app,
        "DRIFT_RUNS_DIR",
        tmp_path / "drift_recalibration_runs",
    )
    monkeypatch.setattr(
        live_app,
        "NEURAL_DRIFT_EXPERIMENTS_DIR",
        tmp_path / "neural_drift_experiments",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_PAPER_RUNS_DIR",
        tmp_path / "nlp_in_severity" / "paper_replication",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_LANGUAGE_MODELING_LIVE_DIR",
        tmp_path / "nlp_in_severity" / "language_modeling_live",
    )

    run_dir = live_app.DRIFT_RUNS_DIR / "run_abc123"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": "run_abc123",
        "status": "running",
        "updated_at": "2026-03-22T10:00:00",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    sources = live_app._build_live_sources()

    assert len(sources) == 1
    assert sources[0]["type"] == "drift_recalibration"
    assert "run_abc123" in str(sources[0]["label"])


def test_build_live_sources_includes_paper_replication_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(live_app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_app,
        "CALIBRATION_EXPERIMENTS_DIR",
        tmp_path / "calibration_experiment_runs",
    )
    monkeypatch.setattr(
        live_app,
        "DRIFT_RUNS_DIR",
        tmp_path / "drift_recalibration_runs",
    )
    monkeypatch.setattr(
        live_app,
        "NEURAL_DRIFT_EXPERIMENTS_DIR",
        tmp_path / "neural_drift_experiments",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_PAPER_RUNS_DIR",
        tmp_path / "nlp_in_severity" / "paper_replication",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_LANGUAGE_MODELING_LIVE_DIR",
        tmp_path / "nlp_in_severity" / "language_modeling_live",
    )

    paper_run_dir = live_app.NLP_PAPER_RUNS_DIR / "paper_replication_abcd"
    paper_run_dir.mkdir(parents=True)
    (paper_run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "paper_replication_abcd",
                "status": "running",
                "updated_at": "2026-03-31T09:00:00",
            }
        ),
        encoding="utf-8",
    )

    sources = live_app._build_live_sources()

    assert len(sources) == 1
    assert sources[0]["type"] == "paper_replication"
    assert "paper_replication_abcd" in str(sources[0]["label"])


def test_build_live_sources_includes_neural_drift_experiment_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(live_app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_app,
        "CALIBRATION_EXPERIMENTS_DIR",
        tmp_path / "calibration_experiment_runs",
    )
    monkeypatch.setattr(
        live_app,
        "DRIFT_RUNS_DIR",
        tmp_path / "drift_recalibration_runs",
    )
    monkeypatch.setattr(
        live_app,
        "NEURAL_DRIFT_EXPERIMENTS_DIR",
        tmp_path / "neural_drift_experiments",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_PAPER_RUNS_DIR",
        tmp_path / "nlp_in_severity" / "paper_replication",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_LANGUAGE_MODELING_LIVE_DIR",
        tmp_path / "nlp_in_severity" / "language_modeling_live",
    )

    run_dir = live_app.NEURAL_DRIFT_EXPERIMENTS_DIR / "run_experiment_001"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_experiment_001",
                "status": "running",
                "updated_at": "2026-04-11T15:00:00",
            }
        ),
        encoding="utf-8",
    )

    sources = live_app._build_live_sources()

    assert len(sources) == 1
    assert sources[0]["type"] == "neural_drift_experiment"
    assert "run_experiment_001" in str(sources[0]["label"])


def test_build_live_sources_includes_language_modeling_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(live_app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_app,
        "CALIBRATION_EXPERIMENTS_DIR",
        tmp_path / "calibration_experiment_runs",
    )
    monkeypatch.setattr(
        live_app,
        "DRIFT_RUNS_DIR",
        tmp_path / "drift_recalibration_runs",
    )
    monkeypatch.setattr(
        live_app,
        "NEURAL_DRIFT_EXPERIMENTS_DIR",
        tmp_path / "neural_drift_experiments",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_PAPER_RUNS_DIR",
        tmp_path / "nlp_in_severity" / "paper_replication",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_LANGUAGE_MODELING_LIVE_DIR",
        tmp_path / "nlp_in_severity" / "language_modeling_live",
    )

    run_dir = live_app.NLP_LANGUAGE_MODELING_LIVE_DIR / "language_search_demo"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "language_search_demo",
                "run_type": "transformers_search",
                "status": "running",
                "updated_at": "2026-04-07T10:00:00",
            }
        ),
        encoding="utf-8",
    )

    sources = live_app._build_live_sources()

    assert len(sources) == 1
    assert sources[0]["type"] == "language_modeling"
    assert "transformers_search" in str(sources[0]["label"])


def test_build_live_sources_includes_calibration_experiment_manifests(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(live_app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_app,
        "CALIBRATION_EXPERIMENTS_DIR",
        tmp_path / "calibration_experiment_runs",
    )
    monkeypatch.setattr(
        live_app,
        "DRIFT_RUNS_DIR",
        tmp_path / "drift_recalibration_runs",
    )
    monkeypatch.setattr(
        live_app,
        "NEURAL_DRIFT_EXPERIMENTS_DIR",
        tmp_path / "neural_drift_experiments",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_PAPER_RUNS_DIR",
        tmp_path / "nlp_in_severity" / "paper_replication",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_LANGUAGE_MODELING_LIVE_DIR",
        tmp_path / "nlp_in_severity" / "language_modeling_live",
    )

    run_dir = live_app.CALIBRATION_EXPERIMENTS_DIR / "calibration_sweep_demo"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "calibration_sweep_demo",
                "status": "running",
                "updated_at": "2026-04-18T15:30:00",
            }
        ),
        encoding="utf-8",
    )

    sources = live_app._build_live_sources()

    assert len(sources) == 1
    assert sources[0]["type"] == "calibration_experiment"
    assert "calibration_sweep_demo" in str(sources[0]["label"])


def test_build_live_sources_includes_model_optuna_batch_manifests(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(live_app, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        live_app,
        "CALIBRATION_EXPERIMENTS_DIR",
        tmp_path / "calibration_experiment_runs",
    )
    monkeypatch.setattr(
        live_app,
        "DRIFT_RUNS_DIR",
        tmp_path / "drift_recalibration_runs",
    )
    monkeypatch.setattr(
        live_app,
        "NEURAL_DRIFT_EXPERIMENTS_DIR",
        tmp_path / "neural_drift_experiments",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_PAPER_RUNS_DIR",
        tmp_path / "nlp_in_severity" / "paper_replication",
    )
    monkeypatch.setattr(
        live_app,
        "NLP_LANGUAGE_MODELING_LIVE_DIR",
        tmp_path / "nlp_in_severity" / "language_modeling_live",
    )

    run_dir = (
        tmp_path
        / "model_history"
        / "optuna_batch_live"
        / "model_optuna_batch_demo"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "model_optuna_batch_demo",
                "status": "running",
                "result_status": "running",
                "updated_at": "2026-04-23T17:30:00",
            }
        ),
        encoding="utf-8",
    )

    sources = live_app._build_live_sources()

    assert len(sources) == 1
    assert sources[0]["type"] == "model_optuna_batch"
    assert "Modelos | batch Optuna" in str(sources[0]["label"])
    assert "model_optuna_batch_demo" in str(sources[0]["label"])


def test_read_model_optuna_batch_run_loads_progress_artifacts(tmp_path):
    run_dir = (
        tmp_path
        / "model_history"
        / "optuna_batch_live"
        / "model_optuna_batch_demo"
    )
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "model_optuna_batch_demo",
                "status": "running",
                "result_status": "running",
                "updated_at": "2026-04-23T17:31:00",
                "progress": {
                    "completed_steps": 1,
                    "total_steps": 2,
                    "current_step_id": "combo_done",
                },
            }
        ),
        encoding="utf-8",
    )
    live_status = {
        "timestamp": "2026-04-23T17:31:00",
        "step_id": "combo_done",
        "message": "Base | Sin SMOTE | Candidato 1 | Conservador completado.",
        "progress_ratio": 0.5,
        "model_name": "XGBoost",
        "objective_metric": "balanced_f1",
        "calibration_method": "sigmoid",
        "threshold_objective": "far",
        "threshold_protocol": "conservative",
        "backend": "local",
    }
    (run_dir / "live_status.json").write_text(
        json.dumps(live_status),
        encoding="utf-8",
    )
    (run_dir / "live_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-23T17:30:00",
                        "step_id": "run_start",
                        "step_status": "running",
                        "message": "Entrenando batch Optuna...",
                        "progress": {
                            "completed_steps": 0,
                            "total_steps": 2,
                            "progress_ratio": 0.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-23T17:31:00",
                        "step_id": "combo_done",
                        "step_status": "completed",
                        "message": "Combo completado.",
                        "combo_index": 1,
                        "total_combinations": 2,
                        "model_name": "XGBoost",
                        "objective_metric": "balanced_f1",
                        "calibration_method": "sigmoid",
                        "threshold_objective": "far",
                        "threshold_protocol": "conservative",
                        "balance_mode": "none",
                        "progress": {
                            "completed_steps": 1,
                            "total_steps": 2,
                            "progress_ratio": 0.5,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "status": "completed",
                "combo_id": "a",
                "model_name": "XGBoost",
                "objective_metric": "balanced_f1",
                "calibration_method": "sigmoid",
                "threshold_objective": "far",
                "threshold_protocol": "conservative",
                "balance_mode": "none",
                "val_balanced_f1": 0.42,
                "test_balanced_f1": 0.40,
            },
            {
                "status": "failed",
                "combo_id": "b",
                "model_name": "XGBoost",
                "objective_metric": "balanced_f1",
                "val_balanced_f1": None,
            },
        ]
    ).to_csv(run_dir / "partial_results.csv", index=False)

    payload = live_app._read_model_optuna_batch_run(manifest_path)

    assert payload["current_context"]["model_name"] == "XGBoost"
    assert payload["current_context"]["progress_ratio"] == pytest.approx(0.5)
    assert payload["live_events_df"]["progress_pct"].iloc[-1] == pytest.approx(50.0)
    assert payload["live_events_df"]["combo_index"].iloc[-1] == 1
    assert len(payload["partial_results_df"]) == 2
    curve_df = live_app._calibration_progress_curve_df(
        payload["partial_results_df"],
        metric_col="val_balanced_f1",
    )
    assert curve_df["best_so_far"].tolist() == pytest.approx([0.42])


def test_read_language_modeling_run_loads_live_artifacts(tmp_path):
    run_dir = tmp_path / "nlp_in_severity" / "language_modeling_live" / "language_search_demo"
    run_dir.mkdir(parents=True)
    search_trials_path = run_dir / "search_trials_live.csv"
    best_history_path = run_dir / "best_history_live.csv"
    search_summary_path = run_dir / "search_summary.json"
    best_result_path = run_dir / "best_result.json"

    pd.DataFrame(
        [{"trial_index": 1, "status": "ok", "objective": 0.71}]
    ).to_csv(search_trials_path, index=False)
    pd.DataFrame(
        [{"epoch": 1.0, "loss": 0.5, "eval_balanced_f1": 0.72}]
    ).to_csv(best_history_path, index=False)
    search_summary_path.write_text(
        json.dumps({"greater_is_better": True, "objective_metric": "balanced_f1"}),
        encoding="utf-8",
    )
    best_result_path.write_text(
        json.dumps({"model_name": "demo-model", "output_dir": str(run_dir / "best_model")}),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "language_search_demo",
                "run_type": "transformers_search",
                "title": "Busqueda robusta demo",
                "status": "completed",
                "result_status": "ok",
                "updated_at": "2026-04-07T10:05:00",
                "progress_ratio": 1.0,
                "artifacts": {
                    "search_trials_csv": str(search_trials_path),
                    "best_history_csv": str(best_history_path),
                    "search_summary_json": str(search_summary_path),
                    "best_result_json": str(best_result_path),
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "live_status.json").write_text(
        json.dumps(
            {
                "stage": "search.completed",
                "message": "Busqueda robusta completada.",
                "progress_ratio": 1.0,
                "status": "completed",
                "result_status": "ok",
                "updated_at": "2026-04-07T10:05:00",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "live_events.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-04-07T10:02:00",
                "stage": "search.trial",
                "event_type": "search_trial_result",
                "trial_index": 1,
                "objective": 0.71,
                "progress_ratio": 0.55,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    data = live_app._read_language_modeling_run(run_dir / "manifest.json")

    assert data["manifest"]["run_type"] == "transformers_search"
    assert list(data["search_trials_df"]["trial_index"]) == [1]
    assert list(data["best_history_df"]["epoch"]) == [1.0]
    assert data["search_summary"]["objective_metric"] == "balanced_f1"
    assert data["best_result"]["model_name"] == "demo-model"


def test_read_calibration_experiment_run_loads_partial_results_and_previous_runs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        live_app,
        "CALIBRATION_EXPERIMENTS_DIR",
        tmp_path / "calibration_experiment_runs",
    )

    shared_event_path = str(tmp_path / "events.csv")
    shared_features_path = str(tmp_path / "features.duckdb")

    current_run_dir = live_app.CALIBRATION_EXPERIMENTS_DIR / "run_current"
    previous_run_dir = live_app.CALIBRATION_EXPERIMENTS_DIR / "run_previous"
    (current_run_dir / "results").mkdir(parents=True)
    (previous_run_dir / "results").mkdir(parents=True)

    protocol = {
        "protocol_family": "calibration_score_threshold",
        "model_name": "XGBoost",
        "objective_metrics": ["pr_auc"],
        "calibration_methods": ["sigmoid"],
        "threshold_objectives": ["far", "mcc"],
        "balance_modes": ["none", "smote"],
    }

    current_manifest = {
        "run_id": "run_current",
        "status": "running",
        "result_status": "running",
        "created_at": "2026-04-18T15:00:00",
        "updated_at": "2026-04-18T15:10:00",
        "protocol": dict(protocol),
        "progress": {
            "completed_steps": 3,
            "total_steps": 6,
            "current_step_id": "combo__pr_auc__sigmoid__mcc__none",
        },
    }
    (current_run_dir / "manifest.json").write_text(
        json.dumps(current_manifest),
        encoding="utf-8",
    )
    (current_run_dir / "protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    (current_run_dir / "live_status.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-04-18T15:10:00",
                "status": "running",
                "result_status": "running",
                "step_id": "combo__pr_auc__sigmoid__mcc__none",
                "message": "Evaluando combinación actual.",
                "progress": {
                    "completed_steps": 3,
                    "total_steps": 6,
                    "current_step_id": "combo__pr_auc__sigmoid__mcc__none",
                },
            }
        ),
        encoding="utf-8",
    )
    (current_run_dir / "live_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-18T15:00:00",
                        "status": "running",
                        "result_status": "running",
                        "step_id": "init",
                        "step_status": "running",
                        "message": "Inicio.",
                        "progress": {"completed_steps": 0, "total_steps": 6},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-18T15:04:00",
                        "status": "running",
                        "result_status": "running",
                        "step_id": "combo__pr_auc__sigmoid__far__none",
                        "step_status": "completed",
                        "message": "Primera combinación lista.",
                        "progress": {"completed_steps": 2, "total_steps": 6},
                        "metadata": {
                            "objective_metric": "pr_auc",
                            "calibration_method": "sigmoid",
                            "threshold_objective": "far",
                            "balance_mode": "none",
                            "val_objective_score": 0.21,
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-18T15:10:00",
                        "status": "running",
                        "result_status": "running",
                        "step_id": "combo__pr_auc__sigmoid__mcc__none",
                        "step_status": "running",
                        "message": "Segunda combinación en curso.",
                        "progress": {"completed_steps": 3, "total_steps": 6},
                        "metadata": {
                            "objective_metric": "pr_auc",
                            "calibration_method": "sigmoid",
                            "threshold_objective": "mcc",
                            "balance_mode": "none",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "status": "completed",
                "model_name": "XGBoost",
                "event_path": shared_event_path,
                "features_path": shared_features_path,
                "balance_mode": "none",
                "optuna_objective_metric": "pr_auc",
                "calibration_method": "sigmoid",
                "threshold_objective": "far",
                "decision_threshold": 0.42,
                "val_mcc": 0.11,
                "val_brier_score": 0.09,
                "val_pr_auc": 0.21,
                "test_mcc": 0.1,
                "test_brier_score": 0.12,
                "test_pr_auc": 0.2,
            },
            {
                "status": "failed",
                "model_name": "XGBoost",
                "event_path": shared_event_path,
                "features_path": shared_features_path,
                "balance_mode": "smote",
                "optuna_objective_metric": "pr_auc",
                "calibration_method": "sigmoid",
                "threshold_objective": "far",
                "error": "boom",
            },
        ]
    ).to_csv(current_run_dir / "results" / "grid_results.csv", index=False)

    previous_manifest = {
        "run_id": "run_previous",
        "status": "completed",
        "result_status": "completed",
        "created_at": "2026-04-18T14:00:00",
        "updated_at": "2026-04-18T14:20:00",
        "protocol": dict(protocol),
        "progress": {"completed_steps": 6, "total_steps": 6},
    }
    (previous_run_dir / "manifest.json").write_text(
        json.dumps(previous_manifest),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "status": "completed",
                "model_name": "XGBoost",
                "event_path": shared_event_path,
                "features_path": shared_features_path,
                "balance_mode": "smote",
                "optuna_objective_metric": "pr_auc",
                "calibration_method": "sigmoid",
                "threshold_objective": "mcc",
                "val_pr_auc": 0.31,
                "test_pr_auc": 0.29,
            }
        ]
    ).to_csv(previous_run_dir / "results" / "grid_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "balance_mode": "smote",
                "optuna_objective_metric": "pr_auc",
                "calibration_method": "sigmoid",
                "threshold_objective": "mcc",
                "stability_score": 0.73,
                "val_mcc": 0.18,
                "val_brier_score": 0.08,
                "val_pr_auc": 0.31,
                "test_mcc": 0.16,
                "test_brier_score": 0.1,
                "test_pr_auc": 0.29,
            }
        ]
    ).to_csv(previous_run_dir / "results" / "best_summary.csv", index=False)

    payload = live_app._read_calibration_experiment_run(
        current_run_dir / "manifest.json"
    )

    assert payload["protocol"]["protocol_family"] == "calibration_score_threshold"
    assert payload["live_events_df"]["progress_pct"].iloc[-1] == pytest.approx(50.0)
    assert payload["grid_results_df"]["status"].tolist() == ["completed", "failed"]
    assert not payload["previous_runs_df"].empty
    assert payload["previous_runs_df"]["run_id"].tolist() == ["run_previous"]
    assert bool(payload["previous_runs_df"]["same_context"].iloc[0]) is True
    assert payload["previous_runs_df"]["best_test_pr_auc"].iloc[0] == pytest.approx(
        0.29
    )


def test_read_drift_run_builds_partial_monitoring_frames(tmp_path):
    run_dir = tmp_path / "drift_recalibration_runs" / "run_xyz789"
    blocks_dir = run_dir / "blocks"
    tuning_dir = run_dir / "tuning"
    blocks_dir.mkdir(parents=True)
    tuning_dir.mkdir(parents=True)

    manifest = {
        "run_id": "run_xyz789",
        "status": "running",
        "started_at": "2026-03-22T10:00:00",
        "updated_at": "2026-03-22T10:05:00",
        "progress": {
            "completed_units": 1.5,
            "total_units": 4,
            "completed_tuning_tasks": 1,
            "total_tuning_tasks": 2,
            "completed_blocks": 1,
            "total_blocks": 2,
        },
        "block_index": {
            "block_a": {
                "status": "completed",
                "strategy": "static",
                "model": "Random Forest",
                "balance_mode": "none",
                "run_seed": 11,
                "run_order": 1,
            },
            "block_b": {
                "status": "pending",
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "smote",
                "run_seed": 11,
                "run_order": 1,
            },
        },
        "tuning_index": {
            "tune_a": {
                "status": "completed",
                "filename": "tune_a.json",
                "model_name": "Random Forest",
                "balance_mode": "none",
            }
        },
        "smote_index": {"smote_a": {"status": "available"}},
        "global_execution_log": [
            {
                "order": 1,
                "timestamp": "2026-03-22T10:00:00",
                "phase": "global_tuning_start",
                "status": "started",
                "rss_before_mb": 512.0,
            }
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    block_payload = {
        "saved_at": "2026-03-22T10:04:00",
        "block_id": "block_a",
        "block": {
            "strategy": "static",
            "model": "Random Forest",
            "balance_mode": "none",
        },
        "run_seed": 11,
        "run_order": 1,
        "yearly_rows": [
            {
                "strategy": "static",
                "prediction_year": 2019,
                "model": "Random Forest",
                "balance_mode": "none",
                "auc": 0.81,
                "sensitivity": 1.0,
                "specificity": 1.0,
                "sensitivity_before_calibration": 0.5,
                "specificity_before_calibration": 0.75,
                "sensitivity_after_calibration": 1.0,
                "specificity_after_calibration": 1.0,
                "error_rate": 0.0,
                "training_time_sec": 0.4,
                "run_seed": 11,
                "run_order": 1,
            }
        ],
        "adaptive_rows": [
            {
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "smote",
                "auc": 0.79,
                "sensitivity": 1.0,
                "specificity": 1.0,
                "sensitivity_before_calibration": 0.5,
                "specificity_before_calibration": 0.75,
                "sensitivity_after_calibration": 1.0,
                "specificity_after_calibration": 1.0,
                "error_rate": 0.0,
                "training_time_sec": 0.6,
                "run_seed": 11,
                "run_order": 1,
            }
        ],
        "execution_log": [
            {
                "timestamp": "2026-03-22T10:04:00",
                "phase": "block_complete",
                "status": "ok",
                "training_time_sec": 0.4,
                "rss_after_mb": 620.0,
            }
        ],
        "roc_payload": [
            {
                "strategy": "static",
                "model": "Random Forest",
                "balance_mode": "none",
                "segment": "2019",
                "y_true": [0, 1],
                "scores": [0.1, 0.9],
                "calibrated_scores": [0.2, 0.8],
                "run_seed": 11,
                "run_order": 1,
            },
            {
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "smote",
                "segment": "final",
                "y_true": [0, 1],
                "scores": [0.2, 0.8],
                "calibrated_scores": [0.25, 0.75],
                "run_seed": 11,
                "run_order": 1,
            }
        ],
        "tuning_refs": ["tune_a"],
        "smote_refs": [],
    }
    (blocks_dir / "block_a.json").write_text(
        json.dumps(block_payload),
        encoding="utf-8",
    )

    tuning_artifact = {
        "study_id": "study_rf_none",
        "tuning_key": "tk_rf_none",
        "model_name": "Random Forest",
        "balance_mode": "none",
        "stage": "global_tuning",
        "best_value": 0.81,
        "best_params": {"mtry": 4, "min_node_size": 2},
        "n_trials": 2,
        "requested_trials": 2,
        "search_space_size": 4,
        "invalid_trial_count": 0,
        "has_valid_trial": True,
        "n_train": 40,
        "positive_rows": 10,
        "positive_rate": 0.25,
        "train_signature": "global_signature",
        "strategy_context": {"strategy": "static", "window_kind": "base_year", "training_year": "2018"},
        "trials": [
            {
                "trial_number": 0,
                "state": "COMPLETE",
                "cv_auc": 0.74,
            },
            {
                "trial_number": 1,
                "state": "COMPLETE",
                "cv_auc": 0.81,
            },
        ],
    }
    (tuning_dir / "tune_a.json").write_text(
        json.dumps(tuning_artifact),
        encoding="utf-8",
    )
    tuning_artifact_smote = {
        "study_id": "study_rf_smote",
        "tuning_key": "tk_rf_smote",
        "model_name": "Random Forest",
        "balance_mode": "smote",
        "stage": "global_tuning",
        "best_value": 0.79,
        "best_params": {"mtry": 6, "min_node_size": 1, "sampling_strategy": 1.0, "k_neighbors": 5},
        "n_trials": 2,
        "requested_trials": 2,
        "search_space_size": 6,
        "invalid_trial_count": 0,
        "has_valid_trial": True,
        "n_train": 40,
        "positive_rows": 10,
        "positive_rate": 0.25,
        "train_signature": "global_signature",
        "strategy_context": {"strategy": "static", "window_kind": "base_year", "training_year": "2018"},
        "trials": [
            {
                "trial_number": 0,
                "state": "COMPLETE",
                "cv_auc": 0.75,
            },
            {
                "trial_number": 1,
                "state": "COMPLETE",
                "cv_auc": 0.79,
            },
        ],
    }
    (tuning_dir / "tune_b.json").write_text(
        json.dumps(tuning_artifact_smote),
        encoding="utf-8",
    )

    live_status = {
        "timestamp": "2026-03-22T10:05:00",
        "status": "running",
        "completed_units": 1.5,
        "total_units": 4,
        "progress_ratio": 0.375,
        "label": "Ejecutando bloque de experimentos...",
        "detail": "Seed 11 | Strategy adaptive_adwin",
        "context": {
            "phase": "block_running",
            "strategy": "adaptive_adwin",
            "model": "Random Forest",
            "run_seed": 11,
        },
    }
    (run_dir / "live_status.json").write_text(
        json.dumps(live_status),
        encoding="utf-8",
    )
    (run_dir / "live_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-22T10:00:00",
                        "completed_units": 0.0,
                        "total_units": 4,
                        "progress_ratio": 0.0,
                        "label": "Iniciando",
                        "detail": "",
                        "context": {},
                    }
                ),
                json.dumps(live_status),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = live_app._read_drift_run(manifest_path)

    assert isinstance(payload["block_df"], pd.DataFrame)
    assert len(payload["block_df"]) == 2
    assert not payload["summary_df"].empty
    assert set(payload["summary_df"]["strategy"].unique()) == {"static", "adaptive_adwin"}
    assert "detector_variant" in payload["summary_df"].columns
    assert "n_segments" in payload["summary_df"].columns
    assert "n_repetitions" in payload["summary_df"].columns
    assert payload["yearly_df"]["pr_auc"].iloc[0] == 1.0
    assert payload["yearly_df"]["brier_score"].iloc[0] == pytest.approx(0.04)
    assert payload["yearly_df"]["f1"].iloc[0] == 1.0
    assert payload["yearly_df"]["sensitivity_before_calibration"].iloc[0] == 0.5
    assert payload["yearly_df"]["specificity_after_calibration"].iloc[0] == 1.0
    assert payload["adaptive_df"]["pr_auc"].iloc[0] == 1.0
    assert payload["adaptive_df"]["brier_score"].iloc[0] == pytest.approx(0.0625)
    assert payload["adaptive_df"]["f1"].iloc[0] == 1.0
    assert payload["adaptive_df"]["sensitivity_before_calibration"].iloc[0] == 0.5
    assert payload["adaptive_df"]["specificity_after_calibration"].iloc[0] == 1.0
    assert payload["summary_df"]["brier_score"].notna().all()
    assert {"bias2", "variance", "noise"} <= set(payload["summary_df"].columns)
    assert not payload["tuning_trials_df"].empty
    assert not payload["tuning_params_df"].empty
    smote_tuning = payload["tuning_params_df"].loc[payload["tuning_params_df"]["balance_mode"] == "smote"].iloc[0]
    assert smote_tuning["cv_auc_delta_vs_none"] == pytest.approx(-0.02)
    assert smote_tuning["param_sampling_strategy"] == 1.0
    assert not payload["memory_trace_df"].empty
    assert not payload["live_events_df"].empty
    assert not payload["average_roc_df"].empty
    assert payload["live_events_df"]["progress_pct"].iloc[-1] == 37.5


def test_read_paper_replication_run_builds_partial_monitoring_frames(tmp_path):
    run_dir = tmp_path / "nlp_in_severity" / "paper_replication" / "paper_replication_demo"
    frozen_m1_dir = run_dir / "frozen" / "models" / "M1"
    frozen_m2_k_dir = run_dir / "frozen" / "models" / "M2" / "k_results"
    raw_build_dir = run_dir / "raw_build"
    frozen_m1_dir.mkdir(parents=True)
    frozen_m2_k_dir.mkdir(parents=True)
    raw_build_dir.mkdir(parents=True)

    manifest = {
        "run_id": "paper_replication_demo",
        "status": "running",
        "result_status": "running",
        "created_at": "2026-03-31T10:00:00",
        "updated_at": "2026-03-31T10:12:00",
        "progress": {
            "current_stage": "raw_build",
            "current_step_id": "raw.build.embeddings",
            "completed_steps": 3,
            "total_steps": 8,
            "completed_units": 22.0,
            "total_units": 100.0,
        },
        "steps_index": {
            "frozen.dataset_validation": {
                "step_id": "frozen.dataset_validation",
                "stage": "frozen",
                "description": "frozen dataset validation",
                "status": "completed",
                "order": 1,
                "completed_at": "2026-03-31T10:02:00",
                "last_message": "frozen: dataset validado.",
                "artifact_paths": {"dataset_validation": str(run_dir / "frozen" / "dataset_validation.json")},
            },
            "frozen.M1.k.10": {
                "step_id": "frozen.M1.k.10",
                "stage": "frozen",
                "description": "M1 nested CV k=10",
                "status": "completed",
                "order": 2,
                "completed_at": "2026-03-31T10:05:00",
                "last_message": "M1: k=10 persistido.",
                "artifact_paths": {"k_result": str(frozen_m1_dir / "k_results" / "k_010.json")},
            },
            "frozen.M1.final": {
                "step_id": "frozen.M1.final",
                "stage": "frozen",
                "description": "M1 final fit",
                "status": "completed",
                "order": 3,
                "completed_at": "2026-03-31T10:08:00",
                "last_message": "M1: resultado final persistido.",
                "artifact_paths": {"summary": str(frozen_m1_dir / "final_summary.json")},
            },
            "frozen.M2.k.10": {
                "step_id": "frozen.M2.k.10",
                "stage": "frozen",
                "description": "M2 nested CV k=10",
                "status": "running",
                "order": 4,
                "started_at": "2026-03-31T10:09:00",
                "last_message": "M2: nested CV para k=10.",
                "artifact_paths": {},
            },
            "raw.build.features": {
                "step_id": "raw.build.features",
                "stage": "raw_build",
                "description": "Reconstruccion de features raw",
                "status": "completed",
                "order": 5,
                "completed_at": "2026-03-31T10:10:00",
                "last_message": "Features raw persistidos.",
                "artifact_paths": {"features": str(raw_build_dir / "features.pkl")},
            },
            "raw.build.embeddings": {
                "step_id": "raw.build.embeddings",
                "stage": "raw_build",
                "description": "Extraccion de embeddings fine-tuneados",
                "status": "running",
                "order": 6,
                "started_at": "2026-03-31T10:11:00",
                "last_message": "Generando embeddings [CLS].",
                "artifact_paths": {},
            },
        },
        "step_sequence": [
            "frozen.dataset_validation",
            "frozen.M1.k.10",
            "frozen.M1.final",
            "frozen.M2.k.10",
            "raw.build.features",
            "raw.build.embeddings",
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    (run_dir / "frozen").mkdir(parents=True, exist_ok=True)
    (run_dir / "frozen" / "dataset_validation.json").write_text(
        json.dumps(
            {
                "rows": 2070,
                "flow_features": 432,
                "embedding_features": 200,
                "total_features": 632,
                "train_rows": 1656,
                "test_rows": 414,
            }
        ),
        encoding="utf-8",
    )

    (frozen_m1_dir / "k_results").mkdir(parents=True, exist_ok=True)
    (frozen_m1_dir / "final_summary.json").write_text(
        json.dumps(
            {
                "model_code": "M1",
                "model_title": "M1 - Flow only",
                "feature_group": "flow",
                "candidate_feature_count": 432,
                "selected_k": 10,
                "best_cv_score": 0.78,
                "optimization": {"backend": "gridsearchcv", "requested_backend": "gridsearchcv"},
            }
        ),
        encoding="utf-8",
    )
    (frozen_m1_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.81,
                "precision": 0.77,
                "recall": 0.79,
                "f1_score": 0.78,
                "roc_auc": 0.85,
                "false_negatives_positive_class": 12,
                "class_metrics": {
                    "0": {"f1_score": 0.74},
                    "1": {"f1_score": 0.81},
                },
            }
        ),
        encoding="utf-8",
    )
    pd.to_pickle(
        pd.DataFrame(
            [
                {"k": 10, "accuracy": 0.80, "f1_score": 0.78, "false_negatives_pct": 0.12, "validation_score": 0.79},
                {"k": 15, "accuracy": 0.81, "f1_score": 0.79, "false_negatives_pct": 0.11, "validation_score": 0.80},
            ]
        ),
        frozen_m1_dir / "k_search.pkl",
    )
    (frozen_m2_k_dir / "k_010.json").write_text(
        json.dumps(
            {
                "model_code": "M2",
                "k": 10,
                "accuracy": 0.75,
                "f1_score": 0.71,
                "false_negatives_pct": 0.18,
                "validation_score": 0.73,
            }
        ),
        encoding="utf-8",
    )

    pd.to_pickle(
        {
            "dataset_df": pd.DataFrame({"accident_id": ["a1", "a2"], "severity_target": [1, 0]}),
            "selected_embedding_cols": ["emb_001", "emb_002"],
            "embedding_meta": {"selected_embedding_count": 2, "transformer_model_label": "bert-demo"},
        },
        raw_build_dir / "payload.pkl",
    )

    live_status = {
        "run_id": "paper_replication_demo",
        "status": "running",
        "result_status": "running",
        "step_id": "raw.build.embeddings",
        "step_status": "running",
        "message": "Generando embeddings [CLS].",
        "progress": manifest["progress"],
        "updated_at": "2026-03-31T10:12:00",
    }
    (run_dir / "live_status.json").write_text(json.dumps(live_status), encoding="utf-8")
    (run_dir / "live_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "paper_replication_demo",
                        "status": "running",
                        "result_status": "running",
                        "step_id": "frozen.dataset_validation",
                        "step_status": "completed",
                        "message": "frozen: dataset validado.",
                        "progress": {
                            "current_stage": "frozen",
                            "current_step_id": "frozen.dataset_validation",
                            "completed_steps": 1,
                            "total_steps": 8,
                            "completed_units": 6.0,
                            "total_units": 100.0,
                        },
                        "updated_at": "2026-03-31T10:02:00",
                    }
                ),
                json.dumps(live_status),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = live_app._read_paper_replication_run(manifest_path)

    assert isinstance(payload["step_df"], pd.DataFrame)
    assert len(payload["step_df"]) == 6
    assert isinstance(payload["route_status_df"], pd.DataFrame)
    assert set(payload["route_status_df"]["stage"].tolist()) == {"frozen", "raw", "raw_build", "compare", "export"}
    frozen_row = payload["route_status_df"].loc[payload["route_status_df"]["stage"] == "frozen"].iloc[0]
    assert frozen_row["status"] == "running"
    assert frozen_row["rows"] == 2070
    assert frozen_row["completed_models"] == 1
    raw_build_row = payload["route_status_df"].loc[payload["route_status_df"]["stage"] == "raw_build"].iloc[0]
    assert raw_build_row["status"] == "running"
    assert raw_build_row["embedding_features"] == 2
    assert isinstance(payload["partial_models_df"], pd.DataFrame)
    assert set(payload["partial_models_df"]["model_code"].tolist()) == {"M1", "M2"}
    m1_row = payload["partial_models_df"].loc[payload["partial_models_df"]["model_code"] == "M1"].iloc[0]
    assert m1_row["status"] == "completed"
    m2_row = payload["partial_models_df"].loc[payload["partial_models_df"]["model_code"] == "M2"].iloc[0]
    assert m2_row["status"] == "partial"
    assert isinstance(payload["k_progress_df"], pd.DataFrame)
    assert set(payload["k_progress_df"]["model_code"].astype(str).unique()) == {"M1", "M2"}
    assert payload["live_events_df"]["progress_pct"].iloc[-1] == pytest.approx(22.0)
    assert payload["current_context"]["current_step_id"] == "raw.build.embeddings"
    assert not payload["compare_summary_df"].empty
    assert not payload["export_summary_df"].empty


def test_read_neural_drift_experiment_run_builds_live_and_partial_frames(tmp_path):
    run_dir = tmp_path / "neural_drift_experiments" / "run_demo"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)

    manifest = {
        "run_id": "run_demo",
        "status": "running",
        "result_status": "running",
        "created_at": "2026-04-11T10:00:00",
        "updated_at": "2026-04-11T10:10:00",
        "progress": {
            "completed_units": 12.0,
            "total_units": 50.0,
            "progress_ratio": 0.24,
        },
        "baseline": {
            "status": "completed",
            "seed_metrics": [
                {
                    "seed": 42,
                    "dev": {"score": 0.71, "monthly_pr_auc_median": 0.74, "monthly_pr_auc_std": 0.03},
                    "holdout": {"score": 0.70, "monthly_pr_auc_median": 0.73, "monthly_pr_auc_std": 0.04},
                }
            ],
        },
        "studies": {
            "adwin": {
                "phases": {
                    "phase_1": {
                        "phase": 1,
                        "status": "running",
                        "n_trials_budget": 40,
                        "completed_trials": 12,
                        "best_value": 0.77,
                        "best_trial_number": 9,
                        "storage_path": str(run_dir / "optuna" / "adwin_phase_1.sqlite"),
                    }
                }
            },
            "neural": {
                "phases": {
                    "phase_1": {
                        "phase": 1,
                        "status": "pending",
                        "n_trials_budget": 40,
                        "completed_trials": 0,
                    }
                }
            },
        },
        "winner": {"study": "neural", "eligible_for_promotion": False},
        "artifacts": {
            "leaderboard_dev": str(artifacts_dir / "leaderboard_dev.csv"),
            "leaderboard_holdout": str(artifacts_dir / "leaderboard_holdout.csv"),
            "monthly_metrics": str(artifacts_dir / "monthly_metrics.csv"),
            "pairwise_stats": str(artifacts_dir / "pairwise_stats.csv"),
            "param_importances": str(artifacts_dir / "param_importances.csv"),
            "pareto": str(artifacts_dir / "pareto.csv"),
            "winner_config": str(artifacts_dir / "winner_config.json"),
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pd.DataFrame(
        [
            {"label": "cumulative", "study": "cumulative", "phase": 0, "dev_score": 0.71},
            {"label": "adwin", "study": "adwin", "phase": 1, "dev_score": 0.77},
        ]
    ).to_csv(artifacts_dir / "leaderboard_dev.csv", index=False)
    pd.DataFrame(
        [
            {"label": "cumulative", "study": "cumulative", "phase": 0, "holdout_score": 0.70},
        ]
    ).to_csv(artifacts_dir / "leaderboard_holdout.csv", index=False)
    pd.DataFrame(
        [
            {"month": "2023-01-01", "split": "holdout", "label": "cumulative", "pr_auc": 0.72},
            {"month": "2023-01-01", "split": "holdout", "label": "adwin", "pr_auc": 0.75},
        ]
    ).to_csv(artifacts_dir / "monthly_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"left": "adwin", "right": "cumulative", "passes_gate": False},
        ]
    ).to_csv(artifacts_dir / "pairwise_stats.csv", index=False)
    pd.DataFrame(
        [
            {"study": "adwin", "phase": 1, "parameter": "adwin_delta", "importance": 0.42},
        ]
    ).to_csv(artifacts_dir / "param_importances.csv", index=False)
    pd.DataFrame(
        [
            {"study": "cumulative", "dev_monthly_pr_auc_median": 0.74, "dev_n_actions": 0.0, "pareto_optimal": True},
        ]
    ).to_csv(artifacts_dir / "pareto.csv", index=False)
    (artifacts_dir / "winner_config.json").write_text(
        json.dumps({"study": "neural", "eligible_for_promotion": False}),
        encoding="utf-8",
    )

    live_status = {
        "timestamp": "2026-04-11T10:09:00",
        "status": "running",
        "result_status": "running",
        "completed_units": 12.0,
        "total_units": 50.0,
        "progress_ratio": 0.24,
        "label": "Ejecutando estudio",
        "detail": "adwin | Fase 1 | trial 12/40",
        "context": {"study": "adwin", "phase": 1},
    }
    (run_dir / "live_status.json").write_text(json.dumps(live_status), encoding="utf-8")
    (run_dir / "live_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-11T10:01:00",
                        "event": "baseline_seed_complete",
                        "payload": {
                            "completed_units": 1.0,
                            "total_units": 50.0,
                            "progress_ratio": 0.02,
                            "label": "Ejecutando baseline cumulative",
                            "detail": "seed 1/3",
                            "context": {"seed": 42},
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-11T10:09:00",
                        "event": "trial_complete",
                        "payload": {
                            "completed_units": 12.0,
                            "total_units": 50.0,
                            "progress_ratio": 0.24,
                            "label": "Ejecutando estudio",
                            "detail": "adwin | Fase 1 | trial 12/40",
                            "context": {"study": "adwin", "phase": 1},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = live_app._read_neural_drift_experiment_run(manifest_path)

    assert payload["manifest"]["run_id"] == "run_demo"
    assert not payload["live_events_df"].empty
    assert payload["live_events_df"]["progress_pct"].iloc[-1] == pytest.approx(24.0)
    assert not payload["phase_status_df"].empty
    assert payload["phase_status_df"]["completed_trials"].iloc[0] == 12
    assert not payload["baseline_seed_df"].empty
    assert payload["baseline_seed_df"]["seed"].iloc[0] == 42
    assert not payload["leaderboard_dev"].empty
    assert not payload["monthly_metrics"].empty
    assert payload["winner_config"]["study"] == "neural"


def test_read_paper_replication_run_surfaces_blocked_raw_and_export_state(tmp_path):
    run_dir = tmp_path / "nlp_in_severity" / "paper_replication" / "paper_replication_blocked"
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "compare").mkdir(parents=True)
    (run_dir / "export").mkdir(parents=True)

    manifest = {
        "run_id": "paper_replication_blocked",
        "status": "completed",
        "result_status": "blocked",
        "created_at": "2026-03-31T11:00:00",
        "updated_at": "2026-03-31T11:10:00",
        "progress": {
            "current_stage": "export",
            "current_step_id": "export.latex_promote",
            "completed_steps": 4,
            "total_steps": 4,
            "completed_units": 100.0,
            "total_units": 100.0,
        },
        "steps_index": {
            "raw.build.embeddings": {
                "step_id": "raw.build.embeddings",
                "stage": "raw_build",
                "description": "Extraccion de embeddings fine-tuneados",
                "status": "blocked",
                "order": 1,
                "completed_at": "2026-03-31T11:03:00",
                "last_message": "No se encontraron modelos fine-tuneados reutilizables para la ruta raw.",
                "artifact_paths": {},
            },
            "compare.routes": {
                "step_id": "compare.routes",
                "stage": "compare",
                "description": "Comparacion frozen vs raw",
                "status": "completed",
                "order": 2,
                "completed_at": "2026-03-31T11:05:00",
                "last_message": "La ruta raw no se pudo completar.",
                "artifact_paths": {"summary": str(run_dir / "compare" / "summary.json")},
            },
            "export.latex_promote": {
                "step_id": "export.latex_promote",
                "stage": "export",
                "description": "Promocion de assets LaTeX",
                "status": "completed",
                "order": 3,
                "completed_at": "2026-03-31T11:08:00",
                "last_message": "La ruta raw no se pudo completar.",
                "artifact_paths": {"payload": str(run_dir / "export" / "payload.json")},
            },
        },
        "step_sequence": [
            "raw.build.embeddings",
            "compare.routes",
            "export.latex_promote",
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pd.to_pickle(
        {
            "status": "blocked",
            "status_message": "No se encontraron modelos fine-tuneados reutilizables para la ruta raw.",
            "route_name": "raw",
            "dataset_validation": {},
            "model_results": [],
        },
        run_dir / "raw" / "route_payload.pkl",
    )
    (run_dir / "compare" / "summary.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "reason": "La ruta raw no se pudo completar.",
                "passed": False,
                "max_numeric_diff": None,
                "tolerance": 0.001,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "export" / "payload.json").write_text(
        json.dumps(
            {
                "latex_promoted": False,
                "result_status": "blocked",
                "candidate_paths": {"metrics.png": "/tmp/metrics.png"},
                "promoted_paths": {},
            }
        ),
        encoding="utf-8",
    )

    payload = live_app._read_paper_replication_run(manifest_path)

    raw_row = payload["route_status_df"].loc[payload["route_status_df"]["stage"] == "raw"].iloc[0]
    assert raw_row["status"] == "blocked"
    compare_row = payload["compare_summary_df"].iloc[0]
    assert compare_row["status"] == "blocked"
    assert "raw" in str(compare_row["reason"]).lower()
    export_row = payload["export_summary_df"].iloc[0]
    assert export_row["result_status"] == "blocked"
    assert bool(export_row["latex_promoted"]) is False


def test_build_drift_live_result_tables_fill_pending_slots():
    manifest = {
        "run_manifest": {
            "strategies": ["static", "adaptive_adwin"],
            "models": ["Random Forest"],
            "balance_modes": ["none", "smote"],
            "repetition_seeds": [11],
            "base_year": 2018,
            "prediction_years": [2019, 2020],
        }
    }
    yearly_df = pd.DataFrame(
        [
            {
                "strategy": "static",
                "iteration": 1,
                "training_year": "2018",
                "prediction_year": 2019,
                "model": "Random Forest",
                "balance_mode": "none",
                "auc": 0.81,
                "pr_auc": 0.77,
                "brier_score": 0.12,
                "f1": 0.63,
                "sensitivity": 0.72,
                "specificity": 0.78,
                "sensitivity_before_calibration": 0.61,
                "specificity_before_calibration": 0.81,
                "sensitivity_after_calibration": 0.72,
                "specificity_after_calibration": 0.78,
                "error_rate": 0.19,
                "training_time_sec": 0.4,
                "threshold": 0.5,
                "n_train": 10,
                "n_test": 10,
                "run_seed": 11,
                "run_order": 1,
            }
        ]
    )
    adaptive_df = pd.DataFrame()
    block_df = pd.DataFrame(
        [
            {
                "status": "pending",
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "smote",
                "detector_variant": "",
                "run_seed": 11,
                "run_order": 1,
            }
        ]
    )

    tables = live_app._build_drift_live_result_tables(
        manifest,
        block_df,
        yearly_df,
        adaptive_df,
    )

    assert len(tables["A.6"]) == 4
    assert tables["A.6"]["status"].tolist().count("Completado") == 1
    assert tables["A.6"]["status"].tolist().count("Pendiente") == 3
    pending_row = tables["A.6"].loc[
        (tables["A.6"]["prediction_year"] == 2020)
        & (tables["A.6"]["balance_mode"] == "none")
    ].iloc[0]
    assert pending_row["auc"] == "Pendiente"
    assert pending_row["pr_auc"] == "Pendiente"
    assert pending_row["brier_score"] == "Pendiente"
    assert pending_row["f1"] == "Pendiente"
    assert pending_row["sensitivity_before_calibration"] == "Pendiente"
    assert pending_row["specificity_after_calibration"] == "Pendiente"
    assert pending_row["training_year"] == "2018"
    completed_row = tables["A.6"].loc[tables["A.6"]["status"] == "Completado"].iloc[0]
    assert completed_row["pr_auc"] == 0.77
    assert completed_row["brier_score"] == 0.12
    assert completed_row["f1"] == 0.63
    assert completed_row["sensitivity_before_calibration"] == 0.61
    assert completed_row["specificity_after_calibration"] == 0.78

    assert len(tables["A.9"]) == 1
    assert tables["A.9"].iloc[0]["status"] == "Pendiente"
    assert tables["A.9"].iloc[0]["model"] == "Random Forest"
    assert tables["A.9"].iloc[0]["pr_auc"] == "Pendiente"
    assert tables["A.9"].iloc[0]["brier_score"] == "Pendiente"
    assert tables["A.9"].iloc[0]["f1"] == "Pendiente"


def test_build_drift_live_result_tables_keeps_completed_kswin_blocks_without_rows():
    manifest = {
        "run_manifest": {
            "strategies": ["adaptive_kswin"],
            "models": ["Random Forest"],
            "balance_modes": ["none"],
            "repetition_seeds": [42],
            "base_year": 2018,
            "prediction_years": [2019, 2020],
        }
    }
    adaptive_df = pd.DataFrame()
    block_df = pd.DataFrame(
        [
            {
                "status": "completed",
                "strategy": "adaptive_kswin",
                "model": "Random Forest",
                "balance_mode": "none",
                "detector_variant": "KSWINpaper",
                "adaptive_rows": 0,
                "run_seed": 42,
                "run_order": 1,
            }
        ]
    )

    tables = live_app._build_drift_live_result_tables(
        manifest,
        block_df,
        pd.DataFrame(),
        adaptive_df,
    )

    assert len(tables["A.9"]) == 1
    assert tables["A.9"].iloc[0]["status"] == "Completado sin filas"
    assert tables["A.9"].iloc[0]["strategy"] == "adaptive_kswin"
    assert tables["A.9"].iloc[0]["detector_variant"] == "KSWINpaper"
    assert tables["A.9"].iloc[0]["segment_rows"] == 0


def test_read_drift_run_infers_error_status_for_empty_failed_payload():
    manifest = {
        "run_manifest": {
            "strategies": ["adaptive_kswin"],
        },
        "block_index": {
            "block_a": {
                "status": "completed",
                "strategy": "adaptive_kswin",
                "model": "Random Forest",
                "balance_mode": "none",
                "detector_variant": "KSWINpaper",
                "run_seed": 42,
                "run_order": 1,
            }
        },
    }
    block_df = pd.DataFrame(
        [
            {
                "status": "error",
                "strategy": "adaptive_kswin",
                "model": "Random Forest",
                "balance_mode": "none",
                "detector_variant": "KSWINpaper",
                "adaptive_rows": 0,
                "run_seed": 42,
                "run_order": 1,
                "error_message": "kswin_base_train_error: tuning resolver mismatch",
            }
        ]
    )

    tables = live_app._build_drift_live_result_tables(
        manifest,
        block_df,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert len(tables["A.9"]) == 1
    assert tables["A.9"].iloc[0]["status"] == "Error"
    assert "resolver mismatch" in str(tables["A.9"].iloc[0]["error_message"])


def test_build_drift_partial_summary_normalizes_kswin_for_existing_comparison_table():
    adaptive_df = pd.DataFrame(
        [
            {
                "strategy": "adaptive_adwin",
                "model": "Random Forest",
                "balance_mode": "none",
                "auc": 0.81,
                "pr_auc": 0.73,
                "brier_score": 0.11,
                "f1": 0.61,
                "sensitivity": 0.72,
                "specificity": 0.78,
                "sensitivity_before_calibration": 0.66,
                "specificity_before_calibration": 0.8,
                "sensitivity_after_calibration": 0.72,
                "specificity_after_calibration": 0.78,
                "error_rate": 0.19,
                "training_time_sec": 0.4,
                "threshold": 0.5,
                "run_seed": 11,
                "run_order": 1,
            },
            {
                "strategy": "adaptive_kswin",
                "model": "Random Forest | KSWINpaper",
                "base_model": "Random Forest",
                "detector_variant": "KSWINpaper",
                "balance_mode": "none",
                "auc": 0.79,
                "pr_auc": 0.69,
                "brier_score": 0.13,
                "f1": 0.58,
                "sensitivity": 0.7,
                "specificity": 0.76,
                "sensitivity_before_calibration": 0.64,
                "specificity_before_calibration": 0.79,
                "sensitivity_after_calibration": 0.7,
                "specificity_after_calibration": 0.76,
                "error_rate": 0.21,
                "training_time_sec": 0.6,
                "threshold": 0.45,
                "run_seed": 11,
                "run_order": 1,
            },
        ]
    )

    summary = live_app._build_drift_partial_summary(
        pd.DataFrame(),
        adaptive_df,
    )

    assert len(summary) == 2
    assert set(summary["strategy"].tolist()) == {"adaptive_adwin", "adaptive_kswin"}

    adwin_row = summary.loc[summary["strategy"] == "adaptive_adwin"].iloc[0]
    assert adwin_row["model"] == "Random Forest"
    assert adwin_row["detector_variant"] == "-"
    assert adwin_row["auc"] == 0.81
    assert adwin_row["pr_auc"] == 0.73
    assert adwin_row["brier_score"] == 0.11
    assert adwin_row["f1"] == 0.61
    assert adwin_row["sensitivity_before_calibration"] == 0.66
    assert adwin_row["specificity_after_calibration"] == 0.78

    kswin_row = summary.loc[summary["strategy"] == "adaptive_kswin"].iloc[0]
    assert kswin_row["model"] == "Random Forest"
    assert kswin_row["detector_variant"] == "KSWINpaper"
    assert kswin_row["auc"] == 0.79
    assert kswin_row["pr_auc"] == 0.69
    assert kswin_row["brier_score"] == 0.13
    assert kswin_row["f1"] == 0.58
    assert kswin_row["sensitivity_before_calibration"] == 0.64
    assert kswin_row["specificity_after_calibration"] == 0.76
